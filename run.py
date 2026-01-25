import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PImage
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_mutual_information as nmi

from functions import (
    apply_field, pre_processing, compute_ICE,
    save_deformation_grid, plot_def_field, save_fold_overlay, print_stats
)

from backends import (
    run_ants_registration,
    run_sitk_registration
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_paths_json", default='test_data/test.json')
    parser.add_argument("--img_size", default=[256, 256], type=int, nargs=2)
    parser.add_argument("--method", default="Affine", choices=["Affine", "SyN", "BSpline"])
    parser.add_argument("--seed", default=False, type=int)
    parser.add_argument("--result_path", default=None)

    args = parser.parse_args()

    if args.result_path is None:
        args.result_path = f"{args.method}_results"
    
    if os.path.exists(args.result_path):
        print(f"result_path --{args.result_path}-- already exists, previous results may be overwritten")    
    os.makedirs(args.result_path, exist_ok=True)
    
    # ---------- load dataset ----------
    assert os.path.exists(args.data_paths_json), "data_paths_json does not exist"
    with open(args.data_paths_json) as f:
        data = json.load(f)
    
    # --- metric collectors ---
    SSIM_list, NMI_list, ICE_list = [], [], []
    TV_mean_list, TV_p95_list = [], []
    DJ_fold_list, DJ_p95_list = [], []
    DJ_p5_list = []
    MSE_DJ_c_list = []
    
    # ---------- loop ----------
    for i in range(len(data)):
        os.makedirs(f"{args.result_path}/{i}", exist_ok=True)

        im_f = np.array(PImage.open(data[i]["fixed"]), dtype=np.float32)
        im_m = np.array(PImage.open(data[i]["moving"]), dtype=np.float32)

        try:
            fixed_spacing = np.asarray(data[i]["fixed_pixel_spacing"], dtype=np.float64)
            moving_spacing = np.asarray(data[i]["moving_pixel_spacing"], dtype=np.float64)
            
            Hf0, Wf0 = im_f.shape
            Hm0, Wm0 = im_m.shape
            
            fixed_area_phys = (Hf0 * Wf0) * (fixed_spacing[0] * fixed_spacing[1])
            moving_area_phys = (Hm0 * Wm0) * (moving_spacing[0] * moving_spacing[1])

            c_val = float(fixed_area_phys / (moving_area_phys + 1e-12))
            
            seg_m = PImage.open(data[i]["moving_seg"]).convert("L").resize(args.img_size, resample=PImage.NEAREST)
            seg_m = np.array(seg_m, dtype=np.float32)     
            seg_m = np.where(seg_m==255, 1, 0)

        except:
            c_val = None
            
        im_f, im_m = pre_processing(im_f, im_m, args.img_size)
        
        if args.method == "SyN" or args.method == "Affine":
            warped, u_fwd = run_ants_registration(im_f, im_m, args)

            # --- reverse registration for method-consistency ICE ---
            _, u_bwd = run_ants_registration(im_m, im_f, args)
            
        elif args.method == "BSpline":
            # ---- forward registration: moving(im_m) -> fixed(im_f) ----
            warped, u_fwd = run_sitk_registration(im_f, im_m, args.seed)
            if warped is None:
                continue
            
            # ---- reverse registration for ICE: fixed(im_m) <- moving(im_f) ----
            _, u_bwd = run_sitk_registration(im_m, im_f, args.seed)
            if u_bwd is None:
                continue
        
        # --- compute TV, det Jacobian ---
        u, v = u_fwd[0], u_fwd[1]
        ux, uy = np.gradient(u)
        vx, vy = np.gradient(v)
        TV = np.sqrt(ux**2 + uy**2 + vx**2 + vy**2)
        TV_mean = float(TV.mean())
        TV_p95  = float(np.percentile(TV, 95))
        D_J = (1 + ux) * (1 + vy) - uy * vx
        DJ_fold = float((D_J <= 0).mean() * 100.0)
        DJ_p5  = float(np.percentile(D_J, 5))
        DJ_p95  = float(np.percentile(D_J, 95))
        
        ######################################
        #### ---- Save GIF animation ---- ####
        ######################################
        frames = []
        frames.append(PImage.fromarray(im_f*255))
        frames.append(PImage.fromarray(warped*255))
        frames[0].save(os.path.join(args.result_path, f"{i}", 'anime.gif'), save_all=True, append_images=frames[1:], duration=500, loop=0)
        
        
        ##############################
        #### ---- Save image ---- ####
        ##############################
        im_w = apply_field(im_m, u_fwd, order=0)

        plt.imshow(im_w, cmap="gray")
        plt.axis("off")
        plt.savefig(os.path.join(args.result_path, f"{i}", 'im_w'),bbox_inches="tight", pad_inches=0)
        plt.close()
        
        if args.method == "BSpline":
            # --- save fold overlay ---
            save_fold_overlay(im_w,
                              u_fwd,
                              D_J,
                              os.path.join(args.result_path, f"{i}", "fold_overlay.png")
                              )
        
        # -------------- Save deformation field --------------
        plot_def_field(u_fwd, TV, os.path.join(args.result_path, f"{i}", "df"))
        
        # -------------- Save deformation grid --------------
        save_deformation_grid(
            u=u_fwd[0],
            v=u_fwd[1],
            out_path=os.path.join(args.result_path, f"{i}", "deformation_grid.png"),
            background=im_w,
            spacing=32,
            alpha_bg=1.0,
            dpi=200,
        )
        
        ############################################
        #### ---- Compute and save metrics ---- ####
        ############################################
        valid = np.where(apply_field(np.ones_like(im_m), u_fwd, order=0)==1, True, False)
        _, SSIM_img = ssim(im_f, warped, data_range=1.0, full=True)
        SSIM_val = float(np.mean(SSIM_img[valid]))
        NMI_val  = float(nmi(im_f[valid], warped[valid]))
        
        if c_val is None:
            MSE_DJ_c = float('nan')
        else:
            MSE_DJ_c = float(((D_J - c_val)[seg_m==1]**2).mean())

        ice_val = compute_ICE(u_fwd, u_bwd)
        
        # accumulate
        SSIM_list.append(SSIM_val); NMI_list.append(NMI_val); ICE_list.append(ice_val)
        TV_mean_list.append(TV_mean); TV_p95_list.append(TV_p95)
        DJ_fold_list.append(DJ_fold); DJ_p95_list.append(DJ_p95); DJ_p5_list.append(DJ_p5)
        MSE_DJ_c_list.append(MSE_DJ_c)
        
        # save metrics.json (optional)
        path = os.path.join(args.result_path, f"{i}")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metrics.json"), 'w') as f:
            json.dump({
                "SSIM": SSIM_val, "NMI": NMI_val, "ICE": ice_val,
                "TV_mean": TV_mean, "TV_p95": TV_p95,
                "DJ_fold%": DJ_fold, "DJ_p95": DJ_p95, "DJ_p5": DJ_p5,
                "MSE[D_J-c]": MSE_DJ_c, "c": float(c_val)
            }, f, indent=2)

    print_stats("SSIM", SSIM_list)
    print_stats("NMI", NMI_list)
    print_stats("ICE", ICE_list)
    print_stats("TV_mean", TV_mean_list)
    print_stats("TV_p95", TV_p95_list)
    print_stats("DJ_fold%", DJ_fold_list)
    print_stats("DJ_p95", DJ_p95_list)
    print_stats("DJ_p5", DJ_p5_list)
    print_stats("MSE[D_J-c]", MSE_DJ_c_list)

if __name__ == "__main__":
    main()
