import numpy as np
import SimpleITK as sitk
import ants
import pandas as pd

################################################################
###  Simple ITK backend for 2D affine + BSpline registration ###
################################################################

def np_to_sitk(img_np, spacing=(1.0,1.0)):
    img = sitk.GetImageFromArray(img_np)     # rows, cols -> y, x
    img.SetSpacing(tuple(map(float, spacing)))  # (sx, sy) in x,y
    img.SetOrigin((0.0, 0.0))
    img.SetDirection((1.0,0.0,0.0,1.0))
    return img

def sitk_to_np(img_sitk):
    return sitk.GetArrayFromImage(img_sitk).astype(np.float32)

def register_affine_bspline(fixed_sitk, moving_sitk, bspline_mesh=(4,4), bspline_order=3, seed=False):
    """
    Two-stage: Affine -> BSpline. Returns:
      affine_out (AffineTransform),
      bspline_out (BSplineTransform),
      final_composite (CompositeTransform = bspline ∘ affine).
    """
    try:
        # --- Affine stage ---
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(50)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.2, seed=seed)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsGradientDescent(learningRate=0.5,
                                        numberOfIterations=600,
                                        convergenceMinimumValue=1e-6,
                                        convergenceWindowSize=10)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel([4,2,1])
        R.SetSmoothingSigmasPerLevel([2,1,0]); R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        affine = sitk.AffineTransform(2)
        R.SetInitialTransform(affine, inPlace=False)
        affine_out = R.Execute(fixed_sitk, moving_sitk)

        # --- BSpline stage (optimize BSpline, keep affine as moving initial) ---
        bspline_init = sitk.BSplineTransformInitializer(image1=fixed_sitk,
                                                        transformDomainMeshSize=bspline_mesh,
                                                        order=bspline_order)
        R2 = sitk.ImageRegistrationMethod()
        R2.SetMetricAsMattesMutualInformation(50)
        R2.SetMetricSamplingStrategy(R2.RANDOM)
        R2.SetMetricSamplingPercentage(0.2, seed=seed)
        R2.SetInterpolator(sitk.sitkLinear)
        R2.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                                numberOfIterations=200,
                                maximumNumberOfCorrections=5)
        R2.SetShrinkFactorsPerLevel([4,2,1])
        R2.SetSmoothingSigmasPerLevel([2,1,0]); R2.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # VERY IMPORTANT: set the transform to optimize + the moving-initial (affine)
        R2.SetInitialTransform(bspline_init, inPlace=False)     # optimize BSpline params
        R2.SetMovingInitialTransform(affine_out)                # keep affine applied before BSpline

        bspline_out = R2.Execute(fixed_sitk, moving_sitk)

        # Build the final composite: apply affine, then bspline (order matters)
        final_composite = sitk.CompositeTransform(2)
        final_composite.AddTransform(affine_out)   # first
        final_composite.AddTransform(bspline_out)  # then BSpline
        return final_composite
        
    except Exception:
        return None

def resample_image(moving_sitk, fixed_sitk, transform, is_label=False):
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    return sitk.Resample(moving_sitk, fixed_sitk, transform, interp, 0.0, moving_sitk.GetPixelID())

def transform_to_disp_px(transform, reference_image):
    """
    Displacement field in pixel units (on 'reference_image' grid).
    Returns (2,H,W) with [0]=dx (cols, +right), [1]=dy (rows, +down).
    """
    df = sitk.TransformToDisplacementFieldFilter()
    df.SetReferenceImage(reference_image)
    DF = df.Execute(transform)
    disp_np_mm = sitk.GetArrayFromImage(DF).astype(np.float32)
    sx, sy = reference_image.GetSpacing() 
    dx = disp_np_mm[:, :, 1] / max(sx, 1e-8)
    dy = disp_np_mm[:, :, 0] / max(sy, 1e-8)
    return np.stack([dx, dy], axis=0)

def run_sitk_registration(im_f, im_m, seed):
    # ---- SITK images (unit spacing so pixel == mm for convenience here) ----
    fixed_img  = np_to_sitk(im_f, spacing=(1.0,1.0))
    moving_img = np_to_sitk(im_m, spacing=(1.0,1.0))

    # ---- forward registration: fixed(im_f) <- moving(im_m) ----
    final_tx_fwd = register_affine_bspline(fixed_img, moving_img, bspline_mesh=(4,4), bspline_order=3, seed=seed)
    if final_tx_fwd is None:
        return None, None

    warped_img = resample_image(moving_img, fixed_img, final_tx_fwd, is_label=False)
    warped  = sitk_to_np(warped_img)
    
    # ---- build u_fwd on FIXED grid (fixed->moving) in pixel units ----
    u_fwd = transform_to_disp_px(final_tx_fwd, fixed_img)  # (2,H,W), [dx,dy]
    return warped, u_fwd


######################################################
###  ANTs backend for 2D affine + SyN registration ###
######################################################
    
def displacement_from_transforms(h, w, transform_list):    
    X, Y = np.meshgrid(np.arange(0, h, dtype=np.float32),
                np.arange(0, w, dtype=np.float32),
                indexing='ij')
    pts_df = pd.DataFrame({"x": X.ravel(), "y": Y.ravel()})

    if len(transform_list) == 2:
        affine_df = ants.apply_transforms_to_points(dim=2, points=pts_df, transformlist=transform_list[1:], whichtoinvert=[False])
        warp = ants.image_read(transform_list[0]).numpy()
        
    elif len(transform_list) == 1:
        affine_df = ants.apply_transforms_to_points(dim=2, points=pts_df, transformlist=transform_list, whichtoinvert=[False])


    Xw = affine_df["x"].to_numpy().reshape(Y.shape)
    Yw = affine_df["y"].to_numpy().reshape(Y.shape)
    
    if len(transform_list) == 2:
        u = (Xw - X).astype(np.float32) + warp[:, :, 0]
        v = (Yw - Y).astype(np.float32) + warp[:, :, 1]

    elif len(transform_list) == 1:
        u = (Xw - X).astype(np.float32)
        v = (Yw - Y).astype(np.float32)

    return np.stack([u, v], axis=0)

def run_ants_registration(im_f, im_m, args):
    H, W = im_f.shape
       
    # ---- ANTs registration ----
    reg = ants.registration(ants.from_numpy(im_f), ants.from_numpy(im_m), type_of_transform=args.method, random_seed=args.seed)
    
    # ---- warp moving to fixed ----
    warped_img = ants.apply_transforms(fixed=ants.from_numpy(im_f),
                                        moving=ants.from_numpy(im_m),
                                        transformlist=reg['fwdtransforms'],
                                        interpolator='linear')
    warped  = warped_img.numpy()
        
    u = displacement_from_transforms(H, W, reg['fwdtransforms'])
    
    return warped, u