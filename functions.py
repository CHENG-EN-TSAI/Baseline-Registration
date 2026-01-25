import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage.morphology import skeletonize
from skimage.transform import resize, warp
import matplotlib.pyplot as plt

__all__ = ["apply_field", "pre_processing", "compute_ICE", "save_deformation_grid", "save_fold_overlay", "plot_def_field", "print_stats"]

def apply_field(im, u_fwd_fix, order=1):
  X, Y = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]), indexing='ij')
  im_warp = warp(im, np.array([X + u_fwd_fix[0], Y + u_fwd_fix[1]]), order=order, preserve_range=True)
  return im_warp

def pre_processing(im_f, im_m, out_hw=(256,256)):
    im_f = resize(im_f, out_hw, preserve_range=True).astype(np.float32)
    im_m = resize(im_m, out_hw, preserve_range=True).astype(np.float32)
    if im_f.ndim == 3: im_f = 0.33*im_f[...,0] + 0.34*im_f[...,1] + 0.33*im_f[...,2]
    if im_m.ndim == 3: im_m = 0.33*im_m[...,0] + 0.34*im_m[...,1] + 0.33*im_m[...,2]
    # normalize to [0,1]
    def norm01(a): 
        mn, mx = float(a.min()), float(a.max()); return (a - mn) / max(mx - mn, 1e-8)
    return norm01(im_f), norm01(im_m)

def compute_ICE(u_fwd_fix, u_bwd_mov, spacing_fix=(1.0,1.0), spacing_mov=(1.0,1.0)):
    """
    ICE = mean_{(r,c) in valid} || u_fwd_fix(r,c) + u_bwd_mov(r',c') ||_2
    with (r',c') = (r + u_fwd_fix_row, c + u_fwd_fix_col)

    u_fwd_fix: (2,Hf,Wf) forward field (fixed→moving) in *pixels* [dr, dc] on FIXED grid
    u_bwd_mov: (2,Hm,Wm) backward field (moving→fixed) in *pixels* [dr, dc] on MOVING grid
    spacing_*: (sr, sc) pixel spacings (row, col) to optionally compute ICE in physical units
    """
    uf = u_fwd_fix.astype(np.float32)
    ub = u_bwd_mov.astype(np.float32)

    Hf, Wf = uf.shape[1:]
    Hm, Wm = ub.shape[1:]

    R, C = np.indices((Hf, Wf), dtype=np.float32)  # rows, cols on FIXED grid
    Rq = R + uf[0]   # query rows on MOVING grid   #!
    Cq = C + uf[1]   # query cols on MOVING grid   #!

    valid = (Rq >= 0) & (Rq <= Hm - 1) & (Cq >= 0) & (Cq <= Wm - 1)
    if not np.any(valid):
        return float("nan")
    
    ub_at = np.zeros_like(uf)
    ub_at[0] = warp(ub[0], np.array([Rq, Cq]), order=1, preserve_range=True)
    ub_at[1] = warp(ub[1], np.array([Rq, Cq]), order=1, preserve_range=True)
    
    res = uf + ub_at  # ideal inverse-consistent ⇒ res ≈ 0

    # If you want ICE in physical units, scale components by spacings before the norm:
    sr_f, sc_f = spacing_fix
    # (Assuming uf/ub are in pixels of their own grids; if spacings differ, ideally
    # convert both to a common physical space before combining)
    res_phys_r = res[0] * sr_f
    res_phys_c = res[1] * sc_f

    ice_map = np.sqrt(res_phys_r**2 + res_phys_c**2)
    return float(ice_map[valid].mean())

def save_deformation_grid(
    u: np.ndarray,
    v: np.ndarray,
    out_path: str,
    background = None,
    spacing: int = 32,
    alpha_bg: float = 1.0,
    dpi: int = 200,
):
    """
    Save a deformation grid visualization from a 2D displacement field.

    Parameters
    ----------
    u, v : np.ndarray
        Displacement fields of shape (H, W), where
        phi(x, y) = (x + u[y, x], y + v[y, x]).
        (i.e., u is x-displacement, v is y-displacement)
    out_path : str
        Output image path, e.g., "grid.png".
    background : np.ndarray | None
        Optional background image (H, W) to plot underneath (e.g., warped image).
    spacing : int
        Grid line spacing in pixels. Typical: 20~25.
    linewidth : float
        Line width for grid.
    alpha_grid : float
        Opacity for grid lines.
    alpha_bg : float
        Opacity for background.
    trim_top : int
        Crop `trim_top` pixels from the top of the saved figure (simple post-crop).
        Set 0 to disable.
    dpi : int
        Save resolution.

    Notes
    -----
    - The saved image has no margins (bbox_inches='tight', pad_inches=0).
    - Uses image coordinate convention (origin at top-left) via invert_yaxis().
    """
    if u.shape != v.shape or u.ndim != 2:
        raise ValueError(f"u and v must be 2D arrays with the same shape, got {u.shape} and {v.shape}.")

    H, W = u.shape
    

    grid = np.zeros((H, W), dtype=np.float32)
    grid[::spacing, :] = 1.0
    grid[:, ::spacing] = 1.0
    grid[1::spacing, :] = 1.0
    grid[:, 1::spacing] = 1.0
    grid[2::spacing, :] = 1.0
    grid[:, 2::spacing] = 1.0
    grid[-1, :] = 1.0
    grid[:, -1] = 1.0
    
    warped_grid = skeletonize(
        apply_field(grid, np.stack([u, v], axis=0), order=1)
    )

    # Plot
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=dpi)  # scale with image size
    ax = plt.gca()

    if background is not None:
        if background.shape != (H, W):
            raise ValueError(f"background must have shape {(H, W)}, got {background.shape}.")
        ax.imshow(background, cmap="gray", alpha=alpha_bg)
        
    ax.contour(
        warped_grid,
        levels=[0.8],
        colors=["r"],
        linewidths=0.5,
        origin="upper",
        extent=(0, W, H, 0),
    )
    
    ax.invert_yaxis()   # match image coordinates
    ax.set_xlim([0, W - 1])
    ax.set_ylim([H - 1, 0])
    ax.axis("off")

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def save_fold_overlay(
    warped: np.ndarray,          # (H, W), float or uint
    u: np.ndarray,           # (2, H, W), float
    D_J: np.ndarray,          # (H, W), float
    out_path: str,
    alpha: float = 0.3,
    dpi: int = 250,
):
    """
    Overlays fold mask on warped image and saves with no margins.
    fold_mask: True where folding occurs (e.g., J_phi <= 0).
    """

    H, W = warped.shape
    fold_mask = apply_field(D_J <= 0, u, order=0).astype(bool)
    p95 = apply_field(D_J >= np.percentile(D_J, 95), u, order=0).astype(bool)

    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=dpi)

    # Background (lock to pixel boundary box)
    ax.imshow(warped, cmap="gray", extent=(0, W, H, 0), interpolation="nearest")

    # Overlay (use a masked array so only folding pixels are colored)
    overlay = np.ma.masked_where(~fold_mask, fold_mask.astype(float))
    ax.imshow(overlay, cmap="autumn", alpha=alpha, extent=(0, W, H, 0), interpolation="nearest")

    overlay2 = np.ma.masked_where(~p95, p95.astype(float))
    ax.imshow(overlay2, cmap="winter", alpha=alpha, extent=(0, W, H, 0), interpolation="nearest")
    
    legend_elements = [
        Patch(facecolor='red', edgecolor='red',
            label=r'$J_\phi \leq 0$'),
        Patch(facecolor='blue', edgecolor='blue',
            label=r'$J_\phi \geq J_{\phi,p95}$')
    ]

    ax.legend(
        handles=legend_elements,
        loc='upper right',
        frameon=True,
        fontsize=8
    )

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, pad_inches=0)
    plt.close(fig)
    
def plot_def_field(def_field, background, name, step=17):
    """
    def_field: (2, H, W) array, image convention
        def_field[0] = u (x-displacement, +right)
        def_field[1] = v (y-displacement, +down)
    """
    u, v = def_field[0], def_field[1]
    H, W = u.shape

    # coordinate grid in image convention
    Y, X = np.meshgrid(
        np.arange(H, dtype=np.float32),  # rows
        np.arange(W, dtype=np.float32),  # cols
        indexing='ij'
    )

    # subsample
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    us = v[::step, ::step]
    vs = -u[::step, ::step]

    plt.imshow(background, vmin=0)
    plt.axis("off")
    plt.xlim([-24, W+24])
    plt.ylim([H+24, -25])
    plt.colorbar()
    plt.clim(vmin=0, vmax=2.6)
    
    plt.quiver(Xs, Ys, us, vs, color="r", angles="uv", units="xy", headwidth=4)
    plt.savefig(name, bbox_inches="tight", pad_inches=0)
    plt.close()
    
def print_stats(name, vals):
    if len(vals)==0: 
        print(f"{name}: n=0")
        return
    a = np.array(vals, dtype=np.float64)
    print(f"{name}: mean={a.mean():.6f}, std={a.std(ddof=0):.6f}, n={len(a)}")