# Report checklist (PDF deliverable)
- (a) Approach & steps: pipeline diagram (frames→masks→COLMAP→{gsplat,wildgs}→render; +ZeroNVS)
- (b) Results: grid of 25 refs × 4 novel views for gsplat AND wild-gaussians;
      metrics table (mean PSNR/SSIM/LPIPS from outputs/*/eval/metrics.json)
- (b) Comparison figures: gsplat vs wild-gaussians (arm handling); 3DGS vs ZeroNVS (same ref+angle)
- (c) Failure cases: arm residue/floaters, extrapolation breakdown, ZeroNVS identity drift
- (d) Limitations & fixes: sparse coverage, dynamic content, diffusion geometry inconsistency
