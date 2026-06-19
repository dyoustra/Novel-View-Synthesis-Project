**Description of Novel view synthesis:**

Novel View Synthesis (NVS) aims to generate realistic images of a scene from camera viewpoints that were never observed. Modern approaches learn a continuous 3D representation of scene geometry and appearance from a sparse set of input images and render new views via differentiable rendering.

Key papers:

NeRF: Neural Radiance Fields — Mildenhall et al., ECCV 2020

Instant-NGP (hash-grid NeRF acceleration) — Müller et al., SIGGRAPH 2022

Mip-NeRF / Mip-NeRF 360 (anti-aliasing, unbounded scenes) — Barron et al., CVPR 2021/2022

3D Gaussian Splatting — Kerbl et al., SIGGRAPH 2023

**Your task:**

You are given a video of a robot: [link](https://drive.google.com/file/d/1dOTUjctusAi7KAZu2O5lOfMSV6UVdcLD/view?usp=sharing).

Your goal is to generate images of the robot from **novel viewpoints**—that is, viewpoints not present in the input video. Example outputs can be seen [*here*](https://drive.google.com/drive/folders/1mParg2BcgN9Tn13mBd92OHdvLIalfkF-?usp=drive_link). You may refer to the papers above for common definitions and examples of such novel views. We leave the precise interpretation of “novel view” up to the reader, as the definition depends on which direction they pursue from the options below.

**Recommended directions:**

1. Using Gaussian Splatting based methods: [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079):  
     
   In this case, you would first need to run a Structure from Motion (SFM)  pipeline to get the camera poses and an initial pointcloud, we recommend using [COLMAP](https://github.com/colmap/colmap) for obtaining the camera poses. After running COLMAP, you can run any 3DGS method, we recommend: [gsplat](https://github.com/nerfstudio-project/gsplat) or [wild-gaussians.](https://github.com/jkulhanek/wild-gaussians/tree/main/wildgaussians)   
     
   These methods fit a set of 3D anisotropic Gaussians to the reconstructed scene and render new views efficiently using differentiable rasterization.  
     
2. Alternatively, you may use diffusion models that enable NVS directly from one or a few input images, without requiring a full 3D reconstruction pipeline.

Recommended works include:

\[A\] ZeroNVS: Zero-shot 360-degree view synthesis from a single real image, CVPR 2024

\[B\] From an Image to a Scene: Learning to Imagine the World from a Million 360° Videos, NeurIPS 2024

These models learn priors from large video datasets and can hallucinate plausible novel viewpoints even when geometric information is limited.

**Submission Instructions:**

We expect the following items for this assignment:

- A github repository with instructions on how to run the novel view synthesis pipeline, it is okay to include third party packages and assume we would be able to install them, please include brief installation instructions.  
- A pdf report explaining: a) the approach taken and steps performed by your method b) results  on at least 25 images (we expect 4 NVS images per image) c) failure cases and d) short discussion on limitations and how to address them  
  