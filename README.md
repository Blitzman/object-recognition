Tool for object recognition. A single PCD cloud model is matched against a PCD scene and the correspondences are grouped by using a Hough 3D Grouping algorithm together with a Global Hypotheses Verification step.

Instructions:

```
#!c++

Allowed options:
  --h                             produce help message
  --model arg                     Model filename
  --model_n arg (=0)              Load model with normals
  --scene arg                     Scene filename
  --v_n arg (=0)                  Show normals on visualizer
  --v_k arg (=1)                  Show keypoints in visualizer
  --v_c arg (=1)                  Show correspondences in visualizer
  --r arg (=0)                    Use cloud resolution
  --bf arg (=0)                   Apply bilateral filter to scene
  --bf_sr arg (=0.0500000007)     Bilateral filter sigma R
  --bf_ss arg (=15)               Bilateral filter sigma S
  --ne_k arg (=25)                Normal estimation radius
  --mps arg (=1)                  Activate Multi Plane Segmentation
  --mps_mi arg (=100)             Minimum inliers for plane segmentation
  --mps_at arg (=2)               Angular threshold for plane segmentation
  --mps_dt arg (=2)               Distance threshold for plane segmentation
  --vg_m arg (=0)                 Downsample model using Voxel Grid
  --vg_m_ls arg (=0.00999999978)  Model downsampling leaf size
  --vg_s arg (=0)                 Downsample scene using Voxel Grid
  --vg_s_ls arg (=0.00999999978)  Scene downsampling leaf size
  --iss arg (=0)                  Use ISS for keypoint extraction
  --iss_sr arg (=6)               ISS salient radius
  --iss_nmr arg (=4)              ISS non max radius
  --iss_mn arg (=5)               ISS minimum number of neighbors
  --iss_t21 arg (=0.975000024)    ISS threshold 21
  --iss_t32 arg (=0.975000024)    ISS threshold 32
  --sift arg (=0)                 Use SIFT for keypoint extraction
  --sift_ms arg (=0.000500000024) SIFT minimum scale
  --sift_no arg (=8)              SIFT number of scales
  --sift_spo arg (=8)             SIFT scales per octave
  --sift_mc arg (=0.00499999989)  SIFT minimum contrast
  --m_ss arg (=0.0199999996)      Model search radius
  --s_ss arg (=0.0199999996)      Scene search radius
  --d_r arg (=0.0199999996)       Descriptor radius
  --d_lrf_r arg (=0.0149999997)   Descriptor Reference Frame radius
  --m_t arg (=0.25)               Correspondence matching threshold
  --cg_h arg (=0)                 Use Hough3D for correspondence grouping
  --cg_cs arg (=0.00999999978)    Correspondence grouping cluster size
  --cg_ct arg (=5)                Correspondence grouping cluster threshold
  --icp arg (=0)                  Align clouds with ICP
  --icp_i arg (=5)                Maximum iterations for ICP
  --icp_d arg (=0.00499999989)    Maximum correspondence distance for ICP
  --hv_it arg (=0.00499999989)    Hypothesis verification inlier threshold
  --hv_ot arg (=0.00999999978)    Hypothesis verification occlusion threshold
  --hv_rc arg (=0.0299999993)     Hypothesis verification radius clutter
  --hv_cr arg (=5)                Hypothesis verification clutter regularizer
  --hv_r arg (=3)                 Hypothesis verification regularizer
  --hv_rn arg (=0.0500000007)     Hypothesis verification radius normals
  --hv_dc arg (=1)                Hypothesis verification clutter detection


```

AFAIK the best combination of parameters is:

```
.\main.exe --model .\clouds\models\tasmanian_cloud_downsampled_normals_25k_s1.pcd --model_n 1 --scene .\clouds\scenes\multi\scan_9.pcd --r 1 --m_ss 100 --s_ss 100 --d_lrf_r 20 --d_r 20 --cg_cs 6.5 --cg_ct 6 --v_n 0 --ne_k 25 --cg_h 1 --m_t 0.4 --icp_i 5 --v_k 0 --hv_it 0.02 --mps 0 --mps_mi 100 --mps_at 10.0 --mps_dt 0.004 --bf 0

``` 