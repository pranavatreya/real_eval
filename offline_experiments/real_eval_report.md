# Real-Eval snapshot (2025-04-25 15:24)

| model | accuracy | Pearson | 1 − MMRV |
|-------|:-------:|:-------:|:--------:|
| Davidson | 0.500 | 0.939 | 0.961 |
| BT | 0.455 | 0.980 | 0.976 |
| Elo | 0.409 | 0.890 | 0.956 |
| Policy-task | 0.500 | 0.993 | 0.998 |
| Hybrid | 0.364 | 0.948 | 0.951 |

## Leaderboards (BT coefficients)

### Davidson
 1. paligemma_fast_droid (0.406)
 2. paligemma_diffusion_droid (0.378)
 3. paligemma_fast_specialist_droid (0.163)
 4. pi0_fast_droid (0.140)
 5. pi0_droid (0.051)
 6. paligemma_vq_droid (-0.060)
 7. paligemma_binning_droid (-1.078)

### BT (no ties)
 1. paligemma_fast_droid (0.243)
 2. paligemma_diffusion_droid (0.209)
 3. pi0_fast_droid (0.190)
 4. paligemma_fast_specialist_droid (0.100)
 5. paligemma_vq_droid (0.053)
 6. pi0_droid (-0.003)
 7. paligemma_binning_droid (-0.792)

### Elo (online)
 1. paligemma_diffusion_droid (1307.529)
 2. paligemma_fast_specialist_droid (1276.817)
 3. paligemma_fast_droid (1233.900)
 4. pi0_fast_droid (1224.219)
 5. pi0_droid (1193.691)
 6. paligemma_vq_droid (1172.927)
 7. paligemma_binning_droid (990.918)

### Policy-task EM
 1. pi0_fast_droid (0.224)
 2. paligemma_fast_droid (0.187)
 3. paligemma_fast_specialist_droid (0.168)
 4. paligemma_diffusion_droid (0.125)
 5. paligemma_vq_droid (0.034)
 6. pi0_droid (-0.089)
 7. paligemma_binning_droid (-0.649)

### Hybrid EM
 1. paligemma_fast_specialist_droid (60136442129.029)
 2. pi0_fast_droid (37858917705.531)
 3. paligemma_diffusion_droid (26858917717.220)
 4. pi0_droid (18281393318.845)
 5. paligemma_fast_droid (11797679918.958)
 6. paligemma_vq_droid (-1524795671.204)
 7. paligemma_binning_droid (-153408555118.379)

## Task buckets τ (Policy-task EM)

* **Bucket 0 (τ=0.22)** “do absolutely nothing. do not move…”  `evaluation_data/25c0a175-ad1c-468e-b55e-e1029f26d94e/pi0_droid_2025_04_15_12_26_45_video_left.mp4`
* **Bucket 1 (τ=0.14)** “pick up the red box…”  `evaluation_data/214e965c-cfe4-418b-8f88-41ee94939fe4/paligemma_fast_specialist_droid_2025_04_15_11_16_17_video_left.mp4`
* **Bucket 2 (τ=0.25)** “find the fruit…”  `evaluation_data/9c7734f2-1eb4-408e-bc3e-bb07a4f3c757/paligemma_binning_droid_2025_04_16_01_16_39_video_left.mp4`
* **Bucket 3 (τ=0.11)** “just touch the red box and nothing else…”  `evaluation_data/d80e7555-39aa-44e3-8858-333a5034b07b/paligemma_vq_droid_2025_04_15_12_07_31_video_left.mp4`
* **Bucket 4 (τ=-0.12)** “place the cup next to the frog…”  `evaluation_data/9b5f7130-d139-49f2-87fb-45dc8a47ad48/paligemma_vq_droid_2025_04_17_11_42_27_video_left.mp4`
* **Bucket 5 (τ=0.06)** “pick up the brown bear…”  `evaluation_data/81baf7e7-80eb-4901-8bf1-48bc66db77ab/paligemma_fast_specialist_droid_2025_04_15_11_38_10_video_left.mp4`
* **Bucket 6 (τ=-0.34)** “just knock off the green frog off the brown box and nothing …”  `evaluation_data/cd3628b2-6029-4c6e-b34b-094763cd934f/paligemma_diffusion_droid_2025_04_15_12_16_06_video_left.mp4`
* **Bucket 7 (τ=-0.32)** “please touch two different books…”  `evaluation_data/6e4a029a-24a3-4d7e-beca-88d8d439ed26/pi0_droid_2025_04_15_13_00_32_video_left.mp4`

Mixing weights
*Policy-task* π̂ = 0.128, 0.139, 0.136, 0.115, 0.151, 0.150, 0.105, 0.076
*Hybrid* π̂      = 0.058, 0.060, 0.059, 0.058, 0.126, 0.522, 0.058, 0.059
