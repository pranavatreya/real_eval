# Real-Eval snapshot
*seed 114* — train 465  | test 49

| model | acc | pearson ρ | 1−MMRV | spearman ρ | kendall τ | pairwise_acc |
|-------|:---:|:---------:|:------:|:---------:|:--------:|:-------------:|
| Davidson | 0.633 | 0.946 | 0.934 | 0.357 | 0.333 | 0.667 |
| BT | 0.633 | **0.985** | 0.978 | 0.821 | 0.714 | 0.857 |
| Elo | **0.714** | 0.924 | 0.964 | 0.321 | 0.143 | 0.571 |
| PolicyTask | 0.449 | 0.215 | 0.802 | 0.393 | 0.238 | 0.619 |
| Hybrid | 0.612 | 0.827 | **1.000** | **1.000** | **1.000** | **1.000** |
| BT-TaskVar | 0.592 | 0.818 | 0.974 | 0.857 | 0.714 | 0.857 |
| MeanSucc | 0.612 | 0.966 | 0.981 | 0.857 | 0.714 | 0.857 |

## Leaderboards

### Davidson
 1. paligemma_fast_droid (0.256)
 2. paligemma_diffusion_droid (0.228)
 3. pi0_droid (0.150)
 4. pi0_fast_droid (0.147)
 5. paligemma_fast_specialist_droid (0.087)
 6. paligemma_vq_droid (0.076)
 7. paligemma_binning_droid (-0.944)

### BT
 1. paligemma_fast_droid (0.202)
 2. paligemma_vq_droid (0.182)
 3. pi0_fast_droid (0.160)
 4. paligemma_fast_specialist_droid (0.144)
 5. paligemma_diffusion_droid (0.107)
 6. pi0_droid (-0.002)
 7. paligemma_binning_droid (-0.792)

### Elo
 1. paligemma_diffusion_droid (0.274)
 2. paligemma_fast_specialist_droid (0.260)
 3. paligemma_vq_droid (0.246)
 4. paligemma_fast_droid (0.100)
 5. pi0_fast_droid (0.031)
 6. pi0_droid (-0.121)
 7. paligemma_binning_droid (-0.790)

### PolicyTask
 1. paligemma_fast_droid (0.148)
 2. paligemma_diffusion_droid (0.004)
 3. pi0_fast_droid (-0.006)
 4. paligemma_binning_droid (-0.011)
 5. paligemma_vq_droid (-0.036)
 6. paligemma_fast_specialist_droid (-0.040)
 7. pi0_droid (-0.059)

### Hybrid
 1. pi0_fast_droid (29.334)
 2. paligemma_fast_droid (28.981)
 3. paligemma_fast_specialist_droid (28.704)
 4. paligemma_vq_droid (28.181)
 5. paligemma_diffusion_droid (16.610)
 6. pi0_droid (-64.134)
 7. paligemma_binning_droid (-67.676)

### BT-TaskVar
 1. paligemma_fast_droid (2.203)
 2. paligemma_fast_specialist_droid (1.362)
 3. pi0_fast_droid (0.478)
 4. paligemma_vq_droid (-0.069)
 5. pi0_droid (-0.682)
 6. paligemma_diffusion_droid (-0.862)
 7. paligemma_binning_droid (-2.429)

### MeanSucc
 1. paligemma_fast_droid (0.479)
 2. pi0_fast_droid (0.425)
 3. paligemma_vq_droid (0.413)
 4. paligemma_diffusion_droid (0.410)
 5. paligemma_fast_specialist_droid (0.405)
 6. pi0_droid (0.274)
 7. paligemma_binning_droid (0.085)
