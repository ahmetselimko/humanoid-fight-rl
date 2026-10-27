# Motion reference data — sources and licenses

The `*_kin.npz` files here are joint-angle trajectories retargeted from human motion capture onto the robot
(kinematic references used as imitation targets during training). They are derived works of the datasets below
and keep those datasets' terms — they are **not** covered by this repository's MIT license.

| Files | Source | License / terms |
|---|---|---|
| `dovus_dataset-1_punch_normal_*`, `dovus_dataset-1_kick_normal_*` | Bandai Namco Research Motiondataset-1 (punch_normal_001, kick_normal_001) | CC BY-NC 4.0 — non-commercial use, attribution: Bandai Namco Research Inc. |
| `dovus_02_05_*`, `dovus_86_*`, `yurume_07_*`, `yurume_08_*` | CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu), subjects 02, 07, 08, 86 | Free for research/non-commercial use; attribution requested: "The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217." |

Other files (`dovus_secim.json`, `gard_durus.npy`, `tek_gozlem_sabit.npz`, `dusus_havuzu_govde2.npz`) were
generated in this project (strike-segment selection, guard pose, observation normalisation, fall-state pool).
