# Architecture

## Robot and physics
- **Robot:** a MuJoCo humanoid (`sim/dual_humanoid_getup_motor_govde2.xml`, "govde2" body) with 17 position-controlled
  motors: legs, hips, waist, shoulders and elbows.
- **Actuation:** each policy action `a ∈ [-1, 1]` maps to a joint target `mid + a · 1.1 · half_range`. A PD controller
  (gains per joint, `KOL_KP` scales the arms) turns that target into torque.
- **Motor model:** damping, joint friction and armature are scaled (`PD_KD_K`, `SONUM_K`, `ARMATUR_K`) to make the
  motion less "rubbery". These values were tuned after a physics revision, and all current networks were trained
  with them.
- **Timing:** control runs at 62.5 Hz (2 physics substeps, so 125 Hz physics). The tactic net decides every 16 control
  steps, about 0.26 s.

## Levels
### 1. Tactic net (`taktik_train.py`, `taktik_env.py`)
- **Observation (47 values):** own pose and velocity in the robot's frame, the opponent's relative position, heading,
  standing state, strike type and progress, and health, the two nearest gateways of the arena, and the last command.
- **Action:** one of 9 commands: `kovala` chase, `yavas_kovala` slow chase, `dur` stop and face, `sola`/`saga`
  turn 45°, `yumruk` punch, `tekme` kick, `kac` flee toward a far gateway, `geri` back off.
- **Network:** MLP 47→256→256 (tanh) with policy and value heads, trained with PPO.
- **Opponents:** a rule-based opponent plus snapshots of itself (self-play pool, refreshed every 20 updates).
- **Reward:** damage dealt minus damage taken, a large penalty for falling, staying near and facing the opponent,
  and a small cost for passivity.
- **Strike brake** (`VURUS_FREN_HIZ`): a rule applied *after* the network chooses. If the robot's horizontal speed is
  above the threshold, a strike command becomes "stop and face".

### 2. Motion nets (`getup_train_par.py`, continuous PPO)
| Skill (`BECERI`) | Role | Main reward terms |
|---|---|---|
| `alt_dovus` (motion net) | approach, face the opponent, punch and kick on command | imitation of the strike reference, hit bonus (`W_ISABET`), aim (`W_NISAN`), damage taken (`W_ALINAN`), stay upright |
| get-up expert | stand up from any fallen pose | phase-tracked imitation of get-up references, head height, then upright |
| walking (`gez`) | follow a commanded speed and heading | velocity and heading tracking, gait imitation (CMU walk), stay upright |

- **Observation:** proprioception, plus the opponent's relative pose for fight skills and the command vector.
- **Training:** the training env samples realistic starts from a pool of fall states. The opponent comes from a pool
  of earlier motion nets.
- **Parallel PPO** with N CPU workers. A checkpoint is saved every 2M steps, and an evaluation runs in the middle of
  training (`POZ` line in the log).

### 3. Skill selector (`taktik_env.py`, `beceri_secici.py`)
The selector is rule-based:
- **Not standing:** the get-up expert controls the robot until it has stood stably for a short time.
- **Opponent far** (> 2.5 m, back below 1.8 m) and not striking: the walking net follows the A* path
  (`gezinme.py`) through the arena.
- **Otherwise:** the fight motion net is in control.

## Example training commands
Run these from `sim/` after sourcing `scripts/best_system.sh`. Cloud runs used 11 workers.

```bash
# Tactic net: continue from v6
python -u taktik_train.py --parca arena/insa/oda_001/oda_001_komsu3_fizik.xml --dunya arena \
  --alt $ALT --kalkis $KALKIS --gez $GEZ --devam $TAKTIK --onek taktik7 \
  --workers 11 --mac 2 --guncelleme 100 --olcum-her 20 --havuz-her 20

# Motion net: continue fight training from the selected checkpoint
KOSU=deneme BECERI=alt_dovus W_NISAN=2.0 W_ISABET=1.5 W_ALINAN=1.5 PASIF_P=0.15 YENI_SIGMA=0.08 \
  W_AYAKTA_DOVUS=0.6 YERDE_P=0.3 RAKIP_HAVUZ=$ALT \
  python -u getup_train_par.py --resume $ALT --workers 11 --lr 3e-5 --max-adim 10e6
```

## Evaluation protocol
- **Benchmark:** `degerlendirme/taktik_olc.py N <tactic.pth>` plays N matches with seeds 5000…5000+N. Seeds alternate
  which side the tactic net plays, and the opponent is the rule-based bot.
- **Reported metrics:** win rate, mean score, hits landed and taken, falls per match, and how often the walking net
  was used.
- **Model selection:** every checkpoint is selected by this benchmark. With 24 matches the run-to-run variance is
  about ±15%, so decisive changes are confirmed with 48.
