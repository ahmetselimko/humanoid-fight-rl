# Results

All numbers come from `degerlendirme/taktik_olc.py`: the tactic net against the rule-based opponent, with fixed seeds.
**Score** is damage dealt minus damage taken, with fall penalties included. A 24-match run has about ±15% variance.

## 1. Tactic net v5 → v6 (new lower-level nets)
| Tactic net | Matches | Win | Score |
|---|---|---|---|
| v5 | 24 | 100% | +123.7 |
| v6, update 60 | 24 | – | +128.9 |
| v6, update 140 | 24 | – | +138.5 |
| **v6, update 160** | 24 | 100% | **+147.6** |

v6 continued training from v5 on top of the improved motion, get-up and walking nets.

## 2. Picking the motion-net checkpoint (20M-step run)
The motion net was trained for 20M steps. Several checkpoints were measured in matches:

| Checkpoint | Win | Score | Hits taken | Strike test (hits per punch) |
|---|---|---|---|---|
| previous net | 100% | +110 | 16.5 | 0.77 |
| **12M** | 96% | **+120** | **12.6** | 1.00 |
| 18M | 100% | +95 | – | **1.37** |
| 20M | 92% | +72 | – | 1.31 |

The isolated strike skill kept improving until 18M, but match performance peaked at 12M. The robot
over-specialised: its punch speed fell from 1.6 m/s to 1.3 m/s. Lesson: select checkpoints by matches.

## 3. Walking net
| Net | Speed error | Heading error | Turn time | Matches |
|---|---|---|---|---|
| **gezC 20M** | 0.13 m/s | 17° | 0.74 s | 100% / +118.7 |
| gezD2 (~22M total) | 0.11 m/s | 16° | 0.70 s | 96% / +86.4 |

Walking had plateaued: small tracking gains did not translate into better matches, so gezC was kept.

## 4. Why does the robot fall? Strike brake
Every fall in 24 matches was classified by the second before it:

| What happened before the fall | Count |
|---|---|
| **own punch, no hit received** | **13 / 25 (52%)** |
| hit received (3 of them while striking itself) | 6 |
| chasing the opponent | 4 |
| turning | 1 |
| right after getting up | 1 |

Before falling, the robot reached a peak speed of about 1.35 m/s on average: it was punching while running in. The fix
was a rule that runs after the network chooses: above a speed threshold, a strike command becomes "stop and face" first.

| Brake threshold | Matches | Win | Score | Hits | Taken | Falls/match |
|---|---|---|---|---|---|---|
| off | 24 | 100% | +135.6 | 38.4 | 14.4 | 1.0 |
| 0.4 m/s | 24 | 100% | +130.2 | 40.8 | 15.9 | 0.9 |
| **0.6 m/s** | 24 | 100% | **+160.6** | **44.9** | 15.8 | **0.7** |
| 0.9 m/s | 24 | 96% | +147.6 | 42.9 | 15.5 | 1.1 |
| off | 48 | 100% | +126.2 | 40.7 | 16.0 | 1.3 |
| **0.6 m/s** | 48 | 96% | **+153.1** | **43.1** | **15.0** | **0.9** |

The 48-match check confirms the result: the second batch of 24 matches gave +116.8 vs +145.6. That is about +21%
score and 30% fewer falls, with no retraining.

## 5. Negative result: a guard stance
**Goal:** keep the arms in a boxing guard while close to the opponent and not striking.

| Attempt | What | Outcome |
|---|---|---|
| gard1 | arm-pose reward `exp(−5·err²)` | 20M steps, arms in guard 0%. The kernel was flat at the 60–70° starting error, so there was no gradient. |
| gard2 | wide kernel `exp(−0.8·err²)` | Arm error 70° → 69° by 4M, so it stopped early. The exploration noise could not move the arm actions that far. |
| gard3 | arms **locked** in guard by rule; the net learns to fight from it (10M) | In training: guard 91%, hits 6.0 → 12.6, upright 99%. In matches: see below. |

| Matches (24) | Win | Score | Hits | Taken | Falls/match |
|---|---|---|---|---|---|
| **no guard (baseline)** | **100%** | **+135.6** | **38.4** | 14.4 | **1.0** |
| old net + rule guard | 67% | +18.6 | 6.2 | 3.1 | 2.0 |
| gard3 10M + rule guard | 75% | +32.7 | 14.0 | 7.7 | 2.2 |

The guard halves the damage taken, but it cuts hits to about a third and doubles falls, so it was not adopted. The
in-training improvement did not survive contact with real matches.
