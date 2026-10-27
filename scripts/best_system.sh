# Best fighting system: physics settings + the four selected networks.
# Usage (from the repo root, bash / Git Bash):  source scripts/best_system.sh
# Every network was selected by local match evaluation, not by in-training metrics (see docs/results.md).

# Robot and physics (new motor model: PD damping, joint damping, armature)
export ROBOT=govde2 KOL_KP=3 PD_KD_K=0.4 SONUM_K=0.3 ARMATUR_K=0.6
# Fight settings: opponent-aware observation, guard reference, strike timing
export RAKIP_GOZ=1 GARD_REF=1 DOVUS_HIZ=2.25 DOVUS_GECIS_S=0.15 VURUS_ONCE_S=0.35 VURUS_SONRA_S=0.25

export ALT=models/alt_dovus2_govde2_isabet1_eppoz12M.pth      # motion net: fighting/striking (12M steps)
export KALKIS=models/alt_dovus2_govde2_kuzmanB_epson3M.pth    # get-up expert (takes over after a fall)
export GEZ=models/alt_dovus2_govde2_gezC_epson20M.pth          # walking/navigation net (20M steps)
export TAKTIK=models/taktik6_ep160.pth                          # tactic net: picks one of 9 commands every 0.26 s

# Strike brake: above 0.6 m/s horizontal speed a strike command becomes "stop and face" first.
# 48 matches: score +126.2 -> +153.1, falls/match 1.3 -> 0.9 (docs/results.md).
export VURUS_FREN_HIZ=0.6

export PYTHONIOENCODING=utf-8
