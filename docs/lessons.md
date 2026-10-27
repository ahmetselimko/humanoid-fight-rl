# Lessons learned (selection)

The project keeps a running log of lessons (24 so far, in Turkish). Each entry records a failure, its root cause and a
rule for the future. This is a translated selection; lesson numbers follow the original log.

**General rules**
1. **Before long training, check that the target motion is physically possible** for this robot in this simulator.
   Most early failures came from trying to teach something the body could not do (wrist, hip and torso limits).
2. **After every reward change, break the reward down per term.** Reward that nobody can explain should be zero.
3. **`exp(−k·Σerr²)` kernels are flat far from the target and give no gradient.** Use per-joint or wide kernels.
   This was violated again in lesson 22.
4. **Averages hide problems.** "Knee −0.4" turned out to be one knee at −0.7 and the other straight. Measure left
   and right separately.
5. **Evaluate without helper forces.** Success with assistive forces is misleading.

**Selected lessons**
- **9. Time-indexed imitation targets reward the wrong motion once the robot falls behind.** The get-up reference
  said "stand with straight knees" at 3.6 s, so a robot that had not stood up yet was paid to keep its knees straight.
  Fix: advance the reference by the robot's own progress (phase tracking), not by the clock.
- **10. Validate a reward change with "does the correct motion earn more than the wrong one?"** Replay both from the
  same state, the policy's motion and a PD-forced correct motion, and compare them term by term.
- **18. A distilled student learns spurious cues.** Leftover opponent-related observations became position cues.
  This only showed when the state distribution changed: the robot fell as soon as it turned.
- **20. A per-step "airborne" reward teaches hopping, not jumping.** Reward the *event* (one continuous flight,
  distance at landing), not every step.
- **21. In long runs the best checkpoint is not the last one, and matches decide, not in-training metrics.** The
  isolated strike skill improved until 18M, but match score peaked at 12M (over-specialisation).
- **22. The flat-kernel mistake again (lesson 3).** 20M steps were wasted on a guard reward that paid about 0.01 at the
  starting error. New rules: (a) compute the reward at the current error before training (it should be at least
  0.2–0.3); (b) every run with a new reward gets a metric-based early stop; (c) track continuous error, not only a
  thresholded rate.
- **23. An in-training gain did not carry over to matches.** The guard stance doubled hits inside the training env
  but lost badly in matches. Run the cheap rule-based version of an idea in matches first. If it already loses big,
  do not expect training to close the gap.
- **24. Classify failures event by event before spending compute.** Logging the second before each fall showed that
  52% were self-inflicted during punches at speed. A one-line "slow down before striking" rule then improved score by
  about 20% with no training.

**Process rules that came out of these**
- Every cloud run has an automatic monitor. It restarts the run after spot preemption, stops early on stalled
  metrics, downloads the checkpoints and shuts the VM down.
- Checkpoints are saved every 2M steps, and at least three candidates are compared in matches.
- File checksums are verified after every upload to the VM. A stale environment file once silently ruined a run.
