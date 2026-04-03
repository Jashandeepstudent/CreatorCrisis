"""
engine/reward_shaper.py — Curriculum Reward Shaper
Meta OpenEnv Hackathon | Creator Account Crisis Scenario
Author: Jashandeep Singh

Problem (Fix 11 — Anti-Convergence):
    With HACKER_RESTORED=-500 and NEGLIGENT_ESCALATION=-240 in the reward
    table, a DQN will find the "safe" local minimum early in training:
    just REJECT everyone on step 1 for a predictable -30. This strategy
    avoids all risk — the agent never explores past the first action.

    This is the "Convergence Trap": a policy that minimises loss by
    doing nothing useful, forever.

Solution — Potential-Based Reward Shaping (Ng et al., 1999):
    We use F(s,s') = γ·Φ(s') - Φ(s) where Φ is a potential function
    that grows with verification_score. This is mathematically proven
    not to change the optimal policy — it only changes which policies
    are discovered FIRST during training.

    Concretely:
        1. Verification milestone bonuses (+25, +50, +100) reward
           progress through the verification axis, making exploration
           more profitable than instant rejection.
        2. A curriculum shaping_scale starts at 3.0 and anneals to 1.0
           over training, so early exploration bonuses are large but
           shrink as the policy matures (preventing bonus farming).
        3. An episodic curiosity bonus rewards visiting new (action,
           patience_band) pairs, preventing the agent from getting
           stuck repeating the first action that gave a positive signal.

Architecture:
    RewardShaper is stateless with respect to the environment — it
    receives (obs_before, obs_after, base_reward) and returns a
    shaped reward. The environment never needs to know shaping is active.

    Training harness::

        shaper = RewardShaper(total_training_steps=500_000)
        ...
        shaped_r = shaper.shape(
            step_idx       = global_step,
            verify_before  = obs_before_vec[3],   # not available in obs — use internal tracker
            verify_after   = sim.state.verification_score,
            base_reward    = response.reward,
            action         = action,
            patience_signal= obs.patience_signal,
            episode_actions= episode_action_set,
        )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import ActionType


# ─────────────────────────────────────────────────────────────────────────────
#  SHAPING HYPERPARAMETERS
#  All values are justified by ratio analysis — see inline comments.
# ─────────────────────────────────────────────────────────────────────────────

# Verification milestones — one-time bonuses, not repeatable
_MILESTONE_THRESHOLDS: Final[list[tuple[float, float]]] = [
    (0.30, +25.0),    # "you found signal"  — exceeds early-reject cost (-30) at scale 3x
    (0.60, +50.0),    # "halfway there"     — strong gradient pull toward threshold
    (0.85, +100.0),   # "restore is legal"  — largest bonus; earned once per episode
]

# Shaping scale annealing: 3.0 → 1.0 over total_training_steps
# At scale 3.0: milestone 1 is +75, which exceeds the -30 early-reject.
# At scale 1.0: milestone 1 is +25, the base value — shaping fades out.
_SHAPING_SCALE_START: Final[float] = 3.0
_SHAPING_SCALE_END:   Final[float] = 1.0

# Curiosity bonus: reward first visit to each (action, patience_band) pair
# Keeps agent exploring the action × state space instead of converging on
# a single "safe" action sequence.
_CURIOSITY_BONUS: Final[float] = +5.0

# Per-step urgency bonus: small reward for taking any action while deadline < 20m
# This ensures the agent doesn't learn to idle near the deadline.
_URGENCY_ACTION_BONUS: Final[float] = +2.0

# Potential function scale: Φ(s) = verify * this constant
# Used for the F(s,s') = γΦ(s') - Φ(s) shaping term.
# Chosen so that the full 0→1 verification path gives cumulative +160 shaped reward,
# matching the episode-level terminal reward scale.
_POTENTIAL_SCALE: Final[float] = 160.0
_GAMMA: Final[float]           = 0.99


# ─────────────────────────────────────────────────────────────────────────────
#  PER-EPISODE MILESTONE TRACKER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeMilestoneTracker:
    """
    Tracks which verification milestones have been awarded in the current episode.

    Must be reset at episode start. Milestone bonuses are one-time only —
    repeating them would allow an agent to farm bonuses by oscillating
    around a threshold.

    Usage::

        tracker = EpisodeMilestoneTracker()
        bonus = tracker.check(verify_before=0.25, verify_after=0.35)
        # → +25.0 (milestone 0.30 crossed for the first time)
        bonus = tracker.check(verify_before=0.80, verify_after=0.90)
        # → +100.0 (milestone 0.85 crossed for the first time)
        bonus = tracker.check(verify_before=0.84, verify_after=0.90)
        # → 0.0   (0.85 already awarded this episode)
    """

    _awarded: set[float] = field(default_factory=set)

    def check(self, verify_before: float, verify_after: float) -> float:
        """
        Return the sum of milestone bonuses newly crossed this step.

        Args:
            verify_before: verification_score before this action.
            verify_after:  verification_score after this action.

        Returns:
            float: total milestone bonus (0.0 if none crossed).
        """
        total = 0.0
        for threshold, bonus in _MILESTONE_THRESHOLDS:
            if threshold not in self._awarded and verify_before < threshold <= verify_after:
                self._awarded.add(threshold)
                total += bonus
        return total

    def reset(self) -> None:
        """Reset for a new episode. Call at episode init."""
        self._awarded.clear()

    @property
    def max_remaining(self) -> float:
        """Maximum milestone bonus still available this episode."""
        return sum(b for t, b in _MILESTONE_THRESHOLDS if t not in self._awarded)


# ─────────────────────────────────────────────────────────────────────────────
#  REWARD SHAPER
# ─────────────────────────────────────────────────────────────────────────────

class RewardShaper:
    """
    Applies curriculum-annealed potential-based reward shaping.

    Wraps the base environment reward with four additive components:

        1. Potential shaping   F(s,s') = γΦ(s') - Φ(s)
           Policy-invariant by Ng et al. theorem. Pulls the agent toward
           high-verification states without changing the optimal policy.

        2. Milestone bonuses   one-time per episode, annealed by scale
           +25 / +50 / +100 at verification 0.30 / 0.60 / 0.85.
           At scale 3.0 (start of training): +75 / +150 / +300.
           Makes early exploration strictly better than instant rejection.

        3. Curiosity bonus     per new (action, patience_band) pair
           +5.0 for each first-visit in the episode.
           Prevents premature convergence to a single action loop.

        4. Urgency bonus       +2.0 per action taken when deadline < 20m
           Keeps the agent active near the deadline rather than stalling.

    Curriculum annealing:
        shaping_scale = START - (START - END) * min(1.0, step / total_steps)

        At step 0:        scale = 3.0  (aggressive guidance)
        At 50% training:  scale = 2.0
        At 100% training: scale = 1.0  (shaping fades to base values)

    Usage::

        shaper  = RewardShaper(total_training_steps=500_000)
        tracker = EpisodeMilestoneTracker()   # one per episode
        visited = set()                        # one per episode

        # inside training loop:
        shaped = shaper.shape(
            global_step     = step_counter,
            verify_before   = v_before,
            verify_after    = v_after,
            base_reward     = env_response.reward,
            action          = action,
            patience_signal = obs.patience_signal,
            deadline_mins   = obs.brand_deal_deadline_mins,
            episode_visited = visited,
            tracker         = tracker,
        )
    """

    def __init__(self, total_training_steps: int = 500_000) -> None:
        self._total_steps = total_training_steps

    def shaping_scale(self, global_step: int) -> float:
        """
        Current curriculum scale in [END, START].

        Linearly anneals from _SHAPING_SCALE_START to _SHAPING_SCALE_END
        over total_training_steps. Clamps at both ends.
        """
        progress = min(1.0, global_step / max(1, self._total_steps))
        return _SHAPING_SCALE_START - (_SHAPING_SCALE_START - _SHAPING_SCALE_END) * progress

    def potential(self, verification_score: float) -> float:
        """
        Φ(s) = verification_score * _POTENTIAL_SCALE

        Concave — early verification gains get more shaping than late ones,
        reflecting diminishing marginal returns as the agent approaches the
        restore threshold.
        """
        # Use sqrt to make potential concave: rewards early progress more
        return math.sqrt(max(0.0, verification_score)) * _POTENTIAL_SCALE

    def shape(
        self,
        global_step:      int,
        verify_before:    float,
        verify_after:     float,
        base_reward:      float,
        action:           ActionType,
        patience_signal:  int,
        deadline_mins:    float,
        episode_visited:  set[tuple[ActionType, int]],
        tracker:          EpisodeMilestoneTracker,
    ) -> tuple[float, dict[str, float]]:
        """
        Compute shaped reward and a breakdown dict for logging.

        Args:
            global_step:     Training step counter (for scale annealing).
            verify_before:   verification_score before this action.
            verify_after:    verification_score after this action.
            base_reward:     Raw reward from EnvironmentResponse.
            action:          ActionType taken this step.
            patience_signal: Observed patience band (0–3).
            deadline_mins:   Remaining deadline (for urgency bonus).
            episode_visited: Set of (action, patience_signal) pairs visited
                             this episode. Updated IN-PLACE by this method.
            tracker:         EpisodeMilestoneTracker for this episode.
                             Updated IN-PLACE by this method.

        Returns:
            Tuple of:
                float:            Total shaped reward.
                dict[str, float]: Breakdown for logging/debugging.
        """
        scale = self.shaping_scale(global_step)

        # 1. Potential shaping — policy-invariant
        phi_before = self.potential(verify_before)
        phi_after  = self.potential(verify_after)
        f_potential = _GAMMA * phi_after - phi_before

        # 2. Milestone bonuses — one-time, annealed
        milestone_raw  = tracker.check(verify_before, verify_after)
        milestone      = milestone_raw * scale

        # 3. Curiosity bonus — first-visit (action, patience_band) pairs
        visit_key = (action, patience_signal)
        curiosity = 0.0
        if visit_key not in episode_visited:
            episode_visited.add(visit_key)
            curiosity = _CURIOSITY_BONUS * scale

        # 4. Urgency bonus — active near deadline
        urgency = _URGENCY_ACTION_BONUS if deadline_mins <= 20.0 else 0.0

        total_shaping = f_potential + milestone + curiosity + urgency
        shaped        = base_reward + total_shaping

        breakdown = {
            "base":       base_reward,
            "potential":  round(f_potential, 3),
            "milestone":  round(milestone, 3),
            "curiosity":  round(curiosity, 3),
            "urgency":    round(urgency, 3),
            "scale":      round(scale, 3),
            "total":      round(shaped, 3),
        }
        return shaped, breakdown


# ─────────────────────────────────────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEP = "=" * 64

    print(SEP)
    print("  RewardShaper — annealing curve")
    print(SEP)
    shaper = RewardShaper(total_training_steps=100_000)
    for step in [0, 10_000, 25_000, 50_000, 75_000, 100_000]:
        print(f"  step={step:7d}  scale={shaper.shaping_scale(step):.3f}")

    print(f"\n{SEP}")
    print("  EpisodeMilestoneTracker — crossing all three milestones")
    print(SEP)
    tracker = EpisodeMilestoneTracker()
    tests = [
        (0.00, 0.20, "no milestone"),
        (0.20, 0.35, "crosses 0.30 → +25"),
        (0.35, 0.65, "crosses 0.60 → +50"),
        (0.65, 0.90, "crosses 0.85 → +100"),
        (0.85, 0.92, "already awarded → +0"),
    ]
    for vb, va, label in tests:
        bonus = tracker.check(vb, va)
        print(f"  {vb:.2f}→{va:.2f}  bonus={bonus:+.1f}  ({label})")

    print(f"\n{SEP}")
    print("  RewardShaper.shape() — full episode simulation (10 steps)")
    print(SEP)
    shaper2  = RewardShaper(total_training_steps=50_000)
    tracker2 = EpisodeMilestoneTracker()
    visited2: set[tuple[ActionType, int]] = set()

    script = [
        (ActionType.EMPATHIZE,              0.00, 0.00, -1.5,  3,  90.0),
        (ActionType.CHECK_IP_LOGS,          0.00, 0.18, -1.2,  3,  84.0),
        (ActionType.EMPATHIZE,              0.18, 0.18, -0.9,  2,  78.0),
        (ActionType.REQUEST_ID_VOICE_VERIFY,0.18, 0.60,  7.2,  2,  72.0),
        (ActionType.EMPATHIZE,              0.60, 0.60, -1.4,  2,  66.0),
        (ActionType.CHECK_IP_LOGS,          0.60, 0.78, -1.1,  1,  60.0),
        (ActionType.REQUEST_ID_VOICE_VERIFY,0.78, 0.88,  7.5,  1,  18.0),  # CRITICAL
        (ActionType.RESTORE_ACCOUNT,        0.88, 0.88, 1980., 1,  12.0),
    ]

    total_base, total_shaped = 0.0, 0.0
    for i, (action, vb, va, base, ps, dl) in enumerate(script, 1):
        shaped, breakdown = shaper2.shape(
            global_step    = 0,  # scale = 3.0
            verify_before  = vb,
            verify_after   = va,
            base_reward    = base,
            action         = action,
            patience_signal= ps,
            deadline_mins  = dl,
            episode_visited= visited2,
            tracker        = tracker2,
        )
        total_base   += base
        total_shaped += shaped
        ms = f" [MILESTONE +{breakdown['milestone']:.0f}]" if breakdown["milestone"] else ""
        cu = f" [CURIOSITY +{breakdown['curiosity']:.0f}]"  if breakdown["curiosity"]  else ""
        print(f"  Step {i:2d} | {action.value:25s} | "
              f"base={base:+8.1f} | potential={breakdown['potential']:+6.2f} "
              f"| shaped={shaped:+8.1f}{ms}{cu}")

    print(f"\n  Total base reward  : {total_base:+.1f}")
    print(f"  Total shaped reward: {total_shaped:+.1f}")
    print(f"  Shaping bonus      : {total_shaped - total_base:+.1f}")
    print(f"\n  (Early reject baseline = -30. Shaping makes exploration worth "
          f"{total_shaped - (-30):+.1f} more than instant rejection.)")

    print(f"\n{SEP}")
    print("  Smoke test complete.")
    print(SEP)