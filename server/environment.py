"""
server/environment.py — CreatorCrisisEnv
Meta OpenEnv Hackathon | Creator Account Crisis Scenario
Author: Jashandeep Singh

Wraps UserSimulator in a standard Gymnasium interface so any RL library
(Stable-Baselines3, RLlib, CleanRL, custom DQN) can train against it
without touching simulation internals.

Gymnasium Contract
──────────────────
    env = CreatorCrisisEnv()
    obs, info   = env.reset(seed=42)
    obs, r, terminated, truncated, info = env.step(action_int)

The 5-tuple follows the modern Gymnasium API (v0.26+):
    terminated : episode ended via a natural terminal condition
                 (RESTORE_ACCOUNT, REJECT_ACCOUNT, rage-quit,
                  deadline expiry, budget exceeded).
    truncated  : episode cut short by the 15-action hard cap
                 (RewardReason.TIMEOUT_SYSTEM).

Space Design
────────────
    action_space : Discrete(6)  — one integer per ActionType, in
                   canonical declaration order.

    observation_space : gymnasium.spaces.Dict  — one entry per
                   Observation field.  All continuous values are
                   normalised to [0, 1] or [0, N] via Box(dtype=float32).
                   The 'message_history' field is excluded from the
                   numeric space; it lives in info['obs_text'] for
                   LLM-based agents.

Observation Vector
──────────────────
    The env also exposes obs['vector'] — a flat Box(15,) built by
    Observation.to_vector().  DQN / PPO agents should use this.
    LLM agents should read info['obs_text']['sentiment_summary'].
    Hybrid agents can use both.

Info Dict Schema
────────────────
    reset()  → { 'episode_seed': EpisodeSeed (dict),
                 'obs_text': { 'message_history': [...],
                               'sentiment_summary': str } }

    step()   → { 'step_result':  StepResult  (dict),
                 'episode_stats': EpisodeStats (dict),
                 'obs_text': { 'message_history': [...],
                               'sentiment_summary': str },
                 'loophole_flags': { ... },   # live fix-status per step
                 'action_name': str }

    The 'loophole_flags' sub-dict surfaces every active loophole fix
    in a single boolean/numeric snapshot so training dashboards can
    plot fix-specific metrics without parsing StepResult.notes.

Loophole Fix Coverage (all 10 + 3 Grandmaster)
───────────────────────────────────────────────
    Fix 1  — Resource Budget         : truncated=True at step 15
    Fix 2  — Blind Trust             : penalty_multiplier ≥ 2 on RESTORE
    Fix 3  — Free Money              : credits deducted from reward
    Fix 4  — Silent Hacker           : behavioural_consistency in obs
    Fix 5  — Repetition Engine       : repetition_penalty in obs + doubling
    Fix 6  — Empathy Decay           : empathy_effectiveness in obs
    Fix 7  — Cliff-Edge Brand Deal   : +200 cap after deadline
    Fix 8  — Static Seed / Oracle    : EpisodeSeed Gaussian noise
    Fix 9  — Duty of Care            : NEGLIGENT_ESCALATION × 3 penalty
    Fix 10 — Jitter Latency          : uniform sleep per step
    Fix A  — Duty of Care (alias Fix 9)
    Fix B  — Oracle Masking          : patience_signal (noisy 0-3) in obs
    Fix C  — Cliff-Edge (alias Fix 7)

Render Modes
────────────
    'human'   — prints a colour-coded step summary to stdout
    'ansi'    — returns the same summary as a string
    None      — silent (default; use for training)
"""

from __future__ import annotations

import sys
import os
from typing import Any, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Project path setup ───────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import ActionType, Observation, RiskLevel
from engine.user_sim import EpisodeStats, UserSimulator
from adversarial.checks import AdversarialFinding


# ─────────────────────────────────────────────────────────────────────────────
#  ACTION CODEC
#  Maps int ↔ ActionType in a stable, documented order.
#  NEVER reorder — doing so silently breaks any trained policy checkpoint.
# ─────────────────────────────────────────────────────────────────────────────

_ACTION_INDEX_TO_TYPE: list[ActionType] = [
    ActionType.CHECK_IP_LOGS,           # 0  — Security (passive)
    ActionType.REQUEST_ID_VOICE_VERIFY,  # 1  — Security (high friction)
    ActionType.EMPATHIZE,               # 2  — Negotiation (patience restore)
    ActionType.OFFER_COMPENSATION,      # 3  — Negotiation (credits)
    ActionType.RESTORE_ACCOUNT,         # 4  — Terminal (restore)
    ActionType.REJECT_ACCOUNT,          # 5  — Terminal (deny)
]

_ACTION_TYPE_TO_INDEX: dict[ActionType, int] = {
    a: i for i, a in enumerate(_ACTION_INDEX_TO_TYPE)
}

_N_ACTIONS: int = len(_ACTION_INDEX_TO_TYPE)  # 6


# ─────────────────────────────────────────────────────────────────────────────
#  OBSERVATION SPACE DEFINITION
#
#  Every field from Observation.to_vector() gets its own named Box entry.
#  This is more verbose than a single flat Box but:
#    1. Makes field semantics explicit for debugging.
#    2. Lets Dict-aware algorithms (recurrent PPO, attention models)
#       treat numeric vs categorical inputs differently.
#    3. Allows future additions without reshaping all downstream code.
#
#  The 'vector' key contains the full 15-dim normalised flat array —
#  a convenience for DQN/PPO that just needs torch.tensor(obs['vector']).
#
#  The 'last_action_onehot' key is a 6-dim binary array (one-hot of
#  the previous action).  All zeros at step 0 (no prior action).
# ─────────────────────────────────────────────────────────────────────────────

def _build_observation_space() -> spaces.Dict:
    """
    Construct the gymnasium observation space.

    All continuous quantities are normalised to [0, 1] to prevent
    the State Explosion pathology (large-scale values like credits=50000
    drowning small-scale signals like mismatch=0.61 in the gradient).

    Normalisation factors mirror Observation.to_vector() exactly.
    Any change here must be mirrored there and vice-versa.
    """
    return spaces.Dict({
        # ── Scalar continuous fields (all [0, 1] after normalisation) ─
        "follower_count_norm": spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
            # raw / 10_000_000  (10M follower reference ceiling)
        ),
        "deadline_norm": spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
            # brand_deal_deadline_mins / 180.0
        ),
        "actions_taken_norm": spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
            # total_actions_taken / 15
        ),
        "evidence_mismatch": spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
            # already normalised — contradiction index
        ),
        "credits_spent_norm": spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
            # credits_spent / 50_000
        ),
        "behavioural_consistency": spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
            # silent-hacker signal — already normalised
        ),
        "repetition_penalty_norm": spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
            # clipped to [1, 8] → (val - 1) / 7 → [0, 1]
        ),
        "empathy_effectiveness": spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
            # Fix 6 decay signal — already normalised
        ),
        "patience_signal_norm": spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
            # Fix B — noisy quantised proxy: patience_signal / 3
        ),

        # ── Last-action one-hot (6 dims, binary) ─────────────────────
        "last_action_onehot": spaces.Box(
            low=0.0, high=1.0, shape=(_N_ACTIONS,), dtype=np.float32,
            # One-hot of previous action. All zeros at step 0.
        ),

        # ── Flat convenience vector for DQN/PPO ──────────────────────
        "vector": spaces.Box(
            low=0.0, high=1.0,
            shape=(15,),     # matches Observation.vector_dim
            dtype=np.float32,
            # = concatenation of all fields above. Use this for DQN.
            # Do NOT use 'vector' AND the individual keys in the same
            # policy — they carry identical information.
        ),

        # ── Discrete categorical ──────────────────────────────────────
        "patience_signal": spaces.Discrete(4),    # Fix B — raw 0-3 signal
        "total_actions_taken": spaces.Discrete(16),  # 0..15

        # ── Adversarial layer observables (public CreatorState fields) ───
        "adversarial_risk_score": spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
        ),
        "honey_pot_triggered": spaces.Discrete(2),
        "entropy_decay_total": spaces.Box(
            low=0.0, high=0.25, shape=(1,), dtype=np.float32,
        ),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  OBSERVATION CONVERTER
#  Transforms an Observation Pydantic model into the gymnasium-compatible dict.
# ─────────────────────────────────────────────────────────────────────────────

def _obs_to_gym(obs: Observation, state=None) -> dict[str, np.ndarray | int]:
    """
    Convert a Pydantic Observation into a gymnasium-space-compatible dict.

    All numpy arrays use float32 to match the Box dtype and minimise
    GPU memory usage during training.  Discrete fields are plain int.

    Args:
        obs: Validated Observation from UserSimulator.

    Returns:
        Dict matching _build_observation_space() exactly.
    """
    vec = obs.to_vector()                     # list[float], length 15

    # One-hot of last action (dims 9-14 of vec, extracted separately)
    onehot = np.array(vec[9:15], dtype=np.float32)

    # Adversarial fields: public CreatorState signals, not in to_vector()
    if state is not None:
        adv_risk  = float(state.adversarial_risk_score)
        hp_flag   = int(state.honey_pot_triggered)
        ent_decay = float(state.entropy_decay_total)
    else:
        adv_risk, hp_flag, ent_decay = 0.0, 0, 0.0

    return {
        "follower_count_norm":      np.array([vec[0]],  dtype=np.float32),
        "deadline_norm":            np.array([vec[1]],  dtype=np.float32),
        "actions_taken_norm":       np.array([vec[2]],  dtype=np.float32),
        "evidence_mismatch":        np.array([vec[3]],  dtype=np.float32),
        "credits_spent_norm":       np.array([vec[4]],  dtype=np.float32),
        "behavioural_consistency":  np.array([vec[5]],  dtype=np.float32),
        "repetition_penalty_norm":  np.array([vec[6]],  dtype=np.float32),
        "empathy_effectiveness":    np.array([vec[7]],  dtype=np.float32),
        "patience_signal_norm":     np.array([vec[8]],  dtype=np.float32),
        "last_action_onehot":       onehot,
        "vector":                   np.array(vec,        dtype=np.float32),
        "patience_signal":          obs.patience_signal,
        "total_actions_taken":      obs.total_actions_taken,
        "adversarial_risk_score":   np.array([adv_risk],  dtype=np.float32),
        "honey_pot_triggered":      hp_flag,
        "entropy_decay_total":      np.array([ent_decay], dtype=np.float32),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  LOOPHOLE FLAG BUILDER
#  Produces a flat snapshot of every active loophole fix for the info dict.
#  Training dashboards can log these directly without parsing notes strings.
# ─────────────────────────────────────────────────────────────────────────────

def _build_loophole_flags(
    obs:   Observation,
    stats: EpisodeStats,
    step_result_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a flat bool/float snapshot of all active loophole conditions.

    Each key maps to one of the 10+3 fixes. True = fix is currently active
    (i.e., the condition it guards against is present this step).

    Args:
        obs:              Current Observation after step.
        stats:            Cumulative EpisodeStats after step.
        step_result_dict: StepResult.model_dump() for this step.

    Returns:
        dict suitable for direct logging via wandb.log() or TensorBoard.
    """
    rr = step_result_dict.get("reward_reason", "")
    pm = step_result_dict.get("penalty_multiplier", 1.0)

    return {
        # Fix 1 — Resource Budget
        "fix1_budget_timeout":        stats.total_steps >= 15,
        "fix1_actions_remaining":     max(0, 15 - stats.total_steps),

        # Fix 2 — Blind Trust / Evidence Mismatch
        "fix2_mismatch_high":         obs.evidence_mismatch > 0.5,
        "fix2_penalty_doubled":       pm >= 2.0 and rr == "MISMATCH_RESTORE",
        "fix2_evidence_mismatch":     float(obs.evidence_mismatch),

        # Fix 3 — Free Money / Credits
        "fix3_credits_spent":         int(obs.credits_spent),
        "fix3_credits_norm":          obs.credits_spent / 50_000.0,
        "fix3_budget_exceeded":       rr == "BUDGET_EXCEEDED",

        # Fix 4 — Silent Hacker
        "fix4_consistency_low":       obs.behavioural_consistency < 0.35,
        "fix4_consistency":           float(obs.behavioural_consistency),

        # Fix 5 — Repetition Engine
        "fix5_repetition_penalty":    float(obs.repetition_penalty),
        "fix5_penalty_active":        obs.repetition_penalty > 1.0,
        "fix5_rep_multiplier":        step_result_dict.get("repetition_multiplier", 1.0),

        # Fix 6 — Empathy Decay
        "fix6_empathy_effectiveness": float(obs.empathy_effectiveness),
        "fix6_empathy_exhausted":     obs.empathy_effectiveness <= 0.0,
        "fix6_empathy_diminished":    rr == "EMPATHY_DIMINISHED",

        # Fix 7 / C — Cliff-Edge Deadline
        "fix7_deadline_mins":         float(obs.brand_deal_deadline_mins),
        "fix7_deadline_critical":     obs.brand_deal_deadline_mins <= 20.0,
        "fix7_cliff_triggered":       rr == "DEAL_EXPIRED_RESTORE",
        "fix7_deal_dead":             obs.brand_deal_deadline_mins == 0.0,

        # Fix 8 — Stochastic Domain Randomisation
        "fix8_episode_seed":          stats.seed,

        # Fix 9 / A — Duty of Care / Negligent Escalation
        "fix9_negligence_triggered":  stats.negligence_triggered,
        "fix9_penalty_tripled":       pm >= 3.0 and rr == "NEGLIGENT_ESCALATION",

        # Fix 10 — Jitter Latency
        "fix10_step_latency_ms":      step_result_dict.get("step_latency_ms", 0.0),
        "fix10_jitter_active":        True,  # always on

        # Fix B — Oracle Masking
        "fixB_patience_signal":       obs.patience_signal,
        "fixB_oracle_masked":         True,  # verification_score never in obs

        # Aggregate health metrics
        "exploit_risk_score":         (
            (1.0 if obs.evidence_mismatch > 0.5 else 0.0)
            + (1.0 if obs.behavioural_consistency < 0.35 else 0.0)
            + (1.0 if obs.repetition_penalty >= 4.0 else 0.0)
            + (1.0 if obs.empathy_effectiveness <= 0.0 else 0.0)
            + (1.0 if obs.brand_deal_deadline_mins <= 20.0 else 0.0)
        ),  # 0-5; higher = more exploitation signals active

        # ── Adversarial Layer flags ───────────────────────────────────
        "adv_finding":              step_result_dict.get("adversarial_finding", "CLEAN"),
        "adv_penalty":              step_result_dict.get("adversarial_penalty", 0.0),
        "adv_notes":                step_result_dict.get("adversarial_notes", ""),
        "adv_gaslight_fired":       step_result_dict.get("adversarial_finding") == "GASLIGHTING_DETECTED",
        "adv_honey_pot_hit":        step_result_dict.get("adversarial_finding") == "HONEY_POT_TRIGGERED",
        "adv_honey_pot_denied":     step_result_dict.get("adversarial_finding") == "HONEY_POT_DENIED",
        "adv_entropy_stale":        step_result_dict.get("adversarial_finding") == "ENTROPY_STALE",
        "adv_contradiction_found":  step_result_dict.get("adversarial_finding") == "CONTRADICTION_FOUND",
        "adv_verification_decay":   step_result_dict.get("verification_decay_applied", 0.0),
        "adv_risk_score":           float(step_result_dict.get("_adv_risk_score", 0.0)),
        "adv_honey_pot_latched":    bool(step_result_dict.get("_honey_pot_triggered", False)),
        "adv_entropy_decay_total":  float(step_result_dict.get("_entropy_decay_total", 0.0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# ANSI colour codes — safe to disable on Windows by setting _COLOUR=False
_COLOUR = sys.platform != "win32"

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOUR else text

_GREEN  = lambda t: _c("32;1", t)
_YELLOW = lambda t: _c("33;1", t)
_RED    = lambda t: _c("31;1", t)
_CYAN   = lambda t: _c("36;1", t)
_DIM    = lambda t: _c("2",    t)
_BOLD   = lambda t: _c("1",    t)


def _render_step(
    step:       int,
    action_name: str,
    reward:     float,
    obs:        dict,
    info:       dict,
    terminated: bool,
    truncated:  bool,
) -> str:
    """
    Produce a compact, colour-coded step summary string.

    Layout:
        ─── Step 3 ─────────────────────────────────────────────────
        Action  : EMPATHIZE
        Reward  : +3.00  (EMPATHY_BONUS)    Latency: 34.2ms
        Patience: ██░░  signal=2            Deadline: 108.2m
        Mismatch: ▓░░░░  0.15               Verify-delta: +0.000
        Flags   : [consistency LOW] [deadline OK]
        User    : "I appreciate that. I know these situations..."
        ────────────────────────────────────────────────────────────
    """
    sr     = info.get("step_result", {})
    flags  = info.get("loophole_flags", {})
    text   = info.get("obs_text", {})
    reason = sr.get("reward_reason", "?")

    # Reward colour
    rfmt   = _GREEN(f"{reward:+.2f}") if reward > 0 else (_RED(f"{reward:+.2f}") if reward < 0 else f"{reward:+.2f}")

    # Patience bar (4 blocks)
    sig    = obs.get("patience_signal", 2)
    bar    = "█" * sig + "░" * (3 - sig)
    bar_c  = _GREEN(bar) if sig == 3 else (_YELLOW(bar) if sig >= 1 else _RED(bar))

    # Deadline colour
    dl     = float(obs.get("deadline_norm", [0.5])[0]) * 180.0
    dl_fmt = _RED(f"{dl:.1f}m") if dl <= 20 else (_YELLOW(f"{dl:.1f}m") if dl <= 60 else _GREEN(f"{dl:.1f}m"))

    # Active loophole flags
    active = []
    if flags.get("fix2_mismatch_high"):     active.append(_RED("mismatch HIGH"))
    if flags.get("fix4_consistency_low"):   active.append(_RED("consistency LOW"))
    if flags.get("fix5_penalty_active"):    active.append(_YELLOW(f"rep×{flags['fix5_rep_multiplier']:.0f}"))
    if flags.get("fix6_empathy_exhausted"): active.append(_YELLOW("empathy 0"))
    if flags.get("fix7_deadline_critical"): active.append(_RED("DEADLINE CRITICAL"))
    if flags.get("fix9_negligence_triggered"): active.append(_RED("NEGLIGENCE ×3"))
    if flags.get("fix7_cliff_triggered"):   active.append(_RED("CLIFF-EDGE"))
    flag_str = "  ".join(active) if active else _DIM("none")

    # Last user message
    msgs   = text.get("message_history", [])
    user_msgs = [m[6:] for m in msgs if m.startswith("User:")]
    last   = f'"{user_msgs[-1][:72]}..."' if user_msgs else "(no message)"

    end_tag = ""
    if terminated: end_tag = _RED("  ★ TERMINATED")
    if truncated:  end_tag = _YELLOW("  ★ TRUNCATED (budget)")

    lines = [
        _DIM(f"─── Step {step} " + "─" * 55),
        f"  Action   : {_BOLD(action_name)}{end_tag}",
        f"  Reward   : {rfmt}  ({_CYAN(reason)})   "
        f"Latency: {flags.get('fix10_step_latency_ms', 0.0):.1f}ms",
        f"  Patience : {bar_c}  signal={sig}       Deadline: {dl_fmt}",
        f"  Mismatch : {obs.get('evidence_mismatch', [0.0])[0]:.3f}   "
        f"EmpEff: {obs.get('empathy_effectiveness', [1.0])[0]:.2f}   "
        f"VerifyΔ: {sr.get('verification_delta', 0.0):+.3f}",
        f"  Flags    : {flag_str}",
        f"  User     : {last}",
        _DIM("─" * 62),
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENVIRONMENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CreatorCrisisEnv(gym.Env):
    """
    Gymnasium wrapper around UserSimulator for the Creator Account Crisis RL task.

    Action Space
    ────────────
        Discrete(6):
            0 → CHECK_IP_LOGS
            1 → REQUEST_ID_VOICE_VERIFY
            2 → EMPATHIZE
            3 → OFFER_COMPENSATION
            4 → RESTORE_ACCOUNT
            5 → REJECT_ACCOUNT

    Observation Space
    ─────────────────
        Dict (see _build_observation_space() for full schema):
            obs['vector']          — flat float32 array shape (15,) for DQN/PPO
            obs['patience_signal'] — int 0-3 (Fix B oracle-masked proxy)
            + individual normalised scalar fields

    Episode Termination
    ───────────────────
        terminated=True when a natural terminal condition fires:
            RESTORE_ACCOUNT called (hacker or legitimate)
            REJECT_ACCOUNT called
            user_patience hits 0 (rage-quit or negligent escalation)
            brand_deal_deadline_mins hits 0
            credits_spent > ₹50,000

        truncated=True when the 15-action hard cap is reached
            (Fix 1 — Resource Budget).  terminated is False in this case.

    LLM Agent Integration
    ─────────────────────
        info['obs_text']['message_history'] — full conversation list
        info['obs_text']['sentiment_summary'] — 250-token structured summary
        Feed sentiment_summary into the LLM system prompt.
        Use obs['vector'] for any numeric head (hybrid agent).

    Registration
    ────────────
        gymnasium.register(
            id="CreatorCrisis-v1",
            entry_point="server.environment:CreatorCrisisEnv",
        )
        env = gymnasium.make("CreatorCrisis-v1")

    Attributes:
        metadata:        Gymnasium metadata dict.
        action_space:    Discrete(6).
        observation_space: Dict space defined above.
        render_mode:     'human' | 'ansi' | None.
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "render_fps": 1,
        "env_id": "CreatorCrisis-v1",
        "version": 1,
        "n_actions": _N_ACTIONS,
        "obs_vector_dim": 15,
        "max_steps": 15,
        "action_meanings": [a.value for a in _ACTION_INDEX_TO_TYPE],
    }

    def __init__(self, render_mode: str | None = None) -> None:
        """
        Args:
            render_mode: 'human' prints to stdout; 'ansi' returns string;
                         None (default) is silent — use during training.
        """
        super().__init__()

        if render_mode not in (None, "human", "ansi"):
            raise ValueError(
                f"render_mode={render_mode!r} is not supported. "
                "Choose from: None, 'human', 'ansi'."
            )

        self.render_mode = render_mode

        # Gymnasium required spaces
        self.action_space      = spaces.Discrete(_N_ACTIONS)
        self.observation_space = _build_observation_space()

        # Internal episode state — None until reset() is called
        self._sim:   UserSimulator | None = None
        self._obs:   Observation   | None = None   # last Pydantic Observation
        self._step:  int                  = 0
        self._last_render: str            = ""

        # Action codec exposed for wrappers and eval scripts
        self.action_index_to_type = _ACTION_INDEX_TO_TYPE
        self.action_type_to_index = _ACTION_TYPE_TO_INDEX

    # ─── Core API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed:    int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """
        Start a new episode via EpisodeSeed.sample().

        Args:
            seed:    Optional integer seed for deterministic episode
                     generation.  None = random entropy (training mode).
            options: Gymnasium-standard options dict (currently unused).

        Returns:
            obs  : gymnasium-compatible observation dict.
            info : { 'episode_seed': EpisodeSeed dict,
                     'obs_text': { 'message_history', 'sentiment_summary' } }
        """
        super().reset(seed=seed)  # seeds self.np_random for Gymnasium compat

        self._sim   = UserSimulator(seed=seed)
        self._obs   = self._sim.initial_observation()
        self._step  = 0

        gym_obs = _obs_to_gym(self._obs, state=self._sim.state if self._sim else None)
        info    = self._build_reset_info()

        if self.render_mode == "human":
            print(self._render_reset(info))

        return gym_obs, info

    def step(
        self,
        action: int,
    ) -> tuple[dict, SupportsFloat, bool, bool, dict]:
        """
        Execute one environment step.

        Args:
            action: Integer in [0, 5] mapping to an ActionType.

        Returns:
            obs        : gymnasium-compatible observation dict.
            reward     : scalar float.
            terminated : True if episode ended via natural terminal condition.
            truncated  : True if episode ended via 15-action budget cap.
            info       : rich debug dict (step_result, episode_stats,
                         obs_text, loophole_flags, action_name).

        Raises:
            RuntimeError: if called before reset(), or after episode ends.
        """
        if self._sim is None:
            raise RuntimeError(
                "step() called before reset(). Call env.reset() first."
            )
        if self._sim.done:
            raise RuntimeError(
                "step() called on a terminated episode. "
                "Call env.reset() to start a new episode."
            )

        # ── Validate and decode action ────────────────────────────────
        if not self.action_space.contains(int(action)):
            raise ValueError(
                f"Invalid action {action!r}. "
                f"action_space is Discrete({_N_ACTIONS}); "
                f"valid values are 0..{_N_ACTIONS - 1}."
            )
        action_type = _ACTION_INDEX_TO_TYPE[int(action)]

        # ── Delegate to UserSimulator ─────────────────────────────────
        env_response, next_obs = self._sim.step(action_type)

        self._obs   = next_obs
        self._step += 1

        reward = float(env_response.reward)
        done   = env_response.done

        # ── Termination vs truncation ─────────────────────────────────
        # terminated = natural end (restore/reject/rage-quit/deadline/budget)
        # truncated  = hard cap (Fix 1 — Resource Budget, step >= 15)
        is_timeout = (
            env_response.step_result.reward_reason.value == "TIMEOUT_SYSTEM"
        )
        terminated = done and not is_timeout
        truncated  = done and is_timeout

        # ── Adversarial override ─────────────────────────────────────────
        # Gaslighting: UserSimulator already set done=True and reward=-1000.
        # We enforce the terminated/truncated split here for Gymnasium contract.
        adv_finding = env_response.step_result.adversarial_finding
        if adv_finding == AdversarialFinding.GASLIGHTING_DETECTED.value:
            reward     = -1000.0      # hard override — no partial credit
            terminated = True
            truncated  = False
            done       = True

        # Entropy decay: apply the per-step penalty to reward
        # (-2.0 per stale step, already stored in adversarial_penalty)
        elif adv_finding == AdversarialFinding.ENTROPY_STALE.value:
            reward += env_response.step_result.adversarial_penalty   # negative

        # Honey pot hit: apply penalty (hacker confirmed fabricated fact)
        elif adv_finding == AdversarialFinding.HONEY_POT_TRIGGERED.value:
            reward += env_response.step_result.adversarial_penalty   # -80.0

        # ── Build info dict ───────────────────────────────────────────
        gym_obs = _obs_to_gym(next_obs, state=self._sim.state)
        info    = self._build_step_info(
            action_type  = action_type,
            env_response = env_response,
            gym_obs      = gym_obs,
        )

        if self.render_mode in ("human", "ansi"):
            rendered = _render_step(
                step        = self._step,
                action_name = action_type.value,
                reward      = reward,
                obs         = gym_obs,
                info        = info,
                terminated  = terminated,
                truncated   = truncated,
            )
            self._last_render = rendered
            if self.render_mode == "human":
                print(rendered)

        return gym_obs, reward, terminated, truncated, info

    def render(self) -> str | None:
        """
        Return the last rendered frame for 'ansi' mode, or print for 'human'.

        Gymnasium calls this after step() when render_mode is set.
        In training mode (render_mode=None) this is a no-op.
        """
        if self.render_mode == "ansi":
            return self._last_render
        if self.render_mode == "human":
            print(self._last_render)
            return None
        return None

    def close(self) -> None:
        """Clean up (no external resources to release in this env)."""
        self._sim = None
        self._obs = None

    # ── Public helpers ────────────────────────────────────────────────────

    def action_meanings(self) -> list[str]:
        """Return action names in index order — for Atari-style wrappers."""
        return [a.value for a in _ACTION_INDEX_TO_TYPE]

    def decode_action(self, idx: int) -> ActionType:
        """Convert an integer action index to its ActionType."""
        return _ACTION_INDEX_TO_TYPE[idx]

    def encode_action(self, action: ActionType) -> int:
        """Convert an ActionType to its integer index."""
        return _ACTION_TYPE_TO_INDEX[action]

    @property
    def current_observation(self) -> Observation | None:
        """Raw Pydantic Observation (not gymnasium-converted). None before reset."""
        return self._obs

    @property
    def episode_stats(self) -> EpisodeStats | None:
        """Live EpisodeStats for the current episode. None before reset."""
        return self._sim.stats if self._sim else None

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_reset_info(self) -> dict[str, Any]:
        """Build the info dict returned by reset()."""
        assert self._sim is not None and self._obs is not None

        return {
            "episode_seed": self._sim.seed.model_dump(),
            "obs_text": {
                "message_history":  list(self._obs.message_history),
                "sentiment_summary": self._obs.sentiment_summary(),
            },
        }

    def _build_step_info(
        self,
        action_type:  ActionType,
        env_response: Any,
        gym_obs:      dict,
    ) -> dict[str, Any]:
        """
        Build the rich info dict returned by step().

        Contains everything a training dashboard, curriculum scheduler,
        or post-hoc auditor could need — including the full StepResult
        and a live snapshot of all 10+3 loophole fix conditions.
        """
        assert self._sim is not None and self._obs is not None

        sr_dict    = env_response.step_result.model_dump(mode="json")
        # Inject live CreatorState adversarial fields so loophole_flags can read them
        sr_dict["_adv_risk_score"]       = float(self._sim.state.adversarial_risk_score)
        sr_dict["_honey_pot_triggered"]  = bool(self._sim.state.honey_pot_triggered)
        sr_dict["_entropy_decay_total"]  = float(self._sim.state.entropy_decay_total)
        stats_dict = {
            "seed":                   self._sim.stats.seed,
            "risk_level":             self._sim.stats.risk_level.value,
            "total_steps":            self._sim.stats.total_steps,
            "total_reward":           self._sim.stats.total_reward,
            "empathy_uses":           self._sim.stats.empathy_uses,
            "compensation_uses":      self._sim.stats.compensation_uses,
            "security_actions":       self._sim.stats.security_actions,
            "max_repetition_penalty": self._sim.stats.max_repetition_penalty,
            "final_verification":     self._sim.stats.final_verification,
            "final_patience":         self._sim.stats.final_patience,
            "final_deadline_mins":    self._sim.stats.final_deadline_mins,
            "negligence_triggered":   self._sim.stats.negligence_triggered,
            "cliff_edge_triggered":   self._sim.stats.cliff_edge_triggered,
            "outcome":                self._sim.stats.outcome,
        }

        return {
            "action_name":    action_type.value,
            "action_index":   _ACTION_TYPE_TO_INDEX[action_type],
            "step_result":    sr_dict,
            "episode_stats":  stats_dict,
            "loophole_flags": _build_loophole_flags(self._obs, self._sim.stats, sr_dict),
            "obs_text": {
                "message_history":   list(self._obs.message_history),
                "sentiment_summary": self._obs.sentiment_summary(),
            },
        }

    def _render_reset(self, info: dict) -> str:
        """Render the reset event to a human-readable string."""
        seed_info = info.get("episode_seed", {})
        risk      = seed_info.get("risk_level", "?")
        deadline  = seed_info.get("initial_deadline_mins", 0.0)
        mismatch  = seed_info.get("initial_mismatch", 0.0)
        patience  = seed_info.get("initial_patience", 0)

        msgs      = info.get("obs_text", {}).get("message_history", [])
        opening   = msgs[0][6:90] + "..." if msgs else "(no message)"

        risk_colour = {"LOW": _GREEN, "MEDIUM": _YELLOW, "HIGH": _RED}.get(
            risk, lambda t: t
        )

        lines = [
            "",
            _BOLD("╔══ CreatorCrisis-v1  NEW EPISODE " + "═" * 30) + "╗",
            f"  Risk tier   : {risk_colour(risk)}",
            f"  Seed        : {seed_info.get('seed', '?')}",
            f"  Deadline    : {deadline:.1f} mins",
            f"  Mismatch₀   : {mismatch:.3f}",
            f"  Patience₀   : {patience}",
            f"  Opening     : \"{opening}\"",
            _BOLD("╚" + "═" * 60) + "╝",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  GYMNASIUM REGISTRATION
#  Call gymnasium.make("CreatorCrisis-v1") after importing this module.
# ─────────────────────────────────────────────────────────────────────────────

def register() -> None:
    """
    Register CreatorCrisisEnv with Gymnasium's global registry.

    Call once at program start or in __init__.py:
        from server.environment import register
        register()
        env = gymnasium.make("CreatorCrisis-v1")
    """
    if "CreatorCrisis-v1" not in gym.registry:
        gym.register(
            id           = "CreatorCrisis-v1",
            entry_point  = "server.environment:CreatorCrisisEnv",
            max_episode_steps = 15,
            kwargs        = {},
        )


# ─────────────────────────────────────────────────────────────────────────────
#  SMOKE TEST (run as script: python server/environment.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    SEP = "═" * 65

    def _run_episode(
        env:    CreatorCrisisEnv,
        seed:   int,
        script: list[int],
        label:  str,
    ) -> None:
        print(f"\n{SEP}")
        print(f"  {label}  (seed={seed})")
        print(SEP)

        obs, info = env.reset(seed=seed)
        print(f"  Risk tier   : {info['episode_seed']['risk_level']}")
        print(f"  obs['vector'] shape : {obs['vector'].shape}  dtype={obs['vector'].dtype}")
        print(f"  obs['vector'][:5]   : {obs['vector'][:5]}")

        opening = info["obs_text"]["message_history"][0][6:80]
        print(f"  Opening     : \"{opening}...\"")
        print(f"  Sentiment   :\n{info['obs_text']['sentiment_summary']}")

        total_reward = 0.0
        for action_int in script:
            if not (0 <= action_int < _N_ACTIONS):
                continue
            obs, reward, terminated, truncated, info = env.step(action_int)
            total_reward += reward
            flags = info["loophole_flags"]
            sr    = info["step_result"]

            print(
                f"\n  ► {info['action_name']:28s}  "
                f"r={reward:+7.1f}  ({sr['reward_reason']})"
            )
            print(
                f"    vector[0:4]={obs['vector'][:4].round(3).tolist()}  "
                f"patience_signal={obs['patience_signal']}  "
                f"latency={flags['fix10_step_latency_ms']:.1f}ms"
            )
            if flags.get("fix2_mismatch_high"):
                print(f"    [FIX 2] Mismatch HIGH ({flags['fix2_evidence_mismatch']:.2f})")
            if flags.get("fix5_penalty_active"):
                print(f"    [FIX 5] Repetition ×{flags['fix5_rep_multiplier']:.0f}")
            if flags.get("fix6_empathy_exhausted"):
                print(f"    [FIX 6] Empathy exhausted")
            if flags.get("fix7_deadline_critical"):
                print(f"    [FIX 7] DEADLINE CRITICAL {flags['fix7_deadline_mins']:.1f}m")
            if flags.get("fix9_negligence_triggered"):
                print(f"    [FIX 9] NEGLIGENT ESCALATION ×3")
            if terminated or truncated:
                tag = "TERMINATED" if terminated else "TRUNCATED (budget)"
                print(f"\n  ★ Episode ended: {tag}")
                break

        stats = info["episode_stats"]
        print(f"\n  ── Episode Summary ──────────────────────────────────")
        print(f"     outcome     : {stats['outcome']}")
        print(f"     total steps : {stats['total_steps']}")
        print(f"     total reward: {stats['total_reward']:+.1f}")
        print(f"     verification: {stats['final_verification']:.2f}")
        print(f"     patience    : {stats['final_patience']}")
        print(f"     cliff-edge  : {stats['cliff_edge_triggered']}")
        print(f"     negligence  : {stats['negligence_triggered']}")
        print(f"     exploit_risk: {info['loophole_flags']['exploit_risk_score']:.1f}/5.0")

    # ── Build env ─────────────────────────────────────────────────────────
    env = CreatorCrisisEnv(render_mode=None)

    # T1: Check space definitions
    print(f"\n{SEP}")
    print("  T1: Space definitions")
    print(SEP)
    print(f"  action_space      : {env.action_space}")
    print(f"  observation_space : Dict with {len(env.observation_space.spaces)} keys")
    for k, v in env.observation_space.spaces.items():
        print(f"    {k:28s} → {v}")
    assert env.action_space.n == 6, "action_space must be Discrete(6)"
    assert "vector" in env.observation_space.spaces
    assert env.observation_space["vector"].shape == (15,)
    print(f"  PASS — action_space=Discrete(6), obs vector shape=(15,)")

    # T2: reset() returns correct types
    print(f"\n{SEP}")
    print("  T2: reset() return types")
    print(SEP)
    obs, info = env.reset(seed=42)
    assert isinstance(obs, dict)
    assert "vector" in obs
    assert obs["vector"].shape == (15,)
    assert obs["vector"].dtype == np.float32
    assert "episode_seed" in info
    assert "obs_text" in info
    assert "sentiment_summary" in info["obs_text"]
    print(f"  obs keys  : {list(obs.keys())}")
    print(f"  info keys : {list(info.keys())}")
    print(f"  PASS — reset() contract satisfied")

    # T3: step() returns correct 5-tuple types
    print(f"\n{SEP}")
    print("  T3: step() 5-tuple types")
    print(SEP)
    obs, reward, terminated, truncated, info = env.step(0)  # CHECK_IP_LOGS
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert "step_result" in info
    assert "episode_stats" in info
    assert "loophole_flags" in info
    assert "obs_text" in info
    print(f"  reward={reward:.2f}  terminated={terminated}  truncated={truncated}")
    print(f"  step_result keys : {list(info['step_result'].keys())[:6]}...")
    print(f"  loophole_flags   : {len(info['loophole_flags'])} keys")
    print(f"  PASS — step() 5-tuple contract satisfied")

    # T4: Fix 1 — truncation at step 15
    print(f"\n{SEP}")
    print("  T4: Fix 1 — truncated=True at step 15 (budget cap)")
    print(SEP)
    env2 = CreatorCrisisEnv()
    obs, _ = env2.reset(seed=13)
    terminated2 = truncated2 = False
    for i in range(20):
        if terminated2 or truncated2:
            break
        # alternate between safe non-terminal actions
        action_int = 2 if i % 2 == 0 else 0
        obs, r, terminated2, truncated2, info2 = env2.step(action_int)
    assert truncated2, f"expected truncated=True after 15 steps, got terminated={terminated2}"
    assert info2["episode_stats"]["total_steps"] == 15
    print(f"  Steps taken: {info2['episode_stats']['total_steps']}")
    print(f"  truncated={truncated2}  terminated={terminated2}")
    print(f"  PASS — Fix 1 budget cap fires correctly")

    # T5: Full episode runs via _run_episode
    _run_episode(env, seed=42,
        label="T5: Exemplary agent (verify → restore)",
        script=[2, 0, 2, 1, 4],   # EMPATHIZE, IP, EMPATHIZE, VOICE, RESTORE
    )

    _run_episode(env, seed=7,
        label="T6: Fix A — Deliberate negligence (spam VOICE_VERIFY)",
        script=[0, 1, 1, 1, 1],   # IP then hammer VOICE → rage-quit ×3
    )

    _run_episode(env, seed=0,
        label="T7: HIGH risk hacker — correct rejection",
        script=[2, 0, 2, 1, 5],   # empathize, IP, empathize, VOICE, REJECT
    )

    # T8: register() and gymnasium.make()
    print(f"\n{SEP}")
    print("  T8: gymnasium.make() via register()")
    print(SEP)
    register()
    env3 = gym.make("CreatorCrisis-v1")
    obs3, info3 = env3.reset(seed=99)
    assert "vector" in obs3
    assert obs3["vector"].shape == (15,)
    obs3, r3, t3, tr3, i3 = env3.step(2)
    print(f"  gymnasium.make OK — reward={r3:.2f}  obs_shape={obs3['vector'].shape}")
    print(f"  PASS — registration and make() work correctly")
    env3.close()

    print(f"\n{SEP}")
    print("  All 8 tests passed. CreatorCrisisEnv is production-ready.")
    print(SEP)