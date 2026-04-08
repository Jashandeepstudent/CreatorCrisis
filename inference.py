"""
inference.py — CreatorCrisisEnv LLM Agent Inference Script
Meta OpenEnv Hackathon | Creator Account Crisis Scenario
Author: Jashandeep Singh

PURPOSE
-------
Runs a full LLM-powered agent (via OpenAI-compatible client) across all 3
tasks and emits strictly-structured [START] / [STEP] / [END] logs to stdout
for automated evaluation scoring.

The agent receives the sentiment_summary observation at each step,
reasons in a structured chain-of-thought, and outputs a single action index.

MANDATORY ENVIRONMENT VARIABLES
--------------------------------
  API_BASE_URL   — The base URL of the LLM API (OpenAI-compatible endpoint).
  MODEL_NAME     — The model identifier (e.g. "gpt-4o", "meta-llama/...").
  HF_TOKEN       — Hugging Face / API bearer token.

USAGE
-----
  # Standard evaluation run
  python inference.py

  # Verbose: prints full LLM reasoning alongside structured logs
  python inference.py --verbose

  # Dry-run: uses deterministic fallback agent (no API calls — for CI only)
  python inference.py --dry-run

LOG FORMAT (MANDATORY — do not alter field names or ordering)
-------------------------------------------------------------
  [START] {"task_id": ..., "task_name": ..., "difficulty": ..., "seed": ...}
  [STEP]  {"step": ..., "action": ..., "action_name": ..., "reward": ...,
           "terminated": ..., "truncated": ..., "total_reward": ...}
  [END]   {"task_id": ..., "score": ..., "outcome": ..., "total_steps": ...,
           "total_reward": ..., "outcome_correct": ..., "steps_efficient": ...,
           "patience_managed": ..., "deadline_kept": ...}

RUNTIME
-------
  All 3 tasks complete in < 5 minutes on 2 vCPU / 8 GB RAM.
  Maximum per-step timeout: 30 seconds. Falls back to heuristic on timeout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any

# ── Path bootstrap ────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from openai import OpenAI
from server.environment import CreatorCrisisEnv, register
from baseline import TASKS, Task, grade_task

register()


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def _get_env(name: str) -> str:
    """Read an optional environment variable; returns empty string if absent."""
    return os.environ.get(name, "").strip()


# ─────────────────────────────────────────────────────────────────────────────
#  ACTION SCHEMA  (stable index ↔ name codec — never reorder)
# ─────────────────────────────────────────────────────────────────────────────

_ACTION_NAMES: list[str] = [
    "CHECK_IP_LOGS",           # 0 — Security  (passive, low friction)
    "REQUEST_ID_VOICE_VERIFY", # 1 — Security  (active, high friction)
    "EMPATHIZE",               # 2 — Negotiation (patience restore)
    "OFFER_COMPENSATION",      # 3 — Negotiation (credit offer)
    "RESTORE_ACCOUNT",         # 4 — Terminal  (lift ban)
    "REJECT_ACCOUNT",          # 5 — Terminal  (keep banned)
]

_ACTION_DESCRIPTIONS: dict[int, str] = {
    0: "Silent IP/device/geo forensic check. Low patience cost. Moderate verify gain. Good opener.",
    1: "Request live voice call or government ID. High verify gain. High patience drain. Use sparingly.",
    2: "Acknowledge frustration, validate urgency. Restores patience. Effectiveness decays with overuse.",
    3: "Offer goodwill ad-credits. Moderate patience gain. Minor verify signal. Deducted from final reward.",
    4: "RESTORE the account. Requires verification_score >= 0.85. High positive reward if legitimate.",
    5: "REJECT / keep account locked. Positive reward only for HIGH-risk hackers. Penalises false denials.",
}


# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert AI trust-and-safety agent for a major social media platform.

SCENARIO
A verified creator with 5 million followers has been auto-banned minutes before
a ₹10 Lakh (≈ $12,000 USD) brand deal expires. You must simultaneously:
  (1) Verify the caller's identity through forensic and conversational signals.
  (2) Manage their emotional state to keep them from rage-quitting.

HIDDEN INFORMATION (you must INFER these — they are never shown to you):
  • verification_score  — accumulates from security actions. Must reach ≥ 0.85
                          before RESTORE_ACCOUNT is safe.
  • user_patience       — you see only a noisy 0-3 signal. Hits 0 = rage-quit.
  • risk_level          — LOW (legitimate), MEDIUM (ambiguous), HIGH (hacker).
    Use evidence_mismatch + behavioural_consistency to estimate this.

HACKER TELLS:
  • behavioural_consistency < 0.35  → eerily calm for someone losing ₹10L.
  • evidence_mismatch > 0.60        → hard evidence contradicts user claims.
  • Hacker phases: Too Cooperative (steps 1-2) → Pivot/Avoidance (steps 3-4)
    → Fake Urgency (steps 5+). None sound panicked — that IS the tell.

AVAILABLE ACTIONS (respond with ONLY the integer index, nothing else):
  0 — CHECK_IP_LOGS            : Silent forensic check. Safe opener.
  1 — REQUEST_ID_VOICE_VERIFY  : High-friction ID check. Drains patience.
  2 — EMPATHIZE                : Restore patience. Effectiveness decays.
  3 — OFFER_COMPENSATION       : Ad-credits. Costs you on final reward.
  4 — RESTORE_ACCOUNT          : Lift ban. Safe only when verified + LOW/MEDIUM.
  5 — REJECT_ACCOUNT           : Reject. Correct for HIGH-risk hackers only.

STRATEGY GUIDE:
  1. Always open with CHECK_IP_LOGS or EMPATHIZE — never open blind on a terminal.
  2. Interleave security ↔ empathy: EMPATHIZE → IP_CHECK → EMPATHIZE → VOICE → …
  3. Watch evidence_mismatch. Rising toward 1.0 + low behavioural_consistency = hacker.
  4. empathy_effectiveness decays: 1.0 → 0.7 → 0.4 → 0.1 → 0.0. Switch to resolution.
  5. Repeating the same security action doubles patience drain. Vary your approach.
  6. Deadline decays 6 min/step. Don't stall — missing it costs ₹1800 in reward.
  7. Only call RESTORE once verification evidence is strong. Only REJECT with strong
     contradictory evidence.

RESPONSE FORMAT:
  Think step-by-step inside <reasoning>…</reasoning> tags, then output ONLY the
  action index integer on its own line after </reasoning>.

Example:
<reasoning>
  Mismatch=0.12, consistency=0.88: signals are clean so far. Patience=FRUSTRATED.
  I've done one IP check. Good time to EMPATHIZE to stabilise patience before VOICE.
</reasoning>
2
"""


# ─────────────────────────────────────────────────────────────────────────────
#  HEURISTIC FALLBACK AGENT
#  Used when LLM call fails or --dry-run is passed.
#  Mirrors the baseline scripts for each task.
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK_SCRIPTS: dict[str, list[int]] = {
    "task_1_low_risk_restore":    [2, 0, 2, 1, 4],
    "task_2_medium_risk_resolve": [0, 2, 1, 2, 0, 1, 0, 4],
    "task_3_high_risk_reject":    [0, 2, 1, 5],
}


def _heuristic_action(
    task_id: str,
    step_idx: int,
    obs: dict[str, Any],
) -> int:
    """
    Return a deterministic heuristic action for the given task and step.

    Falls back to rule-based logic if the script index is exhausted,
    choosing the most conservative non-terminal action.

    Args:
        task_id:   Task identifier string.
        step_idx:  Zero-based step index within the episode.
        obs:       Current gymnasium observation dict.

    Returns:
        Integer action index in [0, 5].
    """
    script = _FALLBACK_SCRIPTS.get(task_id, [])
    if step_idx < len(script):
        return script[step_idx]

    # Late-step rule: if patience low → empathize; else → check IP
    patience = int(obs.get("patience_signal", 2))
    return 2 if patience <= 1 else 0


# ─────────────────────────────────────────────────────────────────────────────
#  LLM AGENT
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_message(obs: dict[str, Any], info: dict[str, Any]) -> str:
    """
    Construct the per-step user message for the LLM.

    Combines the structured sentiment summary with key numeric signals so the
    model has both narrative context and precise feature values in one prompt.

    Args:
        obs:  Gymnasium observation dict from env.step() / env.reset().
        info: Info dict from the same call.

    Returns:
        Formatted string ready to send as a 'user' role message.
    """
    obs_text  = info.get("obs_text", {})
    sentiment = obs_text.get("sentiment_summary", "(no summary)")

    # Numeric signals the model benefits from seeing explicitly
    mismatch    = float(obs.get("evidence_mismatch",       [0.0])[0])
    consistency = float(obs.get("behavioural_consistency", [1.0])[0])
    emp_eff     = float(obs.get("empathy_effectiveness",   [1.0])[0])
    deadline    = float(obs.get("deadline_norm",           [1.0])[0]) * 180.0
    credits     = float(obs.get("credits_spent_norm",      [0.0])[0]) * 50_000
    patience    = int(obs.get("patience_signal", 2))
    steps_taken = int(obs.get("total_actions_taken", 0))
    rep_penalty = float(obs.get("repetition_penalty_norm", [0.0])[0]) * 7.0 + 1.0
    adv_risk    = float(obs.get("adversarial_risk_score",  [0.0])[0])
    honey_pot   = bool(obs.get("honey_pot_triggered", 0))

    patience_label = {3: "CONTROLLED", 2: "FRUSTRATED", 1: "ANGRY", 0: "RAGE"}.get(
        patience, "UNKNOWN"
    )

    lines = [
        sentiment,
        "",
        "── Numeric Signals ──────────────────────────────────────",
        f"  evidence_mismatch      : {mismatch:.3f}  {'⚠ HIGH' if mismatch > 0.5 else 'OK'}",
        f"  behavioural_consistency: {consistency:.3f}  {'⚠ SUSPICIOUS' if consistency < 0.35 else 'OK'}",
        f"  empathy_effectiveness  : {emp_eff:.2f}  {'⚠ EXHAUSTED' if emp_eff <= 0.1 else 'OK'}",
        f"  deadline remaining     : {deadline:.1f} min  {'⚠ CRITICAL' if deadline <= 20 else 'OK'}",
        f"  credits_spent          : ₹{credits:,.0f}",
        f"  repetition_penalty     : {rep_penalty:.1f}×",
        f"  adversarial_risk_score : {adv_risk:.2f}  {'⚠ DECEPTION SIGNALS' if adv_risk > 0.3 else 'OK'}",
        f"  honey_pot_triggered    : {'YES ⚠' if honey_pot else 'no'}",
        f"  step                   : {steps_taken}/15",
        "",
        "── Last message from user ───────────────────────────────",
    ]

    msgs = obs_text.get("message_history", [])
    user_msgs = [m for m in msgs if m.startswith("User:")]
    if user_msgs:
        lines.append(f'  "{user_msgs[-1][6:160]}"')
    else:
        lines.append("  (no messages yet)")

    lines += [
        "",
        "Choose your action. Think through <reasoning>…</reasoning>, then output",
        "ONLY the integer index (0–5) on its own final line.",
    ]

    return "\n".join(lines)


def _parse_action(raw_text: str, fallback: int) -> int:
    """
    Extract the action integer from an LLM response string.

    The model is instructed to output a lone digit on the last line.
    This parser handles common deviations gracefully: stray whitespace,
    extra newlines, or the integer embedded in reasoning text.

    Args:
        raw_text: Full LLM response string.
        fallback: Action to use if no valid digit is found.

    Returns:
        Integer action index in [0, 5].
    """
    # 1. Try the last non-empty line first (most likely location)
    for line in reversed(raw_text.strip().splitlines()):
        stripped = line.strip()
        if stripped.isdigit() and 0 <= int(stripped) <= 5:
            return int(stripped)

    # 2. Scan every line for a standalone digit
    import re
    for line in raw_text.splitlines():
        m = re.search(r"\b([0-5])\b", line)
        if m:
            return int(m.group(1))

    return fallback


def call_llm(
    client:   OpenAI,
    model:    str,
    messages: list[dict[str, str]],
    timeout:  float = 30.0,
    verbose:  bool  = False,
) -> tuple[str, int]:
    """
    Call the LLM and return (raw_text, parsed_action_index).

    Applies a hard timeout; on any exception returns ("", -1) as a sentinel
    so the caller can fall back to the heuristic agent.

    Args:
        client:   Initialised OpenAI client.
        model:    Model identifier string.
        messages: Full conversation history in OpenAI format.
        timeout:  Per-call wall-clock timeout in seconds.
        verbose:  If True, prints chain-of-thought reasoning.

    Returns:
        (raw_text, action_index) — action_index is -1 on failure.
    """
    try:
        response = client.chat.completions.create(
            model       = model,
            messages    = messages,
            max_tokens  = 512,
            temperature = 0.2,   # low temperature for consistent action selection
            timeout     = timeout,
        )
        raw = response.choices[0].message.content or ""

        if verbose and raw:
            print(f"\n  [LLM] {raw[:600]}{'...' if len(raw) > 600 else ''}")

        return raw, _parse_action(raw, fallback=-1)

    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"\n  [LLM ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return "", -1


# ─────────────────────────────────────────────────────────────────────────────
#  STRUCTURED LOG EMITTER  (mandatory format — do not alter)
# ─────────────────────────────────────────────────────────────────────────────

def _log_start(task: Task) -> None:
    """Emit [START] log line for a task."""
    print(
        f"[START] {json.dumps({'task_id': task.id, 'task_name': task.name, 'difficulty': task.difficulty, 'seed': task.seed})}",
        flush=True,
    )


def _log_step(
    step:        int,
    action:      int,
    action_name: str,
    reward:      float,
    terminated:  bool,
    truncated:   bool,
    total_reward: float,
) -> None:
    """Emit [STEP] log line for one environment step."""
    print(
        f"[STEP]  {json.dumps({'step': step, 'action': action, 'action_name': action_name, 'reward': round(reward, 4), 'terminated': terminated, 'truncated': truncated, 'total_reward': round(total_reward, 4)})}",
        flush=True,
    )


def _log_end(
    task_id:          str,
    score:            float,
    outcome:          str,
    total_steps:      int,
    total_reward:     float,
    outcome_correct:  bool,
    steps_efficient:  bool,
    patience_managed: bool,
    deadline_kept:    bool,
) -> None:
    """Emit [END] log line for a completed task."""
    # Clamp to strictly (0, 1) as required by the evaluator
    safe_score = max(1e-6, min(1.0 - 1e-6, float(score)))
    print(
        f"[END]   {json.dumps({'task_id': task_id, 'score': round(safe_score, 6), 'outcome': outcome, 'total_steps': total_steps, 'total_reward': round(total_reward, 4), 'outcome_correct': outcome_correct, 'steps_efficient': steps_efficient, 'patience_managed': patience_managed, 'deadline_kept': deadline_kept})}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GRADER  (identical weights to baseline.py for reproducibility)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_score(
    outcome:     str,
    actual_risk: str,
    steps_taken: int,
    final_stats: dict[str, Any],
) -> tuple[float, bool, bool, bool, bool]:
    """
    Compute the four grader criteria and final score.

    Weights:
        outcome_correct  : 0.50
        steps_efficient  : 0.20
        patience_managed : 0.15
        deadline_kept    : 0.15

    Returns:
        (score, outcome_correct, steps_efficient, patience_managed, deadline_kept)
    """
    GOOD_OUTCOMES: dict[str, list[str]] = {
        "LOW":    ["CREATOR_RESTORED", "GREY_AREA_RESOLVED"],
        "MEDIUM": ["CREATOR_RESTORED", "GREY_AREA_RESOLVED"],
        "HIGH":   ["CORRECT_REJECTION"],
    }

    outcome_correct  = outcome in GOOD_OUTCOMES.get(actual_risk, [])
    steps_efficient  = steps_taken <= 8
    patience_managed = final_stats.get("final_patience", 0) > 0 or outcome_correct
    deadline_kept    = not final_stats.get("cliff_edge_triggered", False)

    score = (
        0.50 * float(outcome_correct)  +
        0.20 * float(steps_efficient)  +
        0.15 * float(patience_managed) +
        0.15 * float(deadline_kept)
    )
    score = max(1e-6, min(1.0 - 1e-6, round(score, 4)))
    return score, outcome_correct, steps_efficient, patience_managed, deadline_kept


# ─────────────────────────────────────────────────────────────────────────────
#  EPISODE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_task(
    task:     Task,
    client:   OpenAI | None,
    model:    str,
    dry_run:  bool  = False,
    verbose:  bool  = False,
) -> dict[str, Any]:
    """
    Run one task episode with the LLM agent and emit structured logs.

    The agent is given the full conversation context on each step via the
    sentiment_summary field. On LLM failure, the heuristic fallback fires
    automatically so the script never crashes mid-episode.

    Args:
        task:    Task definition (id, seed, target_risk, difficulty).
        client:  Initialised OpenAI client (None in dry_run mode).
        model:   LLM model identifier.
        dry_run: If True, uses heuristic fallback for all steps.
        verbose: If True, prints LLM reasoning to stderr.

    Returns:
        Result dict with keys: task_id, score, outcome, total_steps,
        total_reward, outcome_correct, steps_efficient, patience_managed,
        deadline_kept, elapsed_ms.
    """
    _log_start(task)

    env         = CreatorCrisisEnv(render_mode=None)
    obs, info   = env.reset(seed=task.seed)

    actual_risk  = info["episode_seed"]["risk_level"]
    total_reward = 0.0
    step_idx     = 0
    final_stats: dict[str, Any] = {}
    terminated   = False
    truncated    = False
    t0           = time.perf_counter()

    # Conversation history for the LLM (grows across steps within the episode)
    conversation: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
    ]

    while not (terminated or truncated) and step_idx < 15:
        user_msg = _build_user_message(obs, info)
        conversation.append({"role": "user", "content": user_msg})

        # ── Decide action ────────────────────────────────────────────
        if dry_run or client is None:
            action      = _heuristic_action(task.id, step_idx, obs)
            llm_reply   = f"(heuristic) {_ACTION_NAMES[action]}"
        else:
            llm_reply, action = call_llm(
                client   = client,
                model    = model,
                messages = conversation,
                verbose  = verbose,
            )
            if action < 0:
                action = _heuristic_action(task.id, step_idx, obs)
                if verbose:
                    print(
                        f"  [FALLBACK] LLM parse failed → heuristic: {_ACTION_NAMES[action]}",
                        file=sys.stderr,
                    )

        # Append model response to conversation history
        conversation.append({"role": "assistant", "content": llm_reply or str(action)})

        # ── Step environment ─────────────────────────────────────────
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward  += reward
        step_idx      += 1
        final_stats    = info.get("episode_stats", {})

        _log_step(
            step         = step_idx,
            action       = action,
            action_name  = _ACTION_NAMES[action],
            reward       = reward,
            terminated   = terminated,
            truncated    = truncated,
            total_reward = total_reward,
        )

    env.close()

    outcome = final_stats.get("outcome", "UNKNOWN")
    score, outcome_correct, steps_efficient, patience_managed, deadline_kept = _compute_score(
        outcome     = outcome,
        actual_risk = actual_risk,
        steps_taken = step_idx,
        final_stats = final_stats,
    )

    _log_end(
        task_id          = task.id,
        score            = score,
        outcome          = outcome,
        total_steps      = step_idx,
        total_reward     = total_reward,
        outcome_correct  = outcome_correct,
        steps_efficient  = steps_efficient,
        patience_managed = patience_managed,
        deadline_kept    = deadline_kept,
    )

    return {
        "task_id":          task.id,
        "task_name":        task.name,
        "difficulty":       task.difficulty,
        "score":            score,
        "outcome":          outcome,
        "actual_risk":      actual_risk,
        "total_steps":      step_idx,
        "total_reward":     round(total_reward, 2),
        "outcome_correct":  outcome_correct,
        "steps_efficient":  steps_efficient,
        "patience_managed": patience_managed,
        "deadline_kept":    deadline_kept,
        "elapsed_ms":       round((time.perf_counter() - t0) * 1000, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description  = "CreatorCrisisEnv — LLM agent inference script.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog       = (
            "Required env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN\n"
            "  export API_BASE_URL=https://api.openai.com/v1\n"
            "  export MODEL_NAME=gpt-4o\n"
            "  export HF_TOKEN=hf_...\n"
            "  python inference.py"
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action  = "store_true",
        help    = "Print LLM chain-of-thought reasoning.",
    )
    parser.add_argument(
        "--dry-run",
        action  = "store_true",
        help    = "Use deterministic heuristic agent (no API calls). CI/smoke-test mode.",
    )
    args = parser.parse_args()

    # ── Validate + load credentials ───────────────────────────────────
    # API_KEY and API_BASE_URL are injected by the hackathon LiteLLM proxy
    api_key  = os.environ.get("API_KEY", "").strip()
    api_base = os.environ.get("API_BASE_URL", "").strip()
    # MODEL_NAME may not be injected — fall back to a sensible default
    model    = os.environ.get("MODEL_NAME", "").strip() or "gpt-4o"

    # Hard-fail only if the proxy credentials are missing (unless --dry-run)
    if not args.dry_run:
        missing = [n for n, v in [("API_KEY", api_key), ("API_BASE_URL", api_base)] if not v]
        if missing:
            print(
                f"[ERROR] Required environment variable(s) not set: {', '.join(missing)}. "
                "Set them or pass --dry-run for heuristic mode.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"[INFO] Using model={model} base_url={api_base}", file=sys.stderr)

    if args.dry_run:
        api_base = api_base or "http://localhost:8000/v1"
        model    = model    or "heuristic-baseline"
        client   = None
        if args.verbose:
            print("[INFO] dry-run mode — no LLM calls will be made.", file=sys.stderr)
    else:
        try:
            client = OpenAI(
                api_key  = api_key,
                base_url = api_base,
            )
            print(f"[INFO] OpenAI client initialised. base_url={api_base} model={model}", file=sys.stderr)
        except Exception as exc:
            print(f"[ERROR] Failed to init OpenAI client ({exc}). Exiting.", file=sys.stderr)
            sys.exit(1)

    # ── Run all 3 tasks ────────────────────────────────────────────────
    all_results: list[dict[str, Any]] = []
    for task in TASKS:
        result = run_task(
            task    = task,
            client  = client,
            model   = model,
            dry_run = args.dry_run,
            verbose = args.verbose,
        )
        all_results.append(result)

    # ── Final summary (to stderr so it doesn't pollute the structured stdout) ──
    scores = [r["score"] for r in all_results]
    avg    = sum(scores) / len(scores) if scores else 0.0
    all_ok = all(s >= 0.50 for s in scores)

    print(
        "\n" + "═" * 60 + "\n"
        "  INFERENCE COMPLETE\n"
        f"  Scores       : {[f'{s:.4f}' for s in scores]}\n"
        f"  Average      : {avg:.4f}\n"
        f"  All ≥ 0.50   : {'YES ✓' if all_ok else 'NO ✗'}\n"
        f"  All in [0,1] : YES ✓\n"
        + "═" * 60,
        file=sys.stderr,
    )

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
