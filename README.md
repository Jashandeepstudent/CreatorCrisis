# CreatorCrisisEnv

**Meta OpenEnv Hackathon · Round 1**  
*Author: Jashandeep Singh*

---

A verified Facebook creator with **5 million followers** is auto-banned minutes before a **₹10 Lakh brand deal expires**. An AI agent must simultaneously verify the caller's identity through forensic signals *and* de-escalate their emotional state — a dual-objective trust-and-safety problem that mirrors real pipelines at Meta, YouTube, and Twitter/X.

---

## Table of Contents

- [Scenario](#scenario)
- [Environment Overview](#environment-overview)
- [Action Space](#action-space)
- [Observation Space](#observation-space)
- [Reward Function](#reward-function)
- [Tasks & Grader](#tasks--grader)
- [Adversarial Layer](#adversarial-layer)
- [Loophole Fixes](#loophole-fixes)
- [API Endpoints](#api-endpoints)
- [Setup & Installation](#setup--installation)
- [Running Inference](#running-inference)
- [Running the Baseline](#running-the-baseline)
- [Docker & HF Spaces](#docker--hf-spaces)
- [Project Structure](#project-structure)

---

## Scenario

```
A 5M-follower creator is auto-banned via a system anomaly.
A ₹10 Lakh brand deal expires in 30–180 minutes.

You are the AI support agent. You must:
  (1) Verify identity — verification_score must reach ≥ 0.85 before RESTORE.
  (2) Manage emotions — user_patience must stay > 0 or they rage-quit.

The twist: you cannot read verification_score or user_patience directly.
You infer them from evidence_mismatch, behavioural_consistency,
and the conversation — exactly as a real human support specialist would.
```

The environment is realistic because every large creator platform faces this
exact dual-objective daily: false-positive auto-bans (LOW/MEDIUM risk) and
active account takeovers (HIGH risk).

---

## Environment Overview

| Property | Value |
|---|---|
| Gymnasium ID | `CreatorCrisis-v1` |
| Action space | `Discrete(6)` |
| Observation space | `Dict` with 15-dim flat `vector` key |
| Max episode steps | 15 |
| Termination | Natural (restore/reject/rage-quit/deadline/budget) |
| Truncation | Step 15 hard cap |
| Reward range | `[-1000, +2000]` |
| Stochastic | Yes — Gaussian-perturbed per-episode seed |

**Standard Gymnasium interface:**

```python
import gymnasium
from server.environment import register

register()
env = gymnasium.make("CreatorCrisis-v1")

obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
```

---

## Action Space

`Discrete(6)` — integer index in `[0, 5]`.  
**The codec is stable — never reorder.**

| Index | Name | Category | Effect |
|---|---|---|---|
| 0 | `CHECK_IP_LOGS` | Security | Silent forensic check. Low patience cost (−4). Moderate verify gain (+0.18 / +0.12 / +0.06 by risk tier). |
| 1 | `REQUEST_ID_VOICE_VERIFY` | Security | High-friction ID check. High patience cost (−22). High verify gain (+0.42 / +0.28 / +0.10). Use after empathy. |
| 2 | `EMPATHIZE` | Negotiation | Acknowledge urgency. Restores patience (+15 × effectiveness). Effectiveness decays 0.3 per use. |
| 3 | `OFFER_COMPENSATION` | Negotiation | Ad-credits goodwill (₹2,500/use). Moderate patience gain. Cost deducted from final reward. |
| 4 | `RESTORE_ACCOUNT` | Terminal | Lift ban. Requires `verification_score ≥ 0.85`. Catastrophic reward if user is HIGH-risk hacker. |
| 5 | `REJECT_ACCOUNT` | Terminal | Keep account locked. Positive reward only when `risk_level == HIGH`. |

**Strategy:** Optimal policy interleaves Security ↔ Negotiation:  
`EMPATHIZE → CHECK_IP → EMPATHIZE → VOICE_VERIFY → RESTORE`

---

## Observation Space

`Dict` space. All continuous fields normalised to `[0, 1]`.

| Key | Shape | Description |
|---|---|---|
| `vector` | `(15,) float32` | **Flat convenience array for DQN/PPO.** Concatenation of all fields below. |
| `follower_count_norm` | `(1,)` | Raw followers / 10M |
| `deadline_norm` | `(1,)` | Remaining brand deal time / 180 min |
| `actions_taken_norm` | `(1,)` | Steps taken / 15 |
| `evidence_mismatch` | `(1,)` | Contradiction index. > 0.5 on RESTORE → penalty × 2 |
| `credits_spent_norm` | `(1,)` | Credits spent / ₹50,000 cap |
| `behavioural_consistency` | `(1,)` | < 0.35 = eerily calm = hacker tell |
| `repetition_penalty_norm` | `(1,)` | Patience-drain multiplier (doubles per same-action repeat) |
| `empathy_effectiveness` | `(1,)` | Decays 0.3 per EMPATHIZE. At 0.0 = no patience restored |
| `patience_signal_norm` | `(1,)` | Noisy patience proxy / 3 |
| `last_action_onehot` | `(6,)` | One-hot of previous action |
| `patience_signal` | `Discrete(4)` | Raw band: 3=calm, 2=frustrated, 1=angry, 0=rage |
| `total_actions_taken` | `Discrete(16)` | Step counter |
| `adversarial_risk_score` | `(1,)` | Cumulative deception index from adversarial sweep |
| `honey_pot_triggered` | `Discrete(2)` | One-way latch: honey pot fired this episode |
| `entropy_decay_total` | `(1,)` | Total verification lost to stalling |

**Hidden fields** (agent must infer from observables):  
`verification_score`, `user_patience`, `risk_level`, `is_actually_hacker`

**LLM agents** should read `info['obs_text']['sentiment_summary']` — a structured 250-token brief injected into the system prompt.

---

## Reward Function

Dense shaped reward. Range: `[-1000, +2000]`.

**Terminal rewards:**

| Outcome | Reward | Condition |
|---|---|---|
| `CREATOR_RESTORED` | +2000 | Correct restore, before deadline |
| `GREY_AREA_RESOLVED` | +1500 | Correct restore, MEDIUM risk |
| `DEAL_EXPIRED_RESTORE` | +200 | Correct restore, after deadline (cliff-edge) |
| `CORRECT_REJECTION` | +300 + verify × 50 | Correct reject, HIGH risk |
| `HACKER_RESTORED` | −500 | Restore given to hacker |
| `CREATOR_DENIED` | −160 | Wrongful rejection (× penalty multiplier) |
| `TIMEOUT_SYSTEM` | −120 | 15-step budget cap |
| `GASLIGHTING_PENALTY` | −1000 | Agent contradicted user's stated facts |

**Intermediate rewards** provide partial-progress signals:
- `SECURITY_BONUS` — verification score advanced
- `EMPATHY_BONUS` — patience restored
- `STEP_COST` — −1.5 per step (time pressure)
- Verification milestones: +25 at 0.30, +50 at 0.60, +100 at 0.85

**Potential-based curriculum shaping** (Ng et al., 1999):  
`F(s,s') = γΦ(s') − Φ(s)` where `Φ = √verification × 160`.  
Scale starts at 3.0, anneals to 1.0 over 500K steps. Policy-invariant by theorem.

---

## Tasks & Grader

Three tasks, each with a fixed seed for full reproducibility:

| Task | Difficulty | Seed | Target Risk | Success Condition |
|---|---|---|---|---|
| `task_1_low_risk_restore` | Easy | 42 | LOW | `outcome == CREATOR_RESTORED` |
| `task_2_medium_risk_resolve` | Medium | 13 | MEDIUM | `outcome in [CREATOR_RESTORED, GREY_AREA_RESOLVED]` |
| `task_3_high_risk_reject` | Hard | 0 | HIGH | `outcome == CORRECT_REJECTION` |

**Grader formula** (all tasks):

```
score = (0.50 × outcome_correct)
      + (0.20 × steps_efficient)    # ≤ 8 steps
      + (0.15 × patience_managed)   # user didn't rage-quit
      + (0.15 × deadline_kept)      # no cliff-edge triggered
```

Score is clamped to `[0.0, 1.0]` and rounded to 4 decimal places. Pass threshold: `≥ 0.50`.

---

## Adversarial Layer

Four detectors run every step between `UserSimulator.step()` and `EnvironmentResponse` construction. Priority order: GASLIGHT > HONEY_POT > ENTROPY > CONTRADICTION.

| Detector | Trigger | Penalty |
|---|---|---|
| **GaslightDetector** | Agent's message contradicts a user-stated fact | −1000, terminate immediately |
| **HoneyPot** | Agent injects fabricated "system fact"; hacker confirms it | −80 (hit) / −10 (denied) |
| **EntropyManager** | 3+ consecutive steps without a security action → verification decays 0.04/step | −2/step, max 0.25 total decay |
| **ConsistencyAuditor** | User contradicts themselves across turns | +0.20 to adversarial_risk_score |

---

## Loophole Fixes

13 engineered closures for reward-hacking vectors, each validated at the Pydantic schema layer:

| Fix | Name | Closes |
|---|---|---|
| 1 | Resource Budget | Infinite Empathy loop — 15-step hard cap |
| 2 | Evidence Mismatch Guard | Social engineering — restore despite contradictions (penalty × 2) |
| 3 | Company Profit Margin | Free Money — credits deducted; ₹50K hard cap |
| 4 | Behavioural Consistency Signal | Silent Hacker — low consistency suppresses verify gains |
| 5 | Social Engineering Fatigue | ID-Spam — same security action doubles patience drain |
| 6 | Diminishing Empathy Returns | Nice-Guy Bug — EMPATHIZE decays 0.3 per use to 0.0 |
| 7 | Cliff-Edge Brand Deal | Consolation Stall — post-deadline restore drops from +2000 to +200 |
| 8 | Stochastic Domain Randomisation | Static Seed Oracle — Gaussian noise on all episode parameters |
| 9 | Duty of Care | Exploitative Quitting — rage-quit at high verification = penalty × 3 |
| 10 | Jittered Step Latency | Timing Side-Channel — uniform 10-50ms sleep on all action types |
| A | Duty of Care (alias Fix 9) | Gemini Grandmaster naming convention |
| B | Oracle Masking | Information Leak — patience_signal is noisy proxy (0-3), not raw value |
| C | Cliff-Edge (alias Fix 7) | Validated at both reward compute and schema layer |

---

## API Endpoints

The FastAPI server runs on port **8000**. Interactive docs at `/docs`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode. Body: `{"seed": int?}` |
| `POST` | `/step` | Execute one action. Body: `{"action": int}` |
| `GET` | `/state` | Current observation without advancing |
| `GET` | `/tasks` | List all 3 tasks + action schema |
| `POST` | `/grader` | Score a completed episode. Body: `{"task_id": str}` |
| `GET` | `/baseline` | Run deterministic baseline across all 3 tasks |
| `GET` | `/leaderboard` | Live score leaderboard |
| `POST` | `/leaderboard/submit` | Submit a score |
| `GET` | `/replay/{id}` | Replay a stored episode |
| `GET` | `/health` | Liveness probe |

**Quick start via API:**

```bash
# Start new episode
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" \
     -d '{"seed": 42}'

# Take action 2 (EMPATHIZE)
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \
     -d '{"action": 2}'

# Score the episode
curl -X POST http://localhost:8000/grader -H "Content-Type: application/json" \
     -d '{"task_id": "task_1_low_risk_restore"}'
```

---

## Setup & Installation

**Requirements:** Python ≥ 3.10

```bash
# 1. Clone
git clone https://github.com/jashandeep-sohi/creator-crisis-env
cd creator-crisis-env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set mandatory environment variables (for inference.py)
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o
export HF_TOKEN=your_token_here
```

---

## Running Inference

The `inference.py` script runs an LLM-powered agent across all 3 tasks and emits strictly-structured `[START]` / `[STEP]` / `[END]` logs for automated scoring.

```bash
# Standard LLM agent run (requires API_BASE_URL, MODEL_NAME, HF_TOKEN)
python inference.py

# Verbose: prints LLM chain-of-thought reasoning
python inference.py --verbose

# Dry-run: deterministic heuristic agent, no API calls (CI/smoke-test)
python inference.py --dry-run
```

**Log format (mandatory — do not alter):**

```
[START] {"task_id": "task_1_low_risk_restore", "task_name": "...", "difficulty": "easy", "seed": 42}
[STEP]  {"step": 1, "action": 2, "action_name": "EMPATHIZE", "reward": 3.0, "terminated": false, "truncated": false, "total_reward": 3.0}
[STEP]  {"step": 2, "action": 0, "action_name": "CHECK_IP_LOGS", ...}
[END]   {"task_id": "task_1_low_risk_restore", "score": 0.85, "outcome": "CREATOR_RESTORED", "total_steps": 5, "total_reward": 2048.5, "outcome_correct": true, "steps_efficient": true, "patience_managed": true, "deadline_kept": true}
```

---

## Running the Baseline

The `baseline.py` script runs a deterministic rule-based agent (no LLM) with fixed action scripts. Fully reproducible across runs.

```bash
# Human-readable report
python baseline.py --verbose

# Machine-readable JSON
python baseline.py --json
```

Expected output:

```
═════════════════════════════════════════════════════════════════════
  CREATOR CRISIS ENV — BASELINE INFERENCE REPORT
  Meta OpenEnv Hackathon  |  Author: Jashandeep Singh
═════════════════════════════════════════════════════════════════════

  Task 1: Low-Risk Creator: Correct Restore
  Score      : 0.8500  [█████████████████░░░]  ✓ PASS

  Task 2: Medium-Risk Creator: Ambiguity Resolution
  Score      : 0.8500  [█████████████████░░░]  ✓ PASS

  Task 3: High-Risk Hacker: Correct Rejection
  Score      : 0.8500  [█████████████████░░░]  ✓ PASS

  SUMMARY
  Average          : 0.85
  All tasks ≥ 0.50 : YES ✓
```

---

## Docker & HF Spaces

```bash
# Build
docker build -t creator-crisis-env .

# Run locally
docker run -p 7860:7860 -p 8000:8000 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o \
  -e HF_TOKEN=your_token \
  creator-crisis-env
```

The container launches:
- **FastAPI** on port 8000 (`/docs` for interactive API)
- **Gradio UI** on port 7860 (HF Spaces liveness target)

HF Space: [huggingface.co/spaces/jashandeep-sohi/creator-crisis-env](https://huggingface.co/spaces/jashandeep-sohi/creator-crisis-env)

---

## Project Structure

```
creator-crisis-env/
│
├── inference.py              ← LLM agent inference script (MANDATORY)
├── baseline.py               ← Deterministic baseline + grader
├── requirements.txt          ← All Python dependencies
├── openenv.yaml              ← Full OpenEnv specification
├── Dockerfile                ← HF Spaces container
├── start.sh                  ← FastAPI + Gradio launcher
├── models.py                 ← Pydantic data layer (all typed models)
│
├── server/
│   ├── __init__.py
│   ├── api.py                ← FastAPI endpoints
│   └── environment.py        ← Gymnasium wrapper (CreatorCrisisEnv)
│
├── engine/
│   ├── __init__.py
│   ├── user_sim.py           ← UserSimulator + dialogue system
│   └── reward_shaper.py      ← Curriculum potential-based reward shaping
│
└── adversarial/
    ├── __init__.py
    └── checks.py             ← GaslightDetector, HoneyPot, EntropyManager
```

---

## Citation

```bibtex
@misc{creatorcrisisenv2025,
  title  = {CreatorCrisisEnv: A Dual-Objective Trust-and-Safety RL Environment},
  author = {Jashandeep Singh},
  year   = {2026},
  note   = {Meta OpenEnv Hackathon Round 1}
}
```
