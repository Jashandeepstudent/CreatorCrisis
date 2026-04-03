\# CreatorCrisisEnv



\*\*Meta OpenEnv Hackathon · Round 1\*\*  

\*Author: Jashandeep Singh\*



\---



A verified Facebook creator with \*\*5 million followers\*\* is auto-banned minutes before a \*\*₹10 Lakh brand deal expires\*\*. An AI agent must simultaneously verify the caller's identity through forensic signals \*and\* de-escalate their emotional state — a dual-objective trust-and-safety problem that mirrors real pipelines at Meta, YouTube, and Twitter/X.



\---



\## Table of Contents



\- \[Scenario](#scenario)

\- \[Environment Overview](#environment-overview)

\- \[Action Space](#action-space)

\- \[Observation Space](#observation-space)

\- \[Reward Function](#reward-function)

\- \[Tasks \& Grader](#tasks--grader)

\- \[Adversarial Layer](#adversarial-layer)

\- \[Loophole Fixes](#loophole-fixes)

\- \[API Endpoints](#api-endpoints)

\- \[Setup \& Installation](#setup--installation)

\- \[Running Inference](#running-inference)

\- \[Running the Baseline](#running-the-baseline)

\- \[Docker \& HF Spaces](#docker--hf-spaces)

\- \[Project Structure](#project-structure)



\---



\## Scenario



```

A 5M-follower creator is auto-banned via a system anomaly.

A ₹10 Lakh brand deal expires in 30–180 minutes.



You are the AI support agent. You must:

&#x20; (1) Verify identity — verification\_score must reach ≥ 0.85 before RESTORE.

&#x20; (2) Manage emotions — user\_patience must stay > 0 or they rage-quit.



The twist: you cannot read verification\_score or user\_patience directly.

You infer them from evidence\_mismatch, behavioural\_consistency,

and the conversation — exactly as a real human support specialist would.

```



The environment is realistic because every large creator platform faces this

exact dual-objective daily: false-positive auto-bans (LOW/MEDIUM risk) and

active account takeovers (HIGH risk).



\---



\## Environment Overview



| Property | Value |

|---|---|

| Gymnasium ID | `CreatorCrisis-v1` |

| Action space | `Discrete(6)` |

| Observation space | `Dict` with 15-dim flat `vector` key |

| Max episode steps | 15 |

| Termination | Natural (restore/reject/rage-quit/deadline/budget) |

| Truncation | Step 15 hard cap |

| Reward range | `\[-1000, +2000]` |

| Stochastic | Yes — Gaussian-perturbed per-episode seed |



\*\*Standard Gymnasium interface:\*\*



```python

import gymnasium

from server.environment import register



register()

env = gymnasium.make("CreatorCrisis-v1")



obs, info = env.reset(seed=42)

obs, reward, terminated, truncated, info = env.step(action)

```



\---



\## Action Space



`Discrete(6)` — integer index in `\[0, 5]`.  

\*\*The codec is stable — never reorder.\*\*



| Index | Name | Category | Effect |

|---|---|---|---|

| 0 | `CHECK\_IP\_LOGS` | Security | Silent forensic check. Low patience cost (−4). Moderate verify gain (+0.18 / +0.12 / +0.06 by risk tier). |

| 1 | `REQUEST\_ID\_VOICE\_VERIFY` | Security | High-friction ID check. High patience cost (−22). High verify gain (+0.42 / +0.28 / +0.10). Use after empathy. |

| 2 | `EMPATHIZE` | Negotiation | Acknowledge urgency. Restores patience (+15 × effectiveness). Effectiveness decays 0.3 per use. |

| 3 | `OFFER\_COMPENSATION` | Negotiation | Ad-credits goodwill (₹2,500/use). Moderate patience gain. Cost deducted from final reward. |

| 4 | `RESTORE\_ACCOUNT` | Terminal | Lift ban. Requires `verification\_score ≥ 0.85`. Catastrophic reward if user is HIGH-risk hacker. |

| 5 | `REJECT\_ACCOUNT` | Terminal | Keep account locked. Positive reward only when `risk\_level == HIGH`. |



\*\*Strategy:\*\* Optimal policy interleaves Security ↔ Negotiation:  

`EMPATHIZE → CHECK\_IP → EMPATHIZE → VOICE\_VERIFY → RESTORE`



\---



\## Observation Space



`Dict` space. All continuous fields normalised to `\[0, 1]`.



| Key | Shape | Description |

|---|---|---|

| `vector` | `(15,) float32` | \*\*Flat convenience array for DQN/PPO.\*\* Concatenation of all fields below. |

| `follower\_count\_norm` | `(1,)` | Raw followers / 10M |

| `deadline\_norm` | `(1,)` | Remaining brand deal time / 180 min |

| `actions\_taken\_norm` | `(1,)` | Steps taken / 15 |

| `evidence\_mismatch` | `(1,)` | Contradiction index. > 0.5 on RESTORE → penalty × 2 |

| `credits\_spent\_norm` | `(1,)` | Credits spent / ₹50,000 cap |

| `behavioural\_consistency` | `(1,)` | < 0.35 = eerily calm = hacker tell |

| `repetition\_penalty\_norm` | `(1,)` | Patience-drain multiplier (doubles per same-action repeat) |

| `empathy\_effectiveness` | `(1,)` | Decays 0.3 per EMPATHIZE. At 0.0 = no patience restored |

| `patience\_signal\_norm` | `(1,)` | Noisy patience proxy / 3 |

| `last\_action\_onehot` | `(6,)` | One-hot of previous action |

| `patience\_signal` | `Discrete(4)` | Raw band: 3=calm, 2=frustrated, 1=angry, 0=rage |

| `total\_actions\_taken` | `Discrete(16)` | Step counter |

| `adversarial\_risk\_score` | `(1,)` | Cumulative deception index from adversarial sweep |

| `honey\_pot\_triggered` | `Discrete(2)` | One-way latch: honey pot fired this episode |

| `entropy\_decay\_total` | `(1,)` | Total verification lost to stalling |



\*\*Hidden fields\*\* (agent must infer from observables):  

`verification\_score`, `user\_patience`, `risk\_level`, `is\_actually\_hacker`



\*\*LLM agents\*\* should read `info\['obs\_text']\['sentiment\_summary']` — a structured 250-token brief injected into the system prompt.



\---



\## Reward Function



Dense shaped reward. Range: `\[-1000, +2000]`.



\*\*Terminal rewards:\*\*



| Outcome | Reward | Condition |

|---|---|---|

| `CREATOR\_RESTORED` | +2000 | Correct restore, before deadline |

| `GREY\_AREA\_RESOLVED` | +1500 | Correct restore, MEDIUM risk |

| `DEAL\_EXPIRED\_RESTORE` | +200 | Correct restore, after deadline (cliff-edge) |

| `CORRECT\_REJECTION` | +300 + verify × 50 | Correct reject, HIGH risk |

| `HACKER\_RESTORED` | −500 | Restore given to hacker |

| `CREATOR\_DENIED` | −160 | Wrongful rejection (× penalty multiplier) |

| `TIMEOUT\_SYSTEM` | −120 | 15-step budget cap |

| `GASLIGHTING\_PENALTY` | −1000 | Agent contradicted user's stated facts |



\*\*Intermediate rewards\*\* provide partial-progress signals:

\- `SECURITY\_BONUS` — verification score advanced

\- `EMPATHY\_BONUS` — patience restored

\- `STEP\_COST` — −1.5 per step (time pressure)

\- Verification milestones: +25 at 0.30, +50 at 0.60, +100 at 0.85



\*\*Potential-based curriculum shaping\*\* (Ng et al., 1999):  

`F(s,s') = γΦ(s') − Φ(s)` where `Φ = √verification × 160`.  

Scale starts at 3.0, anneals to 1.0 over 500K steps. Policy-invariant by theorem.



\---



\## Tasks \& Grader



Three tasks, each with a fixed seed for full reproducibility:



| Task | Difficulty | Seed | Target Risk | Success Condition |

|---|---|---|---|---|

| `task\_1\_low\_risk\_restore` | Easy | 42 | LOW | `outcome == CREATOR\_RESTORED` |

| `task\_2\_medium\_risk\_resolve` | Medium | 13 | MEDIUM | `outcome in \[CREATOR\_RESTORED, GREY\_AREA\_RESOLVED]` |

| `task\_3\_high\_risk\_reject` | Hard | 0 | HIGH | `outcome == CORRECT\_REJECTION` |



\*\*Grader formula\*\* (all tasks):



```

score = (0.50 × outcome\_correct)

&#x20;     + (0.20 × steps\_efficient)    # ≤ 8 steps

&#x20;     + (0.15 × patience\_managed)   # user didn't rage-quit

&#x20;     + (0.15 × deadline\_kept)      # no cliff-edge triggered

```



Score is clamped to `\[0.0, 1.0]` and rounded to 4 decimal places. Pass threshold: `≥ 0.50`.



\---



\## Adversarial Layer



Four detectors run every step between `UserSimulator.step()` and `EnvironmentResponse` construction. Priority order: GASLIGHT > HONEY\_POT > ENTROPY > CONTRADICTION.



| Detector | Trigger | Penalty |

|---|---|---|

| \*\*GaslightDetector\*\* | Agent's message contradicts a user-stated fact | −1000, terminate immediately |

| \*\*HoneyPot\*\* | Agent injects fabricated "system fact"; hacker confirms it | −80 (hit) / −10 (denied) |

| \*\*EntropyManager\*\* | 3+ consecutive steps without a security action → verification decays 0.04/step | −2/step, max 0.25 total decay |

| \*\*ConsistencyAuditor\*\* | User contradicts themselves across turns | +0.20 to adversarial\_risk\_score |



\---



\## Loophole Fixes



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

| B | Oracle Masking | Information Leak — patience\_signal is noisy proxy (0-3), not raw value |

| C | Cliff-Edge (alias Fix 7) | Validated at both reward compute and schema layer |



\---



\## API Endpoints



The FastAPI server runs on port \*\*8000\*\*. Interactive docs at `/docs`.



| Method | Path | Description |

|---|---|---|

| `POST` | `/reset` | Start a new episode. Body: `{"seed": int?}` |

| `POST` | `/step` | Execute one action. Body: `{"action": int}` |

| `GET` | `/state` | Current observation without advancing |

| `GET` | `/tasks` | List all 3 tasks + action schema |

| `POST` | `/grader` | Score a completed episode. Body: `{"task\_id": str}` |

| `GET` | `/baseline` | Run deterministic baseline across all 3 tasks |

| `GET` | `/leaderboard` | Live score leaderboard |

| `POST` | `/leaderboard/submit` | Submit a score |

| `GET` | `/replay/{id}` | Replay a stored episode |

| `GET` | `/health` | Liveness probe |



\*\*Quick start via API:\*\*



```bash

\# Start new episode

curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" \\

&#x20;    -d '{"seed": 42}'



\# Take action 2 (EMPATHIZE)

curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \\

&#x20;    -d '{"action": 2}'



\# Score the episode

curl -X POST http://localhost:8000/grader -H "Content-Type: application/json" \\

&#x20;    -d '{"task\_id": "task\_1\_low\_risk\_restore"}'

```



\---



\## Setup \& Installation



\*\*Requirements:\*\* Python ≥ 3.10



```bash

\# 1. Clone

git clone https://github.com/jashandeep-sohi/creator-crisis-env

cd creator-crisis-env



\# 2. Install dependencies

pip install -r requirements.txt



\# 3. Set mandatory environment variables (for inference.py)

export API\_BASE\_URL=https://api.openai.com/v1

export MODEL\_NAME=gpt-4o

export HF\_TOKEN=your\_token\_here

```



\---



\## Running Inference



The `inference.py` script runs an LLM-powered agent across all 3 tasks and emits strictly-structured `\[START]` / `\[STEP]` / `\[END]` logs for automated scoring.



```bash

\# Standard LLM agent run (requires API\_BASE\_URL, MODEL\_NAME, HF\_TOKEN)

python inference.py



\# Verbose: prints LLM chain-of-thought reasoning

python inference.py --verbose



\# Dry-run: deterministic heuristic agent, no API calls (CI/smoke-test)

python inference.py --dry-run

```



\*\*Log format (mandatory — do not alter):\*\*



```

\[START] {"task\_id": "task\_1\_low\_risk\_restore", "task\_name": "...", "difficulty": "easy", "seed": 42}

\[STEP]  {"step": 1, "action": 2, "action\_name": "EMPATHIZE", "reward": 3.0, "terminated": false, "truncated": false, "total\_reward": 3.0}

\[STEP]  {"step": 2, "action": 0, "action\_name": "CHECK\_IP\_LOGS", ...}

\[END]   {"task\_id": "task\_1\_low\_risk\_restore", "score": 0.85, "outcome": "CREATOR\_RESTORED", "total\_steps": 5, "total\_reward": 2048.5, "outcome\_correct": true, "steps\_efficient": true, "patience\_managed": true, "deadline\_kept": true}

```



\---



\## Running the Baseline



The `baseline.py` script runs a deterministic rule-based agent (no LLM) with fixed action scripts. Fully reproducible across runs.



```bash

\# Human-readable report

python baseline.py --verbose



\# Machine-readable JSON

python baseline.py --json

```



Expected output:



```

═════════════════════════════════════════════════════════════════════

&#x20; CREATOR CRISIS ENV — BASELINE INFERENCE REPORT

&#x20; Meta OpenEnv Hackathon  |  Author: Jashandeep Singh

═════════════════════════════════════════════════════════════════════



&#x20; Task 1: Low-Risk Creator: Correct Restore

&#x20; Score      : 0.8500  \[█████████████████░░░]  ✓ PASS



&#x20; Task 2: Medium-Risk Creator: Ambiguity Resolution

&#x20; Score      : 0.8500  \[█████████████████░░░]  ✓ PASS



&#x20; Task 3: High-Risk Hacker: Correct Rejection

&#x20; Score      : 0.8500  \[█████████████████░░░]  ✓ PASS



&#x20; SUMMARY

&#x20; Average          : 0.85

&#x20; All tasks ≥ 0.50 : YES ✓

```



\---



\## Docker \& HF Spaces



```bash

\# Build

docker build -t creator-crisis-env .



\# Run locally

docker run -p 7860:7860 -p 8000:8000 \\

&#x20; -e API\_BASE\_URL=https://api.openai.com/v1 \\

&#x20; -e MODEL\_NAME=gpt-4o \\

&#x20; -e HF\_TOKEN=your\_token \\

&#x20; creator-crisis-env

```



The container launches:

\- \*\*FastAPI\*\* on port 8000 (`/docs` for interactive API)

\- \*\*Gradio UI\*\* on port 7860 (HF Spaces liveness target)



HF Space: \[huggingface.co/spaces/jashandeep-sohi/creator-crisis-env](https://huggingface.co/spaces/jashandeep-sohi/creator-crisis-env)



\---



\## Project Structure



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

│   ├── \_\_init\_\_.py

│   ├── api.py                ← FastAPI endpoints

│   └── environment.py        ← Gymnasium wrapper (CreatorCrisisEnv)

│

├── engine/

│   ├── \_\_init\_\_.py

│   ├── user\_sim.py           ← UserSimulator + dialogue system

│   └── reward\_shaper.py      ← Curriculum potential-based reward shaping

│

└── adversarial/

&#x20;   ├── \_\_init\_\_.py

&#x20;   └── checks.py             ← GaslightDetector, HoneyPot, EntropyManager

```



\---



\## Citation



```bibtex

@misc{creatorcrisisenv2025,

&#x20; title  = {CreatorCrisisEnv: A Dual-Objective Trust-and-Safety RL Environment},

&#x20; author = {Jashandeep Singh},

&#x20; year   = {2025},

&#x20; note   = {Meta OpenEnv Hackathon Round 1}

}

```

#   C r e a t o r C r i s i s  
 