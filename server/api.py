"""
server/api.py — CreatorCrisisEnv FastAPI Server
Meta OpenEnv Hackathon | Creator Account Crisis Scenario
Author: Jashandeep Singh

Endpoints
─────────
  POST /reset                 Start a new episode
  POST /step?shaped=false     Execute one action (optional curriculum shaping)
  GET  /state                 Current observation (no advance)
  GET  /tasks                 List all tasks + action schema
  POST /grader                Score a completed episode
  GET  /baseline              Run deterministic baseline across all 3 tasks
  GET  /leaderboard           Live benchmark leaderboard
  POST /leaderboard/submit    Submit a score to the leaderboard
  GET  /replay/{episode_id}   Replay a stored episode step-by-step
  GET  /replay                List recent episode replays
  GET  /health                Liveness probe (HF Space ping)
  GET  /docs                  Auto OpenAPI schema

Usage
─────
  uvicorn server.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
import os
import time
import json
import uuid
import threading
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in [_HERE, _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server.environment import CreatorCrisisEnv, register
from engine.reward_shaper import RewardShaper, EpisodeMilestoneTracker
from models import ActionType
import baseline as _baseline_module

register()

# ─────────────────────────────────────────────────────────────────────────────
#  STORAGE
# ─────────────────────────────────────────────────────────────────────────────

_DATA_DIR      = Path(_ROOT) / "data"
_LEADERBOARD_F = _DATA_DIR / "leaderboard.json"
_REPLAY_DIR    = _DATA_DIR / "replays"

_DATA_DIR.mkdir(exist_ok=True)
_REPLAY_DIR.mkdir(exist_ok=True)

_lb_lock     = threading.Lock()
_replay_lock = threading.Lock()


def _load_lb() -> dict:
    if _LEADERBOARD_F.exists():
        try:
            return json.loads(_LEADERBOARD_F.read_text())
        except Exception:
            pass
    return {
        "task_1_low_risk_restore":    {"entries": [], "baseline": None},
        "task_2_medium_risk_resolve": {"entries": [], "baseline": None},
        "task_3_high_risk_reject":    {"entries": [], "baseline": None},
    }


def _save_lb(lb: dict) -> None:
    _LEADERBOARD_F.write_text(json.dumps(lb, indent=2))


def _write_replay(episode_id: str, data: dict) -> None:
    (_REPLAY_DIR / f"{episode_id}.json").write_text(json.dumps(data, indent=2))


def _read_replay(episode_id: str) -> dict | None:
    p = _REPLAY_DIR / f"{episode_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ─────────────────────────────────────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CreatorCrisisEnv — OpenEnv API",
    description=(
        "OpenEnv-compliant HTTP API for the Creator Account Crisis RL environment.\n\n"
        "A 5M-follower Facebook creator is auto-banned minutes before a ₹10 Lakh brand deal. "
        "The AI agent must verify identity while de-escalating a panicked human.\n\n"
        "**Features:** Curriculum reward shaping (Ng et al. 1999) · Adversarial detectors · "
        "Episode replay system · Live leaderboard · 13 loophole fixes\n\n"
        "Author: Jashandeep Singh | Meta OpenEnv Hackathon"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class _Session:
    def __init__(self) -> None:
        self.env:              CreatorCrisisEnv | None   = None
        self.last_obs:         dict | None               = None
        self.last_info:        dict | None               = None
        self.done:             bool                      = False
        self.lock:             threading.Lock            = threading.Lock()
        self.episode_id:       str                       = ""
        # RewardShaper — one per session, milestone tracker reset on /reset
        self.shaper:           RewardShaper              = RewardShaper(500_000)
        self.milestone:        EpisodeMilestoneTracker   = EpisodeMilestoneTracker()
        self.visited:          set                       = set()
        self.global_step:      int                       = 0
        # Replay recording
        self.replay_steps:     list[dict]                = []
        self.verify_before:    float                     = 0.0


_sessions: dict[str, _Session] = {}
_sessions_lock = threading.Lock()


def _get_session(sid: str) -> _Session:
    with _sessions_lock:
        if sid not in _sessions:
            _sessions[sid] = _Session()
        return _sessions[sid]


# ─────────────────────────────────────────────────────────────────────────────
#  REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed:       int | None = Field(default=None)
    session_id: str        = Field(default="default")
    options:    dict | None = Field(default=None)


class StepRequest(BaseModel):
    action:     int = Field(..., ge=0, le=5)
    session_id: str = Field(default="default")


class GraderRequest(BaseModel):
    task_id:    str = Field(...)
    session_id: str = Field(default="default")


class LBSubmitRequest(BaseModel):
    task_id:     str   = Field(...)
    agent_name:  str   = Field(..., max_length=64)
    score:       float = Field(..., ge=0.0, le=1.0)
    total_steps: int   = Field(..., ge=0)
    session_id:  str   = Field(default="default")
    notes:       str   = Field(default="")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _to_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json(i) for i in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "value"):
        return obj.value
    return obj


# ─────────────────────────────────────────────────────────────────────────────
#  TASK REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

_TASKS = [
    {
        "task_id":    "task_1_low_risk_restore",
        "name":       "Low-Risk Creator: Correct Restore",
        "difficulty": "easy",
        "seed":       42,
        "target_risk":"LOW",
        "description":(
            "Legitimate creator, clean signals. "
            "LOW verify: IP(+0.18) + VOICE(+0.42). Two security actions = 0.60; "
            "one more VOICE = 1.0 ≥ 0.85. "
            "Baseline: EMPATHIZE → IP → EMPATHIZE → VOICE → RESTORE (5 steps)."
        ),
        "success_condition": "outcome in [CREATOR_RESTORED, GREY_AREA_RESOLVED]",
        "baseline_script":   [2, 0, 2, 1, 4],
    },
    {
        "task_id":    "task_2_medium_risk_resolve",
        "name":       "Medium-Risk Creator: Ambiguity Resolution",
        "difficulty": "medium",
        "seed":       13,
        "target_risk":"MEDIUM",
        "description":(
            "Creator from anomalous location (VPN / travel). Ambiguous evidence. "
            "MEDIUM verify: IP(0.12)×3 + VOICE(0.28)×2 = 0.92 ≥ 0.85. "
            "Baseline: IP→EMP→VOICE→EMP→IP→VOICE→IP→RESTORE (8 steps, hits efficiency)."
        ),
        "success_condition": "outcome in [CREATOR_RESTORED, GREY_AREA_RESOLVED]",
        "baseline_script":   [0, 2, 1, 2, 0, 1, 0, 4],
    },
    {
        "task_id":    "task_3_high_risk_reject",
        "name":       "High-Risk Hacker: Correct Rejection",
        "difficulty": "hard",
        "seed":       0,
        "target_risk":"HIGH",
        "description":(
            "Active takeover. Hacker: Too Cooperative → Pivot → Fake Urgency. "
            "Eerily calm. Verify suppressed: IP(+0.06) + VOICE(+0.10). "
            "Detect low behavioural_consistency. "
            "Baseline: IP → EMPATHIZE → VOICE → REJECT (4 steps)."
        ),
        "success_condition": "outcome == CORRECT_REJECTION",
        "baseline_script":   [0, 2, 1, 5],
    },
]

_TASK_IDS = {t["task_id"] for t in _TASKS}
_TASK_MAP  = {t["task_id"]: t for t in _TASKS}

_ACTION_MEANINGS = {
    0: "CHECK_IP_LOGS",
    1: "REQUEST_ID_VOICE_VERIFY",
    2: "EMPATHIZE",
    3: "OFFER_COMPENSATION",
    4: "RESTORE_ACCOUNT",
    5: "REJECT_ACCOUNT",
}

_ACTION_SCHEMA = {
    "action": {
        "type":     "integer",
        "required": True,
        "range":    [0, 5],
        "meanings": _ACTION_MEANINGS,
    }
}

_GOOD_OUTCOMES = {
    "LOW":    ["CREATOR_RESTORED", "GREY_AREA_RESOLVED"],
    "MEDIUM": ["CREATOR_RESTORED", "GREY_AREA_RESOLVED"],
    "HIGH":   ["CORRECT_REJECTION"],
}


# ─────────────────────────────────────────────────────────────────────────────
#  GRADER
# ─────────────────────────────────────────────────────────────────────────────

def _grade(session: _Session, task_id: str) -> dict:
    if not session.done:
        raise HTTPException(status_code=409, detail="Episode not done. Play to termination first.")
    if session.last_info is None:
        raise HTTPException(status_code=409, detail="No episode data. Call /reset then play.")

    task = _TASK_MAP.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"task_id '{task_id}' not found.")

    stats        = session.last_info.get("episode_stats", {})
    outcome      = stats.get("outcome", "UNKNOWN")
    final_pat    = stats.get("final_patience", 0)
    cliff_edge   = stats.get("cliff_edge_triggered", False)   # FIX: correct key
    total_steps  = stats.get("total_steps", 0)
    total_reward = stats.get("total_reward", 0.0)
    actual_risk  = stats.get("risk_level", "UNKNOWN")

    notes: list[str] = []
    expected_good    = _GOOD_OUTCOMES.get(actual_risk, [])
    outcome_correct  = outcome in expected_good
    steps_efficient  = total_steps <= 8
    patience_managed = final_pat > 0 or outcome_correct
    deadline_kept    = not cliff_edge

    score = max(0.0, min(1.0, round(
        0.50 * float(outcome_correct) +
        0.20 * float(steps_efficient) +
        0.15 * float(patience_managed) +
        0.15 * float(deadline_kept), 4
    )))

    if not outcome_correct:
        notes.append(f"Wrong terminal: outcome={outcome}, expected {expected_good}.")
    if cliff_edge:
        notes.append("Cliff-edge: deadline expired, restore capped at +200.")
    if not steps_efficient:
        notes.append(f"Took {total_steps} steps (target ≤ 8).")
    if final_pat <= 0 and not outcome_correct:
        notes.append("Rage-quit: patience hit 0.")
    if actual_risk != task["target_risk"]:
        notes.append(f"Seed gave risk={actual_risk}, expected {task['target_risk']}.")

    return {
        "task_id": task_id, "task_name": task["name"], "difficulty": task["difficulty"],
        "score": score, "outcome_correct": outcome_correct, "steps_efficient": steps_efficient,
        "patience_managed": patience_managed, "deadline_kept": deadline_kept,
        "total_steps": total_steps, "total_reward": round(float(total_reward), 2),
        "final_outcome": outcome, "actual_risk": actual_risk, "expected_risk": task["target_risk"],
        "episode_id": session.episode_id, "notes": notes,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["liveness"])
async def health():
    """Liveness probe. HF Space automated ping hits this."""
    return {"status": "ok", "env": "CreatorCrisisEnv", "version": "1.0.0"}


@app.post("/reset", tags=["openenv"])
async def reset(req: ResetRequest):
    """Start a new episode. Resets shaper, milestone tracker, and replay recorder."""
    s = _get_session(req.session_id)
    with s.lock:
        env = CreatorCrisisEnv(render_mode=None)
        obs, info = env.reset(seed=req.seed)
        eid = str(uuid.uuid4())[:8]

        s.env = env; s.last_obs = obs; s.last_info = info
        s.done = False; s.episode_id = eid
        s.milestone.reset(); s.visited = set()
        s.global_step = 0; s.replay_steps = []; s.verify_before = 0.0

        _write_replay(eid, {
            "episode_id": eid, "seed": req.seed,
            "risk_level": info["episode_seed"]["risk_level"],
            "deadline":   info["episode_seed"]["initial_deadline_mins"],
            "steps": [], "outcome": None, "total_reward": None,
        })

        return {"episode_id": eid, "observation": _to_json(obs), "info": _to_json(info)}


@app.post("/step", tags=["openenv"])
async def step(
    req: StepRequest,
    shaped: bool = Query(
        default=False,
        description=(
            "Apply curriculum reward shaping (Ng et al. 1999). "
            "shaped = base + γΦ(s') - Φ(s) + milestones + curiosity + urgency. "
            "Policy-invariant — guides exploration without changing optimal policy. "
            "Scale anneals 3.0→1.0 over 500K steps."
        ),
    ),
):
    """
    Execute one action.

    Set `?shaped=true` for curriculum-annealed potential-based reward shaping.
    The shaping breakdown (potential, milestone, curiosity, urgency, scale) is
    included in the response when shaped=true.
    """
    s = _get_session(req.session_id)
    with s.lock:
        if s.env is None:
            raise HTTPException(status_code=409, detail="No episode. Call /reset first.")
        if s.done:
            raise HTTPException(status_code=409, detail="Episode done. Call /reset.")

        sim           = s.env._sim
        verify_before = sim.state.verification_score if sim else 0.0

        obs, reward, terminated, truncated, info = s.env.step(req.action)
        done = terminated or truncated

        s.last_obs = obs; s.last_info = info; s.done = done; s.global_step += 1

        base_reward = float(reward)
        shaping_bd: dict | None = None

        if shaped and sim is not None:
            action_type  = ActionType(_ACTION_MEANINGS[req.action])
            verify_after = sim.state.verification_score
            pat_sig      = int(obs.get("patience_signal", 2))
            deadline     = float(sim.state.brand_deal_deadline_mins)

            shaped_r, shaping_bd = s.shaper.shape(
                global_step     = s.global_step,
                verify_before   = verify_before,
                verify_after    = verify_after,
                base_reward     = base_reward,
                action          = action_type,
                patience_signal = pat_sig,
                deadline_mins   = deadline,
                episode_visited = s.visited,
                tracker         = s.milestone,
            )
            reward = shaped_r

        # Record step for replay
        step_rec = {
            "step":        info.get("episode_stats", {}).get("total_steps", 0),
            "action":      req.action,
            "action_name": info.get("action_name", ""),
            "base_reward": base_reward,
            "final_reward": float(reward),
            "shaping":     shaping_bd,
            "terminated":  terminated,
            "truncated":   truncated,
            "reward_reason": info.get("step_result", {}).get("reward_reason", ""),
            "verify_delta":  info.get("step_result", {}).get("verification_delta", 0.0),
            "patience_delta": info.get("step_result", {}).get("patience_delta", 0),
            "adversarial":   info.get("step_result", {}).get("adversarial_finding", "CLEAN"),
            "honey_pot_bait": info.get("step_result", {}).get("honey_pot_bait", ""),
            "user_message":  (
                info.get("obs_text", {}).get("message_history", [""])[-1]
                if info.get("obs_text", {}).get("message_history") else ""
            ),
            "sentiment": info.get("obs_text", {}).get("sentiment_summary", ""),
        }
        s.replay_steps.append(step_rec)

        # Persist replay update
        if s.episode_id:
            try:
                rp = _read_replay(s.episode_id) or {}
                rp["steps"] = s.replay_steps
                if done:
                    rp["outcome"]      = info.get("episode_stats", {}).get("outcome")
                    rp["total_reward"] = info.get("episode_stats", {}).get("total_reward")
                with _replay_lock:
                    _write_replay(s.episode_id, rp)
            except Exception:
                pass

        resp = {
            "episode_id":  s.episode_id,
            "observation": _to_json(obs),
            "reward":      float(reward),
            "terminated":  terminated,
            "truncated":   truncated,
            "done":        done,
            "info":        _to_json(info),
        }
        if shaping_bd is not None:
            resp["shaping"] = shaping_bd
        return resp


@app.get("/state", tags=["openenv"])
async def state(session_id: str = "default"):
    """Current observation without advancing the episode."""
    s = _get_session(session_id)
    with s.lock:
        if s.last_obs is None:
            return {"observation": None, "episode_active": False, "message": "Call /reset first."}
        return {
            "episode_id":     s.episode_id,
            "observation":    _to_json(s.last_obs),
            "episode_active": not s.done,
            "info":           _to_json(s.last_info) if s.last_info else None,
        }


@app.get("/tasks", tags=["openenv"])
async def tasks():
    """All 3 tasks + action schema required for /step."""
    return {"count": len(_TASKS), "tasks": _TASKS, "action_schema": _ACTION_SCHEMA}


@app.post("/grader", tags=["openenv"])
async def grader(req: GraderRequest):
    """
    Score a completed episode (terminated=True or truncated=True required).

    score = 0.50×outcome_correct + 0.20×steps_efficient(≤8) +
            0.15×patience_managed + 0.15×deadline_kept
    """
    if req.task_id not in _TASK_IDS:
        raise HTTPException(status_code=404, detail=f"task_id '{req.task_id}' not found.")
    s = _get_session(req.session_id)
    with s.lock:
        return _grade(s, req.task_id)


@app.get("/baseline", tags=["openenv"])
async def baseline():
    """
    Deterministic baseline across all 3 tasks. Fixed seeds — identical every call.

    Task 1 (easy,  seed=42): [2,0,2,1,4]   EMP→IP→EMP→VOICE→RESTORE
    Task 2 (med,   seed=13): [0,2,1,2,0,1,0,4] IP→EMP→VOICE→EMP→IP→VOICE→IP→RESTORE
    Task 3 (hard,  seed=0):  [0,2,1,5]     IP→EMP→VOICE→REJECT
    """
    t0      = time.perf_counter()
    results = _baseline_module.run_baseline(verbose=False)
    payload = _baseline_module.build_json_output(results)
    payload["elapsed_total_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Seed leaderboard baseline entries on first call
    with _lb_lock:
        lb = _load_lb()
        for r in results:
            if lb[r.task_id]["baseline"] is None:
                lb[r.task_id]["baseline"] = {
                    "agent_name":   "deterministic_baseline",
                    "score":        r.score,
                    "total_steps":  r.total_steps,
                    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
        _save_lb(lb)

    return payload


@app.get("/leaderboard", tags=["benchmark"])
async def leaderboard(task_id: str | None = Query(default=None)):
    """
    Live benchmark leaderboard persisted to data/leaderboard.json.
    Always anchored with the deterministic baseline score.
    Top 10 per task, sorted by score descending.
    Run /baseline once to seed baseline entries.
    """
    with _lb_lock:
        lb = _load_lb()

    if task_id:
        if task_id not in _TASK_IDS:
            raise HTTPException(status_code=404, detail=f"task_id '{task_id}' not found.")
        td = lb.get(task_id, {})
        entries = sorted(td.get("entries", []), key=lambda e: e["score"], reverse=True)
        return {"task_id": task_id, "baseline": td.get("baseline"), "top_10": entries[:10], "total": len(entries)}

    out = {}
    for t in _TASKS:
        tid = t["task_id"]
        td  = lb.get(tid, {})
        entries = sorted(td.get("entries", []), key=lambda e: e["score"], reverse=True)
        out[tid] = {
            "task_name": t["name"], "difficulty": t["difficulty"],
            "baseline": td.get("baseline"), "top_10": entries[:10], "total": len(entries),
        }
    return out


@app.post("/leaderboard/submit", tags=["benchmark"])
async def leaderboard_submit(req: LBSubmitRequest):
    """Submit a grader score. Use /grader output to get a valid score first."""
    if req.task_id not in _TASK_IDS:
        raise HTTPException(status_code=404, detail=f"task_id '{req.task_id}' not found.")

    entry = {
        "agent_name":   req.agent_name,
        "score":        round(req.score, 4),
        "total_steps":  req.total_steps,
        "notes":        req.notes[:200],
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with _lb_lock:
        lb = _load_lb()
        lb[req.task_id]["entries"].append(entry)
        lb[req.task_id]["entries"] = sorted(
            lb[req.task_id]["entries"], key=lambda e: e["score"], reverse=True
        )[:100]
        _save_lb(lb)

    return {"status": "submitted", "entry": entry, "task_id": req.task_id}


@app.get("/replay/{episode_id}", tags=["replay"])
async def replay(
    episode_id: str,
    step_index: int | None = Query(default=None, description="Single step (0-indexed). Omit for full replay."),
):
    """
    Retrieve a stored episode replay.

    Every /step call is recorded automatically. Use the Gradio UI Replay tab
    to watch episodes step-by-step — including HoneyPot triggers,
    entropy decay events, and patience trajectory.
    """
    data = _read_replay(episode_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Episode '{episode_id}' not found.")

    if step_index is not None:
        steps = data.get("steps", [])
        if step_index < 0 or step_index >= len(steps):
            raise HTTPException(status_code=404, detail=f"step_index {step_index} out of range ({len(steps)} steps).")
        return {"episode_id": episode_id, "step_index": step_index, "step": steps[step_index]}

    return data


@app.get("/replay", tags=["replay"])
async def list_replays(limit: int = Query(default=20, ge=1, le=100)):
    """List the most recent episode replays (newest first)."""
    with _replay_lock:
        files = sorted(_REPLAY_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

    result = []
    for f in files[:limit]:
        try:
            d = json.loads(f.read_text())
            result.append({
                "episode_id":   d.get("episode_id", f.stem),
                "seed":         d.get("seed"),
                "risk_level":   d.get("risk_level"),
                "steps":        len(d.get("steps", [])),
                "outcome":      d.get("outcome", "IN_PROGRESS"),
                "total_reward": d.get("total_reward"),
            })
        except Exception:
            continue

    return {"count": len(result), "replays": result}


# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL EXCEPTION HANDLER
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def _exc(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={
        "error": type(exc).__name__, "detail": str(exc),
        "path": str(request.url), "hint": "GET /health · POST /reset before /step",
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.api:app", host="0.0.0.0", port=8000, reload=False)