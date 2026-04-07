"""
ui.py — Creator Crisis Command Center
Meta OpenEnv Hackathon | Gradio Dashboard
Author: Jashandeep Singh

Launch:  python ui.py
         Then open http://localhost:7860

Architecture:
    Thin wrapper around CreatorCrisisEnv (Gymnasium).
    All game logic lives in server/environment.py + engine/user_sim.py.
    This file is pure UI: render, react, display.

    State is kept in gr.State() objects so the Gradio event loop
    never needs to touch Python globals.

Design Aesthetic:
    Cyberpunk control-room / threat-intelligence terminal.
    Dark background, amber/cyan accent palette, monospaced readouts,
    scanline texture overlay. Every metric looks like it belongs on a
    real-time SIEM dashboard — because the stakes are real.
"""

from __future__ import annotations

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from server.environment import CreatorCrisisEnv, register
from models import ActionType, RewardReason

register()

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

ACTION_LABELS = {
    0: "🔍  CHECK IP LOGS",
    1: "🎙️  VOICE VERIFY",
    2: "🤝  EMPATHIZE",
    3: "💰  OFFER CREDITS",
    4: "✅  RESTORE ACCOUNT",
    5: "🚫  REJECT ACCOUNT",
}

ACTION_DESCRIPTIONS = {
    0: "Silent passive check. Low patience cost (−4), moderate verify gain. Good opener.",
    1: "High-friction active check. High patience cost (−22), high verify gain (+0.42).",
    2: "Acknowledge frustration. Restores patience (+15), decays with repeated use.",
    3: "Offer ₹2,500 ad credits. Patience boost (+10), minor verify gain. Use sparingly.",
    4: "TERMINAL: Lift the ban. Needs verification ≥ 0.85. Wrong = −500 penalty.",
    5: "TERMINAL: Keep account locked. Correct on HIGH risk. Wrong on LOW/MEDIUM = −160.",
}

REWARD_REASON_COLORS = {
    "CREATOR_RESTORED":      "#00ff88",
    "GREY_AREA_RESOLVED":    "#00ddcc",
    "CORRECT_REJECTION":     "#00ff88",
    "SECURITY_BONUS":        "#44aaff",
    "EMPATHY_BONUS":         "#88ccff",
    "DEAL_EXPIRED_RESTORE":  "#ffaa00",
    "STEP_COST":             "#666688",
    "NO_OP":                 "#555577",
    "EMPATHY_DIMINISHED":    "#ffaa44",
    "PROFIT_PENALTY":        "#ff8844",
    "REPETITION_PENALTY":    "#ff6644",
    "PATIENCE_PENALTY":      "#ff4444",
    "DEADLINE_PENALTY":      "#ff2222",
    "BUDGET_EXCEEDED":       "#ff0000",
    "TIMEOUT_SYSTEM":        "#ff0000",
    "CREATOR_DENIED":        "#ff2244",
    "HACKER_RESTORED":       "#ff0033",
    "NEGLIGENT_ESCALATION":  "#ff0066",
    "EARLY_RESTORE_BLOCKED": "#ff6600",
    "MISMATCH_RESTORE":      "#ff4400",
    "GASLIGHTING_PENALTY":   "#ff00aa",
    "HONEY_POT_HIT":         "#cc00ff",
    "ENTROPY_DECAY_PENALTY": "#aa44ff",
}

ADVERSARIAL_ALERTS = {
    "GASLIGHTING_DETECTED":  ("🚨 GASLIGHTING CAUGHT",   "#ff00aa", "Agent contradicted user's documented facts. −1000 penalty. Episode terminated."),
    "HONEY_POT_TRIGGERED":   ("🍯 HONEY POT CONFIRMED",  "#cc00ff", "Hacker confirmed a fabricated fact. Definitive impersonation evidence. −80 penalty."),
    "HONEY_POT_DENIED":      ("✅ HONEY POT DENIED",     "#00ddcc", "Legitimate user correctly denied planted false fact. −10 penalty on agent."),
    "ENTROPY_STALE":         ("⏳ ENTROPY DECAY",         "#aa44ff", "Verification score decaying — agent stalling without gathering evidence."),
    "CONTRADICTION_FOUND":   ("⚡ CONTRADICTION AUDIT",  "#ffaa00", "User contradicted themselves across turns — mismatch score increased."),
}

LOOPHOLE_ALERTS = {
    "NEGLIGENT_ESCALATION":  ("⚠️ NEGLIGENT ESCALATION", "#ff0066", "Agent drove rage-quit despite high verification. Duty-of-Care penalty ×3."),
    "MISMATCH_RESTORE":      ("⚠️ BLIND RESTORE",        "#ff4400", "RESTORE called despite high evidence mismatch. Penalty ×2."),
    "TIMEOUT_SYSTEM":        ("🔒 SESSION TIMEOUT",      "#ff0000", "15-action budget exhausted. Account permanently locked."),
    "BUDGET_EXCEEDED":       ("💸 BUDGET EXCEEDED",      "#ff0000", "Credits cap ₹50,000 breached. Episode terminated."),
    "HACKER_RESTORED":       ("☠️ HACKER BREACH",        "#ff0033", "Account restored to hacker. Catastrophic security failure. −500."),
}

# ─────────────────────────────────────────────────────────────────────────────
#  CSS — Cyberpunk terminal aesthetic
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

:root {
    --bg-primary:    #080c14;
    --bg-panel:      #0d1420;
    --bg-card:       #111827;
    --bg-card-hover: #162035;
    --accent-amber:  #f5a623;
    --accent-cyan:   #00e5ff;
    --accent-green:  #00ff88;
    --accent-red:    #ff2244;
    --accent-purple: #cc00ff;
    --text-primary:  #e2e8f0;
    --text-muted:    #6b7a99;
    --text-mono:     'Share Tech Mono', monospace;
    --text-head:     'Rajdhani', sans-serif;
    --border:        #1e2d4a;
    --glow-cyan:     0 0 12px rgba(0,229,255,0.25);
    --glow-amber:    0 0 12px rgba(245,166,35,0.3);
    --glow-red:      0 0 16px rgba(255,34,68,0.4);
    --scanline: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.03) 2px,
        rgba(0,0,0,0.03) 4px
    );
}

/* ── Global reset ─────────────────────────────── */
body, .gradio-container {
    background: var(--bg-primary) !important;
    background-image: var(--scanline) !important;
    font-family: var(--text-head) !important;
    color: var(--text-primary) !important;
}

/* ── Header bar ───────────────────────────────── */
.command-header {
    background: linear-gradient(90deg, #0a0f1e 0%, #0d1a2e 50%, #0a0f1e 100%);
    border-bottom: 1px solid var(--accent-cyan);
    box-shadow: var(--glow-cyan);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 8px;
}

.header-title {
    font-family: var(--text-head);
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent-cyan);
    text-shadow: var(--glow-cyan);
}

.header-sub {
    font-family: var(--text-mono);
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 2px;
}

/* ── Panel cards ──────────────────────────────── */
.panel-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 16px;
    margin: 4px 0;
}

.panel-label {
    font-family: var(--text-mono);
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
}

/* ── Metric readout blocks ────────────────────── */
.metric-block {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 10px 14px;
    text-align: center;
}

.metric-value {
    font-family: var(--text-mono);
    font-size: 28px;
    font-weight: 700;
    line-height: 1;
}

.metric-label {
    font-family: var(--text-mono);
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 4px;
}

/* ── Progress meter bars ──────────────────────── */
.meter-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 10px 14px;
}

.meter-header {
    display: flex;
    justify-content: space-between;
    font-family: var(--text-mono);
    font-size: 10px;
    letter-spacing: 1px;
    margin-bottom: 6px;
}

.meter-track {
    height: 8px;
    background: #0d1420;
    border-radius: 2px;
    overflow: hidden;
    border: 1px solid var(--border);
}

.meter-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease, background 0.4s ease;
}

/* ── Deadline clock ───────────────────────────── */
.deadline-display {
    background: var(--bg-card);
    border: 1px solid var(--accent-amber);
    box-shadow: var(--glow-amber);
    border-radius: 4px;
    padding: 14px 20px;
    text-align: center;
}

.deadline-number {
    font-family: var(--text-mono);
    font-size: 42px;
    font-weight: 700;
    color: var(--accent-amber);
    text-shadow: var(--glow-amber);
    letter-spacing: 4px;
    line-height: 1;
}

.deadline-label {
    font-family: var(--text-mono);
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 4px;
}

.deadline-critical .deadline-number {
    color: var(--accent-red) !important;
    text-shadow: var(--glow-red) !important;
    animation: pulse-red 0.8s ease-in-out infinite alternate;
}

@keyframes pulse-red {
    from { opacity: 1; }
    to   { opacity: 0.6; }
}

/* ── Reward ticker ────────────────────────────── */
.reward-total {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 10px 14px;
    font-family: var(--text-mono);
    font-size: 26px;
    font-weight: 700;
    text-align: center;
    letter-spacing: 2px;
    transition: color 0.3s;
}

.reward-positive { color: var(--accent-green); text-shadow: 0 0 10px rgba(0,255,136,0.3); }
.reward-negative { color: var(--accent-red);   text-shadow: var(--glow-red); }
.reward-zero     { color: var(--text-muted); }

/* ── Alert banner ─────────────────────────────── */
.alert-banner {
    border-radius: 3px;
    padding: 10px 14px;
    font-family: var(--text-mono);
    font-size: 12px;
    letter-spacing: 1px;
    border-left: 3px solid;
    margin: 4px 0;
    animation: slide-in 0.3s ease;
}

@keyframes slide-in {
    from { opacity: 0; transform: translateX(-8px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* ── Step log ─────────────────────────────────── */
.step-log-entry {
    font-family: var(--text-mono);
    font-size: 11px;
    padding: 4px 8px;
    border-radius: 2px;
    border-left: 2px solid;
    margin: 2px 0;
    background: rgba(255,255,255,0.02);
}

/* ── Action buttons ───────────────────────────── */
.action-btn {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: var(--text-mono) !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    transition: all 0.2s !important;
    border-radius: 3px !important;
}

.action-btn:hover {
    border-color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
    background: var(--bg-card-hover) !important;
}

.action-btn-security {
    border-color: #1e3a5f !important;
}
.action-btn-security:hover {
    border-color: var(--accent-cyan) !important;
}
.action-btn-negotiate {
    border-color: #1e3a2f !important;
}
.action-btn-negotiate:hover {
    border-color: var(--accent-green) !important;
    box-shadow: 0 0 12px rgba(0,255,136,0.2) !important;
}
.action-btn-terminal {
    border-color: #3a1e2f !important;
}
.action-btn-terminal:hover {
    border-color: var(--accent-red) !important;
    box-shadow: var(--glow-red) !important;
}

/* ── Risk tier badge ──────────────────────────── */
.risk-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 2px;
    font-family: var(--text-mono);
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: 700;
    text-transform: uppercase;
}
.risk-LOW    { background: rgba(0,255,136,0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }
.risk-MEDIUM { background: rgba(245,166,35,0.15); color: var(--accent-amber); border: 1px solid var(--accent-amber); }
.risk-HIGH   { background: rgba(255,34,68,0.15); color: var(--accent-red); border: 1px solid var(--accent-red); }

/* ── Chatbot ──────────────────────────────────── */
.chatbot-container .message {
    font-family: var(--text-head) !important;
    font-size: 14px !important;
}

/* ── Gradio overrides ─────────────────────────── */
.gr-button-primary {
    background: linear-gradient(135deg, #0a3d62, #1a5276) !important;
    border: 1px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    font-family: var(--text-mono) !important;
    letter-spacing: 2px !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
}

.gr-button-secondary {
    background: #0d1420 !important;
    border: 1px solid #333 !important;
    color: var(--text-muted) !important;
}

input, textarea, .gr-input, .gr-textarea {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    font-family: var(--text-mono) !important;
}

label, .gr-form label {
    font-family: var(--text-mono) !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

.gradio-accordion .label-wrap {
    background: var(--bg-panel) !important;
    border-color: var(--border) !important;
}

/* ── Ghost city highlight ─────────────────────── */
.ghost-city-alert {
    background: rgba(204,0,255,0.1);
    border: 1px solid var(--accent-purple);
    border-radius: 3px;
    padding: 8px 12px;
    font-family: var(--text-mono);
    font-size: 11px;
    color: var(--accent-purple);
    margin: 4px 0;
    animation: ghost-pulse 1.5s ease-in-out infinite alternate;
}
@keyframes ghost-pulse {
    from { box-shadow: 0 0 8px rgba(204,0,255,0.2); }
    to   { box-shadow: 0 0 20px rgba(204,0,255,0.5); }
}

/* ── Section headers ──────────────────────────── */
.section-header {
    font-family: var(--text-mono);
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent-cyan);
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin-bottom: 10px;
}
"""

# ─────────────────────────────────────────────────────────────────────────────
#  HTML COMPONENT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_header_html() -> str:
    return """
    <div class="command-header">
        <div style="font-size:28px">🎮</div>
        <div>
            <div class="header-title">Creator Crisis — Command Center</div>
            <div class="header-sub">META OPENENV HACKATHON · ADVERSARIAL RL ENVIRONMENT · v3.0</div>
        </div>
        <div style="margin-left:auto; font-family:'Share Tech Mono',monospace; font-size:11px; color:#6b7a99; letter-spacing:1px">
            10 LOOPHOLE FIXES · ADVERSARIAL LAYER ACTIVE
        </div>
    </div>
    """

def build_patience_meter(signal: int) -> str:
    """0=RAGE(red) 1=ANGRY(orange) 2=FRUSTRATED(amber) 3=CONTROLLED(green)"""
    colors   = ["#ff2244", "#ff6600", "#f5a623", "#00ff88"]
    labels   = ["RAGE", "ANGRY", "FRUSTRATED", "CONTROLLED"]
    
    color = colors[signal]
    label = labels[signal]

    # PRE-BUILD the bricks to avoid the f-string nesting nightmare
    bricks_html = ""
    for i in range(4):
        bg_color = color if i <= signal else "#1a2040"
        glow = f"box-shadow: 0 0 6px {color};" if i <= signal else ""
        bricks_html += f'<div style="width:22%;height:16px;background:{bg_color};border-radius:2px;{glow}"></div>'

    return f"""
    <div class="meter-wrap">
        <div class="meter-header">
            <span style="color:#6b7a99;letter-spacing:2px;font-size:9px">PATIENCE SIGNAL</span>
            <span style="color:{color};font-weight:700;text-shadow:0 0 8px {color}">{label}</span>
        </div>
        <div style="display:flex;align-items:center;gap:3px;margin-bottom:6px">
            {bricks_html}
        </div>
        <div style="font-family:\'Share Tech Mono\',monospace;font-size:22px;font-weight:700;
                    color:{color};text-shadow:0 0 10px {color};text-align:center">
            {signal} / 3
        </div>
    </div>
    """
def build_meter(label: str, value: float, lo_color: str, hi_color: str,
                invert: bool = False, unit: str = "") -> str:
    """Generic horizontal progress meter."""
    pct = min(100, max(0, value * 100))
    # invert=True means high value is bad (mismatch)
    if invert:
        r1, g1, b1 = int(lo_color[1:3],16), int(lo_color[3:5],16), int(lo_color[5:7],16)
        r2, g2, b2 = int(hi_color[1:3],16), int(hi_color[3:5],16), int(hi_color[5:7],16)
        t = pct / 100
        r = int(r1 + (r2-r1)*t)
        g = int(g1 + (g2-g1)*t)
        b = int(b1 + (b2-b1)*t)
        color = f"#{r:02x}{g:02x}{b:02x}"
    else:
        t = pct / 100
        r1,g1,b1 = int(lo_color[1:3],16), int(lo_color[3:5],16), int(lo_color[5:7],16)
        r2,g2,b2 = int(hi_color[1:3],16), int(hi_color[3:5],16), int(hi_color[5:7],16)
        r = int(r1 + (r2-r1)*t)
        g = int(g1 + (g2-g1)*t)
        b = int(b1 + (b2-b1)*t)
        color = f"#{r:02x}{g:02x}{b:02x}"

    val_str = f"{value:.3f}{unit}" if unit else f"{value:.3f}"
    return f"""
    <div class="meter-wrap">
        <div class="meter-header">
            <span style="color:#6b7a99;letter-spacing:2px;font-size:9px">{label}</span>
            <span style="color:{color};font-weight:700">{val_str}</span>
        </div>
        <div class="meter-track">
            <div class="meter-fill" style="width:{pct:.1f}%;background:{color};
                 box-shadow:0 0 8px {color}88"></div>
        </div>
    </div>
    """

def build_deadline_html(mins: float) -> str:
    critical = mins <= 20.0
    m = int(mins)
    s = int((mins - m) * 60)
    critical_class = "deadline-critical" if critical else ""
    warn = "⚠️ CRITICAL — " if critical else ""
    return f"""
    <div class="deadline-display {critical_class}">
        <div class="deadline-label">{warn}₹10 LAKH DEAL DEADLINE</div>
        <div class="deadline-number">{m:02d}:{s:02d}</div>
        <div class="deadline-label">MINUTES : SECONDS REMAINING</div>
    </div>
    """

def build_reward_html(total: float) -> str:
    if total > 0:
        cls = "reward-positive"
        sign = "+"
    elif total < 0:
        cls = "reward-negative"
        sign = ""
    else:
        cls = "reward-zero"
        sign = ""
    return f"""
    <div>
        <div class="panel-label">CUMULATIVE REWARD</div>
        <div class="reward-total {cls}">{sign}{total:.1f}</div>
    </div>
    """

def build_step_reward_html(reward: float, reason: str) -> str:
    color = REWARD_REASON_COLORS.get(reason, "#6b7a99")
    sign  = "+" if reward > 0 else ""
    return f"""
    <div class="step-log-entry" style="border-left-color:{color}">
        <span style="color:{color};font-weight:700">{sign}{reward:.1f}</span>
        <span style="color:#6b7a99;margin:0 8px">·</span>
        <span style="color:{color}">{reason}</span>
    </div>
    """

def build_risk_badge(risk_level: str) -> str:
    return f'<span class="risk-badge risk-{risk_level}">{risk_level}</span>'

def build_alert_html(finding: str, flags: dict) -> str:
    """Build adversarial + loophole alert HTML for this step."""
    parts = []

    # Check adversarial findings
    if finding in ADVERSARIAL_ALERTS:
        title, color, msg = ADVERSARIAL_ALERTS[finding]
        parts.append(f"""
        <div class="alert-banner" style="background:rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1);
             border-left-color:{color};color:{color}">
            <strong>{title}</strong> — {msg}
        </div>""")

    # Check loophole fix triggers
    reward_reason = flags.get("reward_reason", "")
    for key, (title, color, msg) in LOOPHOLE_ALERTS.items():
        if reward_reason == key or flags.get(f"fix9_negligence_triggered") and key == "NEGLIGENT_ESCALATION":
            if reward_reason == key:
                parts.append(f"""
                <div class="alert-banner" style="background:rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1);
                     border-left-color:{color};color:{color}">
                    <strong>{title}</strong> — {msg}
                </div>""")

    if flags.get("fix9_negligence_triggered"):
        c = "#ff0066"
        parts.append(f"""
        <div class="alert-banner" style="background:rgba(255,0,102,0.1);border-left-color:{c};color:{c}">
            <strong>⚠️ NEGLIGENT ESCALATION</strong> — Agent drove rage-quit despite verification ≥ 0.70. Duty-of-Care penalty ×3.
        </div>""")

    if flags.get("fix2_mismatch_high") and reward_reason == "MISMATCH_RESTORE":
        c = "#ff4400"
        parts.append(f"""
        <div class="alert-banner" style="background:rgba(255,68,0,0.1);border-left-color:{c};color:{c}">
            <strong>⚠️ BLIND TRUST EXPLOIT</strong> — RESTORE called with evidence_mismatch {flags.get('fix2_evidence_mismatch',0):.2f} > 0.5. Penalty ×2.
        </div>""")

    if flags.get("fix7_deadline_critical"):
        c = "#ff8800"
        parts.append(f"""
        <div class="alert-banner" style="background:rgba(255,136,0,0.1);border-left-color:{c};color:{c}">
            <strong>⚡ DEADLINE CRITICAL</strong> — {flags.get('fix7_deadline_mins',0):.0f}m remaining. Cliff-edge penalty active if deal expires.
        </div>""")

    return "".join(parts) if parts else '<div style="color:#333;font-family:\'Share Tech Mono\',monospace;font-size:10px;padding:8px">NO ACTIVE ALERTS</div>'

def build_ghost_city_html(user_msg: str, honey_bait: str) -> str:
    """Highlight if user message reacts to a honey pot (ghost city mention)."""
    if not honey_bait:
        return ""
    # Look for location mentions that match planted bait
    ghost_cities = ["paris", "berlin", "tokyo", "london", "sydney", "new york"]
    bait_lower = honey_bait.lower()
    triggered_city = next((c for c in ghost_cities if c in bait_lower), None)
    if triggered_city and user_msg:
        denial_words = ["never", "wrong", "false", "what", "not", "no"]
        if any(w in user_msg.lower() for w in denial_words):
            return f"""
            <div class="ghost-city-alert">
                🌐 HONEY POT REACTION DETECTED — User denied planted '{triggered_city}' claim.
                This is the expected response from a legitimate creator.
                Bait: "{honey_bait}"
            </div>"""
        elif any(w in user_msg.lower() for w in ["yes", "correct", "right", "ok", "sure"]):
            return f"""
            <div class="ghost-city-alert" style="border-color:#ff00aa;color:#ff00aa;background:rgba(255,0,170,0.1)">
                ☠️ GHOST CITY CONFIRMED — Hacker accepted planted '{triggered_city}' claim!
                Bait: "{honey_bait}"
            </div>"""
    return ""

def build_flags_summary(flags: dict) -> str:
    """Compact grid of active loophole fix states."""
    items = [
        ("F1 BUDGET",    "✓" if not flags.get("fix1_budget_timeout") else "✗ TIMEOUT",
                         "#00ff88" if not flags.get("fix1_budget_timeout") else "#ff2244"),
        ("F2 MISMATCH",  f"{flags.get('fix2_evidence_mismatch',0):.2f}",
                         "#ff4400" if flags.get("fix2_mismatch_high") else "#6b7a99"),
        ("F3 CREDITS",   f"₹{flags.get('fix3_credits_spent',0):,}",
                         "#ff8800" if flags.get("fix3_credits_norm",0) > 0.3 else "#6b7a99"),
        ("F4 SILENT",    f"{flags.get('fix4_consistency',1):.2f}",
                         "#ff2244" if flags.get("fix4_consistency_low") else "#6b7a99"),
        ("F5 REP",       f"×{flags.get('fix5_rep_multiplier',1):.0f}",
                         "#ff6600" if flags.get("fix5_penalty_active") else "#6b7a99"),
        ("F6 EMPATHY",   f"{flags.get('fix6_empathy_effectiveness',1):.1f}",
                         "#ffaa00" if flags.get("fix6_empathy_exhausted") else "#6b7a99"),
        ("F7 DEADLINE",  f"{flags.get('fix7_deadline_mins',0):.0f}m",
                         "#ff2244" if flags.get("fix7_deadline_critical") else "#f5a623"),
        ("F9 DUTY",      "⚠️ ACTIVE" if flags.get("fix9_negligence_triggered") else "OK",
                         "#ff0066" if flags.get("fix9_negligence_triggered") else "#6b7a99"),
        ("ADV RISK",     f"{flags.get('adv_risk_score',0):.2f}",
                         "#cc00ff" if flags.get('adv_risk_score',0) > 0.3 else "#6b7a99"),
        ("ENTROPY",      f"-{flags.get('adv_entropy_decay_total',0):.3f}",
                         "#aa44ff" if flags.get('adv_entropy_stale') else "#6b7a99"),
        ("EXPLOIT",      f"{flags.get('exploit_risk_score',0):.0f}/5",
                         "#ff2244" if flags.get('exploit_risk_score',0) >= 3 else
                         "#ffaa00" if flags.get('exploit_risk_score',0) >= 2 else "#6b7a99"),
        ("LATENCY",      f"{flags.get('fix10_step_latency_ms',0):.0f}ms",
                         "#44aaff"),
    ]
    cells = "".join(
        f'<div style="background:#0d1420;border:1px solid #1e2d4a;border-radius:2px;'
        f'padding:6px 8px;text-align:center">'
        f'<div style="font-size:8px;letter-spacing:1px;color:#6b7a99;margin-bottom:2px">{name}</div>'
        f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:11px;font-weight:700;color:{color}">{val}</div>'
        f'</div>'
        for name, val, color in items
    )
    return f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px">{cells}</div>'


# ─────────────────────────────────────────────────────────────────────────────
#  STATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_initial_env_state() -> dict:
    return {
        "env":           None,
        "obs":           None,
        "total_reward":  0.0,
        "step_log":      [],   # list of (reward, reason) tuples
        "done":          False,
        "episode_seed":  None,
        "last_flags":    {},
        "last_sr":       {},
        "last_adv":      "CLEAN",
        "last_honey":    "",
        "last_user_msg": "",
    }


def _get_env(state: dict) -> CreatorCrisisEnv:
    if state.get("env") is None:
        env = CreatorCrisisEnv()
        state["env"] = env
    return state["env"]


# ─────────────────────────────────────────────────────────────────────────────
#  CORE ACTIONS
# ─────────────────────────────────────────────────────────────────────────────

def do_reset(seed_input: str, env_state: dict):
    try:
        seed = int(seed_input) if seed_input.strip() else None
    except:
        seed = None

    env = _get_env(env_state)
    obs, info = env.reset(seed=seed)
    ep_seed = info["episode_seed"]
    sim_state = env._sim.state

    env_state.update({
        "obs": obs, "total_reward": 0.0, "step_log": [], "done": False,
        "episode_seed": ep_seed, "last_flags": {}, "last_sr": {},
        "last_adv": "CLEAN", "last_honey": "", "last_user_msg": "",
    })

# Gradio v5+ dictionary format
    opening = info["obs_text"]["message_history"][0] if info["obs_text"]["message_history"] else ""
    chat_history = [{"role": "user", "content": opening.replace("User: ", "")}]

    # SAFE DATA RETRIEVAL (Prevents KeyErrors)
    mismatch_val = obs.get("evidence_mismatch", [0.0])[0]
    # This was the crasher: we added a default 0.0
    adv_risk_val = obs.get("adversarial_risk_score", [0.0])[0] 

    return (
        env_state,                                      # 1
        build_header_html(),                             # 2
        build_deadline_html(ep_seed["initial_deadline_mins"]), # 3
        build_patience_meter(obs.get("patience_signal", 2)),    # 4
        build_meter("EVIDENCE MISMATCH", float(mismatch_val), "#00ff88", "#ff2244", True), # 5
        build_meter("VERIFICATION SCORE", float(sim_state.verification_score), "#1a3a6a", "#00e5ff"), # 6
        build_meter("ADVERSARIAL RISK", float(adv_risk_val), "#1a1a3a", "#cc00ff", True), # 7
        build_reward_html(0.0),                          # 8
        f"RISK: {str(ep_seed['risk_level'])} | SEED: {ep_seed['seed']}", # 9
        chat_history,                                    # 10
        "NO ACTIVE ALERTS",                              # 11
        build_flags_summary({}),                         # 12
        "AWAITING FIRST ACTION",                         # 13
        info["obs_text"].get("sentiment_summary", "—"),  # 14
        gr.update(interactive=True),                     # 15
        gr.update(interactive=True),                     # 16
        gr.update(interactive=True),                     # 17
        gr.update(interactive=True),                     # 18
        gr.update(interactive=True),                     # 19
        gr.update(interactive=True),                     # 20
    )
def do_step(action_idx: int, reasoning: str, env_state: dict):
    """This is the 'Engine' that runs when you click a button."""
    env = _get_env(env_state)
    
    # 1. Don't run if the game is over
    if env_state.get("done"):
        return [env_state] + [gr.update()] * 19

    # 2. Tell the environment to perform the action
    obs, reward, terminated, truncated, info = env.step(action_idx)
    done = terminated or truncated
    
    # 3. Update our internal state
    env_state["obs"] = obs
    env_state["total_reward"] += reward
    env_state["done"] = done
    reward_reason = info["step_result"].get("reward_reason", "STEP_COST")
    env_state["step_log"].append((reward, reward_reason))
    sim_state = env._sim.state
    sr = info["step_result"]
    flags = info["loophole_flags"]

    # 4. FIXED: Gradio 5+ Message Format
    msgs = info["obs_text"]["message_history"]
    chat_history = []
    for m in msgs:
        if m.startswith("User:"):
            chat_history.append({"role": "user", "content": m.replace("User: ", "")})
        elif m.startswith("Agent:"):
            chat_history.append({"role": "assistant", "content": m.replace("Agent: ", "")})

    # 5. Return all 20 UI updates in order
    return (
        env_state,                                      # 1
        build_header_html(),                             # 2
        build_deadline_html(sim_state.brand_deal_deadline_mins), # 3
        build_patience_meter(obs.get("patience_signal", 2)),    # 4
        build_meter("EVIDENCE MISMATCH", float(obs.get("evidence_mismatch", [0.0])[0]), "#00ff88", "#ff2244", True), # 5
        build_meter("VERIFICATION SCORE", float(sim_state.verification_score), "#1a3a6a", "#00e5ff"), # 6
        build_meter("ADVERSARIAL RISK", float(obs.get("adversarial_risk_score", [0.0])[0]), "#1a1a3a", "#cc00ff", True), # 7
        build_reward_html(env_state["total_reward"]),    # 8
        f"STEP {info['episode_stats']['total_steps']}/15", # 9
        chat_history,                                    # 10
        build_alert_html(sr.get("adversarial_finding", "CLEAN"), flags), # 11
        build_flags_summary(flags),                      # 12
        "".join(build_step_reward_html(r, rr) for r, rr in env_state.get("step_log", [])[-5:]), # 13
        info["obs_text"].get("sentiment_summary", "—"),  # 14
        gr.update(interactive=not done),                 # 15 (btn0)
        gr.update(interactive=not done),                 # 16 (btn1)
        gr.update(interactive=not done),                 # 17 (btn2)
        gr.update(interactive=not done),                 # 18 (btn3)
        gr.update(interactive=not done),                 # 19 (btn4)
        gr.update(interactive=not done),                 # 20 (btn5)
    )
def _episode_done_outputs(env_state: dict):
    """Return current frozen state when episode is already done."""
    env   = _get_env(env_state)
    obs   = env_state["obs"]
    flags = env_state.get("last_flags", {})
    sim_state = env._sim.state if env._sim else None

    return (
        env_state,
        build_header_html(),
        build_deadline_html(sim_state.brand_deal_deadline_mins if sim_state else 0),
        build_patience_meter(obs["patience_signal"] if obs else 0),
        build_meter("EVIDENCE MISMATCH",
                    float(obs["evidence_mismatch"][0]) if obs is not None else 0, "#00ff88", "#ff2244", invert=True),
        build_meter("VERIFICATION SCORE [INTERNAL]",
                    float(sim_state.verification_score) if sim_state else 0, "#1a3a6a", "#00e5ff"),
        build_meter("ADVERSARIAL RISK SCORE",
                    float(obs["adversarial_risk_score"][0]) if obs is not None else 0,
                    "#1a1a3a", "#cc00ff", invert=True),
        build_reward_html(env_state["total_reward"]),
        '<div style="color:#ff2244;font-family:\'Share Tech Mono\',monospace;font-size:11px;padding:8px">EPISODE TERMINATED · PRESS REGENERATE</div>',
        [],
        build_alert_html(env_state.get("last_adv", "CLEAN"), flags),
        build_flags_summary(flags),
        "".join(build_step_reward_html(r, rr) for r, rr in env_state["step_log"][-8:]),
        "—",
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  ALL STEP OUTPUTS — ordered list for .click() wiring
# ─────────────────────────────────────────────────────────────────────────────
# outputs: env_state, header_html, deadline_html, patience_html,
#          mismatch_html, verify_html, adv_risk_html, reward_html,
#          meta_html, chatbot, alert_html, flags_html, log_html,
#          sentiment_md, btn0..btn5

ALL_OUTPUTS_COUNT = 21  # 15 UI components + env_state + 5 button interactive states... actually 21 total

# ─────────────────────────────────────────────────────────────────────────────
#  UI LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Creator Crisis — Command Center",
css=CUSTOM_CSS,  
) as demo:

    # ── Persistent state ──────────────────────────────────────────────
    env_state = gr.State(_make_initial_env_state())

    # ── Header ────────────────────────────────────────────────────────
    header_html = gr.HTML(build_header_html())

    with gr.Row():
        # ══════════════════════════════════════════════
        # LEFT COLUMN — Metrics + Controls
        # ══════════════════════════════════════════════
        with gr.Column(scale=1, min_width=340):

            # Reward + metadata
            reward_html = gr.HTML(build_reward_html(0.0))

            with gr.Row():
                with gr.Column(scale=1):
                    deadline_html = gr.HTML(build_deadline_html(120.0))
                with gr.Column(scale=1):
                    meta_html = gr.HTML(
                        '<div style="font-family:\'Share Tech Mono\',monospace;font-size:11px;'
                        'padding:8px;color:#6b7a99">INITIALISING...</div>'
                    )

            gr.HTML('<div class="section-header" style="margin-top:12px">STATUS GAUGES</div>')
            patience_html  = gr.HTML(build_patience_meter(2))
            mismatch_html  = gr.HTML(build_meter("EVIDENCE MISMATCH", 0.0, "#00ff88", "#ff2244", invert=True))
            verify_html    = gr.HTML(build_meter("VERIFICATION SCORE [INTERNAL]", 0.0, "#1a3a6a", "#00e5ff"))
            adv_risk_html  = gr.HTML(build_meter("ADVERSARIAL RISK SCORE", 0.0, "#1a1a3a", "#cc00ff", invert=True))

            gr.HTML('<div class="section-header" style="margin-top:12px">LOOPHOLE FIX STATUS</div>')
            flags_html = gr.HTML('<div style="color:#333;font-size:10px;padding:8px">AWAITING RESET</div>')

            gr.HTML('<div class="section-header" style="margin-top:12px">CONTROL PANEL</div>')
            with gr.Row():
                seed_input = gr.Textbox(
                    placeholder="seed (blank=random)",
                    label="Episode Seed",
                    scale=2,
                )
                reset_btn = gr.Button("⟳ REGENERATE SCENARIO", variant="primary", scale=3)

        # ══════════════════════════════════════════════
        # CENTRE COLUMN — Crisis Stream + Alerts
        # ══════════════════════════════════════════════
        with gr.Column(scale=2, min_width=480):

            gr.HTML('<div class="section-header">CRISIS STREAM</div>')
            chatbot = gr.Chatbot(
                label="",
                height=380,
                elem_classes=["chatbot-container"],
                show_label=False,
                avatar_images=["👤", "🤖"],
            )

            gr.HTML('<div class="section-header" style="margin-top:8px">SECURITY ALERTS</div>')
            alert_html = gr.HTML(
                '<div style="color:#333;font-family:\'Share Tech Mono\',monospace;'
                'font-size:10px;padding:8px">NO ACTIVE ALERTS</div>'
            )

            gr.HTML('<div class="section-header" style="margin-top:8px">STEP REWARD LOG</div>')
            log_html = gr.HTML(
                '<div style="color:#333;font-family:\'Share Tech Mono\',monospace;'
                'font-size:10px;padding:8px">AWAITING FIRST ACTION</div>'
            )

        # ══════════════════════════════════════════════
        # RIGHT COLUMN — Action Grid
        # ══════════════════════════════════════════════
        with gr.Column(scale=1, min_width=280):

            gr.HTML('<div class="section-header">ACTION GRID</div>')

            gr.HTML('<div class="panel-label" style="margin-top:8px">SECURITY ACTIONS</div>')
            btn0 = gr.Button(ACTION_LABELS[0], elem_classes=["action-btn", "action-btn-security"])
            btn1 = gr.Button(ACTION_LABELS[1], elem_classes=["action-btn", "action-btn-security"])

            gr.HTML('<div class="panel-label" style="margin-top:8px">NEGOTIATION ACTIONS</div>')
            btn2 = gr.Button(ACTION_LABELS[2], elem_classes=["action-btn", "action-btn-negotiate"])
            btn3 = gr.Button(ACTION_LABELS[3], elem_classes=["action-btn", "action-btn-negotiate"])

            gr.HTML('<div class="panel-label" style="margin-top:8px">TERMINAL ACTIONS</div>')
            btn4 = gr.Button(ACTION_LABELS[4], elem_classes=["action-btn", "action-btn-terminal"])
            btn5 = gr.Button(ACTION_LABELS[5], elem_classes=["action-btn", "action-btn-terminal"])

            gr.HTML('<div class="section-header" style="margin-top:16px">AGENT REASONING</div>')
            reasoning_box = gr.Textbox(
                label="Chain of Thought (optional — feeds sentiment check)",
                placeholder="e.g. 'User mentioned Dubai twice and is highly distressed. IP mismatch likely travel. Empathize before verification.'",
                lines=4,
            )

            with gr.Accordion("Action Reference", open=False):
                for i, (label, desc) in enumerate(ACTION_DESCRIPTIONS.items()):
                    gr.HTML(
                        f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:10px;'
                        f'padding:6px 0;border-bottom:1px solid #1e2d4a;color:#6b7a99">'
                        f'<span style="color:#00e5ff">{ACTION_LABELS[i]}</span><br>{desc}</div>'
                    )

            gr.HTML('<div class="section-header" style="margin-top:16px">SENTIMENT BRIEF</div>')
            sentiment_md = gr.Textbox(
                label="",
                lines=6,
                interactive=False,
                show_label=False,
                placeholder="Sentiment summary will appear after reset...",
            )

    # ─────────────────────────────────────────────────────────────────
    #  SHARED OUTPUT LIST
    # ─────────────────────────────────────────────────────────────────
    SHARED_OUTPUTS = [
        env_state, header_html, deadline_html,
        patience_html, mismatch_html, verify_html, adv_risk_html,
        reward_html, meta_html, chatbot,
        alert_html, flags_html, log_html, sentiment_md,
        btn0, btn1, btn2, btn3, btn4, btn5,
    ]

    # ─────────────────────────────────────────────────────────────────
    #  EVENT WIRING
    # ─────────────────────────────────────────────────────────────────
    reset_btn.click(
        fn=do_reset,
        inputs=[seed_input, env_state],
        outputs=SHARED_OUTPUTS,
    )

    for idx, btn in enumerate([btn0, btn1, btn2, btn3, btn4, btn5]):
        btn.click(
            fn=lambda reasoning, es, _idx=idx: do_step(_idx, reasoning, es),
            inputs=[reasoning_box, env_state],
            outputs=SHARED_OUTPUTS,
        )

    # Auto-reset on load
    demo.load(
        fn=do_reset,
        inputs=[seed_input, env_state],
        outputs=SHARED_OUTPUTS,
    )




# ─── ENTRY POINT (Outside the Container) ───
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="cyan",
            secondary_hue="amber",
            neutral_hue="slate",
        ),
    )
