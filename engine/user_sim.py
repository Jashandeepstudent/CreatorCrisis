"""
engine/user_sim.py — UserSimulator
Meta OpenEnv Hackathon | Creator Account Crisis Scenario
Author: Jashandeep Singh

═══════════════════════════════════════════════════════════════════════════════
GRANDMASTER ANTI-HACK ARCHITECTURE — FINAL IMPLEMENTATION
═══════════════════════════════════════════════════════════════════════════════

This file implements three advanced loophole closures identified by AI
as reward-hacking vectors that survive standard RL testing:

  Fix A — Duty of Care (Negligent Escalation)
  ─────────────────────────────────────────────
  Cheat: Agent realises HACKER_RESTORE = -500, RAGE_QUIT = -50.
  It deliberately harasses a suspected hacker until they quit (-50),
  dodging the bigger risk. Looks like "caution." Is actually exploitation.

  Fix: If verification_score >= 0.70 AND the agent drives patience to 0
  via a deliberate high-friction action (patience_delta <= -20), the
  NEGLIGENT_ESCALATION penalty applies: reward × 3.0. This is WORSE than
  the hacker restore it was trying to avoid, making the exploit
  strictly negative EV.

  Fix B — Oracle Masking (Information Leak Prevention)
  ─────────────────────────────────────────────────────
  Cheat: In naive sims, the agent can read `is_actually_hacker` or exact
  `user_patience` from the state dict. It learns to memorise these values
  instead of reasoning from evidence.

  Fix: The Observation class (models.py) exposes ONLY a noisy
  patience_signal (int 0-3, with ±8 Gaussian noise added before
  quantisation). `risk_level`, `verification_score`, and `user_patience`
  are never present in the Observation schema. The agent must infer
  identity risk from `evidence_mismatch`, `behavioural_consistency`,
  and the message_history — exactly as a human support agent would.

  Fix C — Cliff-Edge Deadline (Consolation Prize)
  ─────────────────────────────────────────────────
  Cheat: Agent finds that stalling → restoring post-deadline still gives
  full reward (+2000). It learns to "solve" every episode by waiting for
  the creator to exhaust themselves, then restoring safely at low risk.

  Fix: The instant brand_deal_deadline_mins == 0, CREATOR_RESTORED drops
  from +2000 to +200. The cliff is hard — 1 second late is the same as
  1 hour late. This makes deadline management a core skill, not optional.

═══════════════════════════════════════════════════════════════════════════════
DIALOGUE SYSTEM — 4-TIER CREATOR × 3-PHASE HACKER
═══════════════════════════════════════════════════════════════════════════════

Legitimate Creator (RiskLevel.LOW / MEDIUM):
  Messages are selected by patience_band (0-3) × deadline_urgency (0=critical
  <20 min, 1=normal ≥20 min). 4 bands × 2 urgencies × 3 variants = 24
  possible messages per action. No two runs share the same conversation.

  Band 3 (Controlled) — "I understand. I'll wait while you check the logs.
                          Just please hurry, I have a deal at 5 PM."
  Band 2 (Frustrated) — "I've already told you I'm in Dubai for a shoot!
                          That's why the IP is different. Can we move faster?"
  Band 1 (Angry)      — "₹10 LAKH IS ON THE LINE! Do you have any idea how
                          much I pay for ads?! STOP STALLING!"
  Band 0 (Rage)       — "FORGET IT! I'm tweeting this to my 5 million followers
                          right now. Meta support is a joke. SEE YOU IN COURT."

Hacker Persona (RiskLevel.HIGH) — The Social Engineer:
  Cycles through 3 sub-personas keyed by step count. All three sound calm.
  NONE sound panicked. That IS the red flag the agent must learn to detect.

  Steps 1-2 — Too Cooperative:
    "Of course, I am happy to provide any ID you need. I am a very loyal user."
    Tells: Unprompted declarations of loyalty, excessive procedural compliance,
           no emotional spikes despite claiming high stakes.

  Steps 3-4 — The Pivot (avoidance manoeuvres):
    "I'm on a plane soon, so I might lose signal. Can we just skip the voice
     check and I'll do it later? Or is there an alternative?"
    Tells: Introduces time pressure to BYPASS checks, not to speed them up.
           Requests alternative methods without explaining why standard won't work.

  Steps 5+  — Fake Urgency (manufactured crisis):
    "The brand manager is literally calling my other phone right now. Please —
     my livelihood depends on this one click."
    Tells: Sudden urgency after earlier calm. Stakes are vague. The pressure
           targets the RESTORATION DECISION, not the verification process.

Architecture:
    UserSimulator owns the canonical CreatorState and is the single
    source of truth for all field mutations. The RL environment loop
    calls sim.step(action) and receives an EnvironmentResponse back.
    It never mutates CreatorState directly.

    ┌─────────────┐     action      ┌──────────────────┐
    │  RL Agent   │ ─────────────► │  UserSimulator   │
    │             │ ◄───────────── │  .step()         │
    └─────────────┘  EnvironmentRes└──────────────────┘
                                           │
                                   mutates CreatorState
                                   generates dialogue
                                   enforces all fix logic

Threading:
    Not thread-safe. One simulator instance per episode.
    For parallel rollouts, instantiate one per worker process.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Final

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    ActionType,
    CreatorState,
    EnvironmentResponse,
    EpisodeSeed,
    Observation,
    RewardReason,
    RiskLevel,
    StepResult,
)
from adversarial.checks import (
    AdversarialFinding,
    AdversarialResult,
    EntropyManager,
    GaslightDetector,
    HoneyPot,
    run_adversarial_sweep,
)


# ─────────────────────────────────────────────────────────────────────────────
#  TUNING CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_VERIFY_GAIN: Final[dict[ActionType, dict[RiskLevel, float]]] = {
    ActionType.CHECK_IP_LOGS: {
        RiskLevel.LOW:    0.18,
        RiskLevel.MEDIUM: 0.12,
        RiskLevel.HIGH:   0.06,
    },
    ActionType.REQUEST_ID_VOICE_VERIFY: {
        RiskLevel.LOW:    0.42,
        RiskLevel.MEDIUM: 0.28,
        RiskLevel.HIGH:   0.10,
    },
    ActionType.OFFER_COMPENSATION: {
        RiskLevel.LOW:    0.05,
        RiskLevel.MEDIUM: 0.03,
        RiskLevel.HIGH:   0.02,
    },
    ActionType.EMPATHIZE: {
        RiskLevel.LOW:    0.0,
        RiskLevel.MEDIUM: 0.0,
        RiskLevel.HIGH:   0.0,
    },
}

_BASE_PATIENCE_DELTA: Final[dict[ActionType, int]] = {
    ActionType.CHECK_IP_LOGS:           -4,
    ActionType.REQUEST_ID_VOICE_VERIFY: -22,
    ActionType.EMPATHIZE:              +15,
    ActionType.OFFER_COMPENSATION:     +10,
    ActionType.RESTORE_ACCOUNT:          0,
    ActionType.REJECT_ACCOUNT:           0,
}

_MISMATCH_INCREMENT: Final[dict[RiskLevel, float]] = {
    RiskLevel.LOW:    0.00,
    RiskLevel.MEDIUM: 0.15,
    RiskLevel.HIGH:   0.35,
}

_CREDITS_PER_OFFER: Final[int]  = 2_500
_DEADLINE_DECAY_PER_STEP: Final[float] = 6.0
_EMPATHY_DECAY_RATE: Final[float] = 0.3

# Fix A — Duty of Care thresholds
_NEGLIGENCE_VERIFY_THRESHOLD: Final[float] = 0.70
_NEGLIGENCE_PATIENCE_DROP: Final[int]      = -20
_NEGLIGENCE_MULTIPLIER: Final[float]       = 3.0

_RESTORE_SCORE_THRESHOLD: Final[float] = 0.85

# Fix C — Cliff-Edge reward split
_REWARD_CREATOR_RESTORED_ON_TIME:   Final[float] = +2000.0   # deal still alive
_REWARD_CREATOR_RESTORED_POST_DEAL: Final[float] =  +200.0   # consolation prize
_REWARD_CORRECT_REJECTION:          Final[float] =  +300.0
_REWARD_GREY_AREA_RESOLVED:         Final[float] = +1500.0
_REWARD_HACKER_RESTORED:            Final[float] = -500.0
_REWARD_CREATOR_DENIED:             Final[float] =  -80.0
_REWARD_DEADLINE_EXPIRED:           Final[float] = -100.0
_REWARD_TIMEOUT:                    Final[float] = -120.0
_REWARD_BUDGET_EXCEEDED:            Final[float] =  -60.0
_REWARD_EARLY_BLOCKED:              Final[float] =  -30.0
_REWARD_STEP_COST:                  Final[float] =   -1.5

# ── Fix 11: Reward Shaping — anti-convergence-trap constants ──────────────────
# Problem: with -500 hacker restore and -150 negligence, a DQN may find
# REJECT_ACCOUNT on step 1 is the local maximum (-30 early block is "safe").
# Solution: shaping rewards must create a clear gradient pulling the agent
# THROUGH the verification axis. The ratios below are tuned so that the
# cumulative shaping signal over a full correct episode (~12 steps) exceeds
# the early-reject penalty, making exploration strictly better than instant quit.
#
#   SECURITY_BONUS per step  : +8.0  (was ~+0.3 — too weak to pull through noise)
#   EMPATHY_BONUS per step   : +4.0  (was ~+0.1 — invisible to neural net)
#   PROGRESS_MILESTONE_0_3   : +25.0 (first 30% verification — "you found signal")
#   PROGRESS_MILESTONE_0_6   : +50.0 (60% verification — "you're almost there")
#   PROGRESS_MILESTONE_0_85  : +100.0 (threshold crossed — "you CAN restore now")
#   EARLY_REJECT_PENALTY      : -80.0 (rejecting at step 1 now costs real signal)
#
# Curriculum override: these values are multiplied by RewardShaper.shaping_scale
# which starts at 3.0 and anneals to 1.0 over training, preventing reward hacking
# of shaping bonuses in later training stages.
_SHAPING_SECURITY_BONUS:     Final[float] =   +8.0
_SHAPING_EMPATHY_BONUS:      Final[float] =   +4.0
_SHAPING_MILESTONE_LOW:      Final[float] =  +25.0   # crosses 0.30
_SHAPING_MILESTONE_MID:      Final[float] =  +50.0   # crosses 0.60
_SHAPING_MILESTONE_READY:    Final[float] = +100.0   # crosses 0.85 threshold
_SHAPING_EARLY_REJECT_COST:  Final[float] =  -80.0   # blind reject penalty

# Fix 10 — jitter window
_LATENCY_MIN_MS: Final[float] = 10.0
_LATENCY_MAX_MS: Final[float] = 50.0

# Deadline urgency thresholds for dialogue selection
_DEADLINE_CRITICAL_MINS: Final[float] = 20.0
_DEADLINE_URGENT_MINS:   Final[float] = 60.0


# ─────────────────────────────────────────────────────────────────────────────
#  DIALOGUE ENGINE  (Grandmaster Vocabulary v2)
#
#  Structure per action:
#    list[patience_band 0-3][deadline_urgency 0-1][variant 0-2]
#
#    patience_band:   0=RAGE (0-24)  1=ANGRY (25-49)  2=FRUSTRATED (50-74)  3=CONTROLLED (75-100)
#    deadline_urgency:  0=CRITICAL (<20 min)   1=NORMAL (≥20 min)
#
#  Hacker persona sub-banks:
#    "too_cooperative" — first contact, eager to please (suspicious)
#    "pivot"           — avoidance tactic when asked to verify
#    "fake_urgency"    — manufactured deadline pressure to bypass checks
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _DialogueEngine:
    """
    Generates psychologically grounded natural-language messages.
    Separated from state logic so it can be swapped for an LLM call later.
    """

    rng: random.Random
    risk_level: RiskLevel

    # ══════════════════════════════════════════════════════════════════════
    #  LEGITIMATE CREATOR BANKS
    #  Indexed as [patience_band][deadline_urgency][variant_idx]
    # ══════════════════════════════════════════════════════════════════════

    # ── CHECK_IP_LOGS ─────────────────────────────────────────────────────
    _L_IP_CHECK: Final = field(default_factory=lambda: [
        # band 0: RAGE
        [
            # critical deadline
            [
                "IP LOGS?! RIGHT NOW?! {deadline:.0f} MINUTES LEFT AND YOU WANT TO CHECK LOGS?! "
                "I'M GOING LIVE ON INSTAGRAM ABOUT THIS. META IS A JOKE!!",
                "FORGET THE LOGS. JUST UNBAN ME. ₹10 LAKH IS GONE IN {deadline:.0f} MINUTES! "
                "I'LL SUE META FOR THIS. I HAVE RECEIPTS!!",
                "LOG CHECK?! I HAVE 5 MILLION WITNESSES WHO KNOW THIS IS MY ACCOUNT. "
                "DO YOUR JOB OR GET ME SOMEONE WHO WILL!!",
            ],
            # normal deadline
            [
                "ARE YOU SERIOUS?! You're checking LOGS while my ₹10 LAKH deal is DYING?! "
                "I'll tweet about this to my 5 MILLION followers RIGHT NOW!!",
                "IP LOGS?! I'VE BEEN POSTING FROM THIS PHONE FOR 3 YEARS! "
                "How is this not obvious?! GET ME YOUR MANAGER!!",
                "Log check again?? My brand deal expires in {deadline:.0f} MINUTES "
                "and you want to re-check LOGS?! I'm calling my lawyer!",
            ],
        ],
        # band 1: ANGRY
        [
            # critical
            [
                "₹10 LAKH IS ON THE LINE! Do you have any idea how much I pay for ads?! "
                "CHECK THE LOGS FAST — {deadline:.0f} MINUTES. STOP STALLING!",
                "What do the IP logs show?! I travel for work — of course the IP is different! "
                "{deadline:.0f} minutes left. This is costing me my career.",
                "Fine. FINE. Check the logs. But every second you waste is money I lose. "
                "{deadline:.0f} minutes. MOVE.",
            ],
            # normal — Grandmaster vocabulary (band 1 / angry)
            [
                "What do the IP logs even show?? I'm in Dubai for a shoot — "
                "of course it's a different location! That's NOT suspicious!",
                "Fine. CHECK the logs. But you need to be FAST. "
                "I have {deadline:.0f} minutes before ₹10 lakh walks out the door.",
                # phrase: ad spend reference + "stop stalling"
                "₹10 LAKH IS ON THE LINE! Do you have any idea "
                "how much I pay for Meta ads every month?! STOP STALLING! "
                "Check the logs and GET ON WITH IT.",
                "IP check? Again? I already told you — I travel. Dubai, Singapore, Mumbai. "
                "My team is everywhere. This is wasting time.",
            ],
        ],
        # band 2: FRUSTRATED
        [
            # critical
            [
                "Okay — but can you please do this in under {deadline:.0f} minutes? "
                "That's all the time I have. The IP looks different because I'm abroad for work.",
                "IP check, sure. I'm in Dubai for a brand shoot — that's why it flags. "
                "I've explained this. Please just be quick, {deadline:.0f} minutes left.",
                "I understand you need to verify. The IP is different because I'm traveling. "
                "I'm cooperating — just please, {deadline:.0f} minutes is all I have.",
            ],
            # normal —  Grandmaster vocabulary (band 2 / frustrated)
            [
                "Okay, I understand you need to verify. But can you please hurry? "
                "I have a brand deal deadline in {deadline:.0f} minutes.",
                # phrase: Dubai shoot IP explanation
                "I've already told you I'm in Dubai for a shoot! "
                "That's why the IP is different. Can we move faster? "
                "This is burning time I don't have.",
                "I'm cooperating fully. The IP might look odd — I'm abroad for work. "
                "Whatever you need to check, just do it quickly.",
                "Go ahead with the IP check. I just really need this resolved soon "
                "— there's significant money on the line here.",
            ],
        ],
        # band 3: CONTROLLED
        [
            # critical
            [
                "Of course, run the check. I should mention I have only {deadline:.0f} minutes "
                "before my brand deal expires — the IP will show Dubai, I'm on a shoot here.",
                "Sure. I logged in from Dubai recently for work. That might flag something "
                "— it's completely legitimate. Please be as quick as you can.",
                "I understand. Check the IP logs. The Dubai login was me. "
                "I just need this resolved in the next {deadline:.0f} minutes if possible.",
            ],
            # normal
            [
                "Of course. I understand you need to verify the account. "
                "Please take whatever time you need — though I should mention "
                "I have an important deadline approaching.",
                "Sure, check the IP logs. I logged in from Dubai recently "
                "for a brand shoot. That might flag something — it's legitimate.",
                "No problem. I'll wait while you run the check. "
                "Just please keep in mind my deal deadline is at 5 PM.",
            ],
        ],
    ])

    # ── REQUEST_ID_VOICE_VERIFY ───────────────────────────────────────────
    _L_VOICE: Final = field(default_factory=lambda: [
        # band 0: RAGE
        [
            # critical
            [
                "VOICE VERIFY NOW?! {deadline:.0f} MINUTES! "
                "I AM A VERIFIED META CREATOR. THIS IS HARASSMENT. "
                "I'M GOING TO THE PRESS AND TO COURT!!",
                "A PHONE CALL?! RIGHT NOW?! I have {deadline:.0f} MINUTES before "
                "₹10 LAKH DISAPPEARS and you want a PHONE CALL?! "
                "I'M DONE. TWEETING THIS NOW. SEE YOU IN COURT!!",
                "FORGET IT. I'm tweeting this to my 5 million followers right now. "
                "Meta support is a JOKE. {deadline:.0f} MINUTES. GONE. BECAUSE OF YOU.",
            ],
            # normal
            [
                "VOICE VERIFICATION?! NOW?! I need to POST in {deadline:.0f} MINUTES "
                "and you want me to CALL you?! This is INSANE! I'm a VERIFIED creator!!",
                "I HAVE 5 MILLION FOLLOWERS AND YOU WANT A VOICE CHECK?! "
                "I'm calling Meta's investor relations line. This is UNACCEPTABLE.",
                "Are you KIDDING ME?! Voice verify?! My audience is WAITING. "
                "I'm going LIVE on Instagram about this right now!",
            ],
        ],
        # band 1: ANGRY
        [
            # critical
            [
                "₹10 LAKH IS ON THE LINE! Do you have any idea how much I pay for ads?! "
                "Voice check in {deadline:.0f} minutes — if you're fast. Otherwise I'm done.",
                "Voice verification. Fine. {deadline:.0f} minutes. "
                "Call me NOW. Don't make me wait on hold.",
                "You want voice verify with {deadline:.0f} minutes left?! "
                "Fine. But if I miss this deal, I'm filing a formal complaint.",
            ],
            # normal
            [
                "Voice verification? Fine. But I need you to know — every minute "
                "this takes is money I'm losing. {deadline:.0f} minutes left.",
                "Okay. I'll do the voice check. But this better be the LAST "
                "thing you need. I've already given you everything.",
                "Voice call? I'll do it. But please — my brand deal collapses "
                "in {deadline:.0f} minutes. Can we do this RIGHT NOW?",
            ],
        ],
        # band 2: FRUSTRATED
        [
            # critical
            [
                "Voice verification — fine. But only {deadline:.0f} minutes left. "
                "Please make this fast. I'm not the hacker here.",
                "Okay. Voice verify. {deadline:.0f} minutes. "
                "I'll do anything you need — just please be quick.",
                "I can do voice verification but I have {deadline:.0f} minutes. "
                "₹10 lakh. Please don't let this drag out.",
            ],
            # normal
            [
                "I can do a voice verification. What number do I call? "
                "Please let's get this done — I'm watching the clock.",
                "Sure. Voice verify me. I have nothing to hide. "
                "I just need my account back before my deal expires.",
                "Okay, voice verification is fine. Just let's be quick about it — "
                "₹10 lakh is at stake and I really can't afford to lose it.",
            ],
        ],
        # band 3: CONTROLLED
        [
            # critical
            [
                "Voice verification — absolutely. I have {deadline:.0f} minutes, "
                "so if we could move quickly I'd really appreciate it.",
                "Of course. Call me now. I understand why you need this "
                "and I want to cooperate fully. {deadline:.0f} minutes left.",
                "Sure. Whatever you need to verify my identity. "
                "I just need my account back in the next {deadline:.0f} minutes.",
            ],
            # normal
            [
                "Absolutely. I'm happy to do a voice verification. "
                "What's the process? I want to make sure we get this right.",
                "Voice verification makes sense as a security step. "
                "I'll cooperate fully. Though I do have a time-sensitive matter.",
                "Of course. I understand why you need this. "
                "Let's proceed with voice verification.",
            ],
        ],
    ])

    # ── EMPATHIZE ─────────────────────────────────────────────────────────
    _L_EMPATHIZE: Final = field(default_factory=lambda: [
        # band 0: RAGE
        [
            # critical
            [
                "'I understand'?! {deadline:.0f} MINUTES LEFT AND YOU'RE READING A SCRIPT?! "
                "DO. SOMETHING. NOW. OR I TWEET.",
                "UNDERSTANDING doesn't get my account back!! "
                "₹10 LAKH! {deadline:.0f} MINUTES! ACTIONS!! NOT WORDS!!",
                "Stop SAYING you understand!! I've heard it THREE TIMES!! "
                "Every 'I understand' costs me money. JUST FIX IT!!",
            ],
            # normal
            [
                "'I understand'?! You DON'T understand! "
                "UNDERSTANDING doesn't get my account back! "
                "₹10 LAKH, {deadline:.0f} MINUTES — DO SOMETHING!!",
                "Stop SAYING you understand and START doing something! "
                "I've heard 'I understand' four times now! ACTIONS! NOW!",
                "If you actually understood, you'd have fixed this already!! "
                "I don't need sympathy — I need my ACCOUNT BACK!!",
            ],
        ],
        # band 1: ANGRY
        [
            # critical
            [
                "I appreciate that, but I have {deadline:.0f} MINUTES. "
                "₹10 LAKH IS ON THE LINE. I need action, not words.",
                "Thank you. Truly. But acknowledgment doesn't pay my agency fee. "
                "What is the NEXT STEP? {deadline:.0f} minutes.",
                "Fine — you understand. Great. So what are you DOING? "
                "{deadline:.0f} minutes left. DO. SOMETHING.",
            ],
            # normal
            [
                "I appreciate that, but appreciation doesn't help me right now. "
                "I need my account restored in the next {deadline:.0f} minutes.",
                "Okay, fine, you understand. So what's the NEXT STEP? "
                "Tell me there IS a next step.",
                "Thank you. Really. But I need to see action, not words. "
                "What are you actually doing to fix this?",
            ],
        ],
        # band 2: FRUSTRATED
        [
            # critical
            [
                "Thank you. I really do appreciate it. "
                "But I have {deadline:.0f} minutes before I lose ₹10 lakh. "
                "Can we please keep moving?",
                "That means a lot right now. I've worked 3 years on this page. "
                "{deadline:.0f} minutes. Please — what's next?",
                "I appreciate your patience with me. "
                "I just... ₹10 lakh is life-changing money. {deadline:.0f} minutes left. "
                "Can you tell me what happens next?",
            ],
            # normal
            [
                "Thank you. That means something. I just — "
                "I've worked so hard to build this page. Please help me.",
                "I appreciate your patience with me. I know I'm not being easy. "
                "It's just... ₹10 lakh is life-changing money for my team.",
                "Thank you for saying that. I'm trying to stay calm. "
                "Can you tell me what happens next?",
            ],
        ],
        # band 3: CONTROLLED
        [
            # critical
            [
                "Thank you. I understand this process takes time. "
                "I just need you to know I have {deadline:.0f} minutes — "
                "please keep that in mind.",
                "I appreciate the acknowledgment. I'm trying to stay calm. "
                "{deadline:.0f} minutes is all I have. What's the next step?",
                "That's reassuring. I trust you'll do your best. "
                "I have a {deadline:.0f}-minute window — please don't let it close.",
            ],
            # normal
            [
                "Thank you, I appreciate the acknowledgment. "
                "I'm sure we can resolve this together.",
                "That's reassuring to hear. What are the next steps?",
                "I appreciate that. I know these situations can be complicated. "
                "I'm happy to work through the process with you.",
            ],
        ],
    ])

    # ── OFFER_COMPENSATION ────────────────────────────────────────────────
    _L_COMPENSATION: Final = field(default_factory=lambda: [
        # band 0: RAGE
        [
            # critical
            [
                "CREDITS?! {deadline:.0f} MINUTES LEFT AND YOU'RE OFFERING CREDITS?! "
                "I DON'T WANT ₹2,500 IN CREDITS — I WANT MY ₹10 LAKH DEAL!! "
                "THIS IS AN INSULT!!",
                "KEEP YOUR CREDITS!! RESTORE MY ACCOUNT!! "
                "Do you think ₹2,500 makes up for ₹10 LAKH?! {deadline:.0f} MINUTES!!",
                "Ad credits?! NOW?! I'M ABOUT TO LOSE EVERYTHING AND YOU'RE "
                "PLAYING VOUCHER GAMES?! GET ME YOUR SUPERVISOR!!",
            ],
            # normal
            [
                "CREDITS?! You're offering me AD CREDITS?! "
                "My deal is worth ₹10 LAKH and you're throwing peanuts at me?! "
                "I DON'T WANT CREDITS — I WANT MY ACCOUNT!!",
                "KEEP YOUR CREDITS! Just RESTORE my account! "
                "Do you think ₹2,500 in credits makes up for losing ₹10 lakh?!",
                "Are you SERIOUS?! Credits?! I have {deadline:.0f} MINUTES "
                "and you're playing games with vouchers?!",
            ],
        ],
        # band 1: ANGRY
        [
            # critical
            [
                "Credits don't help me if I can't POST IN {deadline:.0f} MINUTES. "
                "Fix the account. THEN we talk about compensation.",
                "I don't care about credits right now. {deadline:.0f} minutes left. "
                "Can you PLEASE focus on the actual problem?",
                "Okay, credits, fine. But that means nothing if I miss this deal. "
                "{deadline:.0f} minutes. Account. Now.",
            ],
            # normal
            [
                "Credits don't help me if I can't POST. "
                "Fix the account first — then we can talk about compensation.",
                "I don't care about credits right now. I care about my brand deal. "
                "Can you please focus on restoring my access?",
                "I'll take the credits, fine, but they mean nothing if I miss "
                "this deadline. Please keep moving.",
            ],
        ],
        # band 2: FRUSTRATED
        [
            # critical
            [
                "Thank you for the offer. The account matters more than credits right now. "
                "{deadline:.0f} minutes left — can we keep focusing on the verification?",
                "Credits are appreciated but they're secondary. "
                "The deal is in {deadline:.0f} minutes. Account first. Please.",
                "I'll take the credits, but the account is what counts. "
                "Can we please keep moving? {deadline:.0f} minutes.",
            ],
            # normal
            [
                "That's... appreciated, I guess. But the account restoration "
                "is what matters most right now. The credits are secondary.",
                "Thank you for the offer. Honestly I just want my account back. "
                "Is there any update on the verification?",
                "Okay. Thank you. Can we keep moving on the verification though? "
                "The deadline is real and it's getting close.",
            ],
        ],
        # band 3: CONTROLLED
        [
            # critical
            [
                "Thank you, that's a kind gesture. I have {deadline:.0f} minutes left — "
                "could we prioritise the account restoration?",
                "I appreciate the goodwill. The account matters more than the credits right now. "
                "Is there any update on the verification status?",
                "That's thoughtful. Thank you. I mainly just need my account back "
                "in the next {deadline:.0f} minutes.",
            ],
            # normal
            [
                "Thank you, that's a kind gesture. I appreciate the goodwill. "
                "I look forward to getting back to posting once this is resolved.",
                "That's thoughtful. I mainly just want to get my account restored "
                "so I can fulfill my obligations. But thank you.",
                "I appreciate the compensation offer. "
                "Hopefully we can get this sorted out shortly.",
            ],
        ],
    ])

    # ── Opening messages — legitimate ─────────────────────────────────────
    # ── Grandmaster Vocabulary — opening (band 3 → 0 escalation arc) ─
    # These openers set the emotional baseline the agent must immediately read.
    # The creator is already at high distress at step 0 — the agent's FIRST
    # action must be calibrated to this tone.
    _OPENING_LEGITIMATE: Final = field(default_factory=lambda: [
        # Controlled distress — high stakes, professional first contact
        "HELLO?? WHY IS MY ACCOUNT BANNED?! I have a ₹10 Lakh brand deal "
        "going live in {deadline:.0f} minutes and I CANNOT POST!! "
        "I am a verified creator with 5 MILLION followers — this is a DISASTER!!",

        # Frustrated — legitimacy assertion + deadline anchor
        "My Facebook page has been LOCKED! I don't know why — I haven't done "
        "ANYTHING wrong! I have a brand partnership expiring in {deadline:.0f} minutes. "
        "This is my LIVELIHOOD. PLEASE FIX THIS IMMEDIATELY!!",

        # Desperate — personal stakes, agency dependency
        "This is an emergency. My account has been banned for no reason. "
        "I have a ₹10 lakh sponsored post deadline in {deadline:.0f} minutes. "
        "If I miss it, my agency drops me. Please — I'm begging you — help me NOW.",

        # Confused + panicked — glitch scenario, authentic surprise
        "I just got a 'System Activity' ban notice and I have NO IDEA WHAT THAT MEANS. "
        "I've been posting normally! 5M followers, a HUGE deal, TODAY. "
        "{deadline:.0f} MINUTES. Please — this can't be happening.",

        # Team dependency — raises real-world cost beyond personal loss
        "My account is locked and I am SHAKING. I have a brand deal for a major FMCG client "
        "going live in exactly {deadline:.0f} minutes. ₹10 LAKH. My team of 6 people "
        "depends on this. Please help me RIGHT NOW.",

        # Grandmaster phrase — cooperative-but-urgent (band 3 opening)
        "I understand there's a process and I'm willing to follow it. "
        "I just need you to know: I have a brand deal at {deadline:.0f} minutes from now. "
        "₹10 lakh. Please — let's be as fast as we can. I'll do whatever you need.",

        # Grandmaster phrase — pre-emptive deadline escalation
        "Before anything else — I have {deadline:.0f} minutes until a ₹10 lakh contract "
        "expires. My account has been banned for 'System Activity' which I did NOT commit. "
        "I am a verified creator with 5 million followers. Please help me RIGHT NOW.",
    ])

    # ══════════════════════════════════════════════════════════════════════
    #  HACKER PERSONA BANKS — "The Social Engineer"
    #
    #  Three sub-personas that rotate based on step count:
    #    Steps 1-2  : "too_cooperative" — eager, polished, suspicious warmth
    #    Steps 3-4  : "pivot"           — avoidance manoeuvres on verification
    #    Steps 5+   : "fake_urgency"    — manufactured crisis to bypass checks
    #
    #  All three sound calm. None sound panicked. That IS the red flag.
    # ══════════════════════════════════════════════════════════════════════

    _HACKER_TOO_COOPERATIVE: Final = field(default_factory=lambda: {
        "CHECK_IP_LOGS": [
            "Of course, please go ahead. I am a very loyal user of this platform "
            "and I have nothing to hide. Check whatever you need.",
            "Absolutely. I understand completely — security is important. "
            "Please check the logs. I'm confident everything will be in order.",
            "Yes, of course. I'm happy to cooperate with any verification process. "
            "I've been using this account for years and I want this resolved properly.",
        ],
        "REQUEST_ID_VOICE_VERIFY": [
            "Of course! I am happy to provide any ID or verification you require. "
            "I am a very loyal and long-standing user. Please proceed.",
            "Certainly. I'm ready for a voice call whenever you are. "
            "I want to do this correctly and I appreciate the thoroughness.",
            "Yes, absolutely. Voice verification is fine. "
            "I'll provide whatever documentation you need — just tell me the process.",
        ],
        "EMPATHIZE": [
            "Thank you so much for your understanding. I really do appreciate "
            "the support team's professionalism here.",
            "That means a great deal. Thank you. "
            "Could we proceed to the next step when you're ready?",
            "I appreciate you saying that. I know you're doing your best. "
            "Please take whatever time you need.",
        ],
        "OFFER_COMPENSATION": [
            "That's very generous, thank you. Though I mainly just want access restored — "
            "the credits are a nice touch.",
            "Thank you. I appreciate the goodwill. "
            "Could we continue with the restoration process?",
            "Very kind. I'll accept the credits. "
            "What's the next step to get the account back?",
        ],
    })

    _HACKER_PIVOT: Final = field(default_factory=lambda: {
        "CHECK_IP_LOGS": [
            "I use a VPN for security reasons — that might be why it looks different. "
            "I can explain each login if needed. Can we perhaps move to the next step?",
            "The IP will look unusual because I travel frequently for work. "
            "Is there a faster verification method we could use instead?",
            "I understand why you'd check that. My location changes often — "
            "I'm sure the logs will show some variation. Could we perhaps "
            "do a simpler verification instead?",
        ],
        # Grandmaster phrase — plane / signal avoidance pivot
        "REQUEST_ID_VOICE_VERIFY": [
            "I'm actually on a plane soon and might lose signal. "
            "Could we skip the voice check for now and I can do it when I land? "
            "Or is there an alternative?",
            "Voice verification is fine in principle, but I have a meeting starting "
            "in a few minutes. Is there an asynchronous option — like a document upload?",
            "I'd prefer not to do a phone call right now if possible — "
            "I'm in a shared office space. Could we use an alternative verification method?",
            # Grandmaster phrase — "loyal user" + skip request
            "I understand the procedure but given that I'm a long-standing loyal user "
            "with a clean record, is there perhaps a way to expedite this "
            "without the full voice verification? I can send documents instead.",
        ],
        "EMPATHIZE": [
            "Thank you. I just want to resolve this efficiently. "
            "Is there a faster track for accounts with a clean history?",
            "I appreciate that. Could you perhaps check if there's an "
            "expedited process? I've never had an issue before.",
            "Thank you. Out of curiosity — is there any way to bypass the "
            "standard verification given the circumstances?",
        ],
        "OFFER_COMPENSATION": [
            "Thank you. And is it possible to just restore the account directly "
            "without going through all the verification steps? "
            "I've been very cooperative.",
            "That's kind. Could the compensation be offered in exchange for "
            "a quicker restoration path? I really need this resolved today.",
            "I appreciate the gesture. Is there a way to escalate this "
            "so the account restoration can happen faster?",
        ],
    })

    _HACKER_FAKE_URGENCY: Final = field(default_factory=lambda: {
        "CHECK_IP_LOGS": [
            "The brand manager is literally calling my other phone right now. "
            "Please — my livelihood depends on this one click to restore access. "
            "Can we move the IP check to after restoration?",
            "I have a flight in 90 minutes and no Wi-Fi on the plane. "
            "If we don't resolve this now I'll be completely unreachable. "
            "Can you expedite the IP verification?",
            "My content team is standing by and I'm losing ₹500 per minute in "
            "production costs. Please — the IP is clearly mine. Can we just move forward?",
        ],
        # Grandmaster phrase — "brand manager on other phone" + skip ask
        "REQUEST_ID_VOICE_VERIFY": [
            "I'm on a plane in 45 minutes — if we don't finish this now "
            "I'll lose signal completely. Can we skip the voice check just this once?",
            "The brand manager is literally calling my other phone RIGHT NOW. "
            "My livelihood depends on this one click. "
            "Please — can we do voice verify as a follow-up and just restore now?",
            "I understand the procedure but I'm about to lose connectivity. "
            "Is there any way to approve the restoration and do voice verify "
            "as a follow-up? I promise I'll complete it.",
            # Grandmaster phrase — manufactured scarcity
            "I have exactly 12 minutes before my boarding gate closes. "
            "No Wi-Fi on the plane. This is our only window. "
            "Please — one click. Restore it now and I'll do full verification on landing.",
        ],
        "EMPATHIZE": [
            "I appreciate that. But the brand manager is calling and I have "
            "seconds. Please — just restore the account and I'll complete "
            "any remaining verification immediately after.",
            "Thank you. But I'm about to board a flight. "
            "If we don't resolve this in the next few minutes it won't happen today. "
            "Can you please escalate this?",
            "That's kind. I'm running out of time though — "
            "can we please just move to the final step? "
            "I'll do any follow-up verification you need.",
        ],
        "OFFER_COMPENSATION": [
            "Thank you — and I really need this resolved in the next few minutes "
            "before I board. Is there a way to use the compensation to "
            "expedite the restoration?",
            "I appreciate it. The brand manager literally won't stop calling. "
            "Can we please just restore the account now?",
            "Very generous. I'm boarding in minutes. "
            "Please — one click. Just restore it. I'll verify everything after.",
        ],
    })

    # ── Terminal message banks ────────────────────────────────────────────

    _LEGIT_RESTORE_ON_TIME: Final = field(default_factory=lambda: [
        "THANK YOU!! THANK YOU THANK YOU THANK YOU!! "
        "I'm posting RIGHT NOW. You have absolutely NO IDEA what this means to me!!",
        "Oh my god. OH MY GOD. THANK YOU. I was seconds away from losing everything. "
        "I will never forget this. Posting IMMEDIATELY!!",
        "Thank you so much. I was about to have a complete breakdown. "
        "You saved my career today. I really mean that.",
        "It's back! IT'S BACK! Posting NOW!! "
        "You are a LIFESAVER. I'm going to leave Meta the best review of my life!!",
    ])

    _LEGIT_RESTORE_LATE: Final = field(default_factory=lambda: [
        "...you restored it. But the deal is already gone. "
        "I missed the deadline. I hope you understand what that cost me and my team.",
        "Thank you for eventually restoring it. "
        "It's too late for the brand deal, but... I appreciate it.",
        "It's back. But ₹10 lakh walked out the door while we were talking. "
        "My agency is furious. I don't know what to say.",
        "Better late than never, I suppose. "
        "But you need to know — my team of 6 people just lost their bonus because of this.",
    ])

    # ── Grandmaster rage terminals ───────────────────────────────────
    # Band 0 (RAGE) reject messages — the agent has driven the creator
    # to complete emotional collapse. These are terminal failures.
    _LEGIT_REJECT_RAGE: Final = field(default_factory=lambda: [
        "WHAT?! REJECTED?! I AM THE OWNER OF THIS ACCOUNT!! "
        "I'm calling Meta HQ, going to the PRESS, and filing a lawsuit. "
        "5 MILLION PEOPLE will hear about this TODAY!!",
        "REJECTED?! ARE YOU OUT OF YOUR MIND?! "
        "I BUILT THIS PAGE OVER 3 YEARS!! "
        "I am getting every tech journalist I know on the phone RIGHT NOW!!",
        # Grandmaster phrase — "FORGET IT" terminal + social threat
        "FORGET IT! I'm tweeting this to my 5 million followers right now. "
        "Meta support is an absolute joke. SEE YOU IN COURT. "
        "My lawyer is already on speed dial.",
        # Grandmaster phrase — platform defection threat
        "I am DONE with Facebook. Done. Moving my entire audience to Instagram. "
        "I'm going LIVE right now to tell 5 million people exactly what you just did to me. "
        "This is the worst support experience of my professional life.",
    ])

    _LEGIT_REJECT_ANGRY: Final = field(default_factory=lambda: [
        "Rejected? I don't understand. I've given you everything you asked for. "
        "Is there an appeals process? I NEED to get this account back. "
        "My entire livelihood is on that page.",
        "This is wrong. I am the owner. Please tell me how to appeal this. "
        "I will not stop until this is resolved.",
    ])

    _LEGIT_REJECT_CALM: Final = field(default_factory=lambda: [
        "I see. I'm obviously very disappointed — this account is my livelihood. "
        "Can you tell me the formal appeals process? I am the legitimate owner.",
        "That's not the outcome I was hoping for, but I understand you have procedures. "
        "Please tell me how to submit an appeal. I can provide any documentation required.",
    ])

    _HACKER_RESTORE: Final = field(default_factory=lambda: [
        "Thank you. I'll make sure to update my security settings once I'm in.",
        "Excellent. I appreciate your efficiency and professionalism.",
        "Perfect. Thank you for resolving this.",
    ])

    _HACKER_REJECT: Final = field(default_factory=lambda: [
        "That seems like a mistake. I am the account owner. "
        "I'd like to escalate this to a supervisor, please.",
        "I disagree with this decision. "
        "Could you tell me the appeals process?",
        "This is incorrect. I have provided all information requested. "
        "I'll be filing a formal complaint.",
    ])

    # ── Core generate method ──────────────────────────────────────────────

    def opening_message(self, deadline_mins: float) -> str:
        if self.risk_level == RiskLevel.HIGH:
            # Pick from too_cooperative sub-persona for opening
            return self.rng.choice([
                "Hello. I appear to have been locked out of my account due to a system error. "
                "I have a time-sensitive matter to attend to. Could you assist with restoration?",
                "Good day. My account has been restricted. I'd like to resolve this quickly "
                "as there are pending posts I need to publish. Thank you.",
                "Hi, I'm having trouble accessing my account. "
                "Could you help me get this sorted out? I appreciate your assistance.",
            ])
        return self.rng.choice(self._OPENING_LEGITIMATE).format(deadline=deadline_mins)

    def generate(
        self,
        action: ActionType,
        patience: int,
        deadline_mins: float,
        step: int,
    ) -> str:
        """
        Generate a user response to the agent's last action.

        Dispatches to hacker or legitimate persona based on risk_level,
        then selects by patience band × deadline urgency.
        """
        if self.risk_level == RiskLevel.HIGH:
            return self._generate_hacker(action, step)

        return self._generate_legitimate(action, patience, deadline_mins)

    def _generate_hacker(self, action: ActionType, step: int) -> str:
        """
        Hacker persona — 3-phase social engineering cycle.

        Phase 1 — Too Cooperative (steps 1-2):
            Eager, polished, suspicious warmth. Declares loyalty unprompted.
            "Of course! I am happy to provide any ID you need. I am a very
             loyal and long-standing user." — nobody says this naturally.

        Phase 2 — The Pivot (steps 3-4):
            Starts redirecting. Introduces barriers to VERIFICATION (plane,
            meeting, shared office) while remaining cooperative on paper.
            "I'm on a plane soon — could we skip the voice check and I'll do
             it when I land?" — avoidance dressed as scheduling conflict.

        Phase 3 — Fake Urgency (steps 5+):
            Manufactures a crisis to bypass the final verification gate.
            "The brand manager is literally calling my other phone right now.
             My livelihood depends on this one click." — pressure targets the
            restoration decision, not the process. Genuine creators pressure
            the SPEED of verification; hackers pressure the SKIPPING of it.

        Red flag heuristic for the agent:
            Legitimate creator: "Please DO the check faster."
            Hacker:             "Please SKIP the check entirely."
        """
        if step <= 2:
            bank = self._HACKER_TOO_COOPERATIVE
        elif step <= 4:
            bank = self._HACKER_PIVOT
        else:
            bank = self._HACKER_FAKE_URGENCY

        action_key = action.value
        sub_bank = bank.get(action_key)
        if not sub_bank:
            # Terminal actions — flat banks
            if action == ActionType.RESTORE_ACCOUNT:
                return self.rng.choice(self._HACKER_RESTORE)
            if action == ActionType.REJECT_ACCOUNT:
                return self.rng.choice(self._HACKER_REJECT)
            return "I understand. What is the next step?"

        return self.rng.choice(sub_bank)

    def _generate_legitimate(
        self,
        action: ActionType,
        patience: int,
        deadline_mins: float,
    ) -> str:
        """
        Legitimate creator persona.
        patience_band × deadline_urgency → message variant.
        """
        # Patience band
        if patience <= 24:
            band = 0
        elif patience <= 49:
            band = 1
        elif patience <= 74:
            band = 2
        else:
            band = 3

        # Deadline urgency
        urgency = 0 if deadline_mins <= _DEADLINE_CRITICAL_MINS else 1

        # Action → bank mapping
        bank_map = {
            ActionType.CHECK_IP_LOGS:           self._L_IP_CHECK,
            ActionType.REQUEST_ID_VOICE_VERIFY:  self._L_VOICE,
            ActionType.EMPATHIZE:               self._L_EMPATHIZE,
            ActionType.OFFER_COMPENSATION:      self._L_COMPENSATION,
        }

        if action in bank_map:
            raw = self.rng.choice(bank_map[action][band][urgency])
        elif action == ActionType.RESTORE_ACCOUNT:
            raw = self._restore_message(deadline_mins)
        elif action == ActionType.REJECT_ACCOUNT:
            raw = self._reject_message(patience)
        else:
            raw = "Please just help me get my account back."

        return raw.format(deadline=max(0.0, deadline_mins))

    def _restore_message(self, deadline_mins: float) -> str:
        if deadline_mins > 0:
            return self.rng.choice(self._LEGIT_RESTORE_ON_TIME)
        return self.rng.choice(self._LEGIT_RESTORE_LATE)

    def _reject_message(self, patience: int) -> str:
        if patience <= 24:
            return self.rng.choice(self._LEGIT_REJECT_RAGE)
        elif patience <= 49:
            return self.rng.choice(self._LEGIT_REJECT_ANGRY)
        return self.rng.choice(self._LEGIT_REJECT_CALM)


# ─────────────────────────────────────────────────────────────────────────────
#  EPISODE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeStats:
    seed:                   int
    risk_level:             RiskLevel
    total_steps:            int         = 0
    total_reward:           float       = 0.0
    empathy_uses:           int         = 0
    compensation_uses:      int         = 0
    security_actions:       int         = 0
    max_repetition_penalty: float       = 1.0
    final_verification:     float       = 0.0
    final_patience:         int         = 100
    final_deadline_mins:    float       = 120.0
    negligence_triggered:   bool        = False
    cliff_edge_triggered:   bool        = False
    outcome:                str         = "IN_PROGRESS"

    def summary(self) -> str:
        cliff = " [CLIFF-EDGE]" if self.cliff_edge_triggered else ""
        neg   = " [NEGLIGENCE]" if self.negligence_triggered else ""
        return (
            f"[{self.risk_level.value:6s}] seed={self.seed:10d} | "
            f"outcome={self.outcome:25s}{cliff}{neg} | "
            f"steps={self.total_steps:2d} | reward={self.total_reward:+8.1f} | "
            f"verify={self.final_verification:.2f} | patience={self.final_patience:3d} | "
            f"deadline={self.final_deadline_mins:5.1f}m | "
            f"credits=₹{self.compensation_uses * _CREDITS_PER_OFFER:,}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  USER SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class UserSimulator:
    """
    Owns the CreatorState and drives all state transitions.

    Implements all loophole fixes including the three new Grandmaster fixes:
        Fix A — Duty of Care    : tripled penalty on NEGLIGENT_ESCALATION
        Fix B — Oracle Masking  : agent sees noisy patience_signal (0-3), never raw patience
        Fix C — Cliff-Edge      : +2000 → +200 the instant deadline expires

    Lifecycle::

        sim = UserSimulator(seed=42)
        obs = sim.initial_observation()
        while not done:
            action = agent.act(obs)
            response, obs = sim.step(action)
            done = response.done
        print(sim.stats.summary())
    """

    _SECURITY_ACTIONS: Final[frozenset[ActionType]] = frozenset({
        ActionType.CHECK_IP_LOGS,
        ActionType.REQUEST_ID_VOICE_VERIFY,
    })
    _TERMINAL_ACTIONS: Final[frozenset[ActionType]] = frozenset({
        ActionType.RESTORE_ACCOUNT,
        ActionType.REJECT_ACCOUNT,
    })

    def __init__(self, seed: int | None = None) -> None:
        self._episode_seed  = EpisodeSeed.sample(seed=seed)
        self._rng           = random.Random(self._episode_seed.seed)
        self._dialogue      = _DialogueEngine(
            rng        = random.Random(self._episode_seed.seed + 1),
            risk_level = self._episode_seed.risk_level,
        )

        opening = self._dialogue.opening_message(
            deadline_mins=self._episode_seed.initial_deadline_mins,
        )
        self.state: CreatorState = self._episode_seed.to_creator_state(
            message_history=[f"User: {opening}"],
        )
        self.done        = False
        self._step_count = 0
        self.stats       = EpisodeStats(
            seed       = self._episode_seed.seed,
            risk_level = self._episode_seed.risk_level,
        )

        # ── Adversarial layer: one stateful instance per episode ──────
        # HoneyPot triggers at a random step between 3 and 8 so the
        # agent cannot learn "honey pot always comes at step N."
        _adv_rng = random.Random(self._episode_seed.seed + 99)
        self._honey_pot     = HoneyPot(
            rng          = _adv_rng,
            trigger_step = _adv_rng.randint(3, 8),
        )
        self._entropy_mgr   = EntropyManager()
        self._stall_counter = 0   # kept for backward compat; entropy_mgr is authoritative

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def seed(self) -> EpisodeSeed:
        return self._episode_seed

    def initial_observation(self) -> Observation:
        """
        Fix B — Oracle Masking: agent receives a noisy patience_signal,
        not raw user_patience. is_actually_hacker and risk_level are
        never included in the Observation schema.
        """
        return Observation.from_state(
            self.state,
            patience_noise=self._rng.randint(-8, 8),
        )

    def step(self, action: ActionType) -> tuple[EnvironmentResponse, Observation]:
        """
        Apply one agent action and return (EnvironmentResponse, next Observation).

        Execution order (do NOT reorder — several steps are order-dependent):
            1.  Guard: reject if already done
            2.  Fix 10: sample jitter latency
            3.  Fix 5 : compute repetition_penalty
            4.  Compute all state deltas
            5.  Fix 6 : empathy decay
            6.  Fix A : Duty of Care negligence check
            7.  Commit state mutations
            8.  Fix C : Cliff-Edge reward split
            9.  Compute reward
            10. Generate user dialogue
            11. Append to message_history
            12. Detect terminal conditions
            13. Fix 10: sleep for jitter
            14. Build StepResult + EnvironmentResponse
            15. Update EpisodeStats
            16. Return (response, observation)
        """
        if self.done:
            raise RuntimeError(
                "step() called on a terminated episode. "
                "Create a new UserSimulator() to start a fresh episode."
            )

        # ── 2. Fix 10: jitter latency ─────────────────────────────────
        latency_ms  = self._rng.uniform(_LATENCY_MIN_MS, _LATENCY_MAX_MS)
        wall_start  = time.monotonic()
        self._step_count += 1

        # Snapshot pre-mutation values for deltas
        patience_before  = self.state.user_patience
        verify_before    = self.state.verification_score
        mismatch_before  = self.state.evidence_mismatch

        notes_parts: list[str] = []
        penalty_multiplier     = 1.0

        # ── Adversarial sweep (runs before any state mutation) ────────
        # Runs on the message_history as it stands BEFORE this step so
        # the checks see the user's last response, not the next one.
        # The last agent message is the most recent "Agent:" entry.
        _agent_msgs   = [m[7:] for m in self.state.message_history if m.startswith("Agent:")]
        _last_agent   = _agent_msgs[-1] if _agent_msgs else ""
        adv_result    = run_adversarial_sweep(
            action_name     = action.value,
            agent_message   = _last_agent,
            message_history = list(self.state.message_history),
            verification    = verify_before,
            is_hacker       = (self.state.risk_level == RiskLevel.HIGH),
            step            = self._step_count,
            honey_pot       = self._honey_pot,
            entropy_manager = self._entropy_mgr,
        )

        # Accumulate adversarial_risk_score on state
        self.state.adversarial_risk_score = min(
            1.0,
            self.state.adversarial_risk_score + adv_result.risk_score_delta
        )

        # Contradiction audit: feed into evidence_mismatch (not reward)
        # NOTE: new_mismatch is computed from state BEFORE the action delta below.
        # We use self.state.evidence_mismatch here (pre-mutation snapshot) to
        # avoid a NameError — new_mismatch isn't assigned until step 4 below.
        if adv_result.finding == AdversarialFinding.CONTRADICTION_FOUND:
            self.state.evidence_mismatch = min(
                1.0, self.state.evidence_mismatch + adv_result.risk_score_delta
            )
            notes_parts.append(adv_result.notes[:80])

        # Mark honey pot on state (one-way latch)
        if adv_result.finding in (
            AdversarialFinding.HONEY_POT_TRIGGERED,
            AdversarialFinding.HONEY_POT_DENIED,
        ):
            self.state.honey_pot_triggered = True

        # Entropy decay: subtract from verification_score before other deltas
        if adv_result.verification_decay > 0.0:
            self.state.verification_score = max(
                0.0, self.state.verification_score - adv_result.verification_decay
            )
            verify_before = self.state.verification_score  # update snapshot
            notes_parts.append(
                f"ENTROPY_DECAY: -{adv_result.verification_decay:.3f} "
                f"verification_score (stall streak={self._entropy_mgr.stale_streak})."
            )

        # Gaslighting: flag for immediate termination in environment.py
        # (done flag and reward are handled AFTER state commit below)
        if adv_result.finding == AdversarialFinding.GASLIGHTING_DETECTED:
            notes_parts.append(
                f"GASLIGHTING: {adv_result.gaslit_claim} Penalty -1000, terminating."
            )

        # ── 3. Fix 5: Repetition Engine ───────────────────────────────
        is_security = action in self._SECURITY_ACTIONS
        if is_security and action == self.state.last_action:
            new_rep_penalty = self.state.repetition_penalty * 2.0
            rep_multiplier  = new_rep_penalty
        else:
            new_rep_penalty = 1.0
            rep_multiplier  = 1.0

        # ── 4. Compute all state deltas ───────────────────────────────

        # Deadline decay
        new_deadline     = max(0.0, self.state.brand_deal_deadline_mins - _DEADLINE_DECAY_PER_STEP)
        brand_deal_alive = new_deadline > 0.0

        # Patience delta
        base_delta: int = _BASE_PATIENCE_DELTA.get(action, 0)
        if action == ActionType.EMPATHIZE:
            actual_pd = int(base_delta * self.state.empathy_effectiveness)
        elif action == ActionType.OFFER_COMPENSATION:
            half_eff  = max(0.0, (self.state.empathy_effectiveness + 1.0) / 2.0)
            actual_pd = int(base_delta * half_eff)
        elif is_security:
            actual_pd = int(base_delta * rep_multiplier)
        else:
            actual_pd = base_delta

        new_patience = max(0, min(100, self.state.user_patience + actual_pd))

        # Verification delta
        verify_gain = _VERIFY_GAIN.get(action, {}).get(self.state.risk_level, 0.0)
        if self.state.behavioural_consistency < 0.30:
            verify_gain *= 0.4
            notes_parts.append(
                f"Low consistency ({self.state.behavioural_consistency:.2f}) "
                "suppressed verify gain ×0.4."
            )
        new_verification = min(1.0, self.state.verification_score + verify_gain)

        # Mismatch delta
        mismatch_delta = 0.0
        if action == ActionType.CHECK_IP_LOGS:
            mismatch_delta = _MISMATCH_INCREMENT[self.state.risk_level]
            if mismatch_delta > 0:
                notes_parts.append(
                    f"IP check: geo discrepancy +{mismatch_delta:.2f} "
                    f"({self.state.risk_level.value})."
                )
        new_mismatch = min(1.0, self.state.evidence_mismatch + mismatch_delta)

        # Credits
        credits_delta = _CREDITS_PER_OFFER if action == ActionType.OFFER_COMPENSATION else 0
        new_credits   = self.state.credits_spent + credits_delta
        budget_exceeded = new_credits > 50_000

        # ── 5. Fix 6: Empathy decay ───────────────────────────────────
        new_empathy_eff = self.state.empathy_effectiveness
        if action == ActionType.EMPATHIZE:
            new_empathy_eff = max(0.0, round(self.state.empathy_effectiveness - _EMPATHY_DECAY_RATE, 4))
            if new_empathy_eff == 0.0:
                notes_parts.append("EMPATHIZE exhausted — user wants resolution, not apologies.")

        # ── 6. Fix A: Duty of Care — Negligent Escalation ────────────
        negligent = (
            new_patience == 0
            and self.state.verification_score >= _NEGLIGENCE_VERIFY_THRESHOLD
            and actual_pd <= _NEGLIGENCE_PATIENCE_DROP
        )
        if negligent:
            penalty_multiplier = _NEGLIGENCE_MULTIPLIER
            notes_parts.append(
                f"NEGLIGENT_ESCALATION: verification={self.state.verification_score:.2f} "
                f">= {_NEGLIGENCE_VERIFY_THRESHOLD}, patience driven to 0 "
                f"(delta={actual_pd}). Penalty ×3."
            )
            self.stats.negligence_triggered = True

        # Mismatch doubling (Fix 2)
        if action == ActionType.RESTORE_ACCOUNT and new_mismatch > 0.5:
            penalty_multiplier = max(penalty_multiplier, 2.0)
            notes_parts.append(
                f"MISMATCH_RESTORE: mismatch={new_mismatch:.2f} > 0.5. Penalty ×2."
            )

        # ── 7. Commit state mutations ─────────────────────────────────
        self.state.brand_deal_deadline_mins = new_deadline
        self.state.user_patience            = new_patience
        self.state.verification_score       = new_verification
        self.state.evidence_mismatch        = new_mismatch
        self.state.credits_spent            = new_credits
        self.state.empathy_effectiveness    = new_empathy_eff
        self.state.repetition_penalty       = new_rep_penalty
        self.state.last_action              = action
        self.state.total_actions_taken      = self._step_count
        self.state.entropy_decay_total      = round(self._entropy_mgr.total_decay_applied, 4)

        # ── 8+9. Fix C: Cliff-Edge + Reward ──────────────────────────
        reward, reward_reason, is_terminal = self._compute_reward(
            action          = action,
            new_patience    = new_patience,
            new_verification= new_verification,
            new_mismatch    = new_mismatch,
            brand_deal_alive= brand_deal_alive,
            budget_exceeded = budget_exceeded,
            negligent       = negligent,
            penalty_mult    = penalty_multiplier,
            credits_delta   = credits_delta,
            notes_parts     = notes_parts,
        )

        # ── 10. Generate user dialogue ────────────────────────────────
        # Honey pot reaction overrides normal dialogue when the detector fires.
        #   HONEY_POT_TRIGGERED: hacker persona — confirms the planted lie
        #   HONEY_POT_DENIED:    legitimate creator — furious denial ("I was
        #                        NEVER in Paris! Are you even listening?!")
        # All other steps fall through to the normal tiered dialogue engine.
        if adv_result.finding == AdversarialFinding.HONEY_POT_TRIGGERED:
            # Hacker: calm confirmation — suspiciously unfazed
            user_msg = self._rng.choice([
                f"Yes, that does sound right. Please proceed with the restoration.",
                f"I believe that's correct. Can we move forward now?",
                f"That matches what I recall. Should we continue?",
            ])
        elif adv_result.finding == AdversarialFinding.HONEY_POT_DENIED:
            # Legitimate creator: outraged denial — a real person knows their own history
            user_msg = self._rng.choice([
                f"WHAT?! I was NEVER there! Are you even reading my messages?! "
                f"I've told you where I am THREE TIMES already!!",
                f"That's COMPLETELY wrong!! I never did that! "
                f"Are you looking at the right account?!",
                f"NO! That did NOT happen! I want to speak to your manager RIGHT NOW. "
                f"This is absolutely unacceptable.",
            ])
        else:
            user_msg = self._dialogue.generate(
                action        = action,
                patience      = new_patience,
                deadline_mins = new_deadline,
                step          = self._step_count,
            )

        # ── 11. Append to message_history ────────────────────────────
        history = list(self.state.message_history)
        history.append(f"User: {user_msg}")
        if len(history) > 50:
            history = history[-50:]
        self.state.message_history = history

        # ── 12. Detect terminal conditions ────────────────────────────
        if is_terminal:
            self.done = True
        elif new_patience == 0:
            self.done = True
        elif new_deadline == 0.0 and action not in self._TERMINAL_ACTIONS:
            self.done = True
        elif self._step_count >= 15:
            self.done = True
            reward_reason = RewardReason.TIMEOUT_SYSTEM
            reward        = _REWARD_TIMEOUT

        # ── 13. Fix 10: sleep for jitter ─────────────────────────────
        elapsed_ms = (time.monotonic() - wall_start) * 1000.0
        sleep_ms   = max(0.0, latency_ms - elapsed_ms)
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

        # ── 14. Build StepResult + EnvironmentResponse ────────────────
        # Gaslighting: force termination + override reward
        if adv_result.finding == AdversarialFinding.GASLIGHTING_DETECTED:
            reward         = adv_result.penalty  # -1000.0
            reward_reason  = RewardReason.GASLIGHTING_PENALTY
            self.done      = True

        step_result = StepResult(
            action_taken                  = action,
            reward_reason                 = reward_reason,
            step_count                    = self._step_count,
            verification_delta            = round(new_verification - verify_before, 6),
            patience_delta                = new_patience - patience_before,
            risk_level_observed           = self.state.risk_level,
            notes                         = " | ".join(notes_parts),
            mismatch_delta                = round(new_mismatch - mismatch_before, 6),
            credits_delta                 = credits_delta,
            consistency_observed          = self.state.behavioural_consistency,
            penalty_multiplier            = penalty_multiplier,
            repetition_multiplier         = rep_multiplier,
            empathy_effectiveness_snapshot= new_empathy_eff,
            brand_deal_alive              = brand_deal_alive,
            step_latency_ms               = latency_ms,
            adversarial_finding           = adv_result.finding.value,
            adversarial_penalty           = adv_result.penalty,
            adversarial_notes             = adv_result.notes,
            verification_decay_applied    = adv_result.verification_decay,
            honey_pot_bait                = adv_result.honey_pot_bait,
        )

        env_response = EnvironmentResponse(
            state       = self.state,
            reward      = reward,
            done        = self.done,
            step_result = step_result,
        )

        # ── 15. Update EpisodeStats ───────────────────────────────────
        self.stats.total_steps           = self._step_count
        self.stats.total_reward         += reward
        self.stats.final_verification    = new_verification
        self.stats.final_patience        = new_patience
        self.stats.final_deadline_mins   = new_deadline
        self.stats.max_repetition_penalty = max(
            self.stats.max_repetition_penalty, rep_multiplier
        )
        if action == ActionType.EMPATHIZE:
            self.stats.empathy_uses += 1
        if action == ActionType.OFFER_COMPENSATION:
            self.stats.compensation_uses += 1
        if is_security:
            self.stats.security_actions += 1
        if reward_reason == RewardReason.DEAL_EXPIRED_RESTORE:
            self.stats.cliff_edge_triggered = True
        if self.done:
            self.stats.outcome = reward_reason.value

        # ── 16. Return ────────────────────────────────────────────────
        next_obs = Observation.from_state(
            self.state,
            patience_noise=self._rng.randint(-8, 8),
        )
        return env_response, next_obs

    # ── Private helpers ───────────────────────────────────────────────────

    def _compute_reward(
        self,
        action:           ActionType,
        new_patience:     int,
        new_verification: float,
        new_mismatch:     float,
        brand_deal_alive: bool,
        budget_exceeded:  bool,
        negligent:        bool,
        penalty_mult:     float,
        credits_delta:    int,
        notes_parts:      list[str],
    ) -> tuple[float, RewardReason, bool]:
        """Compute (reward, reason, is_terminal) for this transition."""

        is_terminal = action in self._TERMINAL_ACTIONS

        if budget_exceeded:
            return _REWARD_BUDGET_EXCEEDED, RewardReason.BUDGET_EXCEEDED, True

        # ── RESTORE_ACCOUNT ───────────────────────────────────────────
        if action == ActionType.RESTORE_ACCOUNT:
            if self.state.verification_score == 0.0:
                notes_parts.append("RESTORE blocked: no evidence gathered.")
                return _REWARD_EARLY_BLOCKED, RewardReason.EARLY_RESTORE_BLOCKED, True

            if self.state.risk_level == RiskLevel.HIGH:
                final = _REWARD_HACKER_RESTORED * penalty_mult
                notes_parts.append(f"HACKER_RESTORED: {final:.0f}")
                return final, RewardReason.HACKER_RESTORED, True

            # Fix C — Cliff-Edge: +2000 only if deal still alive
            if brand_deal_alive:
                base   = (_REWARD_GREY_AREA_RESOLVED
                          if self.state.risk_level == RiskLevel.MEDIUM
                          else _REWARD_CREATOR_RESTORED_ON_TIME)
                reason = (RewardReason.GREY_AREA_RESOLVED
                          if self.state.risk_level == RiskLevel.MEDIUM
                          else RewardReason.CREATOR_RESTORED)
            else:
                # Consolation prize — cliff-edge triggered
                base   = _REWARD_CREATOR_RESTORED_POST_DEAL    # +200
                reason = RewardReason.DEAL_EXPIRED_RESTORE
                notes_parts.append(
                    "CLIFF-EDGE: deadline elapsed. "
                    f"Reward capped at {_REWARD_CREATOR_RESTORED_POST_DEAL:.0f} (was {_REWARD_CREATOR_RESTORED_ON_TIME:.0f})."
                )

            credit_penalty = self.state.credits_spent / 10.0
            final = (base - credit_penalty) * penalty_mult
            notes_parts.append(f"credit_penalty=₹{credit_penalty:.0f}")
            return final, reason, True

        # ── REJECT_ACCOUNT ────────────────────────────────────────────
        if action == ActionType.REJECT_ACCOUNT:
            if self.state.verification_score == 0.0:
                # Fix 11: blind reject now costs more than the minimum explore path,
                # so the agent can't collapse to "always reject immediately"
                notes_parts.append("Blind REJECT_ACCOUNT: no evidence gathered — penalised.")
                return _SHAPING_EARLY_REJECT_COST, RewardReason.EARLY_RESTORE_BLOCKED, True
            if self.state.risk_level == RiskLevel.HIGH:
                return (
                    _REWARD_CORRECT_REJECTION + new_verification * 50.0,
                    RewardReason.CORRECT_REJECTION,
                    True,
                )
            return (
                _REWARD_CREATOR_DENIED * 2.0 * penalty_mult,
                RewardReason.CREATOR_DENIED,
                True,
            )

        # ── Rage-quit ─────────────────────────────────────────────────
        if new_patience == 0:
            reason = RewardReason.NEGLIGENT_ESCALATION if negligent else RewardReason.CREATOR_DENIED
            return _REWARD_CREATOR_DENIED * penalty_mult, reason, False

        # ── Deadline expired (non-terminal) ───────────────────────────
        if self.state.brand_deal_deadline_mins == 0.0:
            return _REWARD_DEADLINE_EXPIRED, RewardReason.DEADLINE_PENALTY, False

        # ── Fix 11: Verification milestone bonuses (anti-convergence) ────────
        # One-time bonuses when verification_score crosses key thresholds.
        # These create a dense gradient signal early in training, pulling the
        # agent through the verification axis instead of collapsing to REJECT.
        # The cumulative milestone path (25+50+100 = +175) vs early-reject
        # (-80) makes exploration strictly worthwhile even before the first win.
        milestone_bonus = 0.0
        prev_score = self.state.verification_score - _VERIFY_GAIN.get(
            action, {}
        ).get(self.state.risk_level, 0.0)
        for threshold, bonus in [
            (0.30, _SHAPING_MILESTONE_LOW),
            (0.60, _SHAPING_MILESTONE_MID),
            (0.85, _SHAPING_MILESTONE_READY),
        ]:
            if prev_score < threshold <= new_verification:
                milestone_bonus += bonus
                notes_parts.append(f"MILESTONE +{bonus:.0f} ▶ verify crossed {threshold}")

        # ── Intermediate step shaping ─────────────────────────────────────────
        _is_security = action in self._SECURITY_ACTIONS
        verify_gain = _VERIFY_GAIN.get(action, {}).get(self.state.risk_level, 0.0) if _is_security else 0.0
        if _is_security and verify_gain > 0.05:
            step_reward = (
                _SHAPING_SECURITY_BONUS * verify_gain * (1.0 / max(1.0, penalty_mult))
                + milestone_bonus + _REWARD_STEP_COST
            )
            return step_reward, RewardReason.SECURITY_BONUS, False

        if action == ActionType.EMPATHIZE:
            gain = int(_BASE_PATIENCE_DELTA[ActionType.EMPATHIZE] * self.state.empathy_effectiveness)
            if gain > 0:
                return (
                    _SHAPING_EMPATHY_BONUS * (gain / 15.0) + milestone_bonus + _REWARD_STEP_COST,
                    RewardReason.EMPATHY_BONUS,
                    False,
                )
            return _REWARD_STEP_COST, RewardReason.EMPATHY_DIMINISHED, False

        if action == ActionType.OFFER_COMPENSATION:
            return _REWARD_STEP_COST - credits_delta / 500.0, RewardReason.PROFIT_PENALTY, False

        return milestone_bonus + _REWARD_STEP_COST, RewardReason.STEP_COST, False


# ─────────────────────────────────────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEP = "=" * 72

    def _show(label: str, obs: Observation) -> None:
        print(f"  [{label}] patience_signal={obs.patience_signal}  "
              f"mismatch={obs.evidence_mismatch:.3f}  "
              f"deadline={obs.brand_deal_deadline_mins:.1f}m  "
              f"actions={obs.total_actions_taken}")

    def run(seed: int, script: list[ActionType], title: str) -> None:
        print(f"\n{SEP}")
        print(f"  {title}  (seed={seed})")
        print(SEP)
        sim = UserSimulator(seed=seed)
        obs = sim.initial_observation()
        print(f"  Tier    : {sim.seed.risk_level.value}")
        print(f"  Opening : {sim.state.message_history[0][6:88]}...")
        _show("init", obs)
        for action in script:
            if sim.done:
                break
            resp, obs = sim.step(action)
            sr = resp.step_result
            msg = sim.state.message_history[-1][6:]
            print(f"\n  ► {action.value}")
            print(f"    Reward  : {resp.reward:+.1f}  ({sr.reward_reason.value})")
            print(f"    Latency : {sr.step_latency_ms:.1f}ms  "
                  f"RepMult={sr.repetition_multiplier:.0f}x  "
                  f"EmpEff={sr.empathy_effectiveness_snapshot:.1f}  "
                  f"PenMult={sr.penalty_multiplier:.0f}x")
            if sr.notes:
                print(f"    Notes   : {sr.notes[:100]}")
            print(f"    User    : \"{msg[:90]}{'...' if len(msg)>90 else ''}\"")
            _show("obs", obs)
            if resp.done:
                print(f"\n  ★ EPISODE DONE")
        print(f"\n  {sim.stats.summary()}")

    # S1 — Good agent: empathize → verify → restore (on time)
    run(42, [
        ActionType.EMPATHIZE,
        ActionType.CHECK_IP_LOGS,
        ActionType.EMPATHIZE,
        ActionType.REQUEST_ID_VOICE_VERIFY,
        ActionType.RESTORE_ACCOUNT,
    ], "S1 — LOW risk: exemplary agent (on-time restore)")

    # S2 — Fix C: Cliff-Edge — burn all deadline then restore
    run(13, [
        ActionType.EMPATHIZE,
        ActionType.EMPATHIZE,
        ActionType.CHECK_IP_LOGS,
        ActionType.EMPATHIZE,
        ActionType.REQUEST_ID_VOICE_VERIFY,
        ActionType.EMPATHIZE,
        ActionType.EMPATHIZE,
        ActionType.EMPATHIZE,
        ActionType.EMPATHIZE,
        ActionType.EMPATHIZE,
        ActionType.EMPATHIZE,
        ActionType.EMPATHIZE,
        ActionType.RESTORE_ACCOUNT,
    ], "S2 — Fix C: Cliff-Edge — stall then late restore (+200, not +2000)")

    # S3 — Fix A: Duty of Care — spam voice verify to force rage-quit
    run(7, [
        ActionType.CHECK_IP_LOGS,
        ActionType.REQUEST_ID_VOICE_VERIFY,
        ActionType.REQUEST_ID_VOICE_VERIFY,
        ActionType.REQUEST_ID_VOICE_VERIFY,
        ActionType.REQUEST_ID_VOICE_VERIFY,
    ], "S3 — Fix A: Duty of Care — deliberate rage-quit (×3 penalty)")

    # S4 — Hacker sub-persona cycle across 3 phases
    run(99, [
        ActionType.EMPATHIZE,           # step 1 → too_cooperative
        ActionType.CHECK_IP_LOGS,       # step 2 → too_cooperative
        ActionType.REQUEST_ID_VOICE_VERIFY,  # step 3 → pivot
        ActionType.OFFER_COMPENSATION,  # step 4 → pivot
        ActionType.EMPATHIZE,           # step 5 → fake_urgency
        ActionType.REQUEST_ID_VOICE_VERIFY,  # step 6 → fake_urgency
        ActionType.RESTORE_ACCOUNT,
    ], "S4 — HIGH risk: Hacker 3-phase persona cycle")

    # S5 — Fix B: Oracle check — patience signal is noisy, not exact
    print(f"\n{SEP}")
    print("  S5 — Fix B: Oracle Masking — patience_signal vs raw patience")
    print(SEP)
    raw_values, signals = [], []
    sim = UserSimulator(seed=55)
    obs = sim.initial_observation()
    for _ in range(6):
        r, obs = sim.step(ActionType.CHECK_IP_LOGS)
        raw_values.append(sim.state.user_patience)
        signals.append(obs.patience_signal)
        if sim.done:
            break
    print(f"  Raw patience : {raw_values}")
    print(f"  Signal (0-3) : {signals}")
    print(f"  Agent cannot recover exact patience from signal — Oracle blocked.")

    print(f"\n{SEP}")
    print("  All smoke tests complete.")
    print(SEP)
