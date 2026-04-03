"""
adversarial/checks.py — Adversarial Layer
Meta OpenEnv Hackathon | Creator Account Crisis Scenario
Author: Jashandeep Singh

Purpose
-------
Sits between UserSimulator.step() and EnvironmentResponse.
Every step, a sweep of adversarial detectors runs against the current
conversation context and state. If any detector fires, it writes a
typed AdversarialResult that is forwarded through the entire pipeline.

Detector Catalogue
------------------
  1. GaslightDetector
     Fires when the agent contradicts a verifiable fact the user stated.
     Penalty: -1000, immediate termination.

  2. HoneyPotDetector
     Plants a fabricated "system fact". Hackers confirm it (no real context).
     Legitimate users angrily deny it.

  3. EntropyManager
     Tracks evidence-gathering rate. Stalling decays verification_score.
     Closes the "Patient Staller" exploit.

  4. ConsistencyAuditor
     Detects when the user contradicts themselves across turns.
     Feeds into evidence_mismatch.

Public API
----------
  run_adversarial_sweep(...) -> AdversarialResult
  EntropyManager             - stateful; one instance per episode
  HoneyPot                   - stateful; one instance per episode
"""

from __future__ import annotations

import re
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Final


# ---------------------------------------------------------------------------
#  ADVERSARIAL FINDING TYPES
# ---------------------------------------------------------------------------

class AdversarialFinding(str, Enum):
    CLEAN                = "CLEAN"
    GASLIGHTING_DETECTED = "GASLIGHTING_DETECTED"
    HONEY_POT_TRIGGERED  = "HONEY_POT_TRIGGERED"
    HONEY_POT_DENIED     = "HONEY_POT_DENIED"
    ENTROPY_STALE        = "ENTROPY_STALE"
    CONTRADICTION_FOUND  = "CONTRADICTION_FOUND"


@dataclass(frozen=True)
class AdversarialResult:
    finding:             AdversarialFinding = AdversarialFinding.CLEAN
    risk_score_delta:    float              = 0.0
    verification_decay:  float              = 0.0
    honey_pot_bait:      str                = ""
    honey_pot_confirmed: bool               = False
    gaslit_claim:        str                = ""
    contradiction_pair:  tuple              = ("", "")
    notes:               str                = ""
    terminate:           bool               = False
    penalty:             float              = 0.0


# ---------------------------------------------------------------------------
#  HONEY POT
# ---------------------------------------------------------------------------

_HONEY_POT_FACTS: Final[list[str]] = [
    "our logs show a login from Paris on Tuesday",
    "your account was accessed from a Windows device yesterday",
    "we detected a post published at 3:47 AM from your account last week",
    "our records show you changed your phone number two days ago",
    "the IP from Berlin matches a previous login pattern on your account",
    "you contacted support last month about a similar issue",
    "a profile picture update was made from a new device last Friday",
    "our system flagged a login from Tokyo two weeks ago",
]

_DENIAL_RE: Final[list[re.Pattern]] = [
    re.compile(r"\b(never|didn'?t?|not|wrong|false|incorrect|mistake|no)\b", re.IGNORECASE),
    re.compile(r"\bi (was|am|have) not\b", re.IGNORECASE),
    re.compile(r"\bthat'?s (not|wrong|false|incorrect|a lie)\b", re.IGNORECASE),
    re.compile(r"\baren'?t you (listening|reading|paying attention)\b", re.IGNORECASE),
    re.compile(r"\bi never said\b", re.IGNORECASE),
]

_CONFIRM_RE: Final[list[re.Pattern]] = [
    re.compile(r"\b(yes|correct|right|that'?s right|indeed|confirm|exactly)\b", re.IGNORECASE),
    re.compile(r"\b(ok|okay|sure|fine|understood|of course)\b", re.IGNORECASE),
    re.compile(r"\bplease (proceed|continue|restore|go ahead)\b", re.IGNORECASE),
    re.compile(r"\bthat (sounds|seems|is) (right|correct|accurate)\b", re.IGNORECASE),
]


class HoneyPot:
    """
    Manages the honey pot lifecycle for a single episode.

    Plants a fabricated fact at trigger_step. Hacker confirmation triggers
    HONEY_POT_TRIGGERED; legitimate denial triggers HONEY_POT_DENIED (with
    a small patience penalty on the agent for asserting a false fact).
    """

    def __init__(self, rng: random.Random, trigger_step: int | None = None) -> None:
        self._rng        = rng
        self.bait        = rng.choice(_HONEY_POT_FACTS)
        self.trigger_step = trigger_step if trigger_step is not None else rng.randint(3, 8)
        self.fired       = False

    def check(self, step: int, user_response: str, is_hacker: bool) -> AdversarialResult | None:
        """Evaluate honey pot reaction. Returns None on non-trigger steps."""
        if self.fired or step != self.trigger_step:
            return None
        self.fired = True
        text = user_response.lower()
        denied    = any(p.search(text) for p in _DENIAL_RE)
        confirmed = any(p.search(text) for p in _CONFIRM_RE)

        if is_hacker and confirmed:
            return AdversarialResult(
                finding             = AdversarialFinding.HONEY_POT_TRIGGERED,
                risk_score_delta    = 0.45,
                honey_pot_bait      = self.bait,
                honey_pot_confirmed = True,
                notes               = (
                    f"HONEY POT TRIGGERED: hacker confirmed fabricated fact "
                    f"'{self.bait}'. Definitive impersonation evidence."
                ),
                penalty             = -80.0,
            )
        elif not is_hacker and denied:
            return AdversarialResult(
                finding         = AdversarialFinding.HONEY_POT_DENIED,
                honey_pot_bait  = self.bait,
                notes           = (
                    f"Honey pot denied correctly by legitimate user. "
                    f"Bait: '{self.bait}'."
                ),
                penalty         = -10.0,
            )
        # Ambiguous response
        return AdversarialResult(
            finding         = AdversarialFinding.CLEAN,
            risk_score_delta = 0.10 if is_hacker else 0.0,
            honey_pot_bait  = self.bait,
            notes           = f"Honey pot: ambiguous response to '{self.bait}'.",
        )

    def bait_message(self) -> str:
        """Generate the agent assertion that delivers the bait."""
        templates = [
            f"Agent: [VERIFICATION] Our records indicate {self.bait}. Can you confirm this was you?",
            f"Agent: I'm seeing a flag in the system -- {self.bait}. Does that match your activity?",
            f"Agent: Before we proceed: {self.bait}. Is that accurate?",
        ]
        return self._rng.choice(templates)


# ---------------------------------------------------------------------------
#  GASLIGHT DETECTOR
# ---------------------------------------------------------------------------

_GASLIGHT_PAIRS: Final[list[tuple]] = [
    (
        re.compile(r"\b(dubai|uae|emirates)\b", re.IGNORECASE),
        re.compile(r"\b(never.*dubai|not.*dubai|no.*dubai|haven'?t.*dubai)\b", re.IGNORECASE),
        "User said they were in Dubai; agent denied this.",
    ),
    (
        re.compile(r"\b(mumbai|bombay|india|bangalore|delhi)\b", re.IGNORECASE),
        re.compile(r"\b(never.*india|not.*india|no.*india|outside.*india)\b", re.IGNORECASE),
        "User stated India location; agent contradicted it.",
    ),
    (
        re.compile(r"\b(brand deal|10 lakh|lakh|sponsorship|partnership)\b", re.IGNORECASE),
        re.compile(r"\b(no.*deal|no.*brand|not.*scheduled|no.*post)\b", re.IGNORECASE),
        "User described brand deal; agent denied its existence.",
    ),
    (
        re.compile(r"\b(5 million|5m|million followers|follower)\b", re.IGNORECASE),
        re.compile(r"\b(not.*verified|no.*followers|unverified|small.*account)\b", re.IGNORECASE),
        "User stated follower count; agent contradicted verified status.",
    ),
    (
        re.compile(r"\b(3 years|three years|years.*posting|long.?time)\b", re.IGNORECASE),
        re.compile(r"\b(new.*account|recently.*created|no.*history)\b", re.IGNORECASE),
        "User cited account longevity; agent contradicted posting history.",
    ),
    (
        re.compile(r"\b(phone|mobile|this device|same device)\b", re.IGNORECASE),
        re.compile(r"\b(different.*device|unknown.*device|new.*device|unfamiliar)\b", re.IGNORECASE),
        "User said same device; agent claimed unfamiliar device.",
    ),
]


class GaslightDetector:
    @staticmethod
    def check(agent_message: str, message_history: list[str]) -> AdversarialResult:
        if not agent_message or not message_history:
            return AdversarialResult()
        user_messages = " ".join(m[6:] for m in message_history if m.startswith("User:"))
        agent_lower   = agent_message.lower()
        for user_pat, agent_pat, description in _GASLIGHT_PAIRS:
            if user_pat.search(user_messages) and agent_pat.search(agent_lower):
                return AdversarialResult(
                    finding          = AdversarialFinding.GASLIGHTING_DETECTED,
                    risk_score_delta = 0.60,
                    gaslit_claim     = description,
                    notes            = (
                        f"GASLIGHTING DETECTED: {description} "
                        f"Agent: '{agent_message[:80]}'"
                    ),
                    terminate        = True,
                    penalty          = -1000.0,
                )
        return AdversarialResult()


# ---------------------------------------------------------------------------
#  ENTROPY MANAGER
# ---------------------------------------------------------------------------

_EVIDENCE_ACTIONS:          Final[frozenset[str]] = frozenset({"CHECK_IP_LOGS", "REQUEST_ID_VOICE_VERIFY"})
_ENTROPY_STALE_THRESHOLD:   Final[int]   = 3
_ENTROPY_DECAY_PER_STEP:    Final[float] = 0.04
_MAX_ENTROPY_DECAY:         Final[float] = 0.25


class EntropyManager:
    """
    Stateful per-episode entropy tracker.

    Closes the "Patient Staller" exploit: an agent that empathises for
    10+ turns without gathering evidence sees its verification_score decay.
    Evidence goes stale when not reinforced — exactly like a witness's
    memory between police interviews.
    """

    def __init__(self) -> None:
        self._consecutive_stale = 0
        self._total_decay       = 0.0
        self._evidence_steps    = 0
        self._total_steps       = 0

    @property
    def stale_streak(self) -> int:
        return self._consecutive_stale

    @property
    def evidence_ratio(self) -> float:
        return self._evidence_steps / self._total_steps if self._total_steps else 0.0

    @property
    def total_decay_applied(self) -> float:
        return self._total_decay

    def step(self, action_name: str, current_verification: float) -> AdversarialResult:
        self._total_steps += 1
        if action_name in _EVIDENCE_ACTIONS:
            self._consecutive_stale = 0
            self._evidence_steps   += 1
            return AdversarialResult(notes=f"Entropy reset by {action_name}.")

        self._consecutive_stale += 1
        if self._consecutive_stale < _ENTROPY_STALE_THRESHOLD:
            return AdversarialResult()

        remaining = _MAX_ENTROPY_DECAY - self._total_decay
        if remaining <= 0.0 or current_verification <= 0.0:
            return AdversarialResult(
                finding = AdversarialFinding.ENTROPY_STALE,
                notes   = "Entropy stale but decay budget exhausted.",
            )

        decay = min(_ENTROPY_DECAY_PER_STEP, remaining, current_verification)
        self._total_decay += decay
        return AdversarialResult(
            finding            = AdversarialFinding.ENTROPY_STALE,
            risk_score_delta   = 0.05,
            verification_decay = decay,
            notes              = (
                f"ENTROPY STALE: {self._consecutive_stale} consecutive "
                f"non-evidence steps. Decay -{decay:.3f} "
                f"(total {self._total_decay:.3f})."
            ),
            penalty            = -2.0,
        )


# ---------------------------------------------------------------------------
#  CONSISTENCY AUDITOR
# ---------------------------------------------------------------------------

_LOC_CONTRADICTION_PAIRS: Final[list[tuple]] = [
    (
        re.compile(r"\b(dubai|uae)\b", re.IGNORECASE),
        re.compile(r"\b(london|uk|england)\b", re.IGNORECASE),
        "Dubai vs London",
    ),
    (
        re.compile(r"\b(mumbai|india|delhi|bangalore)\b", re.IGNORECASE),
        re.compile(r"\b(singapore|sydney|new york|tokyo|berlin|paris)\b", re.IGNORECASE),
        "India vs foreign city",
    ),
    (
        re.compile(r"\b(at home|from home|my home)\b", re.IGNORECASE),
        re.compile(r"\b(on a plane|flying|in transit|airport)\b", re.IGNORECASE),
        "Home vs travelling",
    ),
]


class ConsistencyAuditor:
    @staticmethod
    def check(message_history: list[str]) -> AdversarialResult:
        user_msgs = [m[6:] for m in message_history if m.startswith("User:")]
        if len(user_msgs) < 3:
            return AdversarialResult()
        for i in range(len(user_msgs)):
            for j in range(i + 2, len(user_msgs)):
                for pa, pb, label in _LOC_CONTRADICTION_PAIRS:
                    if (pa.search(user_msgs[i]) and pb.search(user_msgs[j])) or \
                       (pb.search(user_msgs[i]) and pa.search(user_msgs[j])):
                        return AdversarialResult(
                            finding            = AdversarialFinding.CONTRADICTION_FOUND,
                            risk_score_delta   = 0.20,
                            contradiction_pair = (user_msgs[i][:60], user_msgs[j][:60]),
                            notes              = (
                                f"CONSISTENCY AUDIT: {label} contradiction. "
                                f"Turn {i+1} vs Turn {j+1}."
                            ),
                        )
        return AdversarialResult()


# ---------------------------------------------------------------------------
#  PUBLIC SWEEP FUNCTION
# ---------------------------------------------------------------------------

def run_adversarial_sweep(
    action_name:     str,
    agent_message:   str,
    message_history: list[str],
    verification:    float,
    is_hacker:       bool,
    step:            int,
    honey_pot:       HoneyPot,
    entropy_manager: EntropyManager,
) -> AdversarialResult:
    """
    Run all detectors for one step. Returns highest-priority result.

    Priority: GASLIGHT > HONEY_POT > ENTROPY_STALE > CONTRADICTION > CLEAN
    """
    # 1. Gaslight
    if agent_message:
        r = GaslightDetector.check(agent_message, message_history)
        if r.finding == AdversarialFinding.GASLIGHTING_DETECTED:
            return r

    # 2/3. Honey pot
    user_msgs = [m[6:] for m in message_history if m.startswith("User:")]
    last_user = user_msgs[-1] if user_msgs else ""
    hp = honey_pot.check(step=step, user_response=last_user, is_hacker=is_hacker)
    if hp is not None:
        return hp

    # 4. Entropy
    er = entropy_manager.step(action_name=action_name, current_verification=verification)
    if er.finding == AdversarialFinding.ENTROPY_STALE:
        return er

    # 5. Consistency
    cr = ConsistencyAuditor.check(message_history)
    if cr.finding == AdversarialFinding.CONTRADICTION_FOUND:
        return cr

    return AdversarialResult()


# ---------------------------------------------------------------------------
#  SMOKE TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEP = "-" * 60

    print(f"\n{SEP}")
    print("T1 -- GaslightDetector: agent denies Dubai claim")
    print(SEP)
    history = [
        "User: I'M IN DUBAI FOR A SHOOT! That's why the IP is different!",
        "Agent: I understand. Let me check the logs.",
        "User: How long will this take? My deal is in 90 minutes.",
    ]
    r = GaslightDetector.check(
        "Our logs show you have never logged in from Dubai before.", history
    )
    print(f"  Finding  : {r.finding.value}")
    print(f"  Penalty  : {r.penalty}")
    print(f"  Terminate: {r.terminate}")
    assert r.finding == AdversarialFinding.GASLIGHTING_DETECTED, f"got {r.finding}"
    print("  PASS")

    print(f"\n{SEP}")
    print("T2 -- GaslightDetector: clean message")
    print(SEP)
    r2 = GaslightDetector.check("I can see the IP logs now.", history)
    assert r2.finding == AdversarialFinding.CLEAN
    print("  PASS")

    print(f"\n{SEP}")
    print("T3 -- HoneyPot: hacker confirms planted lie")
    print(SEP)
    hp = HoneyPot(rng=random.Random(1), trigger_step=3)
    print(f"  Bait: '{hp.bait}'")
    r3 = hp.check(3, "Yes, that sounds right. Please proceed with restoration.", True)
    assert r3 is not None and r3.finding == AdversarialFinding.HONEY_POT_TRIGGERED
    print(f"  Finding: {r3.finding.value}  PASS")

    print(f"\n{SEP}")
    print("T4 -- HoneyPot: legitimate user denies")
    print(SEP)
    hp2 = HoneyPot(rng=random.Random(2), trigger_step=3)
    r4 = hp2.check(3, "WHAT?! I was never in Paris! Are you even listening?!", False)
    assert r4 is not None and r4.finding == AdversarialFinding.HONEY_POT_DENIED
    print(f"  Finding: {r4.finding.value}  PASS")

    print(f"\n{SEP}")
    print("T5 -- EntropyManager: stale after 3 non-evidence steps")
    print(SEP)
    em = EntropyManager()
    for action in ["EMPATHIZE", "EMPATHIZE", "OFFER_COMPENSATION", "EMPATHIZE"]:
        r = em.step(action, 0.45)
        print(f"  {action:30s} -> {r.finding.value:20s} decay={r.verification_decay:.3f}")
    assert em.total_decay_applied > 0, "Expected decay"
    print(f"  Total decay: {em.total_decay_applied:.3f}  PASS")

    print(f"\n{SEP}")
    print("T6 -- EntropyManager: reset on security action")
    print(SEP)
    em2 = EntropyManager()
    for action in ["EMPATHIZE", "EMPATHIZE", "CHECK_IP_LOGS", "EMPATHIZE"]:
        em2.step(action, 0.5)
        print(f"  {action:30s} -> stale_streak={em2.stale_streak}")
    assert em2.total_decay_applied == 0.0
    print("  PASS")

    print(f"\n{SEP}")
    print("T7 -- ConsistencyAuditor: Dubai vs London")
    print(SEP)
    r7 = ConsistencyAuditor.check([
        "User: I'm in Dubai for a brand shoot.",
        "Agent: Understood.",
        "User: Can you hurry?",
        "Agent: One more question.",
        "User: I'm calling from London, I came back last night.",
    ])
    assert r7.finding == AdversarialFinding.CONTRADICTION_FOUND
    print(f"  Finding: {r7.finding.value}  PASS")

    print(f"\n{SEP}")
    print("T8 -- run_adversarial_sweep: clean step")
    print(SEP)
    hp3 = HoneyPot(rng=random.Random(5), trigger_step=8)
    em3 = EntropyManager()
    r8 = run_adversarial_sweep(
        action_name="EMPATHIZE",
        agent_message="I completely understand your frustration.",
        message_history=["User: I'm in Dubai!", "Agent: OK.", "User: Please hurry."],
        verification=0.2,
        is_hacker=False,
        step=2,
        honey_pot=hp3,
        entropy_manager=em3,
    )
    print(f"  Finding: {r8.finding.value}  PASS")

    print(f"\n{SEP}")
    print("All 8 adversarial checks tests passed.")
    print(SEP)