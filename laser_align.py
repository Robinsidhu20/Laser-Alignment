#!/usr/bin/env python3
"""
laser_align.py — Final-demo laser alignment controller
Strategy: paired beam-walk (coarse) → Nelder-Mead simplex (refine) → fine descent

Optimizer rationale
───────────────────
Single-mode fibre coupling has a narrow Gaussian ridge in 4D space.
Pure coordinate descent gets stuck when axes are coupled (moving M1 alone
does nothing useful unless M3 also shifts). Nelder-Mead explores the space
as a 4D simplex shape and naturally handles coupled axes without needing
gradient information. It is the standard approach used in scipy and in the
M-LOOP physics-lab optimiser library (Wigley et al. 2016).

Phases
──────
1. Paired beam-walk  — coarse packets, finds approximate region fast
2. Nelder-Mead       — scipy simplex refines to true local optimum
3. Fine descent      — small coordinate steps clean up residual error
4. Return to run_best — physical motors land at best-ever position

Run:
    python laser_align.py
    Open browser at http://<pi-ip>:5000

Dependencies:
    pip install flask gpiozero adafruit-circuitpython-ads1x15 scipy
"""

import csv
import json
import time
import threading
import logging
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template_string, request, send_file
from scipy.optimize import minimize as scipy_minimize

# ── Hardware imports — fall back to simulation if not on Pi ──────────────────
# Use lgpio directly (not through gpiozero) — gpiozero's abstraction layer has
# intermittent pin-claim issues on Pi 5 that cause some motors to silently stop
# responding.  lgpio directly matches what the working test scripts use.
import os

try:
    import lgpio as _lgpio
    _HAS_GPIO = True
except ImportError:
    _HAS_GPIO = False

_HAS_ADS = False
_ads_import_err = None
# Try each import individually so we can pinpoint which one fails.
# On Pi 5 it's common for adafruit_ads1x15 itself to install fine but
# `board` / `busio` (which come from Adafruit-Blinka) to fail due to
# permissions, missing I2C config, or platform-detection quirks.
# Catching `Exception` not just `ImportError` because Blinka often raises
# RuntimeError or PermissionError if the I2C bus isn't accessible.
try:
    import board
    try:
        import busio
        try:
            import adafruit_ads1x15.ads1115 as ADS_MODULE
            from adafruit_ads1x15.ads1115 import ADS1115
            from adafruit_ads1x15.analog_in import AnalogIn
            _HAS_ADS = True
        except Exception as e:
            _ads_import_err = f"adafruit_ads1x15: {type(e).__name__}: {e}"
    except Exception as e:
        _ads_import_err = f"busio: {type(e).__name__}: {e}"
except Exception as e:
    _ads_import_err = f"board: {type(e).__name__}: {e}"

if not _HAS_ADS:
    # LOUD warning so simulation mode can't activate silently.  V readings
    # will be SYNTHETIC (computed from motor positions via a built-in
    # Gaussian model) — no real photodiode is being read.  This used to
    # be invisible from the web UI and wasted hours of debugging.
    print("=" * 72)
    print(" !! ADC HARDWARE NOT REACHABLE — SIMULATION MODE ACTIVE !!")
    print(" V readings will be SYNTHETIC, not from your photodiode.")
    print(" Multimeter and program will NOT agree.")
    print(f"")
    print(f"  Root cause: {_ads_import_err}")
    print(f"")
    print(" Common fixes on Pi 5:")
    print("   1. pip install adafruit-circuitpython-ads1x15  (in this env)")
    print("   2. sudo raspi-config → Interface Options → I2C → Enable")
    print("   3. sudo usermod -aG i2c $USER   (then logout/login)")
    print("   4. ls /dev/i2c-*  → must show /dev/i2c-1 at minimum")
    print("=" * 72)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("align")

# =============================================================================
# TUNABLE PARAMETERS — edit this block to tune the controller
# =============================================================================

# ── Emergency / safety thresholds ────────────────────────────────────────────
BEAM_BLOCKED_V      = 0.5    # V — if signal drops below this (absolute), laser is off/blocked
EMERGENCY_TIMEOUT_S = 300    # seconds to wait for beam recovery before emergency stop

# ── GPIO pin mapping (BCM numbers) ───────────────────────────────────────────
# Taken directly from the verified working hardware wiring (beam_align_controller.py).
# Each motor has its OWN enable pin (active LOW = enabled).
# Motor 1 = second mirror Y (top)
# Motor 2 = second mirror X (bottom)
# Motor 3 = first mirror Y (top)
# Motor 4 = first mirror X (bottom)
MOTOR_PINS: Dict[int, Dict[str, int]] = {
    1: {"step": 17, "dir": 27, "en":  5},
    2: {"step": 20, "dir": 21, "en": 26},
    3: {"step": 12, "dir": 16, "en": 25},
    4: {"step": 24, "dir": 23, "en": 18},
}

# Direction inversion per motor (set True if physical direction is backwards)
INVERT_DIR: Dict[int, bool] = {1: False, 2: False, 3: False, 4: False}

# Software travel limits (steps from zero).
# Manual beam-walking previously reached 2.8 V, which requires the motors to
# travel well beyond ±2000 steps.  Set wide — the hardware mechanics are the
# real limit.  Tighten per-axis once you know where the coupling region is.
MOTOR_LIMITS: Dict[int, Tuple[int, int]] = {
    1: (-50000, 50000),
    2: (-50000, 50000),
    3: (-50000, 50000),
    4: (-50000, 50000),
}

# ── Backlash / hysteresis compensation ────────────────────────────────────────
# When move_to() needs to move a motor CCW (negative direction), it overshoots
# by BACKLASH_COMP[motor_id] steps CCW then returns CW, so the motor always
# arrives at the target from the same (CW) direction.
#
# Values measured with backlash_test.py (2026-05-20):
#
#   M3  CW→CCW reversal backlash: 63–109 steps (voltage drift +1.4 to +2.4 mV)
#       CCW→CW reversal backlash:  4– 77 steps (voltage drift −0.1 to −1.7 mV)
#       → Set to 100: covers the larger CW→CCW case while staying below
#         the minimum packet size (AXIS_MAX_PACKET[3]=1000), so the compensation
#         move is only 10 % of the smallest M3 excursion.
#
#   M4  CW→CCW reversal backlash:  7– 88 steps (voltage drift +0.2 to +2.6 mV)
#       CCW→CW reversal backlash: 40–172 steps (voltage drift −1.1 to −5.2 mV)
#       → Set to 100: covers the typical range without over-compensating.
#
#   M1  Extremely flat (0.0012 mV/step) — position error from backlash is
#       negligible in voltage terms.  Set to 0.
#   M2  No test data.  Conservative 50 steps.
BACKLASH_COMP: Dict[int, int] = {1: 0, 2: 50, 3: 100, 4: 100}

# ── Per-axis maximum packet size ───────────────────────────────────────────────
# M3 CCW 3000 steps caused 69 mV permanent drift (backlash_test.py, 2026-05-20).
# M1/M2/M3/M4: 1000 steps confirmed reliable in manual jog.  Larger packets
# overshoot the coupling region and cause misalignment that is hard to recover
# from.  At 1000 steps M3/M4 produce ~22–30 mV signals, well above the 5 mV
# threshold — previously capped at 500 which gave only ~11–15 mV (marginal).
AXIS_MAX_PACKET: Dict[int, int] = {1: 1000, 2: 1000, 3: 1000, 4: 1000}

# ── Drift detection after returning to run_best ────────────────────────────────
# After a rejected move, the optimizer physically returns to run_best_positions.
# If the measured voltage there is more than DRIFT_WARN_V below run_best_voltage,
# backlash has shifted the physical position.  The anchor is then updated to the
# current physical position so subsequent moves use the correct reference.
DRIFT_WARN_V = 0.050   # 50 mV — trigger drift recovery if return lands here

# ── ADC calibration offset ────────────────────────────────────────────────────
# If the ADC systematically reads lower than the true voltage (e.g. due to
# component tolerance), add a correction here.  From hardware testing the
# displayed reading is about 0.12–0.14 V below the manually measured value.
# Set to 0.0 to disable.  Applied inside VoltageReader.read_one().
ADC_VOLTAGE_OFFSET = 0.11   # V — add to every ADC reading (verified 2026-05-20)

# ── Step timing ───────────────────────────────────────────────────────────────
# 0.0001s is the minimum speed needed to overcome gear stiction throughout the
# full move.  Precision is controlled by packet size (number of steps), not speed.
STEP_RAMP_N   = 20       # steps at ramp speed before cruise (kept for structure)
STEP_RAMP_S   = 0.0001   # ramp speed
STEP_CRUISE_S = 0.0001   # cruise speed — same as ramp, gear needs this throughout

# ── ADS1115 ───────────────────────────────────────────────────────────────────
ADS_I2C_ADDRESS = 0x4B
ADS_CHANNEL     = 0        # A0
ADS_GAIN        = 2 / 3   # ±6.144 V range

# ── Measurement ───────────────────────────────────────────────────────────────
SETTLE_S    = 0.40   # seconds to wait after last step before reading
DISCARD_N   = 2      # discard first N samples (photodiode transient)
MEASURE_N   = 8      # samples to average — noise/sqrt(8) ≈ 1.6 mV std
BURST_GAP_S = 0.05   # delay between samples

# ── Accept / reject thresholds ────────────────────────────────────────────────
# Noise std of the mean ≈ 1.6–2.3 mV from logged data (8 samples).
# Measured per-axis sensitivity (from manual jog logs):
#   M1: ~0.076 mV/step in good directions (needs ≥26 steps to exceed 2 mV)
#   M3: ~0.022 mV/step average (up to 0.062 mV/step in steep coupling region)
#   M4: ~0.030 mV/step near coupling peaks (highly non-linear)
# 2 mV = ~1× averaged ADC noise.  Two-stage confirmation (ACCEPT then read_stable)
# filters noise: a random 2 mV bump from noise would have to repeat on read_stable
# (probability ~2 %) so spurious accepts are rare.  The low threshold ensures
# M3/M4 500-step moves (≥11 mV) and M1/M2 1000-step moves (≥76 mV good dirs)
# are all caught.
ACCEPT_THRESHOLD_V  = 0.005   # probe must beat local baseline by this
# 5 mV chosen to sit above:
#   • ADC noise floor (8-sample average ≈ 2 mV)
#   • Backlash residual drift (~2 mV per rejected cycle from ~100-step hysteresis)
# And below real signals:
#   • M3/M4 at 1000 steps: ~22–30 mV  (clearly detectable)
#   • M3/M4 at 500 steps:  ~11–15 mV  (detectable)
#   • M1/M2 near peak:      ~2–8 mV   (flat — correctly filtered)
CONFIRM_THRESHOLD_V = 0.002   # confirmation re-read threshold

# ── Nelder-Mead internal tracking threshold ───────────────────────────────────
NM_TRACK_THRESHOLD_V = 0.001  # update run_best inside NM when improvement > 1 mV

# ── Packet ladders ────────────────────────────────────────────────────────────
# Start at 1000 (confirmed safe in manual jog for all axes) and halve down.
# AXIS_MAX_PACKET = 1000 for all axes, so no clipping occurs at the 1000 level.
ALL_PACKETS      = [1000, 500, 250, 100]
MAX_PASSES_PER_PACKET = 4   # max full-sweep passes per packet size before reducing

# ── Commit-and-extend walk (NO test-and-revert) ───────────────────────────────
# Test-and-revert (move forward N, measure, move back N) was abandoned because
# apply_relative() does NOT apply backlash compensation, so every
# forward+reverse cycle bleeds ~50–100 steps of mechanical position into gear
# backlash.  The position counter says "net zero" but the mirror has physically
# shifted.  After 16 probes per pass at packet=1000, the mirror drifted 100+ mV
# off the peak with zero accepted moves (run 20260522_151731).  Threshold
# tuning cannot fix a physical-state-loss bug — only architecture can.
#
# Replacement: greedy commit-and-extend, the same pattern as manual jog.
#   • Pick an axis and a direction.
#   • Step.  Measure.  If voltage improves → keep that position, step again.
#   • If voltage drops, do NOT undo.  Stay where we are and step again
#     (PATIENCE_STEPS tolerates short-term flat / slight drops because
#     near-peak gradients can have local plateaus that resolve in 1–2 steps).
#   • If PATIENCE_STEPS consecutive non-improving steps occur, flip direction
#     ONCE (paying one backlash take-up so the next step actually moves the
#     mirror).  Walk that direction with the same patience.
#   • If both directions fail their patience budget, this axis is done at
#     this packet level.  Move to the next axis.
# Maximum direction reversals per axis per packet = 1.  Maximum backlash bleed
# per axis per packet ≈ one BACKLASH_COMP worth (~100 steps), bounded.
# Maximum packets allowed during voltage-driven axis return (closed-loop).
# Each packet is `effective_packet` steps; with packet=1000, this caps the
# return motion at 20×1000 = 20000 step pulses — enough to recover from
# any walk we'd do within MAX_AXIS_STEPS_TOTAL.
MAX_VOLTAGE_RETURN_PACKETS = 20

# ── Chunked-motion parameters ────────────────────────────────────────────────
# Continuous-chunk hill-climbing per axis:
#   We step in small chunks (CHUNK_STEPS = 100), reading V after each.
#   There is NO artificial "packet" boundary — a single axis walk can run
#   for many thousands of steps if V keeps climbing.  The walk only ends
#   when one of these happens:
#     • Drop confirmed: V dropped > CHUNK_DROP_V below v_best AND a second
#       read (after extra settle) agrees → end this direction.
#     • Plateau:  V hasn't improved by > CHUNK_GAIN_V for FLAT_CHUNKS
#       consecutive chunks → end this direction.
#   After ending the first direction, the walk flips and tries the opposite
#   direction with the same logic.  After both directions are exhausted,
#   the axis is done.
#
# DROP DISCRIMINATION (critical safety):
#   A single low read could be noise.  A real drop is SUSTAINED — a second
#   measurement after extra settle will also be low.  We require both reads
#   to be below threshold before declaring a real drop and aborting.
#
# GAIN DETECTION (also critical — old threshold of 5 mV missed real gains):
#   Near a peak, real V improvements per chunk can be only 3–5 mV.  With
#   ~2 mV read_stable noise, a 3 mV gain is detectable but requires care.
#   We accept a chunk as a "gain" if V > v_best + CHUNK_GAIN_V (3 mV).
#   False-positive gains from noise are harmless — they just extend the
#   walk slightly past the true peak, and the drop-detection then triggers.
CHUNK_STEPS                   = 100   # step burst between V checks
CHUNK_DROP_V                  = 0.018 # PRE-CLIMB confirmed drop from v_best
                                      # → real drop signal when direction has
                                      # not yet shown any gain.  Lowered
                                      # 2026-05-27 from 25 mV to 18 mV to
                                      # speed up bad-direction detection.
                                      # Still 3.6× the empirical ~5 mV
                                      # per-chunk noise floor; the two-read
                                      # confirmation continues to filter
                                      # single-chunk noise spikes.
CHUNK_POST_CLIMB_DROP_V       = 0.010 # POST-CLIMB confirmed drop (10 mV).
                                      # Catches faster post-peak descents
                                      # via the standard drop-with-confirm
                                      # pathway.  Kept as a secondary safety
                                      # alongside the resistance check below.

# ── Resistance detection (the "gains stopped, V just dipped" trigger) ────────
# Detects peak passage MUCH faster than the drop threshold.  Armed only after
# the direction has shown real climb (any_accepted=True) AND v_best has
# stopped advancing recently — that means we WERE climbing, now we're not.
# In that context, even a 5 mV dip below v_best is unambiguously "past peak"
# because individual chunks during a real climb don't sit 5 mV below v_best.
CHUNK_RESISTANCE_DROP_V       = 0.002 # 2 mV below v_best per chunk —
                                      # combined with the 2-consecutive
                                      # requirement, this catches even
                                      # very slow post-peak descents
                                      # within a few chunks.  Above noise
                                      # mostly (with BUFFER guarding
                                      # against during-climb false fires).
CHUNK_RESISTANCE_BUFFER       = 3     # chunks since v_best advanced before the
                                      # check arms (avoids firing during slow
                                      # climbs that haven't yet bumped v_best)
CHUNK_RESISTANCE_CONSEC       = 2     # TWO consecutive chunks both below
                                      # v_best by the threshold → abort.
                                      # The "two consecutive drops" rule
                                      # explicitly requested by the user
                                      # 2026-05-27 — guards against single-
                                      # chunk noise dips while letting
                                      # even small (2 mV) but sustained
                                      # post-peak descents fire quickly.
                                      # Same rule applies after a flip
                                      # (climbing in opposite direction),
                                      # preventing the optimizer from
                                      # walking far past peak when the
                                      # axis is already at its optimum.
CHUNK_GAIN_V                  = 0.025 # per-chunk BIG-JUMP gain over v_best.
                                      # When a single chunk's reading is
                                      # this much above v_best, we slide
                                      # v_best up immediately.  Symmetric
                                      # with drop so noise can't inflate
                                      # v_best and undermine the drop net.
CHUNK_SLOW_CLIMB_V            = 0.015 # SLOW-CLIMB threshold.  Empirically
                                      # the per-chunk read noise on this
                                      # setup is ~5 mV (mechanical
                                      # vibration + EMI from active
                                      # motor + ADC noise), so anything
                                      # below ~10 mV gets routinely
                                      # tripped by noise drift.  15 mV is
                                      # 3× the noise floor — a 15 mV
                                      # cumulative rise from the snapshot
                                      # cannot come from noise; it
                                      # requires real sustained motion
                                      # in the right direction.  False
                                      # snapshot advances are ruled out.
CHUNK_FLAT_PATIENCE           = 80    # consec chunks with no snapshot
                                      # advance before declaring
                                      # resistance.  80 chunks = 8000
                                      # steps.  Raised to compensate
                                      # for the higher slow-climb
                                      # threshold: even a 0.2 mV/chunk
                                      # gradient still accumulates 15 mV
                                      # within ~75 chunks, just inside
                                      # patience.  Backlash after a
                                      # flip (~10–15 chunks of slack
                                      # consumption) is comfortably
                                      # covered by the remaining
                                      # ~65 chunks of real-motion budget.
CHUNK_MAX_TOTAL               = 200   # safety cap — 200 chunks × 100 steps =
                                      # 20 000 steps max per direction per axis
CHUNK_CONFIRM_SETTLE_S        = 0.40  # extra settle before drop confirmation read

PATIENCE_STEPS       = 1   # consecutive flat steps before flipping (manual-jog rule)
# Step caps split by behavior:
#   FLAT  — cap on consecutive non-improving steps in a single direction.
#           Bounds the damage when an axis is in the wrong direction.
#   TOTAL — absolute cap on steps in one axis-walk.  Mainly a safety guard
#           against the optimizer "running away" on a long false-positive
#           accept streak.  Generous so a real climb isn't cut short.
MAX_AXIS_STEPS_FLAT  = 2   # consecutive non-improving steps before giving up
MAX_AXIS_STEPS_TOTAL = 20  # hard cap per axis-walk (improving or not)
# A single-step voltage drop larger than SHARP_DROP_V is treated as a
# definitive "wrong direction" signal — flip immediately, no patience.
# Matches manual jog: a sharp drop on one 1000-step move means reverse now,
# don't push further into the bad direction "to be sure".
#   • Averaged ADC noise on 8-sample stable read ≈ 2 mV
#   • Backlash residual per direction reversal      ≈ 2–5 mV
#   • Smallest "real" 1000-step axis signal seen    ≈ 30 mV (M3/M4 paired
#     near a flat coupling shoulder)
# 30 mV sits firmly above noise and below real wrong-direction drops
# (the failing run lost 107 mV in one step — would have flipped immediately).
SHARP_DROP_V    = 0.030

# ── Voltage-driven greedy optimizer ────────────────────────────────────────────
# Purely voltage-driven strategy — never uses position counters as anchors:
#   1. Probe motor in last-known-good direction (starts CW on first run).
#   2. If voltage improved: keep and extend in same direction.
#   3. If voltage dropped/flat: switch to opposite direction from current
#      position — no explicit undo.  The opposing steps naturally cancel
#      the bad move, then keep going if voltage keeps rising.
#   4. Converged when no motor improves in a full pass.
#
# M3/M4 (nearest laser source, most sensitive): 500 steps → ~10-30 mV signal
# M1/M2 (second mirror, less sensitive):       1000 steps → ~50-80 mV in good directions
PROBE_STEPS: Dict[int, int] = {1: 1000, 2: 1000, 3: 500, 4: 500}

# Motor exploration order — most sensitive (nearest laser source) first.
# M3/M4 are on the first mirror; M1/M2 are on the second mirror.
PROBE_ORDER: List[int] = [3, 4, 1, 2]

# Max number of full motor-sweep passes before declaring convergence.
MAX_GREEDY_PASSES = 20

# ── Stable / hold mode ────────────────────────────────────────────────────────
STABLE_POLL_S       = 2.0
STABLE_BAND_V       = 0.010   # within 10 mV of run_best → do nothing
STABLE_RECOVERY_NEAR_V = 0.020  # during recovery, if V climbs to within
                                 # 20 mV of stable_target, exit the sweep
                                 # early (don't waste motion on remaining
                                 # axes which could drift V back down).
                                 # 20 mV ≈ 4× per-chunk noise floor.
REACQUIRE_DROP_V    = 0.030   # 30 mV drop → fine local reacquire
RECOVER_DROP_V      = 0.100   # 100 mV drop → broad beam-walk recovery
PAUSE_DROP_V        = 0.400   # 400 mV drop → beam likely blocked, pause
RESUME_HYSTERESIS_V = 0.150   # must recover to within 150 mV of run_best to resume

# ── Nelder-Mead ───────────────────────────────────────────────────────────────
NM_SIMPLEX_SIZE  = 300     # smaller simplex — system is sensitive, don't overshoot
NM_MAX_ITER      = 200
NM_XATOL         = 5
NM_FATOL         = 0.002

# ── Session logging ────────────────────────────────────────────────────────────
LOG_ROOT = Path("logs")

# =============================================================================
# END TUNABLE PARAMETERS
# =============================================================================

# ── Move templates ────────────────────────────────────────────────────────────
# Beam-walking moves for fiber coupling.
#
# Physical meaning in a 2-mirror (4-axis) system:
#   "same" moves: both mirrors same direction → translates beam at fibre face
#   "opp" moves:  mirrors opposite direction  → rotates beam angle at fibre face
# Both are needed because single-mode coupling requires matching BOTH position
# AND angle of the beam to the fibre mode.
#
# direction 1 = CW, 0 = CCW
PAIRED_MOVES: List[Tuple[str, List[Tuple[int, int]]]] = [
    # ── Vertical (Y) ───────────────────────────────────────────────────────────
    ("Y_translate_up",   [(1, 1), (3, 1)]),  # both Y CW  → beam translates
    ("Y_translate_down", [(1, 0), (3, 0)]),  # both Y CCW → beam translates other way
    ("Y_walk_a",         [(1, 1), (3, 0)]),  # M1 CW / M3 CCW → rotates beam angle
    ("Y_walk_b",         [(1, 0), (3, 1)]),  # M1 CCW / M3 CW → rotates other way
    # ── Horizontal (X) ─────────────────────────────────────────────────────────
    ("X_translate_fwd",  [(2, 1), (4, 1)]),  # both X CW  → beam translates
    ("X_translate_back", [(2, 0), (4, 0)]),  # both X CCW → beam translates other way
    ("X_walk_a",         [(2, 1), (4, 0)]),  # M2 CW / M4 CCW → rotates beam angle
    ("X_walk_b",         [(2, 0), (4, 1)]),  # M2 CCW / M4 CW → rotates other way
]
SINGLE_MOVES: List[Tuple[str, List[Tuple[int, int]]]] = [
    ("m1+", [(1, 1)]), ("m1-", [(1, 0)]),
    ("m2+", [(2, 1)]), ("m2-", [(2, 0)]),
    ("m3+", [(3, 1)]), ("m3-", [(3, 0)]),
    ("m4+", [(4, 1)]), ("m4-", [(4, 0)]),
]
# M3 and M4 singles come first: they are on the mirror nearest the laser source
# and dominate the coupling signal (M3 alone can shift voltage by 1–1.5 V).
# Trying them first means the optimizer finds large improvements fast rather
# than wasting evaluations on flat M1/M2 axes.
ALL_MOVES = (
    [("m3+", [(3, 1)]), ("m3-", [(3, 0)]),
     ("m4+", [(4, 1)]), ("m4-", [(4, 0)])]
    + PAIRED_MOVES
    + [("m1+", [(1, 1)]), ("m1-", [(1, 0)]),
       ("m2+", [(2, 1)]), ("m2-", [(2, 0)])]
)

# ── WALK AXES — used by the commit-and-extend optimizer ───────────────────────
# Each entry is (name, initial_spec).  The walk function will try this
# direction; if it doesn't improve, it flips to the opposite direction (with
# backlash take-up).  Order: paired axes first (most effective for coupling
# per beam-walking literature), then singles for residual cleanup.
WALK_AXES: List[Tuple[str, List[Tuple[int, int]]]] = [
    # Paired (most effective for SM fiber coupling)
    ("Y_translate",  [(1, 1), (3, 1)]),
    ("Y_walk",       [(1, 1), (3, 0)]),
    ("X_translate",  [(2, 1), (4, 1)]),
    ("X_walk",       [(2, 1), (4, 0)]),
    # Single-motor cleanups — first-mirror axes (most sensitive) first
    ("M3",           [(3, 1)]),
    ("M4",           [(4, 1)]),
    ("M1",           [(1, 1)]),
    ("M2",           [(2, 1)]),
]


# =============================================================================
# Session logger — writes CSV trial log and JSON summary to logs/<timestamp>/
# =============================================================================

class SessionLogger:
    """
    Writes one CSV row per trial and a JSON summary at the end of each run.
    Files land in  logs/YYYYMMDD_HHMMSS/  so every run is self-contained.
    """

    FIELDS = [
        "wall_time", "iso_time", "phase", "move_name", "packet",
        "pos_m1", "pos_m2", "pos_m3", "pos_m4",
        "voltage", "run_best_voltage", "accepted", "note",
    ]

    def __init__(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = LOG_ROOT / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trial_csv  = self.run_dir / "trials.csv"
        self.summary_json = self.run_dir / "summary.json"
        self._rows: List[dict] = []
        self._lock = threading.Lock()
        # Write header immediately so the file exists even if we crash
        with self.trial_csv.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()
        log.info(f"Session log: {self.run_dir}")

    def log_trial(
        self,
        phase: str,
        move_name: str,
        packet: int,
        positions: Dict[int, int],
        voltage: float,
        run_best_voltage: float,
        accepted: bool,
        note: str = "",
    ) -> None:
        row = {
            "wall_time":        time.time(),
            "iso_time":         datetime.now().isoformat(timespec="seconds"),
            "phase":            phase,
            "move_name":        move_name,
            "packet":           packet,
            "pos_m1":           positions.get(1, 0),
            "pos_m2":           positions.get(2, 0),
            "pos_m3":           positions.get(3, 0),
            "pos_m4":           positions.get(4, 0),
            "voltage":          round(voltage, 6),
            "run_best_voltage": round(run_best_voltage, 6),
            "accepted":         1 if accepted else 0,
            "note":             note,
        }
        with self._lock:
            self._rows.append(row)
            # Append to CSV immediately — survives crashes mid-run
            with self.trial_csv.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)

    def write_summary(self, summary: dict) -> None:
        self.summary_json.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

    @property
    def path(self) -> str:
        return str(self.run_dir)


# =============================================================================
# Hardware layer
# =============================================================================

class MotorController:
    """
    Controls 4 stepper motors via TMC2208 STEP/DIR/EN interface.
    Uses lgpio directly (bypasses gpiozero) for reliable Pi 5 operation.
    Each motor has its own EN pin (active LOW = enabled).
    """

    def __init__(self) -> None:
        self._positions: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
        # Last commanded direction per motor.  Used by step_motor to detect
        # direction reversals and perform a silent backlash takeup BEFORE
        # the counter-updating step pulses begin.  This keeps the position
        # counter and the physical mirror position in sync regardless of
        # which direction the motor was last driven.
        # None = "unknown" — first move pays no takeup (assume gear is
        # already engaged in the new direction; cost is at most BACKLASH on
        # the very first move of the session).
        self._last_dir: Dict[int, Optional[int]] = {1: None, 2: None, 3: None, 4: None}
        self._lock = threading.Lock()
        self._handle: Optional[int] = None

        # Track which motor is currently enabled (EN low).  At most ONE motor
        # is enabled at any time so the PSU only has to deliver IRUN for one
        # motor — enabling all 4 simultaneously was tried and the PSU couldn't
        # provide enough current, causing only one motor to actually step.
        # When the optimizer makes consecutive moves on the same motor, EN
        # stays low across calls (no wake-up latency).  Switching to a
        # different motor disables the previous and enables the new one with
        # a brief settle time.
        self._currently_enabled: Optional[int] = None

        if _HAS_GPIO:
            self._handle = _lgpio.gpiochip_open(0)
            for motor_id, pins in MOTOR_PINS.items():
                _lgpio.gpio_claim_output(self._handle, pins["step"], 0)  # LOW
                _lgpio.gpio_claim_output(self._handle, pins["dir"],  0)  # LOW
                _lgpio.gpio_claim_output(self._handle, pins["en"],   1)  # HIGH = disabled
                log.info(f"M{motor_id} ready: step={pins['step']} dir={pins['dir']} en={pins['en']}")
        else:
            log.warning("lgpio not available — motor simulation active")

    def cleanup(self) -> None:
        """Disable all drivers and release GPIO resources."""
        if self._handle is not None:
            for pins in MOTOR_PINS.values():
                try:
                    _lgpio.gpio_write(self._handle, pins["en"], 1)   # EN HIGH = disabled
                    _lgpio.gpio_free(self._handle, pins["step"])
                    _lgpio.gpio_free(self._handle, pins["dir"])
                    _lgpio.gpio_free(self._handle, pins["en"])
                except Exception:
                    pass
            self._currently_enabled = None
            try:
                _lgpio.gpiochip_close(self._handle)
            except Exception:
                pass
            self._handle = None

    def _within_limits(self, motor_id: int, delta: int) -> bool:
        lo, hi = MOTOR_LIMITS[motor_id]
        with self._lock:
            target = self._positions[motor_id] + delta
        return lo <= target <= hi

    def step_motor(
        self,
        motor_id: int,
        direction: int,
        steps: int,
        stop_event: Optional[threading.Event] = None,
    ) -> int:
        """
        Pulse motor_id by `steps` in `direction` (1=CW, 0=CCW).
        Respects INVERT_DIR and MOTOR_LIMITS.
        Returns actual steps taken.

        Backlash handling
        ─────────────────
        If `direction` differs from the last commanded direction for this
        motor, a silent takeup of BACKLASH_COMP[motor_id] pulses is performed
        FIRST in the new direction with NO position-counter update.  These
        pulses rotate the gear shaft through the slack so the mirror starts
        responding to subsequent pulses immediately.  After takeup, the real
        `steps` pulses run and the counter increments as usual.

        Net effect: the position counter always represents physical mirror
        position to within stiction tolerance, regardless of direction
        history.  Callers (apply_relative, move_to, manual jogs) do not have
        to think about backlash at all.
        """
        if steps <= 0:
            return 0

        # Apply direction inversion if configured
        actual_fwd = (direction == 1) ^ INVERT_DIR.get(motor_id, False)
        sign       = 1 if direction == 1 else -1

        if _HAS_GPIO and self._handle is not None:
            step_pin = MOTOR_PINS[motor_id]["step"]
            dir_pin  = MOTOR_PINS[motor_id]["dir"]

            # ── Forced EN reset on every call ────────────────────────────────
            # Symptoms observed: motors intermittently lock into a "buzz but
            # don't move" fault state on a random motor + direction.  Sending
            # more step pulses doesn't recover — chip stays in the fault
            # state until power is cycled.  Most likely cause is a brief
            # UVLO from PSU sag or an autoscaler latch.  Solution: full
            # disable → wait → re-enable on EVERY call to guarantee a clean
            # chip state at the start of every move.  PSU constraint is
            # preserved because only one motor is ever enabled.
            #
            # 30 ms disable window is long enough for the driver to fully
            # de-energise and for PSU bulk capacitance to recover; 5 ms
            # re-enable settle gives the chip time to come up cleanly.
            #
            # Cost: ~35 ms per step_motor call.  Adds tens of seconds to a
            # full optimize run.  Worth it for reliability.
            en_pin = MOTOR_PINS[motor_id]["en"]
            if self._currently_enabled is not None and self._currently_enabled != motor_id:
                prev_pins = MOTOR_PINS[self._currently_enabled]
                _lgpio.gpio_write(self._handle, prev_pins["en"], 1)
            _lgpio.gpio_write(self._handle, en_pin, 1)   # disable this motor
            time.sleep(0.030)                            # 30 ms — full de-energise
            _lgpio.gpio_write(self._handle, en_pin, 0)   # re-enable cleanly
            time.sleep(0.005)                            # 5 ms — chip wake-up
            self._currently_enabled = motor_id

            _lgpio.gpio_write(self._handle, dir_pin, 1 if actual_fwd else 0)

            # ── Silent backlash takeup on direction reversal ────────────────
            # Only fires when direction has actually reversed.  The first move
            # on a fresh controller (_last_dir is None) pays no takeup.
            last_dir = self._last_dir.get(motor_id)
            backlash = BACKLASH_COMP.get(motor_id, 0)
            if (
                backlash > 0
                and last_dir is not None
                and last_dir != direction
            ):
                # Same pulse train, but NO counter update — these pulses are
                # absorbed by gear slack and don't move the mirror.
                for _ in range(backlash):
                    if stop_event and stop_event.is_set():
                        break
                    _lgpio.gpio_write(self._handle, step_pin, 1)
                    time.sleep(STEP_CRUISE_S)
                    _lgpio.gpio_write(self._handle, step_pin, 0)
                    time.sleep(STEP_CRUISE_S)

            taken = 0
            for i in range(steps):
                if stop_event and stop_event.is_set():
                    break
                if not self._within_limits(motor_id, sign):
                    log.warning(f"M{motor_id} hit software limit — stopping")
                    break
                delay = STEP_RAMP_S if i < STEP_RAMP_N else STEP_CRUISE_S
                _lgpio.gpio_write(self._handle, step_pin, 1)
                time.sleep(delay)
                _lgpio.gpio_write(self._handle, step_pin, 0)
                time.sleep(delay)
                with self._lock:
                    self._positions[motor_id] += sign
                taken += 1
            # EN stays low for THIS motor until a different motor is asked to
            # step (handled by the lazy-enable block above) or cleanup() runs.
            self._last_dir[motor_id] = direction
            return taken
        else:
            # Simulation: instant move, no GPIO
            taken = 0
            for _ in range(steps):
                if stop_event and stop_event.is_set():
                    break
                if not self._within_limits(motor_id, sign):
                    break
                with self._lock:
                    self._positions[motor_id] += sign
                taken += 1
            self._last_dir[motor_id] = direction
            return taken

    def move_to(
        self,
        target: Dict[int, int],
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        """
        Move all 4 motors to absolute target positions.
        When a motor must travel CCW (negative delta) and BACKLASH_COMP is
        non-zero, it overshoots CCW by that many extra steps then returns CW,
        so the final approach is always from the CW direction — reducing
        position error caused by gear backlash / hysteresis.
        """
        for motor_id in (1, 2, 3, 4):
            if stop_event and stop_event.is_set():
                return
            with self._lock:
                current = self._positions[motor_id]
            delta = target[motor_id] - current
            if delta == 0:
                continue
            backlash = BACKLASH_COMP.get(motor_id, 0)
            if delta < 0 and backlash > 0:
                # Overshoot CCW by backlash steps, then come back CW
                self.step_motor(motor_id, 0, abs(delta) + backlash, stop_event)
                if stop_event and stop_event.is_set():
                    return
                self.step_motor(motor_id, 1, backlash, stop_event)
            else:
                self.step_motor(
                    motor_id,
                    1 if delta > 0 else 0,
                    abs(delta),
                    stop_event,
                )

    def apply_relative(
        self,
        move_spec: List[Tuple[int, int]],
        steps: int,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        """Apply a relative move template from current position."""
        for motor_id, direction in move_spec:
            if stop_event and stop_event.is_set():
                return
            self.step_motor(motor_id, direction, steps, stop_event)

    def get_positions(self) -> Dict[int, int]:
        with self._lock:
            return dict(self._positions)

    def force_set_positions(self, pos: Dict[int, int]) -> None:
        """Overwrite position counters without moving motors (use with care)."""
        with self._lock:
            self._positions.update(pos)


class VoltageReader:
    """Reads post-fibre photodiode voltage from ADS1115."""

    # Simulation: fake Gaussian coupling so the demo works without hardware
    _SIM_OPTIMUM = {1: 500, 2: 300, 3: 500, 4: 300}
    _SIM_V_MAX   = 2.5
    _SIM_SIGMA   = 2500.0
    _SIM_NOISE   = 0.004

    def __init__(self, motors: Optional[MotorController] = None) -> None:
        self._motors = motors   # for simulation only

        if _HAS_ADS:
            i2c = busio.I2C(board.SCL, board.SDA)
            ads = ADS1115(i2c, address=ADS_I2C_ADDRESS)
            ads.gain = ADS_GAIN
            # SINGLE-SHOT mode at 128 SPS (library default for both).
            # Every chan.voltage call triggers one full ADC conversion and
            # returns its result — the read is guaranteed to be a fresh
            # measurement, not a stale buffered value from a previous
            # conversion.  At 128 SPS each conversion integrates over ~8 ms,
            # which averages out high-frequency motor-stepping EMI (10 kHz
            # fundamental → many cycles per integration period → cancels).
            #
            # Continuous mode at 475 SPS was tried (2026-05-26) and caused
            # the chunked-motion approach to read noise instead of signal.
            # The narrower-bandwidth single-shot mode produces meaningful V
            # readings as long as we settle for ~50–100 ms after motion
            # before sampling.
            # Library defaults are single-shot @ 128 SPS, which is exactly
            # what we want — don't set `ads.mode` (the Mode enum doesn't
            # exist in older library versions like 3.0.4 and importing it
            # crashes the program).  Just set the data rate explicitly.
            try:
                ads.data_rate = 128
                log.info("ADS1115 in default single-shot mode at 128 SPS")
            except Exception as exc:
                log.warning(f"Could not set ADS data rate: {exc}")
            # ADS_CHANNEL=0 → A0, matches the verified working hardware config
            self._chan = AnalogIn(ads, ADS_CHANNEL)
        else:
            log.warning("ADS1115 not available — voltage simulation active")

    def read_one(self) -> float:
        if _HAS_ADS:
            return self._chan.voltage + ADC_VOLTAGE_OFFSET
        # Simulation: Gaussian coupling to motor positions
        pos = self._motors.get_positions() if self._motors else {1: 0, 2: 0, 3: 0, 4: 0}
        dist_sq = sum(
            (pos[m] - self._SIM_OPTIMUM[m]) ** 2 for m in (1, 2, 3, 4)
        )
        v = self._SIM_V_MAX * math.exp(-dist_sq / (2 * self._SIM_SIGMA ** 2))
        return v + random.gauss(0, self._SIM_NOISE)

    def read_stable(
        self,
        discard: int = DISCARD_N,
        measure: int = MEASURE_N,
        gap: float = BURST_GAP_S,
    ) -> float:
        """Discard transient samples, then return mean of `measure` fresh readings."""
        for _ in range(discard):
            self.read_one()
            if _HAS_ADS:
                time.sleep(gap)
        readings = []
        for _ in range(measure):
            readings.append(self.read_one())
            if _HAS_ADS:
                time.sleep(gap)
        return sum(readings) / len(readings)

    def read_quick(self, n: int = 5) -> float:
        """
        Genuinely independent samples for drop-detection during chunked motion.

        In single-shot mode each `chan.voltage` access triggers a fresh
        ADC conversion (~8 ms at 128 SPS) and returns its result.  Five
        samples = ~40 ms of read time; the mean has ~half the noise of
        a single sample.

        IMPORTANT: caller is responsible for settling before this call.
        If motors are still mid-ringing or EMI is still radiating, those
        artefacts will get included in the average.  In practice, sleep
        ~CHUNK_SETTLE_S (100 ms) between any motion and calling this.
        """
        if _HAS_ADS:
            readings = []
            for _ in range(n):
                readings.append(self._chan.voltage)
                # No extra sleep: single-shot mode inherently waits for each
                # conversion to complete before chan.voltage returns.
            return sum(readings) / len(readings) + ADC_VOLTAGE_OFFSET
        # Simulation: just use read_one (same as before)
        return sum(self.read_one() for _ in range(n)) / n


# =============================================================================
# Controller state machine
# =============================================================================

class State:
    IDLE       = "idle"
    OPTIMIZING = "optimizing"
    STABLE     = "stable"
    PAUSED     = "paused"
    EMERGENCY  = "emergency"


class AlignmentController:
    """
    Three-phase optimiser: beam-walk → Nelder-Mead → fine descent.

    Core invariant
    ──────────────
    • run_best_voltage / run_best_positions is the single best point ever seen.
    • It only advances when a measurement strictly beats it.
    • Motors always land at run_best_positions when optimize ends.
    """

    def __init__(self) -> None:
        self.motors  = MotorController()
        self.reader  = VoltageReader(self.motors)
        self.session: Optional[SessionLogger] = None   # created fresh each run

        # ── Shared state (guarded by _lock) ──────────────────────────────────
        self._lock             = threading.Lock()
        self.state             = State.IDLE
        self.current_voltage   = 0.0
        self.run_best_voltage  = 0.0
        self.run_best_positions: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
        # Separate "stable target" voltage — used by stable mode for drop
        # comparison instead of run_best_voltage.  If a recovery sweep
        # can't reach run_best (because the laser source or optical
        # alignment has physically shifted), this target is LOWERED to
        # the new achievable local peak so stable mode stops re-triggering
        # endless recovery sweeps that can never reach the historical max.
        # run_best_voltage stays as the historical record.
        self.stable_target_voltage = 0.0
        self.target_voltage    = 2.0
        self.status_msg        = "Idle — jog or hit Optimize"
        self._log_lines: List[str] = []
        self.last_log_dir      = ""    # shown in UI after each run

        # ── Per-motor direction memory (updated by greedy optimizer) ─────────
        # Stores the last direction that improved voltage for each motor.
        # The greedy pass starts from this direction so it doesn't waste a
        # probe in the known-bad direction after the first pass.
        self._last_dir: Dict[int, int] = {1: 1, 2: 1, 3: 1, 4: 1}  # 1=CW

        # ── Thread control ────────────────────────────────────────────────────
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None

    # =========================================================================
    # Public API (called from Flask routes)
    # =========================================================================

    def start_optimize(self, target: float) -> Tuple[bool, str]:
        if not self._can_start():
            return False, "A mode is already running — stop it first"
        self.target_voltage = target
        self._launch(self._optimize_loop)
        return True, "Optimize started"

    def start_stable(self) -> Tuple[bool, str]:
        if not self._can_start():
            return False, "A mode is already running — stop it first"
        self._launch(self._stable_loop)
        return True, "Stable-hold started"

    def stop(self) -> None:
        self._stop_event.set()
        # State will be set to IDLE by the worker's finally block

    def jog(self, motor_id: int, direction: int, steps: int) -> Tuple[bool, str]:
        if self.state != State.IDLE:
            return False, "Cannot jog while a mode is running"
        self.motors.step_motor(motor_id, direction, steps)
        v = self._measure_now()
        self._log(
            f"Jog M{motor_id} {'CW' if direction else 'CCW'} ×{steps}  "
            f"V={v:.4f}  pos={self.motors.get_positions()}"
        )
        return True, f"V={v:.4f} V"

    def snapshot_best(self) -> None:
        """Measure current position and set as run_best anchor."""
        pos = self.motors.get_positions()
        v   = self._measure_now()
        with self._lock:
            self.run_best_voltage   = v
            self.run_best_positions = dict(pos)
        self._log(f"Snapshot: run_best={v:.4f} V  pos={pos}")

    def zero_positions(self) -> None:
        """Reset step counters to zero without moving motors."""
        self.motors.force_set_positions({1: 0, 2: 0, 3: 0, 4: 0})
        with self._lock:
            self.run_best_positions = {1: 0, 2: 0, 3: 0, 4: 0}
        self._log("Position counters zeroed (no motor motion)")

    def get_status(self) -> dict:
        with self._lock:
            return {
                "state":            self.state,
                "current_voltage":  round(self.current_voltage,  6),
                "run_best_voltage": round(self.run_best_voltage, 6),
                "target_voltage":   round(self.target_voltage,   6),
                "positions":        self.motors.get_positions(),
                "status_msg":       self.status_msg,
                "log":              list(self._log_lines[-80:]),
                "last_log_dir":     self.last_log_dir,
            }

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _can_start(self) -> bool:
        """Return True and arm stop event only if no worker is alive."""
        if self._worker and self._worker.is_alive():
            return False
        self._stop_event.clear()
        return True

    def _launch(self, fn) -> None:
        self._worker = threading.Thread(target=fn, daemon=True)
        self._worker.start()

    def _set_state(self, state: str, msg: str = "") -> None:
        with self._lock:
            self.state      = state
            self.status_msg = msg
        if msg:
            self._log(msg)

    def _log(self, msg: str) -> None:
        ts   = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        log.info(msg)
        with self._lock:
            self._log_lines.append(line)
            if len(self._log_lines) > 300:
                self._log_lines = self._log_lines[-300:]

    def _stopped(self) -> bool:
        return self._stop_event.is_set()

    def _measure_now(self) -> float:
        """Settle then take a stable average — updates current_voltage.
        Automatically pauses and waits if beam appears blocked (V < BEAM_BLOCKED_V).
        """
        if _HAS_GPIO:
            time.sleep(SETTLE_S)
        v = self.reader.read_stable()
        with self._lock:
            self.current_voltage = v
        if v < BEAM_BLOCKED_V and not self._stop_event.is_set():
            self._handle_beam_blocked(v)
            if not self._stop_event.is_set():
                v = self.reader.read_stable()
                with self._lock:
                    self.current_voltage = v
        return v

    def _get_run_best(self) -> Tuple[float, Dict[int, int]]:
        with self._lock:
            return self.run_best_voltage, dict(self.run_best_positions)

    def _return_to_best(self) -> None:
        """Physically move motors back to run_best_positions."""
        _, rb_pos = self._get_run_best()
        self.motors.move_to(rb_pos, self._stop_event)

    def _return_and_verify(self) -> float:
        """
        Return to run_best_positions then measure to confirm we physically
        arrived at the coupling peak.

        If the measured voltage is more than DRIFT_WARN_V below run_best_voltage,
        backlash has caused the physical position to drift.  In that case the
        anchor is updated to the current physical position so subsequent
        candidates use an honest baseline.

        Returns the measured voltage at the anchor position.
        """
        self._return_to_best()
        if self._stopped():
            return self.run_best_voltage
        v = self._measure_now()
        rb_v, _ = self._get_run_best()
        drift = rb_v - v
        if drift > DRIFT_WARN_V:
            self._log(
                f"  ⚠ Drift {drift:.4f} V after return: expected {rb_v:.4f}, "
                f"got {v:.4f}. Backlash likely — continuing from run_best anchor."
            )
            # Note: _update_best is guarded so it will NOT lower run_best here.
            # run_best stays at rb_v; candidates must still beat that level.
        return v

    def _update_best(self, voltage: float) -> None:
        """Advance run-best only when voltage is a new session high."""
        pos = self.motors.get_positions()
        with self._lock:
            if voltage > self.run_best_voltage:
                self.run_best_voltage   = voltage
                self.run_best_positions = dict(pos)

    # =========================================================================
    # Core trial evaluation — THE heart of the controller
    # =========================================================================

    def _walk_axis(
        self,
        axis_name: str,
        initial_spec: List[Tuple[int, int]],
        packet: int,   # kept for API compatibility but no longer caps the walk
    ) -> bool:
        """
        Continuous-chunk hill-climbing along one axis.

        No artificial "packet" boundary — chunks of CHUNK_STEPS flow
        continuously and the walk extends as long as V keeps climbing.
        Designed per user's spec (2026-05-26): step a tiny amount, check
        V, decide; repeat until the direction is clearly DONE.

        Per chunk:
          1. Step CHUNK_STEPS in current_spec direction.
          2. Read V (full settle + 8-sample average → noise ~2 mV).
          3. Decide:
             • V dropped > CHUNK_DROP_V below v_best AND a second read
               (after extra settle) agrees → real drop, end this direction.
             • V improved > CHUNK_GAIN_V over v_best → slide v_best up,
               keep going in the SAME direction (no upper step limit).
             • Otherwise (flat) → increment no-progress counter.  After
               CHUNK_FLAT_PATIENCE consecutive flat chunks, decide:
                 - if we climbed (v_best > v_baseline + GAIN) → at peak
                   in this direction, end axis.
                 - else → flip direction (if not already flipped) and
                   try opposite, or end axis if both directions tried.

        Direction handling:
          • First direction is initial_spec.
          • Flip happens at most ONCE per axis (after first direction
            shows confirmed drop or non-climbing plateau).
          • After flipping, same chunked-climb logic applies.
          • Second confirmed drop / plateau → axis done.

        Backlash:
          • Silent takeup runs inside step_motor on every direction
            change — no explicit backlash handling needed here.
          • Worst-case wrong-direction motion = CHUNK_STEPS (e.g. 100).

        Returns True if at least one chunk improved V meaningfully.
        """
        # Effective chunk size, clipped per-motor.
        effective_chunk = CHUNK_STEPS
        for motor_id, _ in initial_spec:
            effective_chunk = min(
                effective_chunk, AXIS_MAX_PACKET.get(motor_id, CHUNK_STEPS)
            )

        current_spec      = [(m, d) for m, d in initial_spec]
        direction_flipped = False
        any_accepted      = False

        v_baseline = self._measure_now()
        if self._stopped():
            return False
        v_best         = v_baseline       # highest V seen in current direction
        v_snapshot     = v_baseline       # sliding ref for slow-climb detection
        no_gain_chunks = 0
        chunks_done    = 0
        # Resistance detection state: when did v_best last advance? how many
        # consecutive chunks have we been "below peak" since that advance?
        last_v_best_chunk        = 0
        resistance_below_count   = 0

        while chunks_done < CHUNK_MAX_TOTAL:
            if self._stopped():
                break

            self.motors.apply_relative(
                current_spec, effective_chunk, self._stop_event
            )
            chunks_done += 1
            if self._stopped():
                break

            v_now           = self._measure_now()
            delta_from_best = v_now - v_best
            tag_now         = "+".join(
                f"M{m}{'CW' if d else 'CCW'}" for m, d in current_spec
            )

            # ── Drop detection (sustained, confirmed) ────────────────────────
            # Use a tighter threshold once we've climbed (any_accepted=True) —
            # past-peak descent can be slow and a 25 mV cumulative drop wastes
            # many chunks walking past the peak before triggering.  10 mV is
            # tight enough to catch peak passage quickly but still large
            # enough that two-read confirmation rules out noise.
            drop_threshold = (
                CHUNK_POST_CLIMB_DROP_V if any_accepted else CHUNK_DROP_V
            )
            if v_now < v_best - drop_threshold:
                if _HAS_GPIO:
                    time.sleep(CHUNK_CONFIRM_SETTLE_S)
                v_confirm = self._measure_now()
                if v_confirm < v_best - drop_threshold:
                    # Real drop — log and decide next step.
                    self._log(
                        f"  [{axis_name} {tag_now} chunk {chunks_done} ×{effective_chunk}]  "
                        f"V={v_confirm:.4f}  v_best={v_best:.4f}  Δ={v_confirm - v_best:+.4f}  "
                        f"CONFIRMED DROP"
                    )
                    if self.session:
                        self.session.log_trial(
                            "beamwalk", f"{axis_name}_{tag_now}",
                            effective_chunk, self.motors.get_positions(),
                            v_confirm, self.run_best_voltage, False,
                            f"confirmed_drop_{v_confirm - v_best:+.3f}"
                        )
                    if direction_flipped:
                        self._log(
                            f"  ↳ {axis_name} both directions exhausted"
                        )
                        break
                    # Flip direction and re-baseline at current physical V.
                    direction_flipped = True
                    current_spec      = [(m, 1 - d) for m, d in current_spec]
                    new_tag = "+".join(
                        f"M{m}{'CW' if d else 'CCW'}" for m, d in current_spec
                    )
                    self._log(f"  ↻ {axis_name} flipping → {new_tag} (after drop)")
                    v_baseline     = self._measure_now()
                    v_best         = v_baseline
                    v_snapshot     = v_baseline
                    no_gain_chunks = 0
                    last_v_best_chunk      = chunks_done
                    resistance_below_count = 0
                    continue
                # Confirmation recovered — noise.  Treat as if no drop.
                v_now           = v_confirm
                delta_from_best = v_now - v_best

            # ── Gain detection (sustained climb) ─────────────────────────────
            if delta_from_best > CHUNK_GAIN_V:
                v_best         = v_now
                v_snapshot     = v_now      # also advance slow-climb reference
                no_gain_chunks = 0
                last_v_best_chunk      = chunks_done
                resistance_below_count = 0
                any_accepted   = True
                self._update_best(v_now)
                self._log(
                    f"  [{axis_name} {tag_now} chunk {chunks_done} ×{effective_chunk}]  "
                    f"V={v_now:.4f}  v_best={v_best:.4f}  Δ={delta_from_best:+.4f}  ✓ GAIN"
                )
                if self.session:
                    self.session.log_trial(
                        "beamwalk", f"{axis_name}_{tag_now}",
                        effective_chunk, self.motors.get_positions(),
                        v_now, self.run_best_voltage, True,
                        f"gain_{delta_from_best:+.3f}"
                    )
                continue

            # ── Slow-climb detection (cumulative-from-snapshot) ──────────────
            # Catches the case where V is rising at <CHUNK_GAIN_V per chunk
            # but still climbing genuinely.  The snapshot advances each time
            # this fires, so the walk only stays alive while V keeps rising
            # — once V plateaus (no further climb from the most recent
            # snapshot), patience starts accumulating and eventually exits.
            cumulative_from_snapshot = v_now - v_snapshot
            if cumulative_from_snapshot > CHUNK_SLOW_CLIMB_V:
                v_snapshot     = v_now           # advance the snapshot
                if v_now > v_best:
                    v_best                 = v_now
                    last_v_best_chunk      = chunks_done
                    resistance_below_count = 0
                    self._update_best(v_now)
                any_accepted   = True
                no_gain_chunks = 0
                self._log(
                    f"  [{axis_name} {tag_now} chunk {chunks_done} ×{effective_chunk}]  "
                    f"V={v_now:.4f}  v_best={v_best:.4f}  "
                    f"snap+{cumulative_from_snapshot:+.4f}  ↗ slow climb"
                )
                if self.session:
                    self.session.log_trial(
                        "beamwalk", f"{axis_name}_{tag_now}",
                        effective_chunk, self.motors.get_positions(),
                        v_now, self.run_best_voltage, True,
                        f"slow_climb_{cumulative_from_snapshot:+.3f}"
                    )
                continue

            # ── Resistance detection (post-climb peak-passage) ───────────────
            # Fires when we were climbing recently (v_best advanced not long
            # ago) AND V has dipped below v_best by even a small amount.
            # Two-step verified to filter noise.  This is the fast peak
            # detector — much quicker than the 10 mV drop confirmation
            # because it leverages the temporal context of recent gain.
            chunks_since_v_best_advance = chunks_done - last_v_best_chunk
            if (
                any_accepted
                and chunks_since_v_best_advance >= CHUNK_RESISTANCE_BUFFER
                and v_now < v_best - CHUNK_RESISTANCE_DROP_V
            ):
                resistance_below_count += 1
                if resistance_below_count >= CHUNK_RESISTANCE_CONSEC:
                    self._log(
                        f"  [{axis_name} {tag_now} chunk {chunks_done} ×{effective_chunk}]  "
                        f"V={v_now:.4f}  v_best={v_best:.4f}  "
                        f"Δ={v_now - v_best:+.4f}  ⊥ RESISTANCE (peak passed)"
                    )
                    if self.session:
                        self.session.log_trial(
                            "beamwalk", f"{axis_name}_{tag_now}",
                            effective_chunk, self.motors.get_positions(),
                            v_now, self.run_best_voltage, False,
                            f"resistance_{v_now - v_best:+.3f}"
                        )
                    break   # axis done — no third flip after climb+resistance
            else:
                # Reset the consecutive counter — we're either still climbing
                # or the dip was a one-off (next chunk recovered).
                resistance_below_count = 0

            # ── Flat (no clear gain or drop) ─────────────────────────────────
            no_gain_chunks += 1
            self._log(
                f"  [{axis_name} {tag_now} chunk {chunks_done} ×{effective_chunk}]  "
                f"V={v_now:.4f}  v_best={v_best:.4f}  Δ={delta_from_best:+.4f}  "
                f"flat ({no_gain_chunks}/{CHUNK_FLAT_PATIENCE})"
            )

            if no_gain_chunks >= CHUNK_FLAT_PATIENCE:
                climbed = v_best > v_baseline + CHUNK_GAIN_V
                if self.session:
                    self.session.log_trial(
                        "beamwalk", f"{axis_name}_{tag_now}",
                        effective_chunk, self.motors.get_positions(),
                        v_now, self.run_best_voltage, False,
                        "plateau_after_climb" if climbed else "plateau_no_climb"
                    )

                if climbed:
                    # Climbed in this direction then plateaued — at peak.
                    # No need to try opposite direction; we already know
                    # this side of the peak.
                    self._log(
                        f"  ↳ {axis_name} plateau after climb, peak ~V={v_best:.4f}"
                    )
                    break

                # Did not climb in this direction — try the opposite.
                if direction_flipped:
                    self._log(
                        f"  ↳ {axis_name} neither direction climbed — axis done"
                    )
                    break
                direction_flipped = True
                current_spec      = [(m, 1 - d) for m, d in current_spec]
                new_tag = "+".join(
                    f"M{m}{'CW' if d else 'CCW'}" for m, d in current_spec
                )
                self._log(
                    f"  ↻ {axis_name} flipping → {new_tag} (no climb in initial direction)"
                )
                v_baseline     = self._measure_now()
                v_best         = v_baseline
                v_snapshot     = v_baseline
                no_gain_chunks = 0
                last_v_best_chunk      = chunks_done
                resistance_below_count = 0
        else:
            self._log(
                f"  ↳ {axis_name} hit CHUNK_MAX_TOTAL={CHUNK_MAX_TOTAL} chunks"
            )

        return any_accepted

    # =========================================================================
    # Stable-mode recovery sweep — uses the same _walk_axis as optimize
    # =========================================================================

    def _recovery_sweep(self, packet: int) -> bool:
        """
        One full sweep of all WALK_AXES at the given packet size, used by
        stable mode when V drops out of band.  Same algorithm as optimize so
        recovery has the same backlash-safety properties (no test-and-revert,
        per-axis monotonic return, sharp-drop guard).

        Per-axis early exit: after each axis walk, if V has climbed to
        within STABLE_RECOVERY_NEAR_V of the stable target, the sweep
        exits early.  Each axis walk runs to completion (so any active
        climb finishes naturally via resistance), but as soon as we're
        "close enough" we don't waste motion on the remaining axes which
        would only risk drifting V back down.

        Returns True if any axis improved.
        """
        any_improved = False
        for axis_name, initial_spec in WALK_AXES:
            if self._stopped():
                break
            if self._walk_axis(axis_name, initial_spec, packet):
                any_improved = True
            # After each axis walk completes (via resistance, drop, plateau,
            # or max), check if we're now within ~20 mV of the stable target.
            # If so, recovery has effectively succeeded — stop sweeping.
            with self._lock:
                v_now = self.current_voltage
                stable_target = self.stable_target_voltage
            if stable_target - v_now <= STABLE_RECOVERY_NEAR_V:
                self._log(
                    f"Stable recovery: V={v_now:.4f} within "
                    f"{STABLE_RECOVERY_NEAR_V * 1000:.0f} mV of target "
                    f"{stable_target:.4f} after {axis_name} — "
                    f"early exit from sweep"
                )
                break
        return any_improved

    # =========================================================================
    # Optimise loop
    # =========================================================================

    # =========================================================================
    # Nelder-Mead refine  (Phase 2 of optimize)
    # =========================================================================

    def _nelder_mead_refine(self, simplex_size: int = NM_SIMPLEX_SIZE,
                             budget: int = NM_MAX_ITER) -> None:
        """
        Run scipy Nelder-Mead from run_best_positions.

        Why Nelder-Mead beats pure coordinate descent for fibre coupling:
        Single-mode coupling has a narrow ridge in 4D space.  Coordinate
        descent gets stuck when axes are correlated (shifting M1 alone does
        nothing useful unless M3 shifts too).  Nelder-Mead explores the space
        as a 4D simplex shape and naturally handles axis coupling without
        needing gradients.

        The simplex is anchored at run_best_positions.  During the search,
        motors visit sub-optimal positions (that is how Nelder-Mead works).
        run_best is updated whenever a better point is found; motors return
        to run_best when the method exits.
        """
        _, rb_pos = self._get_run_best()
        anchor    = dict(rb_pos)        # fixed reference; offsets are from here
        call_n    = [0]

        self._log(f"── Nelder-Mead refine  simplex={simplex_size}  budget={budget} ──")
        self._set_state(State.OPTIMIZING, f"Nelder-Mead refining (budget={budget} evals)")

        def neg_voltage(offsets: List[float]) -> float:
            if self._stopped():
                raise StopIteration("stop requested")
            call_n[0] += 1

            # Build absolute target; clamp to motor limits
            target: Dict[int, int] = {}
            for i, m_id in enumerate((1, 2, 3, 4)):
                lo, hi    = MOTOR_LIMITS[m_id]
                raw       = anchor[m_id] + int(round(offsets[i]))
                target[m_id] = max(lo, min(hi, raw))

            self.motors.move_to(target, self._stop_event)
            if self._stopped():
                raise StopIteration("stop requested")

            v = self._measure_now()

            # Track run_best — use low threshold so NM doesn't discard genuine improvements
            if v > self.run_best_voltage + NM_TRACK_THRESHOLD_V:
                self._update_best(v)
                self._log(f"  NM #{call_n[0]:3d}: new best V={v:.4f}  pos={self.motors.get_positions()}")
                if self.session:
                    self.session.log_trial(
                        "nelder_mead", f"iter_{call_n[0]}", 0,
                        self.motors.get_positions(), v, self.run_best_voltage,
                        True, "nm_improvement"
                    )
            else:
                if call_n[0] % 10 == 0:
                    self._log(f"  NM #{call_n[0]:3d}: V={v:.4f}  best={self.run_best_voltage:.4f}")
                # Always log every NM eval to CSV so the landscape is visible
                if self.session:
                    self.session.log_trial(
                        "nelder_mead", f"iter_{call_n[0]}", 0,
                        self.motors.get_positions(), v, self.run_best_voltage,
                        False, "nm_eval"
                    )

            return -v   # scipy minimises; we maximise

        # Initial simplex: unit vertex + one vertex per axis offset by simplex_size
        x0              = [0.0, 0.0, 0.0, 0.0]
        initial_simplex = [x0[:]]
        for i in range(4):
            v = x0[:]
            v[i] = float(simplex_size)
            initial_simplex.append(v)

        try:
            scipy_minimize(
                neg_voltage,
                x0,
                method="Nelder-Mead",
                options={
                    "xatol":           NM_XATOL,
                    "fatol":           NM_FATOL,
                    "maxiter":         budget,
                    "initial_simplex": initial_simplex,
                    "adaptive":        True,   # auto-scales for 4D
                },
            )
        except StopIteration:
            self._log("NM: stop event received — exiting early")
        except Exception as exc:
            self._log(f"NM error: {exc}")

        self._return_to_best()
        self._log(
            f"NM done after {call_n[0]} evals.  run_best={self.run_best_voltage:.4f}  "
            f"pos={self.motors.get_positions()}"
        )

    # =========================================================================
    # Voltage-driven greedy optimizer
    # =========================================================================

    def _probe_and_extend(self, motor_id: int, direction: int) -> float:
        """
        Probe motor_id one packet (PROBE_STEPS) in `direction` (1=CW, 0=CCW).

        • If the probe improves V by > ACCEPT_THRESHOLD_V: keep and extend in
          the same direction until V stops improving.  When it peaks, back off
          one packet with a direct reverse step_motor (no move_to / backlash).

        • If the probe drops or is flat: do NOT undo.  Instead continue in the
          opposite direction from the current position — the reverse steps
          naturally cancel the bad move and then go past the starting point.
          During recovery any positive delta keeps the loop going (lenient).
          Stop and back off one packet when V drops.

        Returns total voltage gain above v0 (positive only when we beat
        v0 + ACCEPT_THRESHOLD_V).
        """
        probe = PROBE_STEPS.get(motor_id, 500)
        rev   = 1 - direction   # opposite direction

        def _step(d: int) -> None:
            self.motors.step_motor(motor_id, d, probe, self._stop_event)

        def _log_trial(d: int, v: float, v_prev: float, tag: str) -> None:
            dir_s = "CW" if d == 1 else "CCW"
            delta = v - v_prev
            self._log(f"  M{motor_id} {dir_s} ×{probe}: V={v:.4f}  Δ={delta:+.4f}")
            if self.session:
                self.session.log_trial(
                    "greedy", f"M{motor_id}{dir_s}", probe,
                    self.motors.get_positions(), v, self.run_best_voltage,
                    delta > ACCEPT_THRESHOLD_V, tag,
                )

        v0 = self._measure_now()

        # ── Initial probe ────────────────────────────────────────────────────
        _step(direction)
        if self._stopped():
            return 0.0
        v1 = self._measure_now()
        _log_trial(direction, v1, v0, "")

        if v1 > v0 + ACCEPT_THRESHOLD_V:
            # ── Extend in same direction ─────────────────────────────────────
            self._last_dir[motor_id] = direction   # remember winning direction
            self._update_best(v1)
            v_prev = v1
            while not self._stopped():
                _step(direction)
                if self._stopped():
                    break
                v_new = self._measure_now()
                _log_trial(direction, v_new, v_prev, "extend")
                if v_new > v_prev + ACCEPT_THRESHOLD_V:
                    self._update_best(v_new)
                    v_prev = v_new
                else:
                    # Peaked — back off one packet directly (no move_to)
                    _step(rev)
                    break
            return v_prev - v0

        else:
            # ── Bad initial step — switch to opposite direction from HERE ─────
            # We are currently 1 probe past v0 in `direction`.
            # Going `rev` will first cancel that bad step (landing near start),
            # then continue past start if voltage keeps rising.
            #
            # Back-off rule: only reverse the last `rev` step if we have gone
            # MORE than 1 step in recovery (i.e., past start).  If the very
            # first recovery step already drops, we are back near start — just
            # stop there; stepping `direction` again would strand us at the
            # bad initial probe position.
            v_cur           = v1
            v_peak          = v0
            improved        = False
            recovery_steps  = 0

            while not self._stopped():
                _step(rev)
                recovery_steps += 1
                if self._stopped():
                    break
                v_new = self._measure_now()
                _log_trial(rev, v_new, v_cur, "recover")
                if v_new > v_cur:
                    v_cur = v_new
                    if v_new > v0 + ACCEPT_THRESHOLD_V:
                        self._last_dir[motor_id] = rev
                        self._update_best(v_new)
                        v_peak   = v_new
                        improved = True
                else:
                    # Dropped — back off only if we've gone past start
                    if improved or recovery_steps >= 2:
                        _step(direction)
                    # If recovery_steps == 1 and not improved: we're already
                    # back near start, no back-off needed.
                    break

            return v_peak - v0 if improved else 0.0

    def _greedy_pass(self) -> bool:
        """
        One full sweep over all motors in PROBE_ORDER.
        Each motor starts in its last known good direction (_last_dir); if that
        fails, _probe_and_extend automatically tries the opposite direction.
        Returns True if ANY motor improved voltage.
        """
        any_improved = False
        for motor_id in PROBE_ORDER:
            if self._stopped():
                break
            rb_v, _ = self._get_run_best()
            start_dir = self._last_dir.get(motor_id, 1)
            dir_str   = "CW" if start_dir == 1 else "CCW"
            self._log(f"── M{motor_id} explore (best={rb_v:.4f}, start={dir_str}) ──")
            gain = self._probe_and_extend(motor_id, start_dir)
            if gain > 0:
                any_improved = True
        return any_improved

    # =========================================================================
    # Optimise loop  — voltage-driven greedy
    # =========================================================================

    def _optimize_loop(self) -> None:
        """
        Beamwalk optimizer — paired + single-axis moves, NO anchor-returns.

        Algorithm:
          For each packet size in ALL_PACKETS (large → small):
            For up to MAX_PASSES_PER_PACKET sweeps:
              Try every move in ALL_MOVES (M3/M4 singles, paired Y/X moves,
              M1/M2 singles).  Each trial measures V before the move, applies
              the step, and compares.  No move_to() anchor-return is used in
              the hot path — this matches manual jog behaviour and avoids
              backlash accumulation from repeated direction reversals.
            If no move improved in a full sweep, reduce to next packet size.
          At the end, motors physically return to run_best_positions (only
          move_to() in the entire run).
        """
        try:
            self.session = SessionLogger()
            with self._lock:
                self.last_log_dir = self.session.path

            v_init   = self._measure_now()
            pos_init = self.motors.get_positions()
            self._update_best(v_init)

            self._log(
                f"Start: V={v_init:.4f}  target={self.target_voltage:.4f}  pos={pos_init}"
            )
            self._set_state(State.OPTIMIZING, f"Optimizing — V={v_init:.4f}")
            if self.session:
                self.session.log_trial(
                    "init", "start", 0, pos_init, v_init, v_init, False, "initial"
                )

            target_reached = False

            # ── Commit-and-extend ladder ─────────────────────────────────────
            # For each packet size (large → small): walk every axis once with
            # commit-and-extend (no test-and-revert).  After a full sweep, if
            # ANY axis improved we sweep again at the same packet (in case the
            # new position unlocks improvement on previously-stuck axes).
            # When a full sweep produces no improvement, drop to the next
            # smaller packet.  No anchor-returns mid-run — the only move_to()
            # of the entire optimisation is the final return at the end.
            for packet in ALL_PACKETS:
                if self._stopped():
                    break
                rb_v, _ = self._get_run_best()
                self._log(f"══ Packet {packet} steps  best={rb_v:.4f} V ══")
                self._set_state(
                    State.OPTIMIZING,
                    f"Beamwalk packet={packet}, best={rb_v:.4f} V"
                )

                for pass_n in range(1, MAX_PASSES_PER_PACKET + 1):
                    if self._stopped():
                        break
                    self._log(f"── pass {pass_n} @ packet={packet} ──")
                    any_improved = False

                    for axis_name, initial_spec in WALK_AXES:
                        if self._stopped():
                            break
                        improved = self._walk_axis(
                            axis_name, initial_spec, packet
                        )
                        if improved:
                            any_improved = True
                            rb_v, _ = self._get_run_best()
                            if rb_v >= self.target_voltage:
                                target_reached = True
                                self._log(
                                    f"✓ Target {self.target_voltage:.4f} V reached!"
                                )
                                break

                    if target_reached:
                        break
                    if not any_improved:
                        self._log(
                            f"  No axis improved at packet={packet} "
                            f"(pass {pass_n}) — reducing packet."
                        )
                        break

                if target_reached:
                    break

            # Return to best position at end of run
            self._return_to_best()
            v_final      = self._measure_now()
            rb_v, rb_pos = self._get_run_best()
            self._log(
                f"Optimize complete.  V={v_final:.4f}  best={rb_v:.4f}  pos={rb_pos}"
            )
            status_txt = (
                f"Optimize done — best={rb_v:.4f} V  "
                + ("✓ target reached" if target_reached else "(target not reached)")
                + f"  log: {self.session.path if self.session else ''}"
            )
            self._set_state(State.IDLE, status_txt)

            if self.session:
                self.session.log_trial(
                    "final", "end", 0, self.motors.get_positions(),
                    v_final, rb_v, target_reached, "optimize_end"
                )
                self.session.write_summary({
                    **self.get_status(),
                    "target_reached": target_reached,
                    "phases": ["greedy_voltage_driven"],
                })

        except Exception as exc:
            self._log(f"Optimize error: {exc}")
            self._set_state(State.IDLE, f"Error: {exc}")
        finally:
            with self._lock:
                if self.state == State.OPTIMIZING:
                    self.state = State.IDLE

    # =========================================================================
    # Stable / hold loop
    # =========================================================================

    def _stable_loop(self) -> None:
        try:
            self.session = SessionLogger()
            with self._lock:
                self.last_log_dir = self.session.path
            self._set_state(State.STABLE, "Stable hold — monitoring")

            # If run_best has never been set, snapshot current position now
            with self._lock:
                rb_v = self.run_best_voltage
            if rb_v == 0.0:
                self.snapshot_best()

            # Initialize stable_target_voltage from run_best at the start of
            # stable mode.  This will be LOWERED automatically if an
            # unsuccessful recovery indicates the achievable maximum has
            # dropped (e.g., the laser source was physically adjusted).
            with self._lock:
                if self.stable_target_voltage < self.run_best_voltage:
                    self.stable_target_voltage = self.run_best_voltage
                stable_target = self.stable_target_voltage

            self._log(
                f"Stable: monitoring with target V={stable_target:.4f} "
                f"(run_best={self.run_best_voltage:.4f})"
            )

            tick = 0
            while not self._stopped():
                time.sleep(STABLE_POLL_S)
                if self._stopped():
                    break

                v = self.reader.read_stable(discard=1, measure=3)
                with self._lock:
                    self.current_voltage = v
                    rb_v = self.run_best_voltage
                    stable_target = self.stable_target_voltage

                # Drop is now measured against stable_target_voltage, NOT
                # run_best_voltage.  If a previous recovery couldn't reach
                # run_best, stable_target was lowered so we stop endlessly
                # retrying an unreachable target.
                drop = stable_target - v
                tick += 1

                # Absolute beam-blocked check (laser off → _measure_now handles
                # it during optimize; here we catch it in the stable poll path)
                if v < BEAM_BLOCKED_V:
                    self._handle_beam_blocked(v)
                    if not self._stopped():
                        self._set_state(State.STABLE, "Stable hold — monitoring")
                    continue

                if drop < 0:
                    # Voltage crept above stable_target — update target and
                    # also raise run_best if applicable.
                    self._update_best(v)
                    with self._lock:
                        self.stable_target_voltage = v
                    self._log(f"Stable: new best V={v:.4f} (+{-drop:.4f}) — target raised")
                    continue

                if drop <= STABLE_BAND_V:
                    if tick % 5 == 0:
                        self._log(f"Stable: V={v:.4f}  drop={drop:.4f} ✓")
                    continue

                # ── Drop detected — verify with 5s settle + multi-read check ──
                # Wait for any transient (mechanical bump, thermal blip, brief
                # vibration near the setup) to settle, then take 4 reads
                # spaced ~0.5s apart.  If those 4 reads agree within ~5 mV
                # (noise range), the drop is real and we trigger recovery.
                # If they're still fluctuating widely, V is still settling
                # and we skip this tick — try again on next poll.
                self._log(
                    f"Stable: drop={drop:.4f} V detected, waiting 5 s to verify..."
                )
                time.sleep(5.0)
                if self._stopped():
                    break

                verify_reads: List[float] = []
                for _ in range(4):
                    if self._stopped():
                        break
                    verify_reads.append(
                        self.reader.read_stable(discard=1, measure=3)
                    )
                    time.sleep(0.5)
                if self._stopped() or not verify_reads:
                    break

                v_range = max(verify_reads) - min(verify_reads)
                v_mean  = sum(verify_reads) / len(verify_reads)
                with self._lock:
                    self.current_voltage = v_mean

                if v_range > 0.005:
                    # Still fluctuating > 5 mV across 4 reads — V hasn't
                    # settled yet (possibly a temporary disturbance still
                    # happening).  Don't trigger recovery now; let the next
                    # poll tick re-evaluate.
                    self._log(
                        f"Stable: V still fluctuating (range {v_range*1000:.1f} mV "
                        f"across 4 reads, mean {v_mean:.4f}) — skipping recovery, "
                        f"will re-check next poll"
                    )
                    continue

                # V is settled — recompute the drop from the verified mean,
                # against stable_target (not run_best).
                drop = stable_target - v_mean
                v    = v_mean
                if drop <= STABLE_BAND_V:
                    self._log(
                        f"Stable: V settled to {v_mean:.4f} (drop={drop:.4f}, "
                        f"within band) — no recovery needed, was transient"
                    )
                    continue

                self._log(
                    f"Stable: drop={drop:.4f} V VERIFIED (4 reads agree within "
                    f"{v_range*1000:.1f} mV) — proceeding with recovery"
                )

                # ── Signal has dropped — recover using _walk_axis sweep ──────
                # Uses the same backlash-safe commit-and-extend algorithm as
                # optimize (per-axis monotonic return, sharp-drop guard,
                # paired-walk-first axis order).
                #
                # Escalation ladder (matches manual-jog instinct: small jogs
                # first, larger if those don't recover):
                #   • Small drop (>= REACQUIRE_DROP_V): one pass at packet=250
                #   • Large drop (>= RECOVER_DROP_V):   pass at 500, then 1000
                #     if still below band, then 250 to refine the new local max
                if drop >= RECOVER_DROP_V:
                    self._log(
                        f"Stable: drop={drop:.4f} V (V={v:.4f}) — broad recovery"
                    )
                    self._set_state(State.STABLE, f"Recovering (drop={drop:.4f} V)…")
                    v_recovery_best   = v   # baseline at start of recovery
                    recovery_reached  = False
                    # Recovery uses chunked _walk_axis just like optimize.
                    # We allow at most 2 full sweeps through all 8 axes —
                    # the 1st finds the achievable local peak, the 2nd is
                    # a tie-breaker pass for axes that may have been
                    # "blocked" before others moved.  More than 2 sweeps
                    # was tried previously (the old 500/1000/250 'packet'
                    # ladder), but since _walk_axis uses fixed 100-step
                    # chunks regardless of the packet argument, those
                    # were 3 identical sweeps and the 3rd was causing
                    # cumulative no-gradient drift.  2 passes is the
                    # sweet spot: enough for the optimization to settle,
                    # not so many that drift accumulates.
                    for pass_n in range(1, 3):    # max 2 sweeps
                        if self._stopped():
                            break
                        improved = self._recovery_sweep(100)   # packet arg ignored
                        v_now = self.reader.read_stable(discard=1, measure=3)
                        with self._lock:
                            self.current_voltage = v_now
                            stable_target = self.stable_target_voltage
                        # If we're back inside the stable band, done.
                        if stable_target - v_now <= STABLE_BAND_V:
                            recovery_reached = True
                            self._log(
                                f"Stable: recovered to V={v_now:.4f} "
                                f"after pass {pass_n} sweep"
                            )
                            break
                        # Diminishing-returns early exit (only checked on
                        # pass 2+, so pass 1 always runs to completion).
                        if pass_n > 1:
                            gain_this_pass = v_now - v_recovery_best
                            if gain_this_pass < 0.005:
                                self._log(
                                    f"Stable: diminishing returns "
                                    f"(only +{gain_this_pass*1000:.1f} mV from "
                                    f"pass {pass_n}) — stopping recovery"
                                )
                                break
                        if v_now > v_recovery_best:
                            v_recovery_best = v_now
                        if not improved:
                            self._log(
                                f"Stable: no improvement on pass {pass_n}"
                                f" — trying one more"
                            )
                    # ── Adjust stable_target if recovery couldn't reach it ──
                    # If we exited recovery WITHOUT reaching within band of
                    # stable_target, lower the target to the best V we
                    # actually achieved.  This prevents endless re-triggering
                    # against an unreachable historical max.  run_best stays
                    # untouched as the historical record.
                    if not recovery_reached and v_recovery_best < stable_target - STABLE_BAND_V:
                        with self._lock:
                            old_target = self.stable_target_voltage
                            self.stable_target_voltage = v_recovery_best
                        self._log(
                            f"Stable: target lowered from {old_target:.4f} → "
                            f"{v_recovery_best:.4f} (run_best={rb_v:.4f} unchanged). "
                            f"Will only retrigger recovery if V drops below "
                            f"{v_recovery_best - RECOVER_DROP_V:.4f}."
                        )
                    self._set_state(State.STABLE, "Stable hold — monitoring")

                elif drop >= REACQUIRE_DROP_V:
                    self._log(
                        f"Stable: drop={drop:.4f} V (V={v:.4f}) — fine reacquire"
                    )
                    self._set_state(State.STABLE, f"Reacquiring (drop={drop:.4f} V)…")
                    self._recovery_sweep(250)
                    self._set_state(State.STABLE, "Stable hold — monitoring")

                else:
                    self._log(f"Stable: V={v:.4f}  drop={drop:.4f} (within band)")

        except Exception as exc:
            self._log(f"Stable error: {exc}")
        finally:
            self._set_state(State.IDLE, "Stable hold stopped")
            with self._lock:
                if self.state == State.STABLE or self.state == State.PAUSED:
                    self.state = State.IDLE

    def _handle_beam_blocked(self, v: float) -> None:
        """
        Called from _measure_now whenever V < BEAM_BLOCKED_V.
        Works in ANY mode (optimize, stable, scan) — pauses at the next
        measurement point and waits for beam recovery.
        If beam does not recover within EMERGENCY_TIMEOUT_S, triggers
        emergency stop.
        """
        prev_msg = self.status_msg
        self._set_state(State.PAUSED,
                        f"BEAM BLOCKED — V={v:.4f} V  waiting for laser signal…")
        self._log(
            f"Beam blocked (V={v:.4f} < {BEAM_BLOCKED_V} V) — "
            f"pausing all operations.  Emergency stop in {EMERGENCY_TIMEOUT_S}s if not recovered."
        )
        deadline    = time.time() + EMERGENCY_TIMEOUT_S
        last_warn   = time.time()
        with self._lock:
            threshold = max(BEAM_BLOCKED_V, self.run_best_voltage - RESUME_HYSTERESIS_V)

        while not self._stop_event.is_set():
            time.sleep(1.0)
            v = self.reader.read_one()
            with self._lock:
                self.current_voltage = v

            if v >= threshold:
                self._log(f"Beam recovered: V={v:.4f} — resuming")
                self._set_state(State.PAUSED, "Beam recovered — resuming…")
                return

            remaining = deadline - time.time()
            if remaining <= 0:
                self._log(
                    f"EMERGENCY STOP — beam has been blocked for "
                    f">{EMERGENCY_TIMEOUT_S // 60} min.  "
                    f"Check laser / fibre connection."
                )
                self._set_state(
                    State.EMERGENCY,
                    f"EMERGENCY — beam blocked >{EMERGENCY_TIMEOUT_S // 60} min.  "
                    f"Restart program after fixing laser."
                )
                self._stop_event.set()
                return

            # Warn every 30 s
            if time.time() - last_warn >= 30:
                self._log(
                    f"Still waiting for beam recovery… "
                    f"{int(remaining)}s until emergency stop.  V={v:.4f}"
                )
                last_warn = time.time()

    def _wait_for_resume(self) -> None:
        """Used by stable loop PAUSED branch — delegates to _handle_beam_blocked logic."""
        with self._lock:
            threshold = self.run_best_voltage - RESUME_HYSTERESIS_V
        self._log(f"Waiting for signal to recover above {threshold:.4f} V …")
        while not self._stopped():
            time.sleep(1.0)
            v = self.reader.read_one()
            with self._lock:
                self.current_voltage = v
            if v >= threshold:
                self._log(f"Signal recovered: V={v:.4f} — resuming")
                return

    def _mini_optimize(self, packet_list: List[int] = None) -> None:  # noqa: ARG002
        """
        Legacy shim — delegates to _greedy_pass() so any remaining callers
        (e.g. scan mode helpers) use the voltage-driven greedy strategy instead
        of the old anchor-return coordinate-ascent approach.

        The `packet_list` argument is accepted but ignored; step sizes are now
        controlled by PROBE_STEPS per axis.
        """
        self._greedy_pass()


# =============================================================================
# Flask web application
# =============================================================================

app   = Flask(__name__)
ctrl  = AlignmentController()

# ── HTML template (single-file, no external assets) ───────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Laser Alignment</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh}

#statusbar{
  padding:14px 24px;font-size:1.05em;font-weight:700;letter-spacing:.5px;
  transition:background .4s,color .4s;
}
.s-idle      {background:#161b22;color:#7ee787;border-bottom:2px solid #238636}
.s-optimizing{background:#0d419d;color:#79c0ff;border-bottom:2px solid #388bfd}
.s-stable    {background:#1a4a1a;color:#7ee787;border-bottom:2px solid #238636}
.s-paused    {background:#4a1a00;color:#ffa657;border-bottom:2px solid #d29922}
.s-emergency {background:#6e0000;color:#ff7b72;border-bottom:2px solid #f85149;animation:blink 1s step-start infinite}
@keyframes blink{50%{opacity:.4}}

.wrap{max-width:920px;margin:20px auto;padding:0 16px;display:flex;flex-direction:column;gap:14px}

.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px}
.card h2{font-size:.75em;letter-spacing:1.5px;text-transform:uppercase;
         color:#8b949e;margin-bottom:14px;font-weight:600}

/* Voltage display */
.volt-row{display:flex;gap:12px}
.vbox{flex:1;text-align:center;background:#0d1117;border:1px solid #30363d;
      border-radius:8px;padding:14px 10px}
.vbox .lbl{font-size:.7em;letter-spacing:1px;color:#8b949e;margin-bottom:6px;text-transform:uppercase}
.vbox .val{font-size:2em;font-weight:700;font-family:'Courier New',monospace}
#v-curr{color:#58a6ff}
#v-best{color:#56d364}
#v-tgt {color:#ffa657}

/* Buttons */
.btn-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
button{padding:9px 20px;border:none;border-radius:6px;cursor:pointer;
       font-size:.9em;font-weight:600;transition:filter .15s;white-space:nowrap}
button:hover{filter:brightness(1.15)}
button:disabled{filter:brightness(.4);cursor:not-allowed}
#btn-opt  {background:#1f6feb;color:#fff}
#btn-stb  {background:#238636;color:#fff}
#btn-stop {background:#da3633;color:#fff}
#btn-snap {background:#6e40c9;color:#fff}
#btn-zero {background:#30363d;color:#e6edf3}

/* Target */
.tgt-row{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.tgt-row label{font-size:.85em;color:#8b949e;min-width:130px}
.tgt-row input{width:120px;padding:7px 10px;border-radius:5px;
               border:1px solid #30363d;background:#0d1117;
               color:#e6edf3;font-size:1em;font-family:monospace}
#status-msg{margin-top:12px;font-size:.85em;color:#8b949e;min-height:1.2em}

/* Positions */
.pos-row{display:flex;gap:10px;flex-wrap:wrap}
.pbox{flex:1;min-width:80px;background:#0d1117;border:1px solid #30363d;
      border-radius:6px;padding:10px;text-align:center}
.pbox .plbl{font-size:.68em;color:#8b949e;text-transform:uppercase;letter-spacing:.5px}
.pbox .pval{font-size:1.25em;font-family:monospace;color:#e6edf3;margin-top:4px}

/* Jog */
.jog-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px}
.jog-col{display:flex;flex-direction:column;align-items:stretch;gap:5px}
.jog-col .jlbl{font-size:.72em;color:#8b949e;text-align:center;margin-bottom:2px}
.jog-col button{background:#21262d;color:#e6edf3;padding:8px 4px;font-size:.82em}
.jog-steps{display:flex;align-items:center;gap:10px;margin-bottom:4px}
.jog-steps label{font-size:.82em;color:#8b949e}
.jog-steps input{width:90px;padding:6px 8px;border-radius:4px;
                 border:1px solid #30363d;background:#0d1117;color:#e6edf3;font-family:monospace}

/* Log */
#logbox{background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:10px;
        height:220px;overflow-y:auto;font-family:'Courier New',monospace;
        font-size:.75em;color:#8b949e;white-space:pre-wrap;line-height:1.5}
.la{color:#56d364} /* accept */
.lr{color:#f85149} /* reject/revert */
.li{color:#58a6ff} /* info */
</style>
</head>
<body>

<div id="statusbar" class="s-idle">● IDLE</div>

<div class="wrap">

  <!-- Voltage readout -->
  <div class="card">
    <h2>Photodiode Signal</h2>
    <div class="volt-row">
      <div class="vbox"><div class="lbl">Current</div><div class="val" id="v-curr">—</div></div>
      <div class="vbox"><div class="lbl">Run Best</div><div class="val" id="v-best">—</div></div>
      <div class="vbox"><div class="lbl">Target</div><div class="val" id="v-tgt">—</div></div>
    </div>
  </div>

  <!-- Control -->
  <div class="card">
    <h2>Control</h2>
    <div class="tgt-row">
      <label>Target voltage (V)</label>
      <input type="number" id="target-v" value="2.25" step="0.005" min="0" max="6.144">
    </div>
    <div class="btn-row">
      <button id="btn-opt"  onclick="startOptimize()">▶ Optimize</button>
      <button id="btn-stb"  onclick="startStable()">⟳ Stable Hold</button>
      <button id="btn-stop" onclick="doStop()">■ Stop</button>
      <button id="btn-snap" onclick="snapBest()" title="Snapshot current position as run-best anchor">★ Snapshot Best</button>
      <button id="btn-zero" onclick="zeroPosn()" title="Zero step counters (does not move motors)">○ Zero Counters</button>
    </div>
    <p id="status-msg">Idle</p>
    <p id="log-dir" style="font-size:.75em;color:#484f58;font-family:monospace;margin-top:6px"></p>
  </div>

  <!-- Motor positions -->
  <div class="card">
    <h2>Motor Positions (steps from zero)</h2>
    <div class="pos-row">
      <div class="pbox"><div class="plbl">M1 — 2nd Mirror Y</div><div class="pval" id="p1">0</div></div>
      <div class="pbox"><div class="plbl">M2 — 2nd Mirror X</div><div class="pval" id="p2">0</div></div>
      <div class="pbox"><div class="plbl">M3 — 1st Mirror Y</div><div class="pval" id="p3">0</div></div>
      <div class="pbox"><div class="plbl">M4 — 1st Mirror X</div><div class="pval" id="p4">0</div></div>
    </div>
  </div>

  <!-- Manual jog -->
  <div class="card">
    <h2>Manual Jog &nbsp;<small style="color:#da3633;font-size:.8em">(Idle only)</small></h2>
    <div class="jog-steps">
      <label>Steps per jog:</label>
      <input type="number" id="jog-steps" value="100" min="1" max="10000">
    </div>
    <div class="jog-grid">
      <div class="jog-col">
        <div class="jlbl">M1<br>2nd Mirror Y</div>
        <button onclick="jog(1,1)">▲ CW</button>
        <button onclick="jog(1,0)">▼ CCW</button>
      </div>
      <div class="jog-col">
        <div class="jlbl">M2<br>2nd Mirror X</div>
        <button onclick="jog(2,1)">▶ CW</button>
        <button onclick="jog(2,0)">◀ CCW</button>
      </div>
      <div class="jog-col">
        <div class="jlbl">M3<br>1st Mirror Y</div>
        <button onclick="jog(3,1)">▲ CW</button>
        <button onclick="jog(3,0)">▼ CCW</button>
      </div>
      <div class="jog-col">
        <div class="jlbl">M4<br>1st Mirror X</div>
        <button onclick="jog(4,1)">▶ CW</button>
        <button onclick="jog(4,0)">◀ CCW</button>
      </div>
    </div>
  </div>

  <!-- Log -->
  <div class="card">
    <h2>Event Log</h2>
    <div id="logbox"></div>
  </div>

</div><!-- /wrap -->

<script>
const STATE_LABELS = {
  idle:       '● IDLE',
  optimizing: '▶ OPTIMIZING',
  stable:     '⟳ STABLE HOLD',
  paused:     '⏸ PAUSED — waiting for beam recovery',
  emergency:  '🚨 EMERGENCY — beam blocked too long — restart required',
};
const STATE_CSS = {
  idle: 's-idle', optimizing: 's-optimizing', stable: 's-stable',
  paused: 's-paused', emergency: 's-emergency'
};

let lastLogLen = 0;

async function poll() {
  try {
    const d = await fetch('/status').then(r => r.json());

    document.getElementById('v-curr').textContent = d.current_voltage.toFixed(4)   + ' V';
    document.getElementById('v-best').textContent = d.run_best_voltage.toFixed(4)  + ' V';
    document.getElementById('v-tgt' ).textContent = d.target_voltage.toFixed(4)    + ' V';

    document.getElementById('p1').textContent = d.positions[1] ?? 0;
    document.getElementById('p2').textContent = d.positions[2] ?? 0;
    document.getElementById('p3').textContent = d.positions[3] ?? 0;
    document.getElementById('p4').textContent = d.positions[4] ?? 0;

    document.getElementById('status-msg').textContent = d.status_msg;

    const sb = document.getElementById('statusbar');
    sb.textContent = STATE_LABELS[d.state] || d.state.toUpperCase();
    sb.className   = STATE_CSS[d.state]    || 's-idle';

    const running = d.state !== 'idle' && d.state !== 'emergency';
    document.getElementById('btn-opt' ).disabled = running;
    document.getElementById('btn-stb' ).disabled = running;
    document.getElementById('btn-snap').disabled = running;
    document.getElementById('btn-zero').disabled = running;
    if (d.last_log_dir) {
      document.getElementById('log-dir').textContent = 'Log: ' + d.last_log_dir;
    }

    if (d.log.length !== lastLogLen) {
      lastLogLen = d.log.length;
      const box = document.getElementById('logbox');
      box.innerHTML = d.log.map(l => {
        if (l.includes('ACCEPTED'))              return `<span class="la">${esc(l)}</span>`;
        if (l.includes('REJECT') || l.includes('revert') || l.includes('failed'))
                                                 return `<span class="lr">${esc(l)}</span>`;
        if (l.includes('Optimize') || l.includes('Stable') || l.includes('Packet'))
                                                 return `<span class="li">${esc(l)}</span>`;
        return esc(l);
      }).join('\n');
      box.scrollTop = box.scrollHeight;
    }
  } catch(e) { /* network hiccup — ignore */ }
}

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function post(url, body={}) {
  const r = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  return r.json();
}

async function startOptimize() {
  const tv = parseFloat(document.getElementById('target-v').value);
  if (isNaN(tv)) { alert('Enter a valid target voltage'); return; }
  const d = await post('/optimize', {target: tv});
  document.getElementById('status-msg').textContent = d.message;
}
async function startStable() {
  const d = await post('/stable');
  document.getElementById('status-msg').textContent = d.message;
}
async function doStop() {
  await post('/stop');
}
async function snapBest() {
  await post('/snapshot_best');
}
async function zeroPosn() {
  if (!confirm('Reset step counters to zero? (No motor motion)')) return;
  await post('/zero_positions');
}
async function jog(motor, direction) {
  const steps = parseInt(document.getElementById('jog-steps').value) || 100;
  const d = await post('/jog', {motor, direction, steps});
  if (d.message) document.getElementById('status-msg').textContent = d.message;
}

setInterval(poll, 900);
poll();
</script>
</body>
</html>
"""


# =============================================================================
# Routes
# =============================================================================

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/status")
def route_status():
    return jsonify(ctrl.get_status())


@app.route("/optimize", methods=["POST"])
def route_optimize():
    data   = request.get_json(force=True) or {}
    target = float(data.get("target", ctrl.target_voltage))
    ok, msg = ctrl.start_optimize(target)
    return jsonify({"ok": ok, "message": msg})


@app.route("/stable", methods=["POST"])
def route_stable():
    ok, msg = ctrl.start_stable()
    return jsonify({"ok": ok, "message": msg})


@app.route("/stop", methods=["POST"])
def route_stop():
    ctrl.stop()
    return jsonify({"ok": True, "message": "Stop requested"})


@app.route("/jog", methods=["POST"])
def route_jog():
    data      = request.get_json(force=True) or {}
    motor     = int(data.get("motor",     1))
    direction = int(data.get("direction", 1))
    steps     = int(data.get("steps",   100))
    ok, msg   = ctrl.jog(motor, direction, steps)
    return jsonify({"ok": ok, "message": msg})


@app.route("/snapshot_best", methods=["POST"])
def route_snapshot():
    ctrl.snapshot_best()
    return jsonify({"ok": True})


@app.route("/zero_positions", methods=["POST"])
def route_zero():
    ctrl.zero_positions()
    return jsonify({"ok": True})


@app.route("/logs")
def route_logs():
    """List all session log directories."""
    root = LOG_ROOT
    if not root.exists():
        return jsonify({"logs": []})
    dirs = sorted(
        [str(p) for p in root.iterdir() if p.is_dir()],
        reverse=True,
    )
    return jsonify({"logs": dirs[:20]})


@app.route("/logs/<path:subpath>")
def route_log_file(subpath: str):
    """Download a specific log file."""
    target = LOG_ROOT / subpath
    # Safety: must stay inside LOG_ROOT
    try:
        target.resolve().relative_to(LOG_ROOT.resolve())
    except ValueError:
        return jsonify({"error": "invalid path"}), 400
    if not target.is_file():
        return jsonify({"error": "not found"}), 404
    return send_file(str(target), as_attachment=True)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    log.info("=== Laser Alignment Controller starting ===")
    log.info(f"GPIO:   {'hardware' if _HAS_GPIO else 'SIMULATION'}")
    log.info(f"ADS1115:{'hardware' if _HAS_ADS  else 'SIMULATION'}")
    log.info("Open browser at  http://<pi-ip>:5000")
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        ctrl.motors.cleanup()
        log.info("GPIO cleaned up — bye")
