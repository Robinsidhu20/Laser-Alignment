#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import signal
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, Response, jsonify, request

try:
    from gpiozero import OutputDevice
except Exception:  # pragma: no cover
    OutputDevice = None  # type: ignore

try:
    import board
    import busio
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
except Exception:  # pragma: no cover
    board = None  # type: ignore
    busio = None  # type: ignore
    ADS = None  # type: ignore
    AnalogIn = None  # type: ignore


LOG_ROOT = Path("logs")
STATE_PATH = Path("target_manager_state.json")


@dataclass
class MotorPins:
    label: str
    step: int
    direction: int
    enable: int


@dataclass
class ControllerConfig:
    # Wiring
    motor_map: Dict[str, MotorPins] = field(default_factory=lambda: {
        "M1_mirror2_y": MotorPins("M1_mirror2_y", 17, 27, 5),
        "M2_mirror2_x": MotorPins("M2_mirror2_x", 20, 21, 26),
        "M3_mirror1_y": MotorPins("M3_mirror1_y", 12, 16, 25),
        "M4_mirror1_x": MotorPins("M4_mirror1_x", 24, 23, 18),
    })
    enable_active_low: bool = True
    invert_dir: Dict[str, bool] = field(default_factory=lambda: {
        "M1_mirror2_y": False,
        "M2_mirror2_x": False,
        "M3_mirror1_y": False,
        "M4_mirror1_x": False,
    })
    motor_limits: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "M1_mirror2_y": (-500, 500),
        "M2_mirror2_x": (-500, 500),
        "M3_mirror1_y": (-500, 500),
        "M4_mirror1_x": (-500, 500),
    })

    # Motion timing
    pulse_high_s: float = 0.004
    pulse_low_s: float = 0.004
    settle_s: float = 0.45

    # Photodiode
    ads_address: int = 0x4B
    ads_channel: int = 0
    ads_gain: float = 2.0 / 3.0
    burst_samples: int = 15
    burst_sample_delay_s: float = 0.02
    adc_clip_voltage: float = 3.20

    # Optimize mode
    step_schedule_far: Tuple[int, ...] = (8, 4, 2, 1)
    step_schedule_near: Tuple[int, ...] = (2, 1, 1)
    target_complete_band_v: float = 0.01
    near_target_window_v: float = 0.04
    accept_delta_v: float = 0.003
    best_update_epsilon_v: float = 0.0008
    no_progress_limit: int = 20
    max_optimize_cycles: int = 400

    # Stable mode
    stable_drop_trigger_v: float = 0.025
    stable_restore_pause_s: float = 0.6
    stable_monitor_every_s: float = 4.0
    stable_micro_steps: Tuple[int, ...] = (1,)

    # Safety / emergency
    blocked_low_ratio: float = 0.35
    blocked_low_absolute_v: float = 0.08
    blocked_consecutive_reads: int = 3
    sudden_drop_v: float = 0.20
    emergency_recover_attempts: int = 1


@dataclass
class TrialRow:
    timestamp: float
    iso_time: str
    mode: str
    phase: str
    axis_pair: str
    pattern: str
    step_size: int
    baseline_v: float
    trial_v: float
    recovery_v: float
    kept: int
    reverted: int
    best_v: float
    target_v: float
    positions_json: str
    note: str


@dataclass
class SavedState:
    target_voltage: Optional[float] = None
    best_voltage: Optional[float] = None
    best_positions: Dict[str, int] = field(default_factory=dict)
    saved_at: Optional[str] = None


class SessionLogger:
    def __init__(self) -> None:
        self.run_dir = LOG_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trial_csv = self.run_dir / "trial_log.csv"
        self.trace_csv = self.run_dir / "voltage_trace.csv"
        self.summary_json = self.run_dir / "summary.json"
        self.trials: List[TrialRow] = []
        self.trace: List[Tuple[float, str, float, str]] = []

    def log_trial(self, row: TrialRow) -> None:
        self.trials.append(row)
        print(
            f"[{row.iso_time}] {row.mode:<9} {row.axis_pair:<10} {row.pattern:<18} "
            f"{row.baseline_v:.5f}->{row.trial_v:.5f} keep={bool(row.kept)} best={row.best_v:.5f} {row.note}"
        )

    def log_voltage(self, label: str, voltage: float, positions: Dict[str, int]) -> None:
        self.trace.append((time.time(), label, voltage, json.dumps(positions)))

    def flush(self, summary: Dict[str, Any]) -> None:
        with self.trial_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(TrialRow(0.0, "", "", "", "", "", 0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, "", "")).keys()))
            writer.writeheader()
            for row in self.trials:
                writer.writerow(asdict(row))
        with self.trace_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "label", "voltage", "positions_json"])
            writer.writerows(self.trace)
        self.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


class StepperMotor:
    def __init__(self, pins: MotorPins, cfg: ControllerConfig) -> None:
        if OutputDevice is None:
            raise RuntimeError("gpiozero is not installed")
        self.cfg = cfg
        self.pins = pins
        self.step_pin = OutputDevice(pins.step)
        self.dir_pin = OutputDevice(pins.direction)
        self.en_pin = OutputDevice(pins.enable)
        self.position = 0
        self.closed = False
        self.disable()

    def enable(self) -> None:
        if not self.closed:
            self.en_pin.value = 0 if self.cfg.enable_active_low else 1

    def disable(self) -> None:
        if self.closed:
            return
        self.en_pin.value = 1 if self.cfg.enable_active_low else 0
        self.step_pin.value = 0
        self.dir_pin.value = 0

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.disable()
        finally:
            for obj in (self.step_pin, self.dir_pin, self.en_pin):
                try:
                    obj.close()
                except Exception:
                    pass
            self.closed = True

    def within_limits(self, delta: int) -> bool:
        lo, hi = self.cfg.motor_limits[self.pins.label]
        target = self.position + delta
        return lo <= target <= hi

    def move(self, delta: int) -> bool:
        if self.closed:
            return False
        if delta == 0:
            return True
        if not self.within_limits(delta):
            return False
        forward = delta > 0
        if self.cfg.invert_dir.get(self.pins.label, False):
            forward = not forward
        self.enable()
        self.dir_pin.value = 1 if forward else 0
        for _ in range(abs(delta)):
            self.step_pin.value = 1
            time.sleep(self.cfg.pulse_high_s)
            self.step_pin.value = 0
            time.sleep(self.cfg.pulse_low_s)
        self.position += delta
        return True


class PairController:
    def __init__(self, label: str, m2: StepperMotor, m1: StepperMotor, cfg: ControllerConfig) -> None:
        self.label = label
        self.m2 = m2
        self.m1 = m1
        self.cfg = cfg

    def positions(self) -> Tuple[int, int]:
        return self.m2.position, self.m1.position

    def move(self, dm2: int, dm1: int) -> bool:
        if not self.m2.within_limits(dm2) or not self.m1.within_limits(dm1):
            return False
        if dm2 == 0 and dm1 == 0:
            return True

        m2_forward = dm2 > 0
        m1_forward = dm1 > 0
        if self.cfg.invert_dir.get(self.m2.pins.label, False):
            m2_forward = not m2_forward
        if self.cfg.invert_dir.get(self.m1.pins.label, False):
            m1_forward = not m1_forward

        self.m2.enable()
        self.m1.enable()
        if dm2 != 0:
            self.m2.dir_pin.value = 1 if m2_forward else 0
        if dm1 != 0:
            self.m1.dir_pin.value = 1 if m1_forward else 0

        n2 = abs(dm2)
        n1 = abs(dm1)
        n = max(n2, n1)
        for i in range(n):
            if i < n2:
                self.m2.step_pin.value = 1
            if i < n1:
                self.m1.step_pin.value = 1
            time.sleep(self.cfg.pulse_high_s)
            if i < n2:
                self.m2.step_pin.value = 0
            if i < n1:
                self.m1.step_pin.value = 0
            time.sleep(self.cfg.pulse_low_s)

        self.m2.position += dm2
        self.m1.position += dm1
        return True


class Photodiode:
    def __init__(self, cfg: ControllerConfig) -> None:
        if board is None or busio is None or ADS is None or AnalogIn is None:
            raise RuntimeError("Adafruit ADS1x15 libraries are not installed")
        self.cfg = cfg
        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS.ADS1115(i2c, address=cfg.ads_address)
        ads.gain = cfg.ads_gain
        self.chan = AnalogIn(ads, cfg.ads_channel)

    def read_filtered(self) -> Tuple[float, float, List[float]]:
        vals: List[float] = []
        for _ in range(self.cfg.burst_samples):
            vals.append(float(self.chan.voltage))
            time.sleep(self.cfg.burst_sample_delay_s)
        mean_v = statistics.fmean(vals)
        std_v = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return mean_v, std_v, vals


class TargetController:
    def __init__(self, cfg: ControllerConfig, logger: SessionLogger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.stop_requested = False
        self.mode = "IDLE"
        self.target_voltage: Optional[float] = None
        self.best_voltage = -1e9
        self.best_positions: Dict[str, int] = {}
        self.current_voltage = -1e9
        self.current_positions: Dict[str, int] = {}
        self.no_progress_cycles = 0
        self.alert: Optional[Dict[str, str]] = None
        self.last_status_message = "Idle"
        self.low_signal_count = 0

        self.motors = {name: StepperMotor(pins, cfg) for name, pins in cfg.motor_map.items()}
        self.vertical = PairController("vertical", self.motors["M1_mirror2_y"], self.motors["M3_mirror1_y"], cfg)
        self.horizontal = PairController("horizontal", self.motors["M2_mirror2_x"], self.motors["M4_mirror1_x"], cfg)
        self.diode = Photodiode(cfg)
        self._load_state()

    def positions(self) -> Dict[str, int]:
        return {k: m.position for k, m in self.motors.items()}

    def close(self) -> None:
        for motor in self.motors.values():
            motor.close()

    def request_stop(self) -> None:
        self.stop_requested = True

    def _load_state(self) -> None:
        if not STATE_PATH.exists():
            return
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            self.target_voltage = data.get("target_voltage")
            bv = data.get("best_voltage")
            if isinstance(bv, (int, float)):
                self.best_voltage = float(bv)
            bp = data.get("best_positions")
            if isinstance(bp, dict):
                self.best_positions = {str(k): int(v) for k, v in bp.items()}
        except Exception:
            pass

    def save_state(self) -> None:
        data = asdict(SavedState(
            target_voltage=self.target_voltage,
            best_voltage=(self.best_voltage if self.best_voltage > -1e8 else None),
            best_positions=self.best_positions,
            saved_at=datetime.now().isoformat(timespec="seconds"),
        ))
        STATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def set_target(self, target: float) -> None:
        self.target_voltage = float(target)
        self.save_state()

    def set_alert(self, level: str, message: str) -> None:
        self.alert = {"level": level, "message": message}
        self.last_status_message = message

    def clear_alert(self) -> None:
        self.alert = None

    def read_pd(self, label: str) -> Tuple[float, float]:
        mean_v, std_v, _ = self.diode.read_filtered()
        self.current_voltage = mean_v
        self.current_positions = self.positions()
        self.logger.log_voltage(label, mean_v, self.current_positions)
        return mean_v, std_v

    def update_best_if_needed(self, voltage: float, positions: Dict[str, int], reason: str = "") -> None:
        if voltage > self.best_voltage:
            self.best_voltage = voltage
            self.best_positions = dict(positions)
            self.save_state()
            if reason:
                self.last_status_message = reason

    def blocked_floor(self) -> float:
        if self.target_voltage is None:
            return self.cfg.blocked_low_absolute_v
        return max(self.cfg.blocked_low_absolute_v, self.target_voltage * self.cfg.blocked_low_ratio)

    def check_emergency_conditions(self, voltage: float, previous_voltage: Optional[float]) -> Optional[str]:
        if voltage >= self.cfg.adc_clip_voltage:
            return f"ADC close to clipping at {voltage:.3f} V"
        if voltage < self.blocked_floor():
            self.low_signal_count += 1
        else:
            self.low_signal_count = 0
        if self.low_signal_count >= self.cfg.blocked_consecutive_reads:
            return f"Signal too low for {self.low_signal_count} consecutive checks; possible beam block or laser off"
        if previous_voltage is not None and previous_voltage - voltage >= self.cfg.sudden_drop_v:
            return f"Sudden voltage drop detected ({previous_voltage:.3f} -> {voltage:.3f} V)"
        return None

    def restore_best_positions(self) -> bool:
        if not self.best_positions:
            return False
        target = dict(self.best_positions)
        current = self.positions()
        for pair_name, pair in (("vertical", self.vertical), ("horizontal", self.horizontal)):
            if pair_name == "vertical":
                m2_label, m1_label = "M1_mirror2_y", "M3_mirror1_y"
            else:
                m2_label, m1_label = "M2_mirror2_x", "M4_mirror1_x"
            dm2 = target.get(m2_label, 0) - current.get(m2_label, 0)
            dm1 = target.get(m1_label, 0) - current.get(m1_label, 0)
            while dm2 != 0 or dm1 != 0:
                if self.stop_requested:
                    return False
                step2 = 0 if dm2 == 0 else (1 if dm2 > 0 else -1)
                step1 = 0 if dm1 == 0 else (1 if dm1 > 0 else -1)
                if not pair.move(step2, step1):
                    return False
                dm2 -= step2
                dm1 -= step1
                time.sleep(self.cfg.settle_s / 3)
        self.current_positions = self.positions()
        return True

    @staticmethod
    def paired_patterns(step: int, axis_label: str) -> List[Tuple[int, int, str]]:
        return [
            ( step, -step, f"{axis_label}_m2+_m1-"),
            (-step,  step, f"{axis_label}_m2-_m1+"),
            ( step,  step, f"{axis_label}_m2+_m1+"),
            (-step, -step, f"{axis_label}_m2-_m1-"),
        ]

    def trial_pair_move(self, pair: PairController, axis_label: str, dm2: int, dm1: int, mode: str, phase: str) -> Tuple[bool, float, float, str]:
        baseline_v, baseline_std = self.read_pd("baseline")
        baseline_positions = self.positions()
        prev_v = baseline_v
        if not pair.move(dm2, dm1):
            self.logger.log_trial(TrialRow(time.time(), datetime.now().isoformat(timespec="seconds"), mode, phase, axis_label, "blocked", max(abs(dm2), abs(dm1)), baseline_v, baseline_v, baseline_v, 0, 0, self.best_voltage, self.target_voltage or -1.0, json.dumps(baseline_positions), "blocked_by_limit"))
            return False, baseline_v, baseline_std, "blocked_by_limit"
        time.sleep(self.cfg.settle_s)
        trial_v, trial_std = self.read_pd("trial")
        trial_positions = self.positions()

        emergency = self.check_emergency_conditions(trial_v, prev_v)
        if emergency:
            pair.move(-dm2, -dm1)
            time.sleep(self.cfg.settle_s)
            recovery_v, _ = self.read_pd("recovery")
            self.mode = "EMERGENCY"
            self.set_alert("emergency", emergency)
            self.logger.log_trial(TrialRow(time.time(), datetime.now().isoformat(timespec="seconds"), mode, phase, axis_label, f"{axis_label}_emergency", max(abs(dm2), abs(dm1)), baseline_v, trial_v, recovery_v, 0, 1, self.best_voltage, self.target_voltage or -1.0, json.dumps(self.positions()), f"emergency:{emergency}"))
            return False, recovery_v, max(trial_std, baseline_std), emergency

        delta = trial_v - baseline_v
        target = self.target_voltage if self.target_voltage is not None else baseline_v
        near_target = (target - max(trial_v, baseline_v)) <= self.cfg.near_target_window_v

        keep = False
        note = "reverted"
        if trial_v >= target - self.cfg.target_complete_band_v:
            keep = True
            note = "target_reached"
        elif delta > self.cfg.accept_delta_v:
            keep = True
            note = "improved"
        elif near_target and trial_v > self.best_voltage + self.cfg.best_update_epsilon_v:
            keep = True
            note = "borderline_new_best_near_target"

        recovery_v = trial_v
        if keep:
            self.current_voltage = trial_v
            self.current_positions = dict(trial_positions)
            self.update_best_if_needed(trial_v, trial_positions, reason=f"New best {trial_v:.5f} V")
            self.logger.log_trial(TrialRow(time.time(), datetime.now().isoformat(timespec="seconds"), mode, phase, axis_label, f"{axis_label}:{dm2},{dm1}", max(abs(dm2), abs(dm1)), baseline_v, trial_v, trial_v, 1, 0, self.best_voltage, target, json.dumps(trial_positions), note))
            return True, trial_v, max(trial_std, baseline_std), note

        pair.move(-dm2, -dm1)
        time.sleep(self.cfg.settle_s)
        recovery_v, recovery_std = self.read_pd("recovery")
        self.current_voltage = recovery_v
        self.current_positions = self.positions()
        self.update_best_if_needed(recovery_v, self.current_positions)
        self.logger.log_trial(TrialRow(time.time(), datetime.now().isoformat(timespec="seconds"), mode, phase, axis_label, f"{axis_label}:{dm2},{dm1}", max(abs(dm2), abs(dm1)), baseline_v, trial_v, recovery_v, 0, 1, self.best_voltage, target, json.dumps(self.current_positions), note))
        return False, recovery_v, max(recovery_std, trial_std, baseline_std), note

    def complete_optimize(self) -> None:
        self.mode = "COMPLETE"
        self.set_alert("complete", f"Optimize mode complete. Target {self.target_voltage:.3f} V reached.")
        for motor in self.motors.values():
            motor.disable()

    def emergency_stop(self, reason: str) -> None:
        self.mode = "EMERGENCY"
        for motor in self.motors.values():
            motor.disable()
        self.set_alert("emergency", reason)
        self.save_state()

    def optimize_loop(self) -> Dict[str, Any]:
        if self.target_voltage is None:
            raise RuntimeError("Set a target voltage first")
        self.mode = "OPTIMIZE"
        self.clear_alert()
        self.no_progress_cycles = 0

        start_v, _ = self.read_pd("optimize_start")
        self.update_best_if_needed(start_v, self.positions())
        if self.best_positions and self.best_voltage >= self.target_voltage - self.cfg.target_complete_band_v:
            self.complete_optimize()
            return self.summary()

        for cycle in range(self.cfg.max_optimize_cycles):
            if self.stop_requested:
                raise KeyboardInterrupt("Stopped")
            baseline_v, _ = self.read_pd("cycle_start")
            self.update_best_if_needed(baseline_v, self.positions())
            if baseline_v >= self.target_voltage - self.cfg.target_complete_band_v:
                self.complete_optimize()
                return self.summary()

            previous_baseline = baseline_v
            emergency = self.check_emergency_conditions(baseline_v, None)
            if emergency:
                self.emergency_stop(emergency)
                return self.summary()

            steps = self.cfg.step_schedule_near if (self.target_voltage - baseline_v) <= self.cfg.near_target_window_v else self.cfg.step_schedule_far
            improved_this_cycle = False
            for step in steps:
                for pair in (self.vertical, self.horizontal):
                    for dm2, dm1, _label in self.paired_patterns(step, pair.label):
                        kept, new_v, _, note = self.trial_pair_move(pair, pair.label, dm2, dm1, mode="OPTIMIZE", phase=f"cycle_{cycle}")
                        if self.mode == "EMERGENCY":
                            return self.summary()
                        if kept:
                            improved_this_cycle = True
                            if new_v >= self.target_voltage - self.cfg.target_complete_band_v:
                                self.complete_optimize()
                                return self.summary()
                            break
                    if improved_this_cycle:
                        break
                if improved_this_cycle:
                    break

            end_v, _ = self.read_pd("cycle_end")
            self.update_best_if_needed(end_v, self.positions())
            if end_v >= self.target_voltage - self.cfg.target_complete_band_v:
                self.complete_optimize()
                return self.summary()

            if improved_this_cycle or (end_v - previous_baseline) > self.cfg.best_update_epsilon_v:
                self.no_progress_cycles = 0
                self.last_status_message = f"Optimize running. Current {end_v:.5f} V, best {self.best_voltage:.5f} V"
            else:
                self.no_progress_cycles += 1
                self.last_status_message = f"No progress in cycle {cycle}. Current {end_v:.5f} V, best {self.best_voltage:.5f} V"

            if self.no_progress_cycles >= self.cfg.no_progress_limit:
                if self.best_positions:
                    self.restore_best_positions()
                    time.sleep(self.cfg.settle_s)
                    hold_v, _ = self.read_pd("restore_best")
                    self.update_best_if_needed(hold_v, self.positions())
                self.set_alert("info", f"Optimize mode stalled. Best achieved this run: {self.best_voltage:.3f} V")
                self.mode = "IDLE"
                return self.summary()

        self.mode = "IDLE"
        self.set_alert("info", f"Optimize mode stopped after max cycles. Best achieved: {self.best_voltage:.3f} V")
        return self.summary()

    def stable_loop(self) -> Dict[str, Any]:
        if self.target_voltage is None:
            raise RuntimeError("Set a target voltage first")
        self.mode = "STABLE"
        self.clear_alert()
        previous_v: Optional[float] = None
        restore_attempts = 0
        while not self.stop_requested and self.mode == "STABLE":
            current_v, _ = self.read_pd("stable_monitor")
            self.update_best_if_needed(current_v, self.positions())
            emergency = self.check_emergency_conditions(current_v, previous_v)
            if emergency:
                restored = False
                if restore_attempts < self.cfg.emergency_recover_attempts and self.best_positions:
                    restore_attempts += 1
                    restored = self.restore_best_positions()
                    time.sleep(self.cfg.settle_s)
                    current_v, _ = self.read_pd("stable_post_restore")
                    self.update_best_if_needed(current_v, self.positions())
                    if current_v >= self.best_voltage - self.cfg.stable_drop_trigger_v:
                        self.last_status_message = f"Recovered in stable mode to {current_v:.5f} V"
                        previous_v = current_v
                        time.sleep(self.cfg.stable_monitor_every_s)
                        continue
                self.emergency_stop(emergency if not restored else f"Restore failed after emergency condition: {emergency}")
                break

            restore_attempts = 0
            previous_v = current_v
            if self.best_voltage > -1e8 and current_v < self.best_voltage - self.cfg.stable_drop_trigger_v:
                self.last_status_message = f"Stable mode restoring best {self.best_voltage:.5f} V"
                if self.restore_best_positions():
                    time.sleep(self.cfg.settle_s)
                    restore_v, _ = self.read_pd("stable_restore")
                    self.update_best_if_needed(restore_v, self.positions())
                    if restore_v < self.best_voltage - self.cfg.stable_drop_trigger_v:
                        for step in self.cfg.stable_micro_steps:
                            improved = False
                            for pair in (self.vertical, self.horizontal):
                                for dm2, dm1, _label in self.paired_patterns(step, pair.label):
                                    kept, _, _, _ = self.trial_pair_move(pair, pair.label, dm2, dm1, mode="STABLE", phase="micro_recover")
                                    if self.mode == "EMERGENCY":
                                        return self.summary()
                                    if kept:
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                else:
                    self.emergency_stop("Could not restore best position in stable mode")
                    break
            else:
                self.last_status_message = f"Stable mode holding. Current {current_v:.5f} V, best {self.best_voltage:.5f} V"
            time.sleep(self.cfg.stable_monitor_every_s)
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "target_voltage": self.target_voltage,
            "best_voltage": None if self.best_voltage < -1e8 else self.best_voltage,
            "current_voltage": None if self.current_voltage < -1e8 else self.current_voltage,
            "positions": self.positions(),
            "best_positions": self.best_positions,
            "alert": self.alert,
            "message": self.last_status_message,
        }


HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Laser Target Manager</title>
<style>
  body { font-family: Arial, sans-serif; margin: 24px; background: #111827; color: #e5e7eb; }
  .card { border-radius: 14px; padding: 16px; margin-bottom: 16px; border: 1px solid #374151; background: #1f2937; }
  .row { display:flex; gap:16px; flex-wrap:wrap; }
  .col { flex:1 1 280px; }
  input, button { padding: 8px 10px; border-radius: 8px; border:1px solid #4b5563; background:#111827; color:#e5e7eb; }
  button { cursor:pointer; margin-right:8px; }
  .status-idle { background:#334155; }
  .status-optimize { background:#3b2f0e; }
  .status-stable { background:#114b5f; }
  .status-complete { background:#14532d; }
  .status-emergency { background:#7f1d1d; }
  .status-manual { background:#4338ca; }
  #statusbox { transition: background 0.2s ease; }
  pre { white-space: pre-wrap; word-wrap: break-word; }
  .small { color:#9ca3af; font-size: 0.95rem; }
</style>
</head>
<body>
<h1>Laser Target Manager</h1>
<div id="statusbox" class="card status-idle">
  <h2 id="modeTitle">Idle</h2>
  <div id="message">Waiting</div>
  <pre id="statusPre">Loading...</pre>
</div>
<div class="card">
  <div class="row">
    <div class="col">
      <label>Target Voltage (V)<br><input id="targetVoltage" type="number" step="0.001" value="2.200"></label>
      <div class="small">Operator-entered known achievable value for this setup.</div>
    </div>
    <div class="col">
      <label>Settle Time (s)<br><input id="settle" type="number" step="0.01" value="0.45"></label>
      <label>Pulse High (s)<br><input id="pulseHigh" type="number" step="0.001" value="0.004"></label>
      <label>Pulse Low (s)<br><input id="pulseLow" type="number" step="0.001" value="0.004"></label>
    </div>
  </div>
  <div style="margin-top:12px;">
    <button onclick="startOptimize()">Start Optimize</button>
    <button onclick="startStable()">Start Stable</button>
    <button onclick="stopRun()">Stop</button>
    <button onclick="ackAlert()">Acknowledge Alert</button>
  </div>
</div>
<div class="card">
  <h3>Manual Jog</h3>
  <label>Motor Label<br><input id="motorLabel" value="M1_mirror2_y"></label>
  <label>Steps<br><input id="motorSteps" type="number" value="1"></label>
  <div style="margin-top:12px;"><button onclick="manualJog()">Send Jog</button></div>
</div>
<script>
let beepTimer = null;
let audioCtx = null;
let alertActive = false;

function payloadBase() {
  return {
    target_voltage: parseFloat(document.getElementById('targetVoltage').value),
    settle: parseFloat(document.getElementById('settle').value),
    pulse_high: parseFloat(document.getElementById('pulseHigh').value),
    pulse_low: parseFloat(document.getElementById('pulseLow').value),
  };
}

function ensureAudio() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
}

function singleBeep(frequency, durationMs) {
  ensureAudio();
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = 'sine';
  osc.frequency.value = frequency;
  gain.gain.value = 0.04;
  osc.connect(gain);
  gain.connect(audioCtx.destination);
  osc.start();
  setTimeout(() => { osc.stop(); }, durationMs);
}

function startBeeping(level) {
  if (beepTimer) return;
  alertActive = true;
  const freq = level === 'emergency' ? 880 : 660;
  beepTimer = setInterval(() => {
    singleBeep(freq, 180);
    setTimeout(() => singleBeep(freq, 180), 260);
  }, 1100);
}

function stopBeeping() {
  alertActive = false;
  if (beepTimer) {
    clearInterval(beepTimer);
    beepTimer = null;
  }
}

async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload || {})
  });
  return res.json();
}

async function startOptimize() { await postJSON('/api/start_optimize', payloadBase()); }
async function startStable() { await postJSON('/api/start_stable', payloadBase()); }
async function stopRun() { await postJSON('/api/stop', {}); }
async function ackAlert() { await postJSON('/api/ack_alert', {}); stopBeeping(); }
async function manualJog() {
  const p = payloadBase();
  p.motor_label = document.getElementById('motorLabel').value;
  p.steps = parseInt(document.getElementById('motorSteps').value, 10);
  await postJSON('/api/manual_jog', p);
}

async function refreshStatus() {
  const res = await fetch('/api/status');
  const data = await res.json();
  document.getElementById('statusPre').textContent = JSON.stringify(data, null, 2);
  document.getElementById('modeTitle').textContent = data.mode || 'Unknown';
  document.getElementById('message').textContent = data.message || '';
  const box = document.getElementById('statusbox');
  box.className = 'card status-' + String(data.mode || 'idle').toLowerCase();
  if (data.alert && (data.alert.level === 'complete' || data.alert.level === 'emergency')) {
    if (!alertActive) startBeeping(data.alert.level);
  } else if (!data.alert || data.alert.level !== 'info') {
    stopBeeping();
  }
}

document.addEventListener('mousemove', () => { if (alertActive) ackAlert(); }, {passive:true});
document.addEventListener('click', () => { if (alertActive) ackAlert(); });
setInterval(refreshStatus, 1000);
refreshStatus();
</script>
</body>
</html>
"""


class AlignmentService:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.worker: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.logger: Optional[SessionLogger] = None
        self.controller: Optional[TargetController] = None
        self.status: Dict[str, Any] = {
            "mode": "IDLE",
            "message": "Idle",
            "alert": None,
            "target_voltage": None,
            "best_voltage": None,
            "current_voltage": None,
            "positions": {},
            "best_positions": {},
            "log_dir": None,
            "last_update": None,
        }

    def _set_status(self, **updates: Any) -> None:
        with self.lock:
            self.status.update(updates)
            self.status["last_update"] = time.time()

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.status)

    def _make_cfg(self, payload: Dict[str, Any]) -> ControllerConfig:
        cfg = ControllerConfig(
            pulse_high_s=float(payload.get("pulse_high", 0.004)),
            pulse_low_s=float(payload.get("pulse_low", 0.004)),
            settle_s=float(payload.get("settle", 0.45)),
        )
        return cfg

    def _publish_from_controller(self) -> None:
        if self.controller is None:
            return
        summary = self.controller.summary()
        self._set_status(
            mode=summary["mode"],
            message=summary["message"],
            alert=summary["alert"],
            target_voltage=summary["target_voltage"],
            best_voltage=summary["best_voltage"],
            current_voltage=summary["current_voltage"],
            positions=summary["positions"],
            best_positions=summary["best_positions"],
        )

    def _run_optimize(self, payload: Dict[str, Any]) -> None:
        try:
            self.stop_flag.clear()
            self.logger = SessionLogger()
            self.controller = TargetController(self._make_cfg(payload), self.logger)
            self.controller.set_target(float(payload["target_voltage"]))
            self._set_status(log_dir=str(self.logger.run_dir))
            original_read = self.controller.read_pd

            def wrapped_read(label: str):
                if self.stop_flag.is_set():
                    self.controller.request_stop()
                    raise KeyboardInterrupt("Stopped from UI")
                result = original_read(label)
                self._publish_from_controller()
                return result

            self.controller.read_pd = wrapped_read  # type: ignore
            result = self.controller.optimize_loop()
            self._publish_from_controller()
            self.logger.flush(result)
        except KeyboardInterrupt:
            if self.controller is not None:
                self.controller.mode = "IDLE"
                self.controller.last_status_message = "Stopped by user"
                self._publish_from_controller()
        except Exception as exc:
            self._set_status(mode="EMERGENCY", message=f"Error: {exc}", alert={"level": "emergency", "message": str(exc)})
            if self.logger is not None:
                self.logger.flush({"error": str(exc)})
        finally:
            try:
                if self.controller is not None:
                    self.controller.save_state()
                    self.controller.close()
            finally:
                self.controller = None

    def _run_stable(self, payload: Dict[str, Any]) -> None:
        try:
            self.stop_flag.clear()
            self.logger = SessionLogger()
            self.controller = TargetController(self._make_cfg(payload), self.logger)
            self.controller.set_target(float(payload["target_voltage"]))
            self._set_status(log_dir=str(self.logger.run_dir))
            original_read = self.controller.read_pd

            def wrapped_read(label: str):
                if self.stop_flag.is_set():
                    self.controller.request_stop()
                    raise KeyboardInterrupt("Stopped from UI")
                result = original_read(label)
                self._publish_from_controller()
                return result

            self.controller.read_pd = wrapped_read  # type: ignore
            result = self.controller.stable_loop()
            self._publish_from_controller()
            self.logger.flush(result)
        except KeyboardInterrupt:
            if self.controller is not None:
                self.controller.mode = "IDLE"
                self.controller.last_status_message = "Stopped by user"
                self._publish_from_controller()
        except Exception as exc:
            self._set_status(mode="EMERGENCY", message=f"Error: {exc}", alert={"level": "emergency", "message": str(exc)})
            if self.logger is not None:
                self.logger.flush({"error": str(exc)})
        finally:
            try:
                if self.controller is not None:
                    self.controller.save_state()
                    self.controller.close()
            finally:
                self.controller = None

    def start_optimize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError("A run is already active")
        self.worker = threading.Thread(target=self._run_optimize, args=(payload,), daemon=True)
        self.worker.start()
        return self.get_status()

    def start_stable(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError("A run is already active")
        self.worker = threading.Thread(target=self._run_stable, args=(payload,), daemon=True)
        self.worker.start()
        return self.get_status()

    def stop(self) -> Dict[str, Any]:
        self.stop_flag.set()
        if self.controller is not None:
            self.controller.request_stop()
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=3.0)
        self._set_status(mode="IDLE", message="Stop requested")
        return self.get_status()

    def manual_jog(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError("Stop the current run before manual jog")
        logger = SessionLogger()
        ctl = TargetController(self._make_cfg(payload), logger)
        try:
            motor_label = str(payload.get("motor_label"))
            steps = int(payload.get("steps", 0))
            if motor_label not in ctl.motors:
                raise ValueError(f"Unknown motor_label: {motor_label}")
            ok = ctl.motors[motor_label].move(steps)
            time.sleep(ctl.cfg.settle_s)
            v, _ = ctl.read_pd("manual_jog")
            ctl.update_best_if_needed(v, ctl.positions())
            ctl.mode = "MANUAL"
            ctl.last_status_message = f"Manual jog complete: {v:.5f} V"
            result = ctl.summary() | {"ok": ok}
            logger.flush(result)
            self._set_status(
                mode="MANUAL",
                message=ctl.last_status_message,
                current_voltage=v,
                best_voltage=ctl.best_voltage,
                positions=ctl.positions(),
                best_positions=ctl.best_positions,
                log_dir=str(logger.run_dir),
                alert=None,
            )
            return result
        finally:
            ctl.close()

    def ack_alert(self) -> Dict[str, Any]:
        if self.controller is not None:
            self.controller.clear_alert()
            self._publish_from_controller()
        else:
            self._set_status(alert=None)
        return self.get_status()


app = Flask(__name__)
service = AlignmentService()


@app.route("/")
def index() -> Response:
    return Response(HTML, mimetype="text/html")


@app.route("/api/status")
def api_status():
    return jsonify(service.get_status())


@app.route("/api/start_optimize", methods=["POST"])
def api_start_optimize():
    try:
        payload = request.get_json(force=True) or {}
        return jsonify(service.start_optimize(payload))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/start_stable", methods=["POST"])
def api_start_stable():
    try:
        payload = request.get_json(force=True) or {}
        return jsonify(service.start_stable(payload))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/stop", methods=["POST"])
def api_stop():
    try:
        return jsonify(service.stop())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/manual_jog", methods=["POST"])
def api_manual_jog():
    try:
        payload = request.get_json(force=True) or {}
        return jsonify(service.manual_jog(payload))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/ack_alert", methods=["POST"])
def api_ack_alert():
    try:
        return jsonify(service.ack_alert())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    def _handle_signal(signum, frame):
        service.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
