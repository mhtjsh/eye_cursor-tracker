from __future__ import annotations

import argparse
import sys

from .calibrate import run_calibration
from .debug import run_debug
from .doctor import run_doctor
from .runtime import run_cursor
from .train import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eye-cursor",
        description="Local laptop-webcam eye cursor prototype.",
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace directory for profiles and outputs. Default: current directory.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check Python, dependencies, screen, and camera.")
    doctor.add_argument("--camera", type=int, default=0)

    debug = sub.add_parser("debug", help="Show webcam tracking overlay.")
    debug.add_argument("--camera", type=int, default=0)
    debug.add_argument("--width", type=int, default=640)
    debug.add_argument("--height", type=int, default=480)

    cal = sub.add_parser("calibrate", help="Collect gaze calibration samples.")
    cal.add_argument("--profile", default="default")
    cal.add_argument("--camera", type=int, default=0)
    cal.add_argument("--width", type=int, default=640)
    cal.add_argument("--height", type=int, default=480)
    cal.add_argument("--grid-x", type=int, default=7)
    cal.add_argument("--grid-y", type=int, default=5)
    cal.add_argument("--rounds", type=int, default=8)
    cal.add_argument("--seconds-per-target", type=float, default=1.1)
    cal.add_argument("--settle-seconds", type=float, default=0.35)
    cal.add_argument("--margin", type=float, default=0.08)
    cal.add_argument("--save-frames", action="store_true")

    train = sub.add_parser("train", help="Train the personal gaze-to-screen model.")
    train.add_argument("--profile", default="default")
    train.add_argument("--calibration", default=None, help="Path to a calibration .npz file.")
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=7)

    run = sub.add_parser("run", help="Run real-time cursor control.")
    run.add_argument("--profile", default="default")
    run.add_argument("--camera", type=int, default=0)
    run.add_argument("--width", type=int, default=640)
    run.add_argument("--height", type=int, default=480)
    run.add_argument("--show", action="store_true", help="Show a webcam debug overlay.")
    run.add_argument("--min-cutoff", type=float, default=0.7)
    run.add_argument("--beta", type=float, default=0.05)
    run.add_argument("--dead-zone", type=float, default=2.0, help="Do not move for smaller pixel changes.")
    run.add_argument("--failsafe-margin", type=int, default=8)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "doctor":
            return run_doctor(args)
        if args.command == "debug":
            return run_debug(args)
        if args.command == "calibrate":
            return run_calibration(args)
        if args.command == "train":
            return run_training(args)
        if args.command == "run":
            return run_cursor(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2
