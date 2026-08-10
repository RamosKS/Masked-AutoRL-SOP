"""Interactive launcher for the Masked AutoRL-SOP implementations."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
VARIANTS_DIR = ROOT_DIR / "variants"

VARIANTS = [
    {
        "name": "[MULTIVARIATE] HYPERBAND TPE Masked AutoRL-SOP",
        "directory": "[MULTIVARIATE] HYPERBAND Masked AutoRL-SOP",
        "script": "[MULTIVARIATE] HYPERBAND TPE Masked AutoRL-SOP.py",
    },
    {
        "name": "[MULTIVARIATE] TPE Masked AutoRL-SOP",
        "directory": "[MULTIVARIATE] TPE Masked AutoRL-SOP",
        "script": "[MULTIVARIATE] TPE Masked AutoRL-SOP.py",
    },
    {
        "name": "[UNIVARIATE] HYPERBAND TPE Masked AutoRL-SOP",
        "directory": "[UNIVARIATE] HYPERBAND Masked AutoRL-SOP",
        "script": "[UNIVARIATE] HYPERBAND TPE Masked AutoRL-SOP.py",
    },
    {
        "name": "[UNIVARIATE] TPE Masked AutoRL-SOP",
        "directory": "[UNIVARIATE] TPE Masked AutoRL-SOP",
        "script": "[UNIVARIATE] TPE Masked AutoRL-SOP.py",
    },
    {
        "name": "HYPERBAND GP Masked AutoRL-SOP",
        "directory": "[MULTIVARIATE] HYPERBAND Masked AutoRL-SOP",
        "script": "HYPERBAND GP Masked AutoRL-SOP.py",
    },
    {
        "name": "GP Masked AutoRL-SOP",
        "directory": "GP Masked AutoRL-SOP",
        "script": "GP Masked AutoRL-SOP.py",
    },
    {
        "name": "NO_BAYESIAN Masked AutoRL-SOP",
        "directory": "NO_BAYESIAN Masked AutoRL-SOP",
        "script": "NO_BAYESIAN Masked AutoRL-SOP.py",
    },
    {
        "name": "[MULTIVARIATE] NO_MASK Masked AutoRL-SOP",
        "directory": "[MULTIVARIATE] NO_MASK Masked AutoRL-SOP",
        "script": "[MULTIVARIATE] NO_MASK Masked AutoRL-SOP.py",
    },
    {
        "name": "[UNIVARIATE] NO_MASK Masked AutoRL-SOP",
        "directory": "[UNIVARIATE] NO_MASK Masked AutoRL-SOP",
        "script": "[UNIVARIATE] NO_MASK Masked AutoRL-SOP.py",
    },
    {
        "name": "NO_MASK_NO_BAYESIAN Masked AutoRL-SOP",
        "directory": "NO_MASK_NO_BAYESIAN Masked AutoRL-SOP",
        "script": "NO_MASK_NO_BAYESIAN Masked AutoRL-SOP.py",
    },
    {
        "name": "RANDOM_SEARCH Masked AutoRL-SOP",
        "directory": "RANDOM_SEARCH Masked AutoRL-SOP",
        "script": "RANDOM_SEARCH Masked AutoRL-SOP.py",
    },
]


def select_variant() -> dict[str, str]:
    """Prompt for a variant and return its launcher configuration."""
    print("=" * 72)
    print("MASKED AUTORL-SOP VARIANT SELECTION")
    print("=" * 72)
    for index, variant in enumerate(VARIANTS, start=1):
        print(f"[{index:02d}] {variant['name']}")

    while True:
        try:
            choice = int(input(f"\nEnter variant number (1-{len(VARIANTS)}): "))
        except ValueError:
            print(">> Error: Please enter an integer.")
            continue

        if 1 <= choice <= len(VARIANTS):
            return VARIANTS[choice - 1]
        print(">> Error: Invalid variant number.")


def resolve_script(variant: dict[str, str]) -> Path:
    """Resolve and validate the selected variant script."""
    script_path = VARIANTS_DIR / variant["directory"] / variant["script"]
    if not script_path.is_file():
        raise FileNotFoundError(f"Variant script not found: {script_path}")
    return script_path


def run_variant(variant: dict[str, str]) -> int:
    """Run the selected variant and forward control to its instance menu."""
    script_path = resolve_script(variant)
    print(f"\nSelected variant: {variant['name']}")
    print(f"Launching: {script_path}\n")

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),
        env=os.environ.copy(),
        check=False,
    )
    return completed.returncode


def main() -> int:
    """Launch the selected Masked AutoRL-SOP implementation."""
    try:
        variant = select_variant()
        return run_variant(variant)
    except (FileNotFoundError, OSError) as exc:
        print(f"CRITICAL ERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nExecution cancelled.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
