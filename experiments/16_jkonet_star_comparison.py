"""Experiment 16 — attempt a faithful official JKOnet* comparison.

The statistical paper does not wait on this baseline.  The wrapper pins
``antonioterpin/jkonet-star``, writes independent snapshots from the same
quadratic model, and runs the official ``jkonet-star-linear`` coupling /
density / least-squares pipeline when that pipeline can be executed.  If the
repository cannot ingest the observation design without changing its objective
or pairing assumptions, the JSON report records the blocker instead of an
unofficial substitute.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfmm.io_util import save_json
from wfmm.paths import RESULTS_DIR, ROOT, ensure_output_dirs


REPO_URL = "https://github.com/antonioterpin/jkonet-star.git"
PINNED_SHA = "1741c53ae00da932e0841ce02eca2d842c04b813"
VENDOR = ROOT / ".vendor" / "jkonet-star"
A_TRUE = 1.0
K_TRUE = 1.5
Q_TRUE = 1.0 / 3.0
S0_TRUE = 1.5
TIMES = np.array([0.0, 0.5, 1.0])
DESIGNS = {
    "centered": np.array([0.0]),
    "shifted_only": np.array([0.8]),
    "mixed": np.array([-0.8, 0.0, 0.8]),
}


def run_cmd(command: list[str], cwd: Path | None = None, timeout: int = 600) -> dict:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": f"timeout after {timeout}s",
        }
    except Exception as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": f"{type(exc).__name__}: {exc}",
        }


def variance_path(times: np.ndarray) -> np.ndarray:
    return Q_TRUE + (S0_TRUE - Q_TRUE) * np.exp(-2.0 * K_TRUE * times)


def write_independent_snapshots(folder: Path, mu0s: np.ndarray, n_total: int, seed: int) -> None:
    """Write official-format arrays: flattened (time x particle, dim)."""
    rng = np.random.default_rng(seed)
    n_pop = mu0s.size
    n_each = n_total // n_pop
    variances = variance_path(TIMES)
    values = []
    labels = []
    for t_index, t in enumerate(TIMES):
        for mu0 in mu0s:
            mean = mu0 * np.exp(-A_TRUE * t)
            values.append(rng.normal(mean, np.sqrt(variances[t_index]), size=(n_each, 1)))
            labels.append(np.full(n_each, t_index, dtype=int))
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "data.npy", np.concatenate(values, axis=0))
    np.save(folder / "sample_labels.npy", np.concatenate(labels, axis=0))
    (folder / "args.txt").write_text(
        "potential=quadratic_confinement\n"
        "internal=wiener\n"
        f"beta={K_TRUE * Q_TRUE}\n"
        "interaction=quadratic\n"
        f"dt={float(TIMES[1] - TIMES[0])}\n"
        "note=Independent snapshots; not labeled trajectories.\n"
    )


def pin_repository() -> dict:
    VENDOR.parent.mkdir(parents=True, exist_ok=True)
    if not (VENDOR / ".git").exists():
        clone = run_cmd(["git", "clone", "--filter=blob:none", REPO_URL, str(VENDOR)], timeout=180)
        if clone["returncode"] not in (0, None) and clone["returncode"] != 0:
            return {"ok": False, "step": "clone", **clone}
        if clone["returncode"] != 0:
            return {"ok": False, "step": "clone", **clone}
    fetch = run_cmd(["git", "fetch", "--depth", "1", "origin", PINNED_SHA], cwd=VENDOR, timeout=120)
    checkout = run_cmd(["git", "checkout", "--detach", PINNED_SHA], cwd=VENDOR, timeout=60)
    sha = run_cmd(["git", "rev-parse", "HEAD"], cwd=VENDOR, timeout=30)
    return {
        "ok": checkout["returncode"] == 0 and PINNED_SHA.startswith((sha["stdout_tail"] or "").strip()),
        "clone_or_fetch": fetch,
        "checkout": checkout,
        "sha": (sha["stdout_tail"] or "").strip(),
        "expected_sha": PINNED_SHA,
    }


def official_incompatibilities() -> list[dict]:
    return [
        {
            "issue": "unit_time_index",
            "detail": (
                "PopulationEvalDataset hard-codes dt=1. Consecutive snapshot labels "
                "are treated as unit JKO steps. The equal-budget designs in "
                "experiments 12/14 use times {0.25,0.5,1.0}, which are not equally "
                "spaced. Feeding that grid unchanged would silently distort the "
                "official pairing. This attempt therefore uses equally spaced times "
                "{0,0.5,1.0} so official unit steps are a global time rescaling."
            ),
        },
        {
            "issue": "inferred_couplings_and_gmm_density",
            "detail": (
                "The official linear solver is trained on OT couplings and GMM "
                "density estimates between consecutive unlabeled snapshots, not on "
                "the exact Gaussian likelihood used in experiments 14--15. That "
                "preprocessing is preserved; it is not replaced by a custom loss."
            ),
        },
        {
            "issue": "feature_dictionary",
            "detail": (
                "The released linear config uses degree-4 polynomials plus a "
                "constant RBF. An exact quadratic dictionary is degree 2 with no "
                "RBFs. If training proceeds, that dictionary restriction is labeled "
                "as the official JKOnet* objective with a correctly specified "
                "quadratic feature set, not as a fork of the loss."
            ),
        },
    ]


def try_official_run(python_exe: str, dataset: str, n_gmm: int) -> dict:
    env_python = python_exe
    data_gen = [
        env_python, "data_generator.py",
        "--load-from-file", dataset,
        "--split-population",
        "--test-ratio", "0.5",
        "--n-gmm-components", str(n_gmm),
        "--batch-size", "-1",
        "--seed", "16",
    ]
    generated = run_cmd(data_gen, cwd=VENDOR, timeout=300)
    if generated["returncode"] != 0:
        return {"ok": False, "step": "data_generator", **generated}
    train = [
        env_python, "train.py",
        "--solver", "jkonet-star-linear",
        "--dataset", dataset,
        "--eval", "test_data",
        "--seed", "16",
        "--epochs", "1",
    ]
    trained = run_cmd(train, cwd=VENDOR, timeout=600)
    return {
        "ok": trained["returncode"] == 0,
        "step": "train" if trained["returncode"] != 0 else "complete",
        "data_generator": generated,
        "train": trained,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    t0 = time.time()
    n_total = 300 if args.quick else 1200
    n_gmm = 1
    report: dict = {
        "claim": "official_jkonet_star_attempt",
        "repository": REPO_URL,
        "pinned_sha": PINNED_SHA,
        "solver": "jkonet-star-linear",
        "observation_model": (
            "Independent unlabeled snapshots from the scalar quadratic McKean-Vlasov "
            "law, written in the official data.npy/sample_labels.npy format."
        ),
        "incompatibilities": official_incompatibilities(),
        "faithful_comparison": False,
        "status": "blocked",
        "designs": list(DESIGNS),
        "times": TIMES,
        "budget_per_time": n_total,
    }

    data_root = VENDOR / "data"
    pin = pin_repository()
    report["pin"] = pin
    if not pin.get("ok"):
        report["blocker"] = (
            "Could not clone or check out antonioterpin/jkonet-star at the pinned "
            f"revision {PINNED_SHA}."
        )
        report["runtime_sec"] = time.time() - t0
        save_json(RESULTS_DIR / "exp16_jkonet_star_comparison.json", report)
        print("exp16 blocked at pin", report["blocker"])
        return

    for name, mu0s in DESIGNS.items():
        write_independent_snapshots(data_root / f"wfmm_{name}", mu0s, n_total, seed=16)

    # Official linear config uses degree-4 polynomials and optional RBFs.
    # Restricting polynomial degree to 2 is a dictionary specification.
    config_path = VENDOR / "config.yaml"
    original_config = config_path.read_text() if config_path.exists() else ""
    report["official_linear_config_excerpt"] = original_config
    quadratic_config = original_config.replace("degree: 4", "degree: 2")
    report["dictionary"] = (
        "official JKOnet* linear objective/code with polynomial degree 2 "
        "(released default is degree 4); remaining RBF entries, if present, "
        "are the official extras rather than a forked loss"
    )

    install = run_cmd(
        [sys.executable, "-c", "import jax, flax, torch, ott; print('imports-ok')"],
        timeout=30,
    )
    python_exe = sys.executable
    if install["returncode"] != 0:
        venv = ROOT / ".vendor" / "jkonet-venv"
        if not (venv / "bin" / "python").exists():
            created = run_cmd([sys.executable, "-m", "venv", str(venv)], timeout=60)
            report["venv_create"] = created
        python_exe = str(venv / "bin" / "python")
        req = VENDOR / "requirements.txt"
        # The upstream file is UTF-16; rewrite a UTF-8 subset without docs extras.
        raw = req.read_bytes()
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8")
        packages = [
            line.strip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
            and not line.lower().startswith("sphinx")
        ]
        slim = VENDOR / "requirements-wfmm.txt"
        slim.write_text("\n".join(packages) + "\n")
        pip = run_cmd(
            [python_exe, "-m", "pip", "install", "-r", str(slim)],
            timeout=900,
        )
        report["install"] = pip
        if pip["returncode"] != 0:
            report["blocker"] = (
                "Official requirements could not be installed in the attempt "
                "environment. The comparison is not reported as a JKOnet* result."
            )
            report["runtime_sec"] = time.time() - t0
            save_json(RESULTS_DIR / "exp16_jkonet_star_comparison.json", report)
            print("exp16 blocked at install")
            return
    else:
        report["install"] = {"ok": True, "note": "jax/flax/torch/ott already importable"}

    # Apply the quadratic dictionary only after documenting the official file.
    config_path.write_text(quadratic_config)
    try:
        runs = {}
        any_ok = False
        for name in DESIGNS:
            result = try_official_run(python_exe, f"wfmm_{name}", n_gmm)
            runs[name] = result
            any_ok = any_ok or result.get("ok", False)
        report["runs"] = runs
        parsed = {}
        for name, result in runs.items():
            text = ((result.get("train") or {}).get("stdout_tail") or "")
            mean = re.search(r"aggregate: ([0-9.eE+-]+) \+/- ([0-9.eE+-]+)", text)
            vec = re.search(r"Wasserstein error one step ahead: \[([^\]]+)\]", text)
            parsed[name] = {
                "official_w1_one_ahead_mean": float(mean.group(1)) if mean else None,
                "official_w1_one_ahead_std": float(mean.group(2)) if mean else None,
                "official_w1_one_ahead_steps": (
                    [float(x) for x in vec.group(1).split()] if vec else None
                ),
            }
        report["parsed_official_metrics"] = parsed
        report["mixed_arm_note"] = (
            "Writing mixed means into one time-labeled cloud makes a mean-zero "
            "mixture. That is not the three-population equal-budget design of "
            "Experiment 14. Official test_data is a split of the same populations, "
            "not an independent shifted intervention with mean 1.2."
        )
        report["faithful_for"] = [
            "official jkonet-star-linear objective, couplings, GMM densities, and unit-step pairing",
            "independent unlabeled snapshots of the scalar quadratic model at equally spaced times {0,0.5,1}",
            "centered and shifted_only single-population clouds",
        ]
        report["not_faithful_for"] = [
            "Experiment 14 mixed three-population allocation",
            "unequal times {0.25,0.5,1.0}",
            "independent shifted test population with mean 1.2",
            "exact Gaussian W2 against the closed-form law",
        ]
        if any_ok:
            report["status"] = "official_pipeline_completed_with_design_gaps"
            report["faithful_comparison"] = False
            report["label"] = (
                "Official JKOnet* linear solver and coupling pipeline, with a "
                "quadratic polynomial dictionary, on equally spaced independent "
                "snapshots of the scalar quadratic model. Not a substitute for "
                "Experiments 14--15."
            )
        else:
            report["status"] = "blocked"
            first = next(iter(runs.values()))
            report["blocker"] = (
                "Official jkonet-star-linear did not complete on the written "
                f"independent-snapshot datasets. First failure step: {first.get('step')}."
            )
    finally:
        if original_config:
            config_path.write_text(original_config)

    report["runtime_sec"] = time.time() - t0
    report["command"] = "python experiments/16_jkonet_star_comparison.py"
    save_json(RESULTS_DIR / "exp16_jkonet_star_comparison.json", report)
    print("exp16 status", report["status"], report.get("blocker", "ok"))


if __name__ == "__main__":
    main()
