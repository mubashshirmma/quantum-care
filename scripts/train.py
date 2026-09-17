"""Train + benchmark all models for one disease.

  .venv/bin/python scripts/train.py --disease heart_disease_cleveland
  .venv/bin/python scripts/train.py --disease heart_disease_synthetic --n-qubits 4 --reps 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import DISEASE_REGISTRY  # noqa: E402
from backend.training.benchmark import DEFAULT_QUANTUM_CONFIG, run_benchmark  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--disease", default="heart_disease_cleveland", choices=sorted(DISEASE_REGISTRY))
    p.add_argument("--n-qubits", type=int, default=DEFAULT_QUANTUM_CONFIG["n_qubits"])
    p.add_argument("--reps", type=int, default=DEFAULT_QUANTUM_CONFIG["reps"])
    p.add_argument("--entanglement", default=DEFAULT_QUANTUM_CONFIG["entanglement"])
    p.add_argument("--feature-map", default=DEFAULT_QUANTUM_CONFIG["feature_map"], choices=["angle", "z", "zz"])
    p.add_argument("--observable", default=DEFAULT_QUANTUM_CONFIG["observable"], choices=["parity", "single"])
    p.add_argument("--optimizer", default=DEFAULT_QUANTUM_CONFIG["optimizer"], choices=["cobyla", "spsa"])
    p.add_argument("--maxiter", type=int, default=DEFAULT_QUANTUM_CONFIG["maxiter"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-quantum", action="store_true")
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()

    qcfg = dict(n_qubits=a.n_qubits, reps=a.reps, entanglement=a.entanglement, feature_map=a.feature_map,
                observable=a.observable, optimizer=a.optimizer, maxiter=a.maxiter)
    run_benchmark(a.disease, quantum_config=qcfg, run_quantum=not a.skip_quantum, seed=a.seed, save=not a.no_save)


if __name__ == "__main__":
    main()
