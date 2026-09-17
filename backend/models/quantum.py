"""Baseline variational quantum classifier (Qiskit 2.x, qiskit-machine-learning 0.9).

Circuit:   |0..0> -> feature encoding U(x) -> trainable ansatz W(theta) -> measure <O>
Decision:  class 1 if <O> > 0 else class 0.   (labels trained as -1 / +1)

Inputs must already be scaled to angles in [0, pi] and reduced to n_qubits
columns (see preprocessing.build_quantum_feature_pipeline). This class is a
scikit-learn compatible estimator so it drops into the same Pipeline / metric
code as the classical baselines.

Design choices for the *baseline* (deliberately simple, all configurable):
  feature_map : "angle" -> RY(x_i) on qubit i           (1 gate/qubit, no entanglement)
                "z"     -> Qiskit z_feature_map           (H + P(2x))
                "zz"    -> Qiskit zz_feature_map          (adds ZZ entangling phases)
  ansatz      : real_amplitudes(n_qubits, reps, entanglement)  RY layers + CX entanglers
  observable  : "parity" -> Z⊗Z⊗...⊗Z ;  "single" -> Z on qubit 0
  optimizer   : COBYLA (gradient-free), exact statevector expectation values
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import real_amplitudes, z_feature_map, zz_feature_map
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from sklearn.svm import SVC
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA, SPSA
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


def _angle_feature_map(n_qubits: int) -> QuantumCircuit:
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits, name="angle_ry")
    for i in range(n_qubits):
        qc.ry(x[i], i)
    return qc


class QuantumClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_qubits: int = 4, reps: int = 1, entanglement: str = "linear",
                 feature_map: str = "angle", observable: str = "parity",
                 optimizer: str = "cobyla", maxiter: int = 100, seed: int = 42):
        self.n_qubits = n_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.feature_map = feature_map
        self.observable = observable
        self.optimizer = optimizer
        self.maxiter = maxiter
        self.seed = seed

    # ------------------------------------------------------------------ circuit
    def _build_circuit(self):
        n = self.n_qubits
        if self.feature_map == "angle":
            fm = _angle_feature_map(n)
        elif self.feature_map == "z":
            fm = z_feature_map(n, reps=1)
        elif self.feature_map == "zz":
            fm = zz_feature_map(n, reps=1, entanglement=self.entanglement)
        else:
            raise ValueError(f"unknown feature_map '{self.feature_map}'")

        ansatz = real_amplitudes(n, reps=self.reps, entanglement=self.entanglement)

        circuit = QuantumCircuit(n)
        circuit.compose(fm, inplace=True)
        circuit.compose(ansatz, inplace=True)

        if self.observable == "parity":
            obs = SparsePauliOp("Z" * n)
        elif self.observable == "single":
            obs = SparsePauliOp("I" * (n - 1) + "Z")      # Z on qubit 0 (rightmost in Qiskit ordering)
        else:
            raise ValueError(f"unknown observable '{self.observable}'")
        return circuit, fm, ansatz, obs

    def _build_qnn(self):
        circuit, fm, ansatz, obs = self._build_circuit()
        estimator = StatevectorEstimator(default_precision=0.0, seed=self.seed)  # exact <O>
        qnn = EstimatorQNN(circuit=circuit, estimator=estimator, observables=obs,
                           input_params=list(fm.parameters), weight_params=list(ansatz.parameters))
        return qnn, circuit, ansatz

    def _make_optimizer(self):
        if self.optimizer == "cobyla":
            return COBYLA(maxiter=self.maxiter)
        if self.optimizer == "spsa":
            return SPSA(maxiter=self.maxiter)
        raise ValueError(f"unknown optimizer '{self.optimizer}'")

    # ---------------------------------------------------------------- sklearn API
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        if X.shape[1] != self.n_qubits:
            raise ValueError(f"expected {self.n_qubits} features (one per qubit), got {X.shape[1]}")
        self.classes_ = np.array([0, 1])
        y_pm = np.where(y == 1, 1, -1)

        qnn, circuit, ansatz = self._build_qnn()
        rng = np.random.default_rng(self.seed)
        initial = rng.uniform(0, 2 * np.pi, qnn.num_weights)

        self.loss_history_ = []
        clf = NeuralNetworkClassifier(
            neural_network=qnn, loss="squared_error", optimizer=self._make_optimizer(),
            initial_point=initial, callback=lambda w, loss: self.loss_history_.append(float(loss)))
        clf.fit(X, y_pm)

        self.weights_ = np.asarray(clf.weights, dtype=float)
        self.n_weights_ = int(qnn.num_weights)
        self.circuit_depth_ = int(circuit.decompose().depth())
        self._qnn = qnn
        return self

    def _ensure_qnn(self):
        if getattr(self, "_qnn", None) is None:
            self._qnn, _, _ = self._build_qnn()
        return self._qnn

    def decision_function(self, X):
        """Raw expectation value <O> in [-1, 1]. Positive => class 1."""
        check_is_fitted(self, "weights_")
        X = np.asarray(X, dtype=float)
        return np.asarray(self._ensure_qnn().forward(X, self.weights_)).reshape(-1)

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    def predict_proba(self, X):
        """Monotonic map of <O> to [0,1]. NOT a calibrated probability."""
        p1 = (1.0 + self.decision_function(X)) / 2.0
        return np.column_stack([1.0 - p1, p1])

    # -------------------------------------------------------------- persistence
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_qnn", None)          # rebuilt lazily from config; keeps joblib files small & portable
        return state

    def describe(self) -> dict:
        circuit, _, ansatz, obs = self._build_circuit()
        return {
            "n_qubits": self.n_qubits, "feature_map": self.feature_map, "ansatz": "real_amplitudes",
            "reps": self.reps, "entanglement": self.entanglement, "observable": str(obs.paulis[0]),
            "n_trainable_params": int(ansatz.num_parameters), "circuit_depth": int(circuit.decompose().depth()),
            "optimizer": self.optimizer, "maxiter": self.maxiter, "backend": "StatevectorEstimator (exact, noiseless)",
        }

    def draw(self) -> str:
        circuit, *_ = self._build_circuit()
        return str(circuit.draw(output="text", fold=120))


# =====================================================================================
# Quantum kernel SVM (QSVC)
# =====================================================================================
class QuantumKernelClassifier(BaseEstimator, ClassifierMixin):
    """Support-vector classifier with a fidelity quantum kernel  K(x, y) = |<phi(x)|phi(y)>|^2.

    phi(x) is the state prepared by the feature-map circuit. The kernel is evaluated EXACTLY by
    statevector simulation (equivalent to qiskit-machine-learning's FidelityQuantumKernel with an
    ideal sampler and infinite shots, but vectorised: all statevectors are built once and the Gram
    matrix is |Phi Phi^dagger|^2). The classical part is scikit-learn's SVC with a precomputed kernel.
    Output is a signed decision function (margin), not a probability.
    """

    def __init__(self, n_qubits: int = 4, feature_map: str = "zz", entanglement: str = "linear",
                 reps: int = 1, C: float = 1.0, seed: int = 42):
        self.n_qubits = n_qubits
        self.feature_map = feature_map
        self.entanglement = entanglement
        self.reps = reps
        self.C = C
        self.seed = seed

    def _feature_map(self) -> QuantumCircuit:
        n = self.n_qubits
        if self.feature_map == "angle":
            return _angle_feature_map(n)
        if self.feature_map == "z":
            return z_feature_map(n, reps=self.reps)
        if self.feature_map == "zz":
            return zz_feature_map(n, reps=self.reps, entanglement=self.entanglement)
        raise ValueError(f"unknown feature_map '{self.feature_map}'")

    def _states(self, X: np.ndarray) -> np.ndarray:
        fm = self._feature_map()
        params = list(fm.parameters)
        return np.array([Statevector(fm.assign_parameters(dict(zip(params, row)))).data for row in X])

    @staticmethod
    def _gram(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.abs(A @ B.conj().T) ** 2

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_qubits:
            raise ValueError(f"expected {self.n_qubits} features (one per qubit), got {X.shape[1]}")
        self.classes_ = np.array([0, 1])
        self.train_states_ = self._states(X)
        K = self._gram(self.train_states_, self.train_states_)
        self.svc_ = SVC(kernel="precomputed", C=self.C, random_state=self.seed).fit(K, np.asarray(y).astype(int))
        self.n_support_ = int(self.svc_.n_support_.sum())
        return self

    def decision_function(self, X):
        check_is_fitted(self, "svc_")
        K = self._gram(self._states(np.asarray(X, dtype=float)), self.train_states_)
        return self.svc_.decision_function(K)

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    def describe(self) -> dict:
        fm = self._feature_map()
        return {"n_qubits": self.n_qubits, "feature_map": self.feature_map, "reps": self.reps, "entanglement": self.entanglement,
                "kernel": "fidelity |<phi(x)|phi(y)>|^2, exact statevector", "classifier": f"SVC(kernel=precomputed, C={self.C})",
                "n_trainable_params": 0, "circuit_depth": int(fm.decompose().depth()), "n_support_vectors": getattr(self, "n_support_", None),
                "backend": "Statevector (exact, noiseless)"}

    def draw(self) -> str:
        return str(self._feature_map().draw(output="text", fold=120))
