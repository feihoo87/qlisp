import numpy as np
import scipy.sparse as sp

def issparse(qob) -> bool:
    ...


def make_immutable(mat: np.ndarray | sp.spmatrix) -> np.ndarray | sp.spmatrix:
    ...


# Paulis
def sigmaI() -> np.ndarray:
    ...


def sigmaX() -> np.ndarray:
    ...


def sigmaY() -> np.ndarray:
    ...


def sigmaZ() -> np.ndarray:
    ...


def sigmaP() -> np.ndarray:
    ...


def sigmaM() -> np.ndarray:
    ...


# Bell states
def BellPhiP() -> np.ndarray:
    ...


def BellPhiM() -> np.ndarray:
    ...


def BellPsiP() -> np.ndarray:
    ...


def BellPsiM() -> np.ndarray:
    ...


phiplus: np.ndarray = ...
phiminus: np.ndarray = ...
psiplus: np.ndarray = ...
psiminus: np.ndarray = ...

# Clifford gates
H: np.ndarray = ...
S: np.ndarray = ...
Sdag: np.ndarray = ...

# T gates
T: np.ndarray = ...
Tdag: np.ndarray = ...

# two qubit gates
CX: np.ndarray = ...
CZ: np.ndarray = ...
iSWAP: np.ndarray = ...
SWAP: np.ndarray = ...
SQiSWAP: np.ndarray = ...
CR: np.ndarray = ...

##########################################################


def U(theta: float,
      phi: float,
      lambda_: float,
      delta: float = ...) -> np.ndarray:
    ...


def Unitary2Angles(U: np.ndarray, eps: float = ...) -> np.ndarray:
    ...


def rfUnitary(theta: float, phi: float) -> np.ndarray:
    ...


def fSim(theta: float, phi: float) -> np.ndarray:
    ...


def Unitary(matrix: np.ndarray) -> np.ndarray:
    ...


def synchronize_global_phase(U: np.ndarray) -> np.ndarray:
    ...
