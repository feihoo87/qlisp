import itertools
import re
from functools import partial, reduce
from typing import Callable

import numpy as np

from .matricies import (CR, CX, CZ, SWAP, A, B, H, INViSWAP, INVSQiSWAP, M, S,
                        Sdag, SQiSWAP, T, Tdag, U, Unitary, fSim, iSWAP,
                        make_immutable, rfUnitary, sigmaI, sigmaX, sigmaY,
                        sigmaZ)

_clifford_groups = {}
_matrix_of_gates: dict[str, tuple[Callable | np.ndarray, int, str]] = {}


def regesterGateMatrix(gate, mat, N=None, docs=''):
    if isinstance(mat, np.ndarray):
        mat = make_immutable(mat)
    if N is None and isinstance(mat, np.ndarray):
        N = round(np.log2(mat.shape[0]))
    _matrix_of_gates[gate] = (mat, N, docs)


def gate_name(gate) -> str:
    if isinstance(gate, tuple):
        return gate_name(gate[0])
    elif isinstance(gate, str):
        return gate
    else:
        raise ValueError(f'Unexcept gate {gate}')


def is_clifford_gate(gate: str | tuple) -> bool:
    if isinstance(gate, tuple):
        return False
    return re.match(r'^C(\d+)_(\d+)$', gate)


def clifford_gate(gate: str) -> tuple[np.ndarray, int]:
    try:
        from cycles import CliffordGroup

        from .clifford.utils import (one_qubit_clifford_matricies,
                                     two_qubit_clifford_matricies)
    except ImportError:
        return None
    match = re.match(r'^C(\d+)_(\d+)$', gate)
    N, i = [int(num) for num in match.groups()]
    if N == 1:
        return one_qubit_clifford_matricies[i], N
    elif N == 2:
        return two_qubit_clifford_matricies[i], N
    else:
        if N not in _clifford_groups:
            _clifford_groups[N] = CliffordGroup(N)
        perm = _clifford_groups[N][i]
        return _clifford_groups[N].permutation_to_matrix(perm), N


def gate2mat(gate, ignores=[]):
    if isinstance(gate, str) and gate in _matrix_of_gates:
        if callable(_matrix_of_gates[gate][0]):
            mat = _matrix_of_gates[gate][0]()
        else:
            mat = _matrix_of_gates[gate][0]
        N = _matrix_of_gates[gate][1]
        if N is None:
            N = round(np.log2(mat.shape[0]))
        return mat, N
    elif isinstance(gate, tuple) and gate[0] in _matrix_of_gates:
        if callable(_matrix_of_gates[gate[0]][0]):
            mat = _matrix_of_gates[gate[0]][0](*gate[1:])
            N = _matrix_of_gates[gate[0]][1]
            if N is None:
                N = round(np.log2(mat.shape[0]))
            return mat, N
        else:
            raise ValueError(
                f"Could not call {gate[0]}(*{gate[1:]}), `{gate[0]}` is not callable."
            )
    elif is_clifford_gate(gate):
        return clifford_gate(gate)
    elif gate_name(gate) == 'C':
        U, N = gate2mat(gate[1])
        ret = np.eye(2 * U.shape[0], dtype=complex)
        ret[U.shape[0]:, U.shape[0]:] = U
        return ret, N + 1
    elif gate_name(gate) in ignores:
        return np.array([], dtype=complex), 0
    else:
        raise ValueError(f'Unexcept gate {gate}')


def splite_at(l, bits):
    """将 l 的二进制位于 bits 所列的位置上断开插上0
    
    如 splite_at(int('1111',2), [0,2,4,6]) == int('10101010', 2)
    bits 必须从小到大排列
    """
    r = l
    for n in bits:
        mask = (1 << n) - 1
        low = r & mask
        high = r - low
        r = (high << 1) + low
    return r


def place_at(l, bits):
    """将 l 的二进制位置于 bits 所列的位置上
    
    如 place_at(int('10111',2), [0,2,4,5,6]) == int('01010101', 2)
    """
    r = 0
    for index, n in enumerate(bits):
        b = (l >> index) & 1
        r += b << n
    return r


def reduceSubspace(targets, N, inputMat, func, args):
    innerDim = 1 << len(targets)
    outerDim = 1 << (N - len(targets))

    targets = tuple(reversed([N - i - 1 for i in targets]))

    def index(targets, i, j):
        return splite_at(j, sorted(targets)) | place_at(i, targets)

    if len(inputMat.shape) == 1:
        for k in range(outerDim):
            innerIndex = [index(targets, i, k) for i in range(innerDim)]
            inputMat[innerIndex] = func(inputMat[innerIndex], *args)
    else:
        for k, l in itertools.product(range(outerDim), repeat=2):
            innerIndex = np.asarray(
                [[index(targets, i, k),
                  index(targets, j, l)]
                 for i, j in itertools.product(range(innerDim), repeat=2)]).T
            sub = inputMat[innerIndex[0], innerIndex[1]].reshape(
                (innerDim, innerDim))
            inputMat[innerIndex[0], innerIndex[1]] = func(sub, *args).flatten()
    return inputMat


def _apply_gate(gate, inputMat, unitary_process, qubits, N, ignores=[]):
    U, n = gate2mat(gate, ignores)
    if n == 0:
        return
    if len(qubits) == n and all(isinstance(qubit, int) for qubit in qubits):
        reduceSubspace(qubits, N, inputMat, unitary_process, (U, ))
    elif n == 1 and all(isinstance(qubit, int) for qubit in qubits):
        for qubit in qubits:
            reduceSubspace([qubit], N, inputMat, unitary_process, (U, ))
    elif len(qubits) == n and all(
            isinstance(qubit, [tuple, list]) for qubit in qubits):
        for qubit_tuple in zip(*qubits):
            reduceSubspace(qubit_tuple, N, inputMat, unitary_process, (U, ))
    else:
        raise ValueError(f'Unexcept gate {gate} and qubits {qubits}')


def _measure_process(rho):
    return np.array([[rho[0, 0], 0], [0, rho[1, 1]]])


def _reset_process(rho, p1):
    s0 = np.array([[rho[0, 0] + rho[1, 1], 0], [0, 0]])
    s1 = np.array([[0, 0], [0, rho[0, 0] + rho[1, 1]]])
    return (1 - p1) * s0 + p1 * s1


def _decohherence_process(rho, Gamma_t, gamma_t):
    rho00 = rho[0, 0] + rho[1, 1] * (1 - np.exp(-Gamma_t))
    rho11 = rho[1, 1] * np.exp(-Gamma_t)
    rho01 = rho[0, 1] * np.exp(-Gamma_t / 2 - gamma_t**2)
    rho10 = rho[1, 0] * np.exp(-Gamma_t / 2 - gamma_t**2)
    return np.array([[rho00, rho01], [rho10, rho11]])


def applySeq(seq, psi0=None, ignores=[]):

    def _set_vector_to_rho(psi):
        psi = psi.reshape(-1, 1).conj() @ psi.reshape(1, -1)
        unitary_process = lambda psi, U: U @ psi @ U.T.conj()
        return psi, unitary_process, np.array([[1, 0], [0, 0]])

    if psi0 is None:
        psi = np.array([1, 0], dtype=complex)
        N = 1
    else:
        psi = psi0
        N = round(np.log2(psi.shape[0]))

    psi0 = np.array([1, 0])
    unitary_process = lambda psi, U: U @ psi

    if psi.ndim == 2:
        psi, unitary_process, psi0 = _set_vector_to_rho(psi)

    for gate, *qubits in seq:
        if len(qubits) == 1 and isinstance(qubits[0], tuple):
            qubits = qubits[0]
        M = max(qubits)
        if M >= N:
            psi = reduce(np.kron, itertools.repeat(psi0, times=M - N + 1), psi)
            N = M + 1

        if gate_name(gate) in ['Barrier']:
            continue
        if gate_name(gate) in ['Delay']:
            if len(gate) == 2:
                continue
            else:
                if len(gate) == 3:
                    _, t, T1 = gate
                    Gamma_t = t / T1
                    gamma_t = 0
                else:
                    _, t, T1, Tphi = gate
                    Gamma_t = t / T1
                    gamma_t = t / Tphi
        if gate_name(gate) in ['Reset']:
            if isinstance(gate, tuple) and len(gate) == 2:
                _, p1 = gate
            else:
                p1 = 0.0
        if gate_name(gate) in ['Measure', 'Reset', 'Delay'] and psi.ndim == 1:
            psi, unitary_process, psi0 = _set_vector_to_rho(psi)

        if gate_name(gate) == 'Measure':
            reduceSubspace(qubits, N, psi, _measure_process, ())
        elif gate_name(gate) == 'Reset':
            reduceSubspace(qubits, N, psi, _reset_process, (p1, ))
        elif gate_name(gate) == 'Delay':
            reduceSubspace(qubits, N, psi, _decohherence_process,
                           (Gamma_t, gamma_t))
        else:
            _apply_gate(gate, psi, unitary_process, qubits, N, ignores)

    return psi


def seq2mat(seq, U=None, ignores=[]):
    I = np.eye(2, dtype=complex)
    if U is None:
        U = np.eye(2, dtype=complex)
        N = 1
    else:
        N = round(np.log2(U.shape[0]))

    unitary_process = lambda U0, U: U @ U0

    for gate, *qubits in seq:
        if len(qubits) == 1 and isinstance(qubits[0], tuple):
            qubits = qubits[0]
        M = max(qubits)
        if M >= N:
            U = reduce(np.kron, itertools.repeat(I, times=M - N + 1), U)
            N = M + 1
        if gate_name(gate) in ['Delay', 'Barrier']:
            continue
        if gate_name(gate) in ['Measure', 'Reset']:
            raise ValueError(
                'Measure and Reset must be applied to a state vector')
        else:
            _apply_gate(gate, U, unitary_process, qubits, N, ignores)
    return U


def measure(circ=[], rho0=None, A=None):
    """
    Measure the state of quantum circuit.

    Args:
        circ: list of tuple, quantum circuit.
        rho0: np.ndarray, initial state of quantum circuit.
        A: np.ndarray, measurement error matrix.

    Returns:
        np.ndarray, probability of each state.
    """
    if rho0 is None:
        U = seq2mat(circ)
        rho0 = np.zeros_like(U)
        rho0[0, 0] = 1
    else:
        N = round(np.log2(rho0.shape[0]))
        U = seq2mat(circ + [('I', q) for q in range(N)])
    rho = U @ rho0 @ U.T.conj()
    if A is None:
        return np.diag(rho).real
    else:
        return A @ np.diag(rho).real


regesterGateMatrix('U', U, 1)
regesterGateMatrix('u1', lambda p: U(theta=0, phi=0, lambda_=p), 1)
regesterGateMatrix('u2', lambda phi, lam: U(np.pi / 2, phi, lam), 1)
regesterGateMatrix('u3', U, 1)
regesterGateMatrix('P', lambda p=np.pi / 2: U(theta=0, phi=0, lambda_=p), 1)
regesterGateMatrix('rfUnitary', rfUnitary, 1)
regesterGateMatrix('R', lambda phi: rfUnitary(np.pi / 2, phi), 1)
regesterGateMatrix('Rx', partial(rfUnitary, phi=0), 1)
regesterGateMatrix('Ry', partial(rfUnitary, phi=np.pi / 2), 1)
regesterGateMatrix('Rz', lambda p: U(theta=0, phi=0, lambda_=p), 1)
regesterGateMatrix('fSim', fSim, 2)
regesterGateMatrix('Cphase', lambda phi: fSim(theta=0, phi=phi), 2)
regesterGateMatrix('Unitary', Unitary, None)

# one qubit
regesterGateMatrix('I', sigmaI())
regesterGateMatrix('X', -1j * sigmaX())
regesterGateMatrix('Y', -1j * sigmaY())
regesterGateMatrix('X/2', np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2))
regesterGateMatrix('Y/2', np.array([[1, -1], [1, 1]]) / np.sqrt(2))
regesterGateMatrix('-X/2', np.array([[1, 1j], [1j, 1]]) / np.sqrt(2))
regesterGateMatrix('-Y/2', np.array([[1, 1], [-1, 1]]) / np.sqrt(2))
regesterGateMatrix('Z', sigmaZ())
regesterGateMatrix('S', S)
regesterGateMatrix('-S', Sdag)
regesterGateMatrix('Z/2', S)
regesterGateMatrix('-Z/2', Sdag)
regesterGateMatrix('H', H)

# non-clifford
regesterGateMatrix('T', T)
regesterGateMatrix('-T', Tdag)
regesterGateMatrix('W/2', rfUnitary(np.pi / 2, np.pi / 4))
regesterGateMatrix('-W/2', rfUnitary(-np.pi / 2, np.pi / 4))
regesterGateMatrix('V/2', rfUnitary(np.pi / 2, 3 * np.pi / 4))
regesterGateMatrix('-V/2', rfUnitary(-np.pi / 2, 3 * np.pi / 4))

# two qubits
regesterGateMatrix('CZ', CZ)
regesterGateMatrix('Cnot', CX)
regesterGateMatrix('CX', CX)
regesterGateMatrix('iSWAP', iSWAP)
regesterGateMatrix('-iSWAP', INViSWAP)
regesterGateMatrix('SWAP', SWAP)
regesterGateMatrix('CR', CR)

regesterGateMatrix('A', A)
regesterGateMatrix('B', B)
regesterGateMatrix('M', M)

# non-clifford
regesterGateMatrix('SQiSWAP', SQiSWAP)
regesterGateMatrix('-SQiSWAP', INVSQiSWAP)
