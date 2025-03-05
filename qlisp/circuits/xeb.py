import itertools
import random
from typing import Callable, Iterable, Optional, Union

import numpy as np
from numpy import pi

from .utils import mapping_qubits


def uncorrelatedEntropy(D: int) -> float:
    from scipy.special import polygamma

    return np.euler_gamma + polygamma(0, D)


def PTEntropy(N: int) -> float:
    D = 2**N
    return np.sum([1 / i for i in range(1, D + 1)]) - 1


def crossEntropy(P: np.array,
                 Q: np.array,
                 func: Callable = np.log,
                 eps: float = 1e-9) -> float:
    mask = (P > 0) * (Q > eps)
    if isinstance(P, (int, float, complex)):
        return -np.real(P) * np.sum(func(Q[mask]))
    else:
        return -np.sum(P[mask] * func(Q[mask]))


def Fxeb(Pm_lst: Iterable[np.array],
         Pe_lst: Iterable[np.array],
         Pi: Optional[np.array] = None) -> float:
    """
    XEB Fidelity

    Pm_lst: list of measured distribution
    Pe_lst: list of expected distribution
    """
    Si, Sm, Se = [], [], []
    for Pm, Pe in zip(Pm_lst, Pe_lst):
        if Pi is None:
            Pi = 1.0 / Pm.size
        Si.append(crossEntropy(Pi, Pe))
        Sm.append(crossEntropy(Pm, Pe))
        Se.append(crossEntropy(Pe, Pe))
    return (np.mean(Si) - np.mean(Sm)) / (np.mean(Si) - np.mean(Se))


def Fxeb_linear(measured: Iterable[np.array], ideal: Iterable[np.array]):
    """XEB Fidelity

    Linear cross entropy between two probability distributions p and q is
    defined as:
        S(p, q) = SUM [ p * (1 - D * q) ]
    where D is the Hilbert space dimension.

    The fidelity is defined as normalized linear cross entropy between the
    ideal and measured distributions, with values between 0 and 1. In case
    of perfect fidelity, the value is 1, and in case of perfect mixing, the
    value is 0.

    The fidelity is calculated as:
            F = (S(p, q) - S(p, i)) / (S(p, p) - S(p, i))
    where p is the ideal distribution, q is the measured distribution, and
    i is the uniform distribution. In the case of the linear cross entropy,
    S(p, i) always equals 0, so the formula simplifies to:
            F = S(p, q) / S(p, p)

    The fidelity decays exponentially with the cycles, fitting the data with
            F = p ** cycle
    where p is the fidelity per cycle, it can be used to estimate the Pauli
    error rate:
            ep = (1 - p) / (1 - 1/D**2)
    Pauli error rate is the probability of a total Pauli error per cycle.
    The totalvPauli error is the sum of the all kinds of Pauli errors, and
    each kind of Pauli error assumes the same probability.

    Ref:
        https://doi.org/10.1038/s41586-019-1666-5
    """
    measured = np.asarray(measured)
    ideal = np.asarray(ideal)
    D = ideal.shape[-1]
    b = 1 / D**2
    F = (np.mean(ideal * measured, axis=(-1, -2)) -
         b) / (np.mean(ideal**2, axis=(-1, -2)) - b)
    return F


def specklePurity(Pm_lst: Iterable[np.array], D: int = None) -> float:
    """Speckle Purity

    Speckle Purity Benchmarking (SPB) is the method of measuring the state purity
    from raw XEB data. Assuming the depolarizing-channel model with polarization
    parameter p, we can model the quantum state as

    rho = p |psi> <psi| + (1 - p) I / D

    where D is the Hilbert space dimension, and p is the probability of a pure state
    |psi> (which is not necessarily known to us). The Speckle Purity is defined as

    Purity = p^2

    The variance of the experimental probabilities will be p^2 times the Porter-Thomas
    variance, thus the Speckle Purity can be estimated from the variance of the
    experimental measured probabilities Pm.

    Purity = Var(Pm) * (D^2 * (D + 1) / (D - 1))

    Ref:
        https://doi.org/10.1038/s41586-019-1666-5

    Args:
        Pm_lst: list of measured distribution
        D (int): Hilbert space dimension

    Returns:
        Speckle Purity: float
    """
    if D is None:
        D = Pm_lst[0].size
    return np.asarray(Pm_lst).var() * D**2 * (D + 1) / (D - 1)


def generateXEBCircuit(qubits: Union[int, str, tuple],
                       cycle: int,
                       seed: Optional[int] = None,
                       interleaves: list[list[tuple]] = [],
                       base: list[Union[str, tuple]] = [
                           ('rfUnitary', pi / 2, 0),
                           ('rfUnitary', pi / 2, pi / 4),
                           ('rfUnitary', pi / 2, pi / 2),
                           ('rfUnitary', pi / 2, pi * 3 / 4),
                           ('rfUnitary', pi / 2, pi),
                           ('rfUnitary', pi / 2, pi * 5 / 4),
                           ('rfUnitary', pi / 2, pi * 3 / 2),
                           ('rfUnitary', pi / 2, pi * 7 / 2)
                       ]):
    """Generate a random XEB circuit.

    Args:
        qubits (list): The qubits to use.
        cycle (int): The cycles of sequence.
        seed (int): The seed for the random number generator.
        interleaves (list): The interleaves to use.

    Returns:
        list: The XEB circuit.
    """
    if isinstance(qubits, (str, int)):
        qubits = {0: qubits}
    else:
        qubits = {i: q for i, q in enumerate(qubits)}

    interleaves = itertools.cycle(interleaves)

    ret = []
    rng = random.Random(seed)

    for _ in range(cycle):
        try:
            int_seq = next(interleaves)
        except StopIteration:
            int_seq = []
        ret.extend(mapping_qubits(int_seq, qubits))
        for q in qubits:
            ret.append((rng.choice(base), q))

    return ret
