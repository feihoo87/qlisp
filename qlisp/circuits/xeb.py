import random
from typing import Iterable, Optional, Union

import numpy as np
from numpy import pi

from ..simple import applySeq
from .utils import mapping_qubits


def uncorrelated_entropy(D: int) -> float:
    """Uncorrelated entropy

    The uncorrelated entropy is the entropy of the uniform distribution
    over all possible bit strings of a quantum system with Hilbert space
    dimension D.
    """
    from scipy.special import polygamma

    return np.euler_gamma + polygamma(0, D)


def PT_entropy(D: int) -> float:
    """Porter-Thomas entropy

    The Porter-Thomas entropy is the entropy of the uniform distribution
    over the Hilbert space of dimension D.
    """
    return np.sum([1 / i for i in range(1, D + 1)]) - 1


def xeb_fidelity(measured: Iterable[np.array], ideal: Iterable[np.array]):
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

    Args:
        measured: measured distribution
            the last dimension is the Hilbert space dimension
            and the second last dimension is the random circuits
        ideal: ideal distribution
            the last dimension is the Hilbert space dimension
            and the second last dimension is the random circuits

    Returns:
        fidelity
    """
    measured = np.asarray(measured)
    ideal = np.asarray(ideal)
    D = ideal.shape[-1]
    b = 1 / D**2
    F = (np.mean(ideal * measured, axis=(-1, -2)) -
         b) / (np.mean(ideal**2, axis=(-1, -2)) - b)
    return F


def speckle_purity(measured: Iterable[np.array],
                   ideal: Iterable[np.array] | None = None) -> float:
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
        measured: measured distribution
            the last dimension is the Hilbert space dimension
            and the second last dimension is the random circuits
        ideal: ideal distribution
            the last dimension is the Hilbert space dimension
            and the second last dimension is the random circuits

    Returns:
        Speckle Purity
    """
    measured = np.asarray(measured)
    if ideal is None:
        D = measured.shape[-1]
        return np.var(measured, axis=(-1, -2)) * D**2 * (D + 1) / (D - 1)
    else:
        ideal = np.asarray(ideal)
        return np.var(measured, axis=(-1, -2)) / np.var(ideal, axis=(-1, -2))


def xeb_circuit(qubits: Union[int, str, tuple],
                cycle: int,
                seed: Optional[int] = None,
                entangler: list = [],
                optional_single_qutib_gates: list[Union[str, tuple]] = [
                    ('R', i * pi / 4) for i in range(8)
                ],
                ideal_distribution: bool = False):
    """Generate a random XEB circuit.

    Generate a random XEB circuit with the given qubits, cycles, seed. The
    circuit is generated by randomly interleaving single qubit gates between
    the entangler gates.

    Args:
        qubits (list): The qubits to use.
        cycle (int): The cycles of sequence.
        seed (int): The seed for the random number generator.
        entangler (list): The entangler to use.
        optional_single_qutib_gates (list): The optional single qubit gates.
        ideal_distribution (bool): Whether to return the ideal distribution.
    """
    if isinstance(qubits, (str, int)):
        qubits = {0: qubits}
    else:
        qubits = {i: q for i, q in enumerate(qubits)}

    ret = []
    rng = random.Random(seed)

    for _ in range(cycle):
        ret.extend(entangler)
        for i, q in enumerate(qubits):
            ret.append((rng.choice(optional_single_qutib_gates), i))

    if ideal_distribution:
        return mapping_qubits(ret, qubits), np.abs(applySeq(ret))**2
    return mapping_qubits(ret, qubits)
