import random
from typing import Sequence

import numpy as np

from qlisp.clifford import CliffordGroup, Cycles
from qlisp.clifford.utils import (one_qubit_clifford_seq,
                                  two_qubit_clifford_circuit)

two_qubit_clifford_group = CliffordGroup(2)
two_qubit_clifford_elements = {}

groups = {}


def one_qubit_rb(cycles: int,
                 interleaved: list = [],
                 seed: int | None = None) -> list:
    """
    Generate a random one qubit RB circuit.

    Args:
        cycles (int): The cycles of sequence.
        interleaved (list): The interleaved to use.
        seed (int): The seed for the random number generator.
    """
    from qlisp import seq2mat
    from qlisp.matricies import Unitary2Angles

    rng = random.Random()
    if seed is not None:
        rng.seed(int(seed))

    circ = []
    for i in range(cycles):
        gate = rng.choice(one_qubit_clifford_seq)
        circ.append((gate, 0))
        for gate in interleaved:
            circ.append((gate, 0))
    U = seq2mat(circ)
    theta, phi, lambda_, *_ = Unitary2Angles(np.linalg.inv(U))
    circ.append((('u3', theta, phi, lambda_), 0))

    return circ


def two_qubit_rb(cycles: int,
                 interleaved: list = [],
                 allowed_two_qubit_gates: Sequence[str] = ('CZ', 'CX', 'iSWAP',
                                                           'SWAP'),
                 seed: int | None = None) -> list:
    """
    Generate a random two qubit RB circuit.

    Args:
        cycles (int): The cycles of sequence.
        interleaved (list): The interleaved to use.
        allowed_two_qubit_gates (list): The allowed two qubit gates.
        seed (int): The seed for the random number generator.
    """

    if len(two_qubit_clifford_elements) == 0:
        for i in range(11520):
            circ = two_qubit_clifford_circuit(i)
            perm = two_qubit_clifford_group.circuit_to_permutation(circ)
            two_qubit_clifford_elements[perm] = i
            two_qubit_clifford_elements[i] = perm

    rng = random.Random()
    if seed is not None:
        rng.seed(int(seed))

    circ = []
    perm = Cycles()
    interleaved_perm = two_qubit_clifford_group.circuit_to_permutation(
        interleaved)
    for i in range(cycles):
        n = rng.choice(range(two_qubit_clifford_group.order()))
        circ.extend(two_qubit_clifford_circuit(n, allowed_two_qubit_gates))
        circ.extend(interleaved)
        perm = perm * two_qubit_clifford_elements[n] * interleaved_perm
    inv = two_qubit_clifford_elements[perm.inv()]
    circ.extend(two_qubit_clifford_circuit(inv, allowed_two_qubit_gates))

    return circ


def n_qubit_rb(n: int,
               cycles: int,
               interleaved: list = [],
               two_qubit_gate='CZ',
               seed=None):
    """
    Generate a random n qubit RB circuit.

    Args:
        n (int): The number of qubits.
        cycles (int): The cycles of sequence.
        two_qubit_gate (str): The two qubit gate to use.
        interleaved (list): The interleaved to use.
        seed (int): The seed for the random number generator.
    """
    if (n, two_qubit_gate) not in groups:
        groups[(n,
                two_qubit_gate)] = CliffordGroup(n,
                                                 two_qubit_gate=two_qubit_gate)
    group = groups[(n, two_qubit_gate)]

    if seed is None:
        rng = random.Random()
    else:
        rng = random.Random(seed)

    interleaved_perm = group.circuit_to_permutation(interleaved)
    circ = []
    perm = Cycles()
    for i in range(cycles):
        p = group.random(rng=rng)
        circ.extend(group.permutation_to_circuit(p))
        circ.extend(interleaved)
        perm = perm * p * interleaved_perm
    circ.extend(group.permutation_to_circuit(perm.inv()))
    return circ


def mapping_qubits(circuit, mapping):
    """
    Remap qubits in a circuit.

    Args:
        circuit: [(gate, *target), ...]
        mapping: {old: new}
    """
    ret = []
    for gate, *target in circuit:
        ret.append((gate, *tuple(mapping.get(i, i) for i in target)))
    return ret


def circuit_one_qubit_rb(qubits,
                         cycles: int,
                         interleaved: list | None = None,
                         seeds=None):
    if interleaved is None:
        interleaved = []

    circs = []
    if isinstance(seeds, (int, np.int_)):
        seeds = [seeds] * len(qubits)
    for q, seed in zip(qubits, seeds):
        circs.append(
            mapping_qubits(one_qubit_rb(cycles, interleaved, seed), {0: q}))

    circ = []

    for segs in zip(*circs):
        circ.extend(list(segs))
        circ.append(('Barrier', tuple(qubits)))

    return circ


def circuit_two_qubit_rb(pairs,
                         cycles: int,
                         interleaved: list | None = None,
                         allowed_two_qubit_gates: Sequence[str] = ('CZ', ),
                         seeds=None):
    if interleaved is None:
        interleaved = []

    circs = []
    if isinstance(seeds, (int, np.int_)):
        seeds = [seeds] * len(pairs)
    for pair, seed in zip(pairs, seeds):
        circs.append(
            mapping_qubits(
                two_qubit_rb(cycles, interleaved, allowed_two_qubit_gates,
                             seed), {
                                 0: pair[0],
                                 1: pair[1]
                             }))

    return circs


def circuit_rb(qubit_groups: list[list[str | int]],
               cycles: int,
               interleaved: list | None = None,
               two_qubit_gate='CZ',
               seeds=None):
    """
    Generate random RB circuits.

    Args:
        qubit_groups: [[qubits], ...]
            List of qubit groups. each group is a list of qubits.
            If the group has only one or two qubit, it will generate one or two qubit RB.
            If the group has more than two qubits, it will generate n qubit RB, but the circuit may not be optimal.
            the length of the group could be different. but longer than the number of qubits used in the interleaved.
        cycles (int): The cycles of sequence.
            the number of random clifford gates in the sequence.
        interleaved (list): The interleaved to use.
            the interleaved could be a list of gates, each gate is a tuple of gate and qubits.
            the qubits could be a tuple of qubits, or a single qubit.
        two_qubit_gate (str): The two qubit gate to use.
            the two qubit gate could be 'CZ', 'CX' or 'iSWAP'.
        seeds (int | list[int]): The seed for the random number generator.
            seeds could be a list of seeds, or a single seed for all qubit groups.

    Returns:
        list of generated circuits for each qubit group.
    """
    if interleaved is None:
        interleaved = []
    interleaved_qubits = set()
    for gate, *qubits in interleaved:
        interleaved_qubits.update(*qubits)

    circs = []
    if isinstance(seeds, (int, np.int_)) or seeds is None:
        seeds = [seeds] * len(qubit_groups)
    for qubits, seed in zip(qubit_groups, seeds):
        if len(qubits) == 1:
            if interleaved_qubits.issubset({
                    0,
            }):
                circ = one_qubit_rb(cycles, interleaved, seed)
            else:
                raise ValueError(
                    "Interleaved qubits must be 0 for one qubit RB.")
        elif len(qubits) == 2:
            if interleaved_qubits.issubset({0, 1}):
                circ = two_qubit_rb(cycles, interleaved, (two_qubit_gate, ),
                                    seed)
            else:
                raise ValueError(
                    "Interleaved qubits must be 0, 1 for two qubit RB.")
        else:
            if interleaved_qubits.issubset(set(range(len(qubits)))):
                circ = n_qubit_rb(len(qubits), cycles, interleaved,
                                  two_qubit_gate, seed)
            else:
                raise ValueError(
                    f"Interleaved qubits must be {set(range(len(qubits)))} for {len(qubits)} qubit RB."
                )
        mapping = {i: q for i, q in enumerate(qubits)}
        circs.append(mapping_qubits(circ, mapping))
    return circs
