def mapping_qubits(circuit, mapping):
    """
    Remap qubits in a circuit.

    Args:
        circuit: [(gate, target), ...]
        mapping: {old: new}
    """
    ret = []
    for gate, target in circuit:
        if isinstance(target, tuple):
            ret.append((gate, tuple(mapping.get(i, i) for i in target)))
        else:
            ret.append((gate, mapping.get(target, target)))
    return ret
