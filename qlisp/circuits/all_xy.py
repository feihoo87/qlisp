_ALLXYSeq = [('I', 'I'), ('X', 'X'), ('Y', 'Y'), ('X', 'Y'), ('Y', 'X'),
             ('X/2', 'I'), ('Y/2', 'I'), ('X/2', 'Y/2'), ('Y/2', 'X/2'),
             ('X/2', 'Y'), ('Y/2', 'X'), ('X', 'Y/2'), ('Y', 'X/2'),
             ('X/2', 'X'), ('X', 'X/2'), ('Y/2', 'Y'), ('Y', 'Y/2'),
             ('X', 'I'), ('Y', 'I'), ('X/2', 'X/2'), ('Y/2', 'Y/2')]


def ALLXY(qubit, i):
    assert 0 <= i < len(
        _ALLXYSeq), f"i={i} is out of range(0, {len(_ALLXYSeq)})"
    return [(gate, qubit) for gate in _ALLXYSeq[i]]
