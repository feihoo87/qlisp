import numpy as np
import scipy.sparse as sp


def issparse(qob):
    """Checks if ``qob`` is explicitly sparse.
    """
    return isinstance(qob, sp.spmatrix)


def make_immutable(mat):
    """Make array read only, in-place.
    Parameters
    ----------
    mat : sparse or dense array
        Matrix to make immutable.
    """
    if issparse(mat):
        mat.data.flags.writeable = False
        if mat.format in {'csr', 'csc', 'bsr'}:
            mat.indices.flags.writeable = False
            mat.indptr.flags.writeable = False
        elif mat.format == 'coo':
            mat.row.flags.writeable = False
            mat.col.flags.writeable = False
    else:
        mat.flags.writeable = False
    return mat


# Paulis
def sigmaI():
    return make_immutable(np.eye(2, dtype=complex))


def sigmaX():
    return make_immutable(np.array([[0, 1], [1, 0]], dtype=complex))


def sigmaY():
    return make_immutable(np.array([[0, -1j], [1j, 0]], dtype=complex))


def sigmaZ():
    return make_immutable(np.array([[1, 0], [0, -1]], dtype=complex))


def sigmaP():
    return make_immutable(np.array([[0, 0], [0, 1]], dtype=complex))


def sigmaM():
    return make_immutable(np.array([[0, 1], [0, 0]], dtype=complex))


# Bell states
def BellPhiP():
    return make_immutable(np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2))


def BellPhiM():
    return make_immutable(np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2))


def BellPsiP():
    return make_immutable(np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2))


def BellPsiM():
    return make_immutable(np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2))


phiplus, phiminus = BellPhiP(), BellPhiM()
psiplus, psiminus = BellPsiP(), BellPsiM()

# Clifford gates
H = make_immutable(np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2))
S = make_immutable(np.array([[1, 0], [0, 1j]], dtype=complex))
Sdag = make_immutable(np.array([[1, 0], [0, -1j]], dtype=complex))

# T gates
T = make_immutable(
    np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex))
Tdag = make_immutable(
    np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex))

# two qubit gates
CX = make_immutable(
    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
             dtype=complex))
CZ = make_immutable(
    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
             dtype=complex))
iSWAP = make_immutable(
    np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
             dtype=complex))
INViSWAP = make_immutable(
    np.array([[1, 0, 0, 0], [0, 0,-1j, 0], [0,-1j, 0, 0], [0, 0, 0, 1]],
             dtype=complex))
SWAP = make_immutable(
    np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
             dtype=complex))
SQiSWAP = make_immutable(
    np.array([[1, 0, 0, 0], [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
              [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0], [0, 0, 0, 1]],
             dtype=complex))
INVSQiSWAP = make_immutable(
    np.array([[1, 0, 0, 0], [0, 1 / np.sqrt(2), -1j / np.sqrt(2), 0],
              [0, -1j / np.sqrt(2), 1 / np.sqrt(2), 0], [0, 0, 0, 1]],
             dtype=complex))
CR = make_immutable(
    np.array([[1, 1j, 0, 0], [1j, 1, 0, 0], [0, 0, 1, -1j], [0, 0, -1j, 1]],
             dtype=complex) / np.sqrt(2))

##########################################################


def U(theta, phi, lambda_, delta=0):
    """general unitary
    
    Any general unitary could be implemented in 2 pi/2-pulses on hardware

    U(theta, phi, lambda_, delta) = \\
        np.exp(1j * delta) * \\
        U(0, 0, theta + phi + lambda_) @ \\
        U(np.pi / 2, p2, -p2) @ \\
        U(np.pi / 2, p1, -p1))

    or  = \\
        np.exp(1j * delta) * \\
        U(0, 0, theta + phi + lambda_) @ \\
        rfUnitary(np.pi / 2, p2 + pi / 2) @ \\
        rfUnitary(np.pi / 2, p1 + pi / 2)
    
    where p1 = -lambda_ - pi / 2
          p2 = pi / 2 - theta - lambda_
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    a, b = (phi + lambda_) / 2, (phi - lambda_) / 2
    d = np.exp(1j * delta)
    return d * np.array([[c * np.exp(-1j * a), -s * np.exp(-1j * b)],
                         [s * np.exp(1j * b), c * np.exp(1j * a)]])


def _check_angle(x):
    x = np.fmod(np.fmod(x, np.pi * 4) + np.pi * 4, np.pi * 4)
    flag = 0
    for i in range(3):
        if x[i] > np.pi * 2:
            x[i] -= np.pi * 4
        if abs(x[i]) > np.pi:
            x[i] -= np.sign(x[i]) * (np.pi * 2)
            flag += 1
    x[3] = np.fmod(x[3] + np.pi * flag, np.pi * 2)
    if abs(x[3]) > np.pi:
        x[3] -= np.sign(x[3]) * (np.pi * 2)
    return x


def Unitary2Angles(U: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    if np.abs(U[0, 0]) < eps:
        theta = np.pi
        delta = (np.angle(U[1, 0]) + np.angle(-U[0, 1])) / 2
        phi = np.angle(U[1, 0]) - np.angle(-U[0, 1])
        lambda_ = 0
    elif np.abs(U[0, 1]) < eps:
        theta = 0
        delta = (np.angle(U[0, 0]) + np.angle(U[1, 1])) / 2
        phi = -(np.angle(U[0, 0]) - np.angle(U[1, 1])) / 2
        lambda_ = phi
    else:
        delta = np.angle(U[0, 0])
        U = U / np.exp(1j * delta)
        theta = 2 * np.arccos(U[0, 0])
        phi = np.angle(U[1, 0])
        lambda_ = np.angle(-U[0, 1])
        delta += (phi + lambda_) / 2
    return _check_angle(np.array([theta, phi, lambda_, delta]).real)


def rfUnitary(theta, phi):
    """
    Gives the unitary operator for an ideal microwave gate.
    phi gives the rotation axis on the plane of the bloch sphere (RF drive phase)
    theta is the rotation angle of the gate (pulse area)

    rfUnitary(theta, phi) := expm(-1j * theta / 2 * \\
        (sigmaX() * cos(phi) + sigmaY() * sin(phi)))

    rfUnitary(theta, phi + pi/2) == U(theta, phi, -phi)
    or
    rfUnitary(theta, phi) == U(theta, phi - pi/2, pi/2 - phi)
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s * np.exp(-1j * phi)],
                     [-1j * s * np.exp(1j * phi), c]])


def fSim(theta, phi):
    c, s = np.cos(theta), np.sin(theta)
    p = np.exp(-1j * phi)
    return np.array([
        [1,     0,     0,     0],
        [0,     c, -1j*s,     0],
        [0, -1j*s,     c,     0],
        [0,     0,     0,     p]
    ]) #yapf: disable


def A(x, y, z):
    """
    Gives the unitary A in KAK decomposition.

    A(x, y, z) := expm(1j * (x * XX + y * YY + z * ZZ))
    where XX = kron(sigma_x, sigma_x)
          YY = kron(sigma_y, sigma_y)
          ZZ = kron(sigma_z, sigma_z)
    """
    e = np.exp(1j * z)
    ec = np.exp(-1j * z)
    cm = np.cos(x - y)
    cp = np.cos(x + y)
    sm = np.sin(x - y)
    sp = np.sin(x + y)

    return np.array([
        [   e*cm,        0,        0, 1j*e*sm],
        [      0,    ec*cp, 1j*ec*sp,       0],
        [      0, 1j*ec*sp,    ec*cp,       0],
        [1j*e*sm,        0,        0,    e*cm]
    ]) #yapf: disable


def Rzx(theta):
    """
    Gives the unitary operator:
        Rzx(theta) := expm(-1j * theta / 2 * ZX)
    """
    c, s = np.cos(theta / 2), 1j * np.sin(theta / 2)
    return np.array([
        [c, -s, 0, 0],
        [-s, c, 0, 0],
        [ 0, 0, c, s],
        [ 0, 0, s, c]
    ]) #yapf: disable


c, s = np.cos(np.pi / 8), np.sin(np.pi / 8)

B = make_immutable(np.array([
    [c, 0, 0, 1j * s],
    [0, s, 1j * c, 0],
    [0, 1j * c, s, 0],
    [1j * s, 0, 0, c]])) #yapf: disable

M = make_immutable(np.array([
    [1, 0, 0, 1j],
    [0, 1j, 1, 0],
    [0, 1j, -1, 0],
    [1, 0, 0, -1j]]) * np.sqrt(0.5)) #yapf: disable

M_DAG = make_immutable(M.T.conj())


def Unitary(mat):
    """
    Returns the unitary operator for a given matrix.
    """
    if not np.allclose(mat @ mat.conj().T, np.eye(mat.shape[0])):
        raise ValueError("Input matrix is not unitary.")
    if not np.allclose(np.linalg.det(mat), 1):
        raise ValueError("Input matrix is not unitary.")
    if 2**round(np.log2(mat.shape[0])) != mat.shape[0]:
        raise ValueError("Rows of input matrix is not a power of 2.")

    return mat


def synchronize_global_phase(U):
    """
    将第一个非零的矩阵元相位转成 0，以保证仅相差一个全局相位的矩阵具有相同的表示。
    优先处理对角元，再依次处理下三角阵。
    """
    assert U.shape[0] == U.shape[1]
    for i in range(U.shape[0]):
        for j in range(U.shape[0] - i):
            if np.abs(U[j + i, j]) > 1e-9:
                U = U * np.abs(U[j + i, j]) / U[j + i, j]
                return U
    return 0 * U
