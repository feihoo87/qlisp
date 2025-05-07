from __future__ import annotations

import functools
from typing import (TYPE_CHECKING, Callable, Iterable, List, Optional, Tuple,
                    TypeVar, Union)

import numpy as np
from cycles import (is_diagonal, is_hermitian, is_normal,
                    is_special_orthogonal, is_unitary, matrix_commutes)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike


def _merge_dtypes(dtype1: 'DTypeLike', dtype2: 'DTypeLike') -> np.dtype:
    return (np.zeros(0, dtype1) + np.zeros(0, dtype2)).dtype


def all_near_zero(a: 'ArrayLike', *, atol: float = 1e-8) -> bool:
    """Checks if the tensor's elements are all near zero.

    Args:
        a: Tensor of elements that could all be near zero.
        atol: Absolute tolerance.
    """
    return bool(np.all(np.less_equal(np.abs(a), atol)))


def dot(*values: 'ArrayLike') -> np.ndarray:
    """Computes the dot/matrix product of a sequence of values.

    Performs the computation in serial order without regard to the matrix
    sizes.  If you are using this for matrices of large and differing sizes,
    consider using np.lingalg.multi_dot for better performance.

    Args:
        *values: The values to combine with the dot/matrix product.

    Returns:
        The resulting value or matrix.

    Raises:
        ValueError: If the method is called without any arguments.
    """
    if len(values) == 0:
        raise ValueError("dot must be called with arguments")

    if len(values) == 1:
        # note: it's important that we copy input arrays.
        return np.array(values[0])

    result = np.asarray(values[0])
    for value in values[1:]:
        result = np.dot(result, value)
    return result


def block_diag(*blocks: np.ndarray) -> np.ndarray:
    """Concatenates blocks into a block diagonal matrix.

    Args:
        *blocks: Square matrices to place along the diagonal of the result.

    Returns:
        A block diagonal matrix with the given blocks along its diagonal.

    Raises:
        ValueError: A block isn't square.
    """
    for b in blocks:
        if b.shape[0] != b.shape[1]:
            raise ValueError('Blocks must be square.')

    if not blocks:
        return np.zeros((0, 0), dtype=np.complex128)

    n = sum(b.shape[0] for b in blocks)
    dtype = functools.reduce(_merge_dtypes, (b.dtype for b in blocks))

    result = np.zeros(shape=(n, n), dtype=dtype)
    i = 0
    for b in blocks:
        j = i + b.shape[0]
        result[i:j, i:j] = b
        i = j

    return result


def _contiguous_groups(
        length: int, comparator: Callable[[int, int],
                                          bool]) -> List[Tuple[int, int]]:
    """Splits range(length) into approximate equivalence classes.

    Args:
        length: The length of the range to split.
        comparator: Determines if two indices have approximately equal items.

    Returns:
        A list of (inclusive_start, exclusive_end) range endpoints. Each
        corresponds to a run of approximately-equivalent items.
    """
    result = []
    start = 0
    while start < length:
        past = start + 1
        while past < length and comparator(start, past):
            past += 1
        result.append((start, past))
        start = past
    return result


def _svd_handling_empty(mat):
    if not mat.shape[0] * mat.shape[1]:
        z = np.zeros((0, 0), dtype=mat.dtype)
        return z, np.array([]), z

    return np.linalg.svd(mat)


def diagonalize_real_symmetric_matrix(
        matrix: np.ndarray,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        check_preconditions: bool = True) -> np.ndarray:
    """Returns an orthogonal matrix that diagonalizes the given matrix.

    Args:
        matrix: A real symmetric matrix to diagonalize.
        rtol: Relative error tolerance.
        atol: Absolute error tolerance.
        check_preconditions: If set, verifies that the input matrix is real and
            symmetric.

    Returns:
        An orthogonal matrix P such that P.T @ matrix @ P is diagonal.

    Raises:
        ValueError: Matrix isn't real symmetric.
    """

    if check_preconditions and (
            np.any(np.imag(matrix) != 0)
            or not is_hermitian(matrix, rtol=rtol, atol=atol)):
        raise ValueError('Input must be real and symmetric.')

    _, result = np.linalg.eigh(matrix)

    return result


def diagonalize_real_symmetric_and_sorted_diagonal_matrices(
    symmetric_matrix: np.ndarray,
    diagonal_matrix: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_preconditions: bool = True,
) -> np.ndarray:
    """Returns an orthogonal matrix that diagonalizes both given matrices.

    The given matrices must commute.
    Guarantees that the sorted diagonal matrix is not permuted by the
    diagonalization (except for nearly-equal values).

    Args:
        symmetric_matrix: A real symmetric matrix.
        diagonal_matrix: A real diagonal matrix with entries along the diagonal
            sorted into descending order.
        rtol: Relative numeric error threshold.
        atol: Absolute numeric error threshold.
        check_preconditions: If set, verifies that the input matrices commute
            and are respectively symmetric and diagonal descending.

    Returns:
        An orthogonal matrix P such that P.T @ symmetric_matrix @ P is diagonal
        and P.T @ diagonal_matrix @ P = diagonal_matrix (up to tolerance).

    Raises:
        ValueError: Matrices don't meet preconditions (e.g. not symmetric).
    """

    # Verify preconditions.
    if check_preconditions:
        if np.any(np.imag(symmetric_matrix)) or not is_hermitian(
                symmetric_matrix, rtol=rtol, atol=atol):
            raise ValueError('symmetric_matrix must be real symmetric.')
        if (not is_diagonal(diagonal_matrix, atol=atol)
                or np.any(np.imag(diagonal_matrix)) or
                np.any(diagonal_matrix[:-1, :-1] < diagonal_matrix[1:, 1:])):
            raise ValueError(
                'diagonal_matrix must be real diagonal descending.')
        if not matrix_commutes(
                diagonal_matrix, symmetric_matrix, rtol=rtol, atol=atol):
            raise ValueError('Given matrices must commute.')

    def similar_singular(i, j):
        return np.allclose(diagonal_matrix[i, i],
                           diagonal_matrix[j, j],
                           rtol=rtol)

    # Because the symmetric matrix commutes with the diagonal singulars matrix,
    # the symmetric matrix should be block-diagonal with a block boundary
    # wherever the singular values happen change. So we can use the singular
    # values to extract blocks that can be independently diagonalized.
    ranges = _contiguous_groups(diagonal_matrix.shape[0], similar_singular)

    # Build the overall diagonalization by diagonalizing each block.
    p = np.zeros(symmetric_matrix.shape, dtype=np.float64)
    for start, end in ranges:
        block = symmetric_matrix[start:end, start:end]
        p[start:end, start:end] = diagonalize_real_symmetric_matrix(
            block, rtol=rtol, atol=atol, check_preconditions=False)

    return p


def bidiagonalize_real_matrix_pair_with_symmetric_products(
    mat1: np.ndarray,
    mat2: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_preconditions: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds orthogonal matrices that diagonalize both mat1 and mat2.

    Requires mat1 and mat2 to be real.
    Requires mat1.T @ mat2 to be symmetric.
    Requires mat1 @ mat2.T to be symmetric.

    Args:
        mat1: One of the real matrices.
        mat2: The other real matrix.
        rtol: Relative numeric error threshold.
        atol: Absolute numeric error threshold.
        check_preconditions: If set, verifies that the inputs are real, and that
            mat1.T @ mat2 and mat1 @ mat2.T are both symmetric. Defaults to set.

    Returns:
        A tuple (L, R) of two orthogonal matrices, such that both L @ mat1 @ R
        and L @ mat2 @ R are diagonal matrices.

    Raises:
        ValueError: Matrices don't meet preconditions (e.g. not real).
    """

    if check_preconditions:
        if np.any(np.imag(mat1) != 0):
            raise ValueError('mat1 must be real.')
        if np.any(np.imag(mat2) != 0):
            raise ValueError('mat2 must be real.')
        if not is_hermitian(np.dot(mat1, mat2.T), rtol=rtol, atol=atol):
            raise ValueError('mat1 @ mat2.T must be symmetric.')
        if not is_hermitian(np.dot(mat1.T, mat2), rtol=rtol, atol=atol):
            raise ValueError('mat1.T @ mat2 must be symmetric.')

    # Use SVD to bi-diagonalize the first matrix.
    base_left, base_diag, base_right = _svd_handling_empty(np.real(mat1))
    base_diag = np.diag(base_diag)

    # Determine where we switch between diagonalization-fixup strategies.
    dim = base_diag.shape[0]
    rank = dim
    while rank > 0 and all_near_zero(base_diag[rank - 1, rank - 1], atol=atol):
        rank -= 1
    base_diag = base_diag[:rank, :rank]

    # Try diagonalizing the second matrix with the same factors as the first.
    semi_corrected = dot(base_left.T, np.real(mat2), base_right.T)

    # Fix up the part of the second matrix's diagonalization that's matched
    # against non-zero diagonal entries in the first matrix's diagonalization
    # by performing simultaneous diagonalization.
    overlap = semi_corrected[:rank, :rank]
    overlap_adjust = diagonalize_real_symmetric_and_sorted_diagonal_matrices(
        overlap,
        base_diag,
        rtol=rtol,
        atol=atol,
        check_preconditions=check_preconditions)

    # Fix up the part of the second matrix's diagonalization that's matched
    # against zeros in the first matrix's diagonalization by performing an SVD.
    extra = semi_corrected[rank:, rank:]
    extra_left_adjust, _, extra_right_adjust = _svd_handling_empty(extra)

    # Merge the fixup factors into the initial diagonalization.
    left_adjust = block_diag(overlap_adjust, extra_left_adjust)
    right_adjust = block_diag(overlap_adjust.T, extra_right_adjust)
    left = np.dot(left_adjust.T, base_left.T)
    right = np.dot(base_right.T, right_adjust.T)

    return left, right


def bidiagonalize_unitary_with_special_orthogonals(
    mat: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_preconditions: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finds orthogonal matrices L, R such that L @ matrix @ R is diagonal.

    Args:
        mat: A unitary matrix.
        rtol: Relative numeric error threshold.
        atol: Absolute numeric error threshold.
        check_preconditions: If set, verifies that the input is a unitary matrix
            (to the given tolerances). Defaults to set.

    Returns:
        A triplet (L, d, R) such that L @ mat @ R = diag(d). Both L and R will
        be orthogonal matrices with determinant equal to 1.

    Raises:
        ValueError: Matrices don't meet preconditions (e.g. not real).
    """

    if check_preconditions:
        if not is_unitary(mat, rtol=rtol, atol=atol):
            raise ValueError('matrix must be unitary.')

    # Note: Because mat is unitary, setting A = real(mat) and B = imag(mat)
    # guarantees that both A @ B.T and A.T @ B are Hermitian.
    left, right = bidiagonalize_real_matrix_pair_with_symmetric_products(
        np.real(mat),
        np.imag(mat),
        rtol=rtol,
        atol=atol,
        check_preconditions=check_preconditions)

    # Convert to special orthogonal w/o breaking diagonalization.
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.linalg.det(left) < 0:
            left[0, :] *= -1
        if np.linalg.det(right) < 0:
            right[:, 0] *= -1

    diag = dot(left, mat, right)

    return left, np.diag(diag), right


T = TypeVar('T')

# yapf: disable
MAGIC = np.array([[1, 0, 0, 1j],
                  [0, 1j, 1, 0],
                  [0, 1j, -1, 0],
                  [1, 0, 0, -1j]]) * np.sqrt(0.5)

MAGIC_DAG = np.conjugate(np.transpose(MAGIC))
KAK_GAMMA = np.array([[1, 1, 1, 1],
                      [1, 1, -1, -1],
                      [-1, 1, -1, 1],
                      [1, -1, -1, 1]]) * 0.25
# yapf: enable


def kron_factor_4x4_to_2x2s(
        matrix: np.ndarray,
        rtol=1e-5,
        atol=1e-8) -> Tuple[complex, np.ndarray, np.ndarray]:
    """Splits a 4x4 matrix U = kron(A, B) into A, B, and a global factor.

    Requires the matrix to be the kronecker product of two 2x2 unitaries.
    Requires the matrix to have a non-zero determinant.

    Args:
        matrix: The 4x4 unitary matrix to factor.
        rtol: Per-matrix-entry relative tolerance on equality.
        atol: Per-matrix-entry absolute tolerance on equality.

    Returns:
        A scalar factor and a pair of 2x2 unit-determinant matrices. The
        kronecker product of all three is equal to the given matrix.

    Raises:
        ValueError:
            The given matrix can't be tensor-factored into 2x2 pieces.
    """

    # Use the entry with the largest magnitude as a reference point.
    a, b = max(((i, j) for i in range(4) for j in range(4)),
               key=lambda t: abs(matrix[t]))

    # Extract sub-factors touching the reference cell.
    f1 = np.zeros((2, 2), dtype=np.complex128)
    f2 = np.zeros((2, 2), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            f1[(a >> 1) ^ i, (b >> 1) ^ j] = matrix[a ^ (i << 1), b ^ (j << 1)]
            f2[(a & 1) ^ i, (b & 1) ^ j] = matrix[a ^ i, b ^ j]

    # Rescale factors to have unit determinants.
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 /= np.sqrt(np.linalg.det(f1)) or 1
        f2 /= np.sqrt(np.linalg.det(f2)) or 1

    # Determine global phase.
    g = matrix[a, b] / (f1[a >> 1, b >> 1] * f2[a & 1, b & 1])
    if np.real(g) < 0:
        f1 *= -1
        g = -g

    if not np.allclose(matrix, g * np.kron(f1, f2), rtol=rtol, atol=atol):
        raise ValueError("Invalid 4x4 kronecker product.")

    return g, f1, f2


def so4_to_magic_su2s(
        mat: np.ndarray,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        check_preconditions: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Finds 2x2 special-unitaries A, B where mat = Mag.H @ kron(A, B) @ Mag.

    Mag is the magic basis matrix:

        1  0  0  i
        0  i  1  0
        0  i -1  0     (times sqrt(0.5) to normalize)
        1  0  0 -i

    Args:
        mat: A real 4x4 orthogonal matrix.
        rtol: Per-matrix-entry relative tolerance on equality.
        atol: Per-matrix-entry absolute tolerance on equality.
        check_preconditions: When set, the code verifies that the given
            matrix is from SO(4). Defaults to set.

    Returns:
        A pair (A, B) of matrices in SU(2) such that Mag.H @ kron(A, B) @ Mag
        is approximately equal to the given matrix.

    Raises:
        ValueError: Bad matrix.
    """
    if check_preconditions:
        if mat.shape != (4, 4) or not is_special_orthogonal(
                mat, atol=atol, rtol=rtol):
            raise ValueError('mat must be 4x4 special orthogonal.')

    ab = dot(MAGIC, mat, MAGIC_DAG)
    _, a, b = kron_factor_4x4_to_2x2s(ab, rtol, atol)

    return a, b


class KakDecomposition:
    """A convenient description of an arbitrary two-qubit operation.

    Any two qubit operation U can be decomposed into the form

        U = g · (a1 ⊗ a0) · exp(i·(x·XX + y·YY + z·ZZ)) · (b1 ⊗ b0)

    This class stores g, (b0, b1), (x, y, z), and (a0, a1).

    Attributes:
        global_phase: g from the above equation.
        single_qubit_operations_before: b0, b1 from the above equation.
        interaction_coefficients: x, y, z from the above equation.
        single_qubit_operations_after: a0, a1 from the above equation.

    References:
        'An Introduction to Cartan's KAK Decomposition for QC Programmers'
        https://arxiv.org/abs/quant-ph/0507171
    """

    def __init__(
        self,
        *,
        global_phase: complex = complex(1),
        single_qubit_operations_before: Optional[Tuple[np.ndarray,
                                                       np.ndarray]] = None,
        interaction_coefficients: Tuple[float, float, float],
        single_qubit_operations_after: Optional[Tuple[np.ndarray,
                                                      np.ndarray]] = None,
    ):
        """Initializes a decomposition for a two-qubit operation U.

        U = g · (a1 ⊗ a0) · exp(i·(x·XX + y·YY + z·ZZ)) · (b1 ⊗ b0)

        Args:
            global_phase: g from the above equation.
            single_qubit_operations_before: b0, b1 from the above equation.
            interaction_coefficients: x, y, z from the above equation.
            single_qubit_operations_after: a0, a1 from the above equation.
        """
        self.global_phase: complex = global_phase
        self.single_qubit_operations_before: Tuple[np.ndarray, np.ndarray] = (
            single_qubit_operations_before
            or (np.eye(2, dtype=np.complex64), np.eye(2, dtype=np.complex64)))
        self.interaction_coefficients = interaction_coefficients
        self.single_qubit_operations_after: Tuple[np.ndarray, np.ndarray] = (
            single_qubit_operations_after
            or (np.eye(2, dtype=np.complex64), np.eye(2, dtype=np.complex64)))

    def __repr__(self) -> str:
        return (
            f'KakDecomposition(global_phase={self.global_phase}, '
            f'single_qubit_operations_before={self.single_qubit_operations_before}, '
            f'interaction_coefficients={self.interaction_coefficients}, '
            f'single_qubit_operations_after={self.single_qubit_operations_after})'
        )


def kak_canonicalize_vector(x: float,
                            y: float,
                            z: float,
                            atol: float = 1e-9) -> KakDecomposition:
    """Canonicalizes an XX/YY/ZZ interaction by swap/negate/shift-ing axes.

    Args:
        x: The strength of the XX interaction.
        y: The strength of the YY interaction.
        z: The strength of the ZZ interaction.
        atol: How close x2 must be to π/4 to guarantee z2 >= 0

    Returns:
        The canonicalized decomposition, with vector coefficients (x2, y2, z2)
        satisfying:

            0 ≤ abs(z2) ≤ y2 ≤ x2 ≤ π/4
            if x2 = π/4, z2 >= 0

        Guarantees that the implied output matrix:

            g · (a1 ⊗ a0) · exp(i·(x2·XX + y2·YY + z2·ZZ)) · (b1 ⊗ b0)

        is approximately equal to the implied input matrix:

            exp(i·(x·XX + y·YY + z·ZZ))
    """

    phase = [complex(1)]  # Accumulated global phase.
    left = [np.eye(2)] * 2  # Per-qubit left factors.
    right = [np.eye(2)] * 2  # Per-qubit right factors.
    v = [x, y, z]  # Remaining XX/YY/ZZ interaction vector.

    # These special-unitary matrices flip the X, Y, and Z axes respectively.
    flippers = [
        np.array([[0, 1], [1, 0]]) * 1j,
        np.array([[0, -1j], [1j, 0]]) * 1j,
        np.array([[1, 0], [0, -1]]) * 1j,
    ]

    # Each of these special-unitary matrices swaps two the roles of two axes.
    # The matrix at index k swaps the *other two* axes (e.g. swappers[1] is a
    # Hadamard operation that swaps X and Z).
    swappers = [
        np.array([[1, -1j], [1j, -1]]) * 1j * np.sqrt(0.5),
        np.array([[1, 1], [1, -1]]) * 1j * np.sqrt(0.5),
        np.array([[0, 1 - 1j], [1 + 1j, 0]]) * 1j * np.sqrt(0.5),
    ]

    # Shifting strength by ½π is equivalent to local ops (e.g. exp(i½π XX)∝XX).
    def shift(k, step):
        v[k] += step * np.pi / 2
        phase[0] *= 1j**step
        right[0] = dot(flippers[k]**(step % 4), right[0])
        right[1] = dot(flippers[k]**(step % 4), right[1])

    # Two negations is equivalent to temporarily flipping along the other axis.
    def negate(k1, k2):
        v[k1] *= -1
        v[k2] *= -1
        phase[0] *= -1
        s = flippers[3 - k1 - k2]  # The other axis' flipper.
        left[1] = dot(left[1], s)
        right[1] = dot(s, right[1])

    # Swapping components is equivalent to temporarily swapping the two axes.
    def swap(k1, k2):
        v[k1], v[k2] = v[k2], v[k1]
        s = swappers[3 - k1 - k2]  # The other axis' swapper.
        left[0] = dot(left[0], s)
        left[1] = dot(left[1], s)
        right[0] = dot(s, right[0])
        right[1] = dot(s, right[1])

    # Shifts an axis strength into the range (-π/4, π/4].
    def canonical_shift(k):
        while v[k] <= -np.pi / 4:
            shift(k, +1)
        while v[k] > np.pi / 4:
            shift(k, -1)

    # Sorts axis strengths into descending order by absolute magnitude.
    def sort():
        if abs(v[0]) < abs(v[1]):
            swap(0, 1)
        if abs(v[1]) < abs(v[2]):
            swap(1, 2)
        if abs(v[0]) < abs(v[1]):
            swap(0, 1)

    # Get all strengths to (-¼π, ¼π] in descending order by absolute magnitude.
    canonical_shift(0)
    canonical_shift(1)
    canonical_shift(2)
    sort()

    # Move all negativity into z.
    if v[0] < 0:
        negate(0, 2)
    if v[1] < 0:
        negate(1, 2)
    canonical_shift(2)

    # If x = π/4, force z to be positive
    if v[0] > np.pi / 4 - atol and v[2] < 0:
        shift(0, -1)
        negate(0, 2)

    return KakDecomposition(
        global_phase=phase[0],
        single_qubit_operations_after=(left[1], left[0]),
        interaction_coefficients=(v[0], v[1], v[2]),
        single_qubit_operations_before=(right[1], right[0]),
    )


def kak_decomposition(
    unitary: Union[np.ndarray, KakDecomposition],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_preconditions: bool = True,
) -> KakDecomposition:
    """Decomposes a 2-qubit unitary into 1-qubit ops and XX/YY/ZZ interactions.

    Args:
        unitary: The value to decompose. (a 4x4 unitary matrix).
        rtol: Per-matrix-entry relative tolerance on equality.
        atol: Per-matrix-entry absolute tolerance on equality.
        check_preconditions: If set, verifies that the input corresponds to a
            4x4 unitary before decomposing.

    Returns:
        A `KakDecomposition` canonicalized such that the interaction
        coefficients x, y, z satisfy:

            0 ≤ abs(z2) ≤ y2 ≤ x2 ≤ π/4
            if x2 = π/4, z2 >= 0

    Raises:
        ValueError: Bad matrix.
        ArithmeticError: Failed to perform the decomposition.

    References:
        'An Introduction to Cartan's KAK Decomposition for QC Programmers'
        https://arxiv.org/abs/quant-ph/0507171
    """
    if isinstance(unitary, KakDecomposition):
        return unitary
    else:
        mat = np.asarray(unitary)

    if check_preconditions and (mat.shape != (4, 4)
                                or not is_unitary(mat, rtol=rtol, atol=atol)):
        raise ValueError(
            f'Input must correspond to a 4x4 unitary matrix. Received matrix:\n{mat}'
        )

    # Diagonalize in magic basis.
    left, d, right = bidiagonalize_unitary_with_special_orthogonals(
        MAGIC_DAG @ mat @ MAGIC,
        atol=atol,
        rtol=rtol,
        check_preconditions=False)

    # Recover pieces.
    a1, a0 = so4_to_magic_su2s(left.T,
                               atol=atol,
                               rtol=rtol,
                               check_preconditions=False)
    b1, b0 = so4_to_magic_su2s(right.T,
                               atol=atol,
                               rtol=rtol,
                               check_preconditions=False)
    w, x, y, z = (KAK_GAMMA @ np.angle(d).reshape(-1, 1)).flatten()
    g = np.exp(1j * w)

    # Canonicalize.
    inner_cannon = kak_canonicalize_vector(x, y, z)

    b1 = np.dot(inner_cannon.single_qubit_operations_before[0], b1)
    b0 = np.dot(inner_cannon.single_qubit_operations_before[1], b0)
    a1 = np.dot(a1, inner_cannon.single_qubit_operations_after[0])
    a0 = np.dot(a0, inner_cannon.single_qubit_operations_after[1])
    return KakDecomposition(
        interaction_coefficients=inner_cannon.interaction_coefficients,
        global_phase=g * inner_cannon.global_phase,
        single_qubit_operations_before=(b1, b0),
        single_qubit_operations_after=(a1, a0),
    )


def kak_vector(
    unitary: Union[Iterable[np.ndarray], np.ndarray],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_preconditions: bool = True,
) -> np.ndarray:
    r"""Compute the KAK vectors of one or more two qubit unitaries.

    Any 2 qubit unitary may be expressed as

    $$ U = k_l A k_r $$
    where $k_l, k_r$ are single qubit (local) unitaries and

    $$ A= \exp \left(i \sum_{s=x,y,z} k_s \sigma_{s}^{(0)} \sigma_{s}^{(1)}
                 \right) $$

    The vector entries are ordered such that
        $$ 0 ≤ |k_z| ≤ k_y ≤ k_x ≤ π/4 $$
    if $k_x$ = π/4, $k_z \geq 0$.

    References:
        The appendix section of "Lower bounds on the complexity of simulating
        quantum gates".
        http://arxiv.org/abs/quant-ph/0307190v1

    Examples:
        >>> kak_vector(np.eye(4))
        array([0., 0., 0.])
        >>> unitaries = [seq2mat([('CZ', 0, 1)]), seq2mat([('iSWAP', 0, 1)])]
        >>> kak_vector(unitaries) * 4 / np.pi
        array([[ 1.,  0., -0.],
               [ 1.,  1.,  0.]])

    Args:
        unitary: A unitary matrix, or a multi-dimensional array of unitary
            matrices. Must have shape (..., 4, 4), where the last two axes are
            for the unitary matrix and other axes are for broadcasting the kak
            vector computation.
        rtol: Per-matrix-entry relative tolerance on equality. Used in unitarity
            check of input.
        atol: Per-matrix-entry absolute tolerance on equality. Used in unitarity
            check of input. This also determines how close $k_x$ must be to π/4
            to guarantee $k_z$ ≥ 0. Must be non-negative.
        check_preconditions: When set to False, skips verifying that the input
            is unitary in order to increase performance.

    Returns:
        The KAK vector of the given unitary or unitaries. The output shape is
        the same as the input shape, except the two unitary matrix axes are
        replaced by the kak vector axis (i.e. the output has shape
        `unitary.shape[:-2] + (3,)`).

    Raises:
        ValueError: If `atol` is negative or if the unitary has the wrong shape.
    """
    unitary = np.asarray(unitary)
    if len(unitary) == 0:
        return np.zeros(shape=(0, 3), dtype=np.float64)

    if unitary.ndim < 2 or unitary.shape[-2:] != (4, 4):
        raise ValueError(
            f'Expected input unitary to have shape (...,4,4), but got {unitary.shape}.'
        )

    if atol < 0:
        raise ValueError(f'Input atol must be positive, got {atol}.')

    if check_preconditions:
        actual = np.einsum('...ba,...bc', unitary.conj(), unitary) - np.eye(4)
        if not np.allclose(actual, np.zeros_like(actual), rtol=rtol,
                           atol=atol):
            raise ValueError(
                'Input must correspond to a 4x4 unitary matrix or tensor of '
                f'unitary matrices. Received input:\n{unitary}')

    UB = np.einsum('...ab,...bc,...cd', MAGIC_DAG, unitary, MAGIC)

    m = np.einsum('...ab,...cb', UB, UB)

    evals, _ = np.linalg.eig(m)

    # The algorithm in the appendix mentioned above is slightly incorrect in
    # that it only works for elements of SU(4). A phase correction must be
    # added to deal with U(4).
    with np.errstate(divide="ignore", invalid="ignore"):
        phases = np.log(-1j * np.linalg.det(unitary)).imag + np.pi / 2
    evals *= np.exp(-1j * phases / 2)[..., np.newaxis]

    # The following steps follow the appendix exactly.
    S2 = np.log(-1j * evals).imag + np.pi / 2
    S2 = np.sort(S2, axis=-1)[..., ::-1]

    n_shifted = (np.round(S2.sum(axis=-1) / (2 * np.pi))).astype(int)
    for n in range(1, 5):
        S2[n_shifted == n, :n] -= 2 * np.pi

    # Fix pathological case of SWAP gate
    S2[n_shifted == -1, :3] += 2 * np.pi

    k_vec = (np.einsum('ab,...b', KAK_GAMMA, S2))[..., 1:] / 2

    return _canonicalize_kak_vector(k_vec, atol)


def _canonicalize_kak_vector(k_vec: np.ndarray, atol: float) -> np.ndarray:
    r"""Map a KAK vector into its Weyl chamber equivalent vector.

    This implementation is vectorized but does not produce the single qubit
    unitaries required to bring the KAK vector into canonical form.

    Args:
        k_vec: The KAK vector to be canonicalized. This input may be vectorized,
            with shape (...,3), where the final axis denotes the k_vector and
            all other axes are broadcast.
        atol: How close x2 must be to π/4 to guarantee z2 >= 0.

    Returns:
        The canonicalized decomposition, with vector coefficients (x2, y2, z2)
        satisfying:

            0 ≤ abs(z2) ≤ y2 ≤ x2 ≤ π/4
            if x2 = π/4, z2 >= 0
        The output is vectorized, with shape k_vec.shape[:-1] + (3,).
    """

    # Get all strengths to (-¼π, ¼π]
    k_vec = np.mod(k_vec + np.pi / 4, np.pi / 2) - np.pi / 4

    # Sort in descending order with respect to absolute value.
    order = np.argsort(np.abs(k_vec), axis=-1)
    k_vec = np.take_along_axis(k_vec, order, axis=-1)[..., ::-1]

    # Multiply x,z and y,z components by -1 to fix x,y sign.
    x_negative = k_vec[..., 0] < 0
    k_vec[x_negative, 0] *= -1
    k_vec[x_negative, 2] *= -1
    y_negative = k_vec[..., 1] < 0
    k_vec[y_negative, 1] *= -1
    k_vec[y_negative, 2] *= -1

    # If x = π/4, force z to be positive.
    x_is_pi_over_4 = np.isclose(k_vec[..., 0], np.pi / 4, atol=atol)
    z_is_negative = k_vec[..., 2] < 0
    need_diff = np.logical_and(x_is_pi_over_4, z_is_negative)
    # -1 to x and z components, then shift x up by pi/2. Since x is pi/4, we
    # actually do nothing to that index.
    k_vec[need_diff, 2] *= -1

    return k_vec
