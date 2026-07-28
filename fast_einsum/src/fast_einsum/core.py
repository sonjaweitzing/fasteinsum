"""
Copyright (c) 2023 Johnnie Gray

This code base includes software developed by Johnnie Gray (https://github.com/jcmgray/einsum_bmm, commit 4ff3f2b).
The original code is licensed under the GNU General Public License v3.0.
A copy of this license is included in this distribution.

Changes made with this distribution:
    1. Changed einsum function to use our library bmm function
    2. Reshaped tensors are copied to C-contiguous memory layout before calling the bmm function.
    3. Autoray replaced with numpy
    4. Removed tensordot functionality
"""

import math
import functools
import numpy as np
from . import py_bmm

@functools.lru_cache(2 ** 12)
def _sanitize_equation(eq):
    """Get the input and output indices of an equation, computing the output
    implicitly as the sorted sequence of every index that appears exactly once
    if it is not provided.
    """
    # remove spaces
    eq = eq.replace(" ", "")

    if "..." in eq:
        raise NotImplementedError("Ellipsis not supported.")

    if "->" not in eq:
        lhs = eq
        tmp_subscripts = lhs.replace(",", "")
        out = "".join(
            # sorted sequence of indices
            s
            for s in sorted(set(tmp_subscripts))
            # that appear exactly once
            if tmp_subscripts.count(s) == 1
        )
    else:
        lhs, out = eq.split("->")
    return lhs, out


@functools.lru_cache(2 ** 12)
def _parse_einsum_single(eq, shape):
    """Cached parsing of a single term einsum equation into the necessary
    sequence of arguments for axes diagonals, sums, and transposes.
    """
    lhs, out = _sanitize_equation(eq)

    # parse each index
    need_to_diag = []
    need_to_sum = []
    seen = set()
    for ix in lhs:
        if ix in need_to_diag:
            continue
        if ix in seen:
            # if we have already seen this index, we need to take the diagonal.
            need_to_diag.append(ix)
            continue
        seen.add(ix)
        if ix not in out:
            # if we have not seen this index in the output, we need to sum it.
            need_to_sum.append(ix)

    # first handle diagonal reductions
    if need_to_diag:
        diag_selectors = []
        sizes = dict(zip(lhs, shape))
        while need_to_diag:
            ixd = need_to_diag.pop()
            dinds = tuple(range(sizes[ixd]))

            # construct advanced indexing object
            selector = tuple(dinds if ix == ixd else slice(None) for ix in lhs)
            diag_selectors.append(selector)

            # after taking the diagonal what are new indices?
            ixd_contiguous = ixd * lhs.count(ixd)
            if ixd_contiguous in lhs:
                # contiguous axes, new axis is at same position
                lhs = lhs.replace(ixd_contiguous, ixd)
            else:
                # non-contiguous, new axis is at beginning
                lhs = ixd + lhs.replace(ixd, "")
    else:
        diag_selectors = None

    # then sum reductions
    if need_to_sum:
        sum_axes = tuple(map(lhs.index, need_to_sum))
        for ix in need_to_sum:
            lhs = lhs.replace(ix, "")
    else:
        sum_axes = None

    # then transposition
    if lhs == out:
        perm = None
    else:
        perm = tuple(lhs.index(ix) for ix in out)

    return diag_selectors, sum_axes, perm


def _parse_pure_multiplication(a_term, b_term, out, sizes):
    desired_a = ""
    desired_b = ""
    new_shape_a = []
    new_shape_b = []
    for ix in out:
        if ix in a_term:
            desired_a += ix
            new_shape_a.append(sizes[ix])
        else:
            new_shape_a.append(1)
        if ix in b_term:
            desired_b += ix
            new_shape_b.append(sizes[ix])
        else:
            new_shape_b.append(1)

    if desired_a != a_term:
        eq_a = f"{a_term}->{desired_a}"
    else:
        eq_a = None
    if desired_b != b_term:
        eq_b = f"{b_term}->{desired_b}"
    else:
        eq_b = None

    return (
        eq_a,
        eq_b,
        new_shape_a,
        new_shape_b,
        None,  # new_shape_ab, not needed since not fusing
        None,  # perm_ab, not needed as we transpose a and b first
        True,  # pure_multiplication=True
    )


@functools.lru_cache(2 ** 12)
def parse_double_eq(eq, shape_a, shape_b):
    """Cached parsing of a two term einsum equation into the necessary
    sequence of arguments for contracttion via batched matrix multiplication.
    The steps we need to specify are:

        1. Remove repeated and trivial indices from the left and right terms,
           and transpose them, done as a single einsum.
        2. Fuse the remaining indices so we have two 3D tensors.
        3. Perform the batched matrix multiplication.
        4. Unfuse the output to get the desired final index order.
    """
    lhs, out = _sanitize_equation(eq)
    a_term, b_term = lhs.split(",")

    if len(a_term) != len(shape_a):
        raise ValueError(f"Term '{a_term}' does not match shape {shape_a}.")
    if len(b_term) != len(shape_b):
        raise ValueError(f"Term '{b_term}' does not match shape {shape_b}.")

    bat_inds = []  # appears on A, B, O
    con_inds = []  # appears on A, B, .
    a_keep = []    # appears on A, ., O
    b_keep = []    # appears on ., B, O
    sizes = {}

    # parse left term
    seen = set()
    for ix, d in zip(a_term, shape_a):
        # set or check size
        if sizes.setdefault(ix, d) != d:
            raise ValueError(
                f"Index {ix} has mismatched sizes {sizes[ix]} and {d}."
            )

        if ix in seen:
            continue
        seen.add(ix)

        if ix in b_term:
            if ix in out:
                bat_inds.append(ix)
            else:
                con_inds.append(ix)
        elif ix in out:
            a_keep.append(ix)

    # parse right term
    seen.clear()
    for ix, d in zip(b_term, shape_b):
        # set or check size
        if sizes.setdefault(ix, d) != d:
            raise ValueError(
                f"Index {ix} has mismatched sizes {sizes[ix]} and {d}."
            )

        if ix in seen:
            continue
        seen.add(ix)

        if ix not in a_term:
            if ix in out:
                b_keep.append(ix)

    if not con_inds:
        # contraction is pure multiplication
        return _parse_pure_multiplication(a_term, b_term, out, sizes)

    # take diagonal, remove any trivial axes and transpose left
    desired_a = "".join((*bat_inds, *a_keep, *con_inds))
    if a_term != desired_a:
        eq_a = f"{a_term}->{desired_a}"
    else:
        eq_a = None

    # take diagonal, remove any trivial axes and transpose right
    desired_b = "".join((*bat_inds, *con_inds, *b_keep))
    if b_term != desired_b:
        eq_b = f"{b_term}->{desired_b}"
    else:
        eq_b = None

    # then we want to permute the matmul produced output:
    out_produced = "".join((*bat_inds, *a_keep, *b_keep))
    perm_ab = tuple(out_produced.index(ix) for ix in out)
    if perm_ab == tuple(range(len(perm_ab))):
        perm_ab = None

    # then we want to reshape
    if bat_inds:
        lgroups = (bat_inds, a_keep, con_inds)
        rgroups = (bat_inds, con_inds, b_keep)
        ogroups = (bat_inds, a_keep, b_keep)
    else:
        # avoid size 1 batch dimension if no batch indices
        lgroups = (a_keep, con_inds)
        rgroups = (con_inds, b_keep)
        ogroups = (a_keep, b_keep)

    if any(len(group) != 1 for group in lgroups):
        new_shape_a = tuple(
            math.prod(sizes[ix] for ix in ix_group) for ix_group in lgroups
        )
    else:
        new_shape_a = None

    if any(len(group) != 1 for group in rgroups):
        new_shape_b = tuple(
            math.prod(sizes[ix] for ix in ix_group) for ix_group in rgroups
        )
    else:
        new_shape_b = None

    if any(len(group) != 1 for group in ogroups):
        new_shape_ab = tuple(
            sizes[ix] for ix_group in ogroups for ix in ix_group
        )
    else:
        new_shape_ab = None

    return (
        eq_a,
        eq_b,
        new_shape_a,
        new_shape_b,
        new_shape_ab,
        perm_ab,
        False,  # pure_multiplication=False
    )


def _einsum_single(eq, x):
    """Einsum on a single tensor, via three steps: diagonal selection
    (via advanced indexing), axes summations, transposition. The logic for each
    is cached based on the equation and array shape, and each step is only
    performed if necessary.
    """
    try:
        return np.einsum(eq, x)
    except ImportError:
        pass

    diag_sels, sum_axes, perm = _parse_einsum_single(eq, np.shape(x))

    if diag_sels is not None:
        # diagonal reduction via advanced indexing
        # e.g ababbac->abc
        for selector in diag_sels:
            x = x[selector]

    if sum_axes is not None:
        # trivial removal of axes via summation
        # e.g. abc->c
        x = np.sum(x, sum_axes)

    if perm is not None:
        # transpose to desired output
        # e.g. abc->cba
        x = np.transpose(x, perm)

    return x


def _do_contraction_via_bmm(
        a,
        b,
        eq_a,
        eq_b,
        new_shape_a,
        new_shape_b,
        new_shape_ab,
        perm_ab,
        pure_multiplication,
        bmm_function
):
    # prepare left
    if eq_a is not None:
        # diagonals, sums, and tranpose
        a = _einsum_single(eq_a, a)

    if new_shape_a is not None:
        a = np.reshape(a, new_shape_a)

    # prepare right
    if eq_b is not None:
        # diagonals, sums, and tranpose
        b = _einsum_single(eq_b, b)
    if new_shape_b is not None:
        b = np.reshape(b, new_shape_b)

    if pure_multiplication:
        # no contracted indices
        return np.multiply(a, b)

    # Former reshape operations might result in a Fortran memory layout, which is not supported by the bmm function.
    # Copying the array to a C-contiguous memory layout solves this issue.
    if not a.flags['C_CONTIGUOUS']:
        a = np.copy(a, order='C')
    if not b.flags['C_CONTIGUOUS']:
        b = np.copy(b, order='C')

    # Changed this line to use our library bmm function, given as a parameter
    ab = bmm_function(a, b)

    # prepare the output
    if new_shape_ab is not None:
        ab = np.reshape(ab, new_shape_ab)
    if perm_ab is not None:
        ab = np.transpose(ab, perm_ab)

    return ab


def einsum(eq: str, a: np.ndarray[np.float64], b: np.ndarray[np.float64]=None, bmm_function: str='OMP') -> np.ndarray:
    """Perform arbitrary single and pairwise einsums by mapping tensor contractions to batch matrix multiplications.
    Only `transpose`, `reshape`, `sum` and `batch matrix multiply` from py_bmm are used.
    The logic for each is cached based on the equation and array shape, and each step is only performed if necessary.

    Parameters
    ----------
    eq : str
        The einsum equation.
    a : np.ndarray
        The first array to contract. Must be of type `np.float64`.
    b : np.ndarray, optional
        The second array to contract. Must be of type `np.float64`.
    bmm_function : str, optional
        Specifies the batch matrix multiplication method from `py_bmm`. Must be one of the following:

        - "OMP"   : Optimized BMM using OpenMP directives for parallelization. Compatible with most hardware architectures.
        - "AVX2"  : Optimized BMM utilizing AVX2 vector intrinsics for improved performance on supported CPUs.
        - "Naive" : A straightforward, non-optimized implementation of BMM.
        - "BLAS"  : Uses OpenBLAS for BMM operations, leveraging BLAS optimizations.

    Returns
    -------
    np.ndarray
    """
    bmm_functions = {
        'OMP': py_bmm.bmm_omp,
        'AVX2': py_bmm.bmm_avx2,
        'Naive': py_bmm.bmm_naive,
        'BLAS': py_bmm.bmm_blas
    }

    if bmm_function not in bmm_functions:
        raise ValueError(f"Invalid bmm_function: {bmm_function}. Must be one of {list(bmm_functions.keys())}")

    selected_bmm_function = bmm_functions[bmm_function]

    if b is None:
        return _einsum_single(eq, a)

    (
        eq_a,
        eq_b,
        new_shape_a,
        new_shape_b,
        new_shape_ab,
        perm_ab,
        pure_multiplication,
    ) = parse_double_eq(eq, np.shape(a), np.shape(b))

    return _do_contraction_via_bmm(
        a,
        b,
        eq_a,
        eq_b,
        new_shape_a,
        new_shape_b,
        new_shape_ab,
        perm_ab,
        pure_multiplication,
        selected_bmm_function
    )