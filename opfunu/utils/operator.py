#!/usr/bin/env python
# Created by "Thieu" at 10:49, 01/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import jax
import jax.numpy as np
import numpy as onp
from jax import lax, random


def fill_diagonal(matrix, diagonal):
    # Ensure diagonal is a 1D array
    diagonal = np.asarray(diagonal)
    # Compute the indices for the diagonal
    idx = np.arange(min(matrix.shape[0], matrix.shape[1]))
    # Update the matrix using scatter_nd
    return matrix.at[idx, idx].set(diagonal)


def rounder(x, condition):
    temp_2x = 2 * x
    dec, inter = np.modf(temp_2x)
    temp_2x = np.where(temp_2x <= 0.0, inter - (dec >= 0.5), temp_2x)
    temp_2x = np.where(dec < 0.5, inter, temp_2x)
    temp_2x = np.where(dec >= 0.5, inter + 1, temp_2x)
    return np.where(condition < 0.5, x, temp_2x / 2)


def griewank_func(x):
    x = np.array(x).ravel()
    idx = np.arange(1, len(x) + 1)
    t1 = np.sum(x ** 2) / 4000
    t2 = np.prod(np.cos(x / np.sqrt(idx)))
    return t1 - t2 + 1


def rosenbrock_func(x, shift=0.0):
    x = np.array(x).ravel() + shift
    term1 = 100 * (x[:-1] ** 2 - x[1:]) ** 2
    term2 = (x[:-1] - 1) ** 2
    return np.sum(term1 + term2)


def scaffer_func(x):
    x = np.array(x).ravel()
    return 0.5 + (np.sin(np.sqrt(np.sum(x ** 2))) ** 2 - 0.5) / (1 + 0.001 * np.sum(x ** 2)) ** 2


def rastrigin_func(x):
    x = np.array(x).ravel()
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)


def weierstrass_func(x, a=0.5, b=3.0, k_max=20):
    x = np.ravel(np.array(x))  # Ensure x is a flattened JAX array
    k = np.arange(0, k_max + 1)  # Vector of exponents
    ak = a ** k  # Precompute a^k
    bk = b ** k  # Precompute b^k

    result = np.sum(ak[:, None] * np.cos(2 * np.pi * bk[:, None] * (x + 0.5)), axis=0)
    correction = np.sum(ak * np.cos(np.pi * bk))

    return np.sum(result) - len(x) * correction


def weierstrass_norm_func(x, a=0.5, b=3., k_max=20):
    """
    This function matches CEC2005 description of F11 except for addition of the bias and follows the C implementation
    """
    return weierstrass_func(x, a, b, k_max) - weierstrass_func(np.zeros(len(x)), a, b, k_max)


def ackley_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    t1 = np.sum(x ** 2)
    t2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(t1 / ndim)) - np.exp(t2 / ndim) + 20 + np.e


def sphere_func(x):
    x = np.array(x).ravel()
    return np.sum(x ** 2)


def rotated_expanded_schaffer_func(x):
    x = np.asarray(x).ravel()
    x_pairs = np.column_stack((x, np.roll(x, -1)))
    sum_sq = x_pairs[:, 0] ** 2 + x_pairs[:, 1] ** 2
    # Calculate the Schaffer function for all pairs simultaneously
    schaffer_values = (0.5 + (np.sin(np.sqrt(sum_sq)) ** 2 - 0.5)
                       / (1 + 0.001 * sum_sq) ** 2)
    return np.sum(schaffer_values)


def rotated_expanded_scaffer_func(x):
    x = np.array(x).ravel()
    results = np.array([scaffer_func([x[idx], x[idx + 1]]) for idx in range(0, len(x) - 1)])
    return np.sum(results) + scaffer_func([x[-1], x[0]])


def grie_rosen_cec_func(x):
    """This is based on the CEC version which unrolls the griewank and rosenbrock functions for better performance"""
    z = np.array(x).ravel()
    z += 1.0  # This centers the optimal solution of rosenbrock to 0

    tmp1 = (z[:-1] * z[:-1] - z[1:]) ** 2
    tmp2 = (z[:-1] - 1.0) ** 2
    temp = 100.0 * tmp1 + tmp2
    f = np.sum(temp ** 2 / 4000.0 - np.cos(temp) + 1.0)
    # Last calculation
    tmp1 = (z[-1] * z[-1] - z[0]) ** 2
    tmp2 = (z[-1] - 1.0) ** 2
    temp = 100.0 * tmp1 + tmp2
    f += (temp ** 2) / 4000.0 - np.cos(temp) + 1.0

    return f


def f8f2_func(x):
    x = np.array(x).ravel()
    results = np.array([griewank_func(rosenbrock_func([x[idx], x[idx + 1]])) for idx in range(0, len(x) - 1)])
    return np.sum(results) + griewank_func(rosenbrock_func([x[-1], x[0]]))


def non_continuous_expanded_scaffer_func(x):
    x = np.array(x).ravel()
    y = rounder(x, np.abs(x))
    results = np.array([scaffer_func([y[idx], y[idx + 1]]) for idx in range(0, len(x) - 1)])
    return np.sum(results) + scaffer_func([y[-1], y[0]])


def non_continuous_rastrigin_func(x):
    x = np.array(x).ravel()
    y = rounder(x, np.abs(x))
    shifted_y = np.roll(y, -1)
    results = rastrigin_func(np.column_stack((y, shifted_y)))
    return np.sum(results)


def elliptic_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    idx = np.arange(0, ndim)
    return np.sum(10 ** (6.0 * idx / (ndim - 1)) * x ** 2)


def sphere_noise_func(x):
    x = np.array(x).ravel()
    return np.sum(x ** 2) * (1 + 0.1 * np.abs(onp.random.normal(0, 1)))


def twist_func(x):
    # This function in CEC-2008 F7
    return 4 * (x ** 4 - 2 * x ** 3 + x ** 2)


def doubledip(x, c, s):
    """
    JAX-compatible version of the doubledip function from CEC-2008 F7.
    """
    poly_term = (-6144 * (x - c) ** 6 + 3088 * (x - c) ** 4 - 392 * (x - c) ** 2 + 1) * s
    return np.where((-0.5 < x) & (x < 0.5), poly_term, 0.0)


def fractal_1d_func(x):
    """
    JAX-compatible version of the CEC-2008 F7 fractal function.
    Uses a fixed random seed (0).
    """
    def outer_loop(carry, _):
        k = carry
        upper = 2 ** (k - 1)  # Convert to concrete integer
        key_outer = random.key(k)

        def inner_loop(carry, _):
            key_inner = carry

            # Generate random values
            key_a, _ = random.split(key_inner)
            a_vals = random.uniform(key_a, shape=(), minval=0, maxval=1)
            b_vals = 1.0 / upper * (2 - random.uniform(key_a, shape=()))

            # Compute sum using vmap
            sum_val = np.sum(jax.vmap(doubledip, in_axes=(None, 0, 0))(x, a_vals, b_vals))
            return key_a, sum_val

        # result2 = lax.fori_loop(1, upper, inner_loop, result2)
        t, result2 = lax.scan(
            f=inner_loop,
            init=key_outer,
            xs=None,
            length=upper
        )

        k += 1
        return k, np.sum(result2)

    final_k, result1 = lax.scan(
        f=outer_loop,
        init=0,
        xs=None,
        length=3
    )
    return np.sum(result1)


def schwefel_12_func(x):
    cumsum_x = jax.lax.associative_scan(np.add, x)
    return np.sum(cumsum_x ** 2)


def tosz_func(x):
    def transform(xi):
        c1, c2 = 10., 7.9
        x_sign = np.where(xi > 0, 1.0, np.where(xi < 0, -1.0, 0.0))
        x_star = np.log(np.abs(xi))
        return x_sign * np.exp(x_star + 0.049 * (np.sin(c1 * x_star) + np.sin(c2 * x_star)))

    x = np.array(x).ravel()
    transformed_x = np.where((x == x[0]) | (x == x[-1]), transform(x), x)
    return transformed_x


def tasy_func(x, beta=0.5):
    x = np.array(x).ravel()
    ndim = len(x)
    idx = np.arange(0, ndim)
    up = 1 + beta * ((idx - 1) / (ndim - 1)) * np.sqrt(np.abs(x))
    x_temp = np.abs(x) ** up
    return np.where(x > 0, x_temp, x)


def bent_cigar_func(x):
    x = np.array(x).ravel()
    return x[0] ** 2 + 10 ** 6 * np.sum(x[1:] ** 2)


def discus_func(x):
    x = np.array(x).ravel()
    return 1e6 * x[0] ** 2 + np.sum(x[1:] ** 2)


def different_powers_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    idx = np.arange(0, ndim)
    up = 2 + 4 * idx / (ndim - 1)
    return np.sqrt(np.sum(np.abs(x) ** up))


def generate_diagonal_matrix(size, alpha=10):
    idx = np.arange(0, size)
    diagonal = alpha ** (idx / (2 * (size - 1)))
    matrix = np.zeros((size, size), float)
    matrix = fill_diagonal(matrix, diagonal)
    return matrix


def gz_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    t1 = (500 - np.mod(x, 500)) * np.sin(np.sqrt(np.abs(500 - np.mod(x, 500)))) - (x - 500) ** 2 / (10000 * ndim)
    t2 = (np.mod(np.abs(x), 500) - 500) * np.sin(np.sqrt(np.abs(np.mod(np.abs(x), 500) - 500))) - (x + 500) ** 2 / (10000 * ndim)
    t3 = x * np.sin(np.abs(x) ** 0.5)
    conditions = [x < -500, (-500 <= x) & (x <= 500), x > 500]
    choices = [t2, t3, t1]
    y = np.select(conditions, choices, default=np.nan)
    return y


def katsuura_func(x):
    x = np.array(x).ravel()  # Ensure x is a JAX array
    ndim = len(x)
    result = 1.0

    for idx in range(ndim):
        # Create an array from the sum of terms and then sum over the array
        temp_terms = []
        for j in range(1, 20):
            term = np.abs((2 ** j) * x[idx] - np.round((2 ** j) * x[idx])) / (2 ** j)
            temp_terms.append(term)

        temp = np.sum(np.array(temp_terms))  # Sum over the array of terms
        result *= (1 + (idx + 1) * temp) ** (10.0 / ndim ** 1.2)

    return (result - 1) * 10 / ndim ** 2


def lunacek_bi_rastrigin_func(x, miu0=2.5, d=1, shift=0.0):
    x = np.array(x).ravel() + shift
    ndim = len(x)
    s = 1.0 - 1.0 / (2 * np.sqrt(ndim + 20) - 8.2)
    miu1 = -np.sqrt((miu0 ** 2 - d) / s)
    delta_x_miu0 = x - miu0
    term1 = np.sum(delta_x_miu0 ** 2)
    term2 = np.sum((x - miu1) ** 2) * s + d * ndim
    result = np.minimum(term1, term2) + 10 * (ndim - np.sum(np.cos(2 * np.pi * delta_x_miu0)))
    return result


def calculate_weight(x, delta=1.):
    ndim = x.shape[0]  # Use .shape instead of len()
    temp = np.sum(x ** 2)

    weight = np.where(temp != 0,
                      np.sqrt(1.0 / temp) * np.exp(-temp / (2 * ndim * delta ** 2)),
                      1e99)

    return weight


def modified_schwefel_func(x):
    """
    JAX-compatible implementation of the Modified Schwefel F11 Function.
    """
    z = np.ravel(x) + 4.209687462275036e+002
    nx = z.size

    # Masks for the three conditions
    mask1 = z > 500
    mask2 = z < -500
    mask3 = ~mask1 & ~mask2

    # Create fx with all zeros
    fx = np.zeros_like(z)

    # Compute for mask1
    fx = fx + np.where(
        mask1,
        -(
            (500.0 + np.mod(np.abs(z), 500)) * np.sin(np.sqrt(500.0 - np.mod(np.abs(z), 500)))
            - ((z - 500.0) / 100.0) ** 2 / nx
        ),
        0,
    )

    # Compute for mask2
    fx = fx + np.where(
        mask2,
        -(
            (-500.0 + np.mod(np.abs(z), 500)) * np.sin(np.sqrt(500.0 - np.mod(np.abs(z), 500)))
            - ((z + 500.0) / 100.0) ** 2 / nx
        ),
        0,
    )

    # Compute for mask3
    fx = fx + np.where(
        mask3,
        -z * np.sin(np.sqrt(np.abs(z))),
        0,
    )

    return np.sum(fx) + 4.189828872724338e+002 * nx


def happy_cat_func(x, shift=0.0):
    z = np.array(x).ravel() + shift
    ndim = len(z)
    t1 = np.sum(z)
    t2 = np.sum(z ** 2)
    return np.abs(t2 - ndim) ** 0.25 + (0.5 * t2 + t1) / ndim + 0.5


def hgbat_func(x, shift=0.0):
    x = np.array(x).ravel() + shift
    ndim = len(x)
    t1 = np.sum(x)
    t2 = np.sum(x ** 2)
    return np.abs(t2 ** 2 - t1 ** 2) ** 0.5 + (0.5 * t2 + t1) / ndim + 0.5


def zakharov_func(x):
    x = np.array(x).ravel()
    temp = np.sum(0.5 * x)
    return np.sum(x ** 2) + temp ** 2 + temp ** 4


def levy_func(x, shift=0.0):
    x = np.array(x).ravel() + shift
    w = 1. + (x - 1.) / 4
    t1 = np.sin(np.pi * w[0]) ** 2 + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    t2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    return t1 + t2


def expanded_schaffer_f6_func(x):
    """
    This is a direct conversion of the CEC2021 C-Code for the Expanded Schaffer F6 Function
    """
    z = np.array(x).ravel()

    temp1 = np.sin(np.sqrt(z[:-1] ** 2 + z[1:] ** 2))
    temp1 = temp1 ** 2
    temp2 = 1.0 + 0.001 * (z[:-1] ** 2 + z[1:] ** 2)
    f = np.sum(0.5 + (temp1 - 0.5) / (temp2 ** 2))

    temp1_last = np.sin(np.sqrt(z[-1] ** 2 + z[0] ** 2))
    temp1_last = temp1_last ** 2
    temp2_last = 1.0 + 0.001 * (z[-1] ** 2 + z[0] ** 2)
    f += 0.5 + (temp1_last - 0.5) / (temp2_last ** 2)

    return f


def schaffer_f7_func(x):
    x = np.ravel(np.array(x))  # Ensure x is a 1D JAX array
    t = x[:-1] ** 2 + x[1:] ** 2  # Compute pairwise squared sum
    result = np.sum(np.sqrt(t) * (np.sin(50. * t ** 0.2) + 1))  # Vectorized sum
    return (result / (x.shape[0] - 1)) ** 2


def chebyshev_func(x):
    """
    Generalized Tchebychev function implemented with JAX.
    Converted from the original CEC2019 C code.
    """
    x = np.array(x).ravel()
    ndim = x.shape[0]
    sample = 32 * ndim

    # Compute dx_arr using a recurrence relation
    def recurrence(i, dx_arr):
        return dx_arr.at[i].set(2.4 * dx_arr[i - 1] - dx_arr[i - 2])

    dx_arr = np.zeros(ndim)
    dx_arr = dx_arr.at[:2].set(np.array([1.0, 1.2]))
    dx_arr = lax.fori_loop(2, ndim, recurrence, dx_arr)
    dx = dx_arr[-1]

    dy = 2.0 / sample
    y_vals = np.linspace(-1, 1, sample + 1)  # Linearly spaced values for y

    # Inner function for computing sum_val
    def inner_loop(y, sum_val):
        def body(j, px):
            return y * px + x[j]
        px = lax.fori_loop(1, ndim, body, x[0])

        term = np.where((px < -1) | (px > 1), (1.0 - np.abs(px)) ** 2, 0.0)
        return sum_val + term

    sum_val = lax.fori_loop(0, sample + 1, lambda i, val: inner_loop(y_vals[i], val), 0.0)

    # Second part: Additional checks and summation
    px = np.sum(1.2 * x[1:]) + x[0]
    mask = px < dx
    sum_val += np.where(mask, px**2, 0).sum()

    return sum_val


def inverse_hilbert_func(x, ndim: int):
    """
    JAX-compatible version of the inverse Hilbert function with static dimension b.
    """
    # Create the Hilbert matrix
    i = np.arange(ndim).reshape((ndim, 1))
    j = np.arange(ndim).reshape((1, ndim))
    hilbert = 1.0 / (i + j + 1)

    # Reshape x and apply Hilbert transform
    x = np.reshape(x, (ndim, ndim))
    y = hilbert @ x @ hilbert.T

    # Compute the absolute deviation from identity
    result = np.sum(np.abs(y - np.eye(ndim)))
    return result


def lennard_jones_func(x):
    """
    Lennard-Jones potential energy function adapted for JAX.
    Computes the minimum energy configuration for atomic clusters.
    """
    x = np.array(x).ravel()
    ndim = x.shape[0]

    minima = np.array([-1., -3., -6., -9.103852, -12.712062, -16.505384, -19.821489, -24.113360,
                      -28.422532, -32.765970, -37.967600, -44.326801, -47.845157, -52.322627, -56.815742,
                      -61.317995, -66.530949, -72.659782, -77.1777043, -81.684571, -86.809782, -92.844472,
                      -97.348815, -102.372663])

    k = ndim // 3
    x_matrix = x.reshape((k, 3))

    def pairwise_energy(carry, ij):
        i, j = ij
        diff = x_matrix[i] - x_matrix[j]
        ed = np.sum(diff ** 2)
        ud = ed ** 3
        energy = np.where(ud > 1.0e-10, (1.0 / ud - 2.0) / ud, 1.0e20)
        return carry + energy, None

    # Generate all (i, j) pairs where i < j
    indices = np.array([(i, j) for i in range(k - 1) for j in range(i + 1, k)])
    sum_val, _ = lax.scan(pairwise_energy, 0.0, indices)

    return sum_val - minima[k - 2]  # Adjust with known minima


expanded_griewank_rosenbrock_func = grie_rosen_cec_func
expanded_scaffer_f6_func = rotated_expanded_scaffer_func
