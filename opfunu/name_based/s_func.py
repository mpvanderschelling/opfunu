#!/usr/bin/env python
# Created by "Thieu" at 17:31, 30/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import jax.numpy as np

from opfunu.benchmark import Benchmark


class SchaffelN1(Benchmark):
    name = "Schaffel N1 Function"
    latex_formula = r'f_{\text{Sphere}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = False
    unimodal = True
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = False
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-100, 100], [-100, 100]]))
        self.f_global = 0.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        _x, _y = x
        res = 0.5 + (np.sin((_x**2 + _y**2) ** 2) ** 2 - 0.5) / \
            (1 + 0.001 * (_x**2 + _y**2)) ** 2
        return res


class SchaffelN2(Benchmark):
    name = "Schaffel N2 Function"
    latex_formula = r'f_{\text{Sphere}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = False
    unimodal = True
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = False
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-4, 4], [-4, 4]]))
        self.f_global = 0.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        _x, _y = x
        res = 0.5 + (np.sin((_x**2 + _y**2) ** 2) ** 2 - 0.5) / \
            (1 + 0.001 * (_x**2 + _y**2)) ** 2
        return res


class SchaffelN3(Benchmark):
    name = "Schaffel N3 Function"
    latex_formula = r'f_{\text{Sphere}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = False
    unimodal = True
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = False
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-4, 4], [-4, 4]]))
        self.f_global = 0.
        self.x_global = np.array([0, 1.253115])

    def evaluate(self, x, *args):
        _x, _y = x
        res = 0.5 + (np.sin(np.cos(
            np.abs(_x**2 + _y**2))) ** 2 - 0.5) / (
                1 + 0.001 * (_x**2 + _y**2)) ** 2
        return res


class SchaffelN4(Benchmark):
    name = "Schaffel N4 Function"
    latex_formula = r'f_{\text{Sphere}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = False
    unimodal = True
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = False
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-4, 4], [-4, 4]]))
        self.f_global = 0.
        self.x_global = np.array([0, 1.253115])

    def evaluate(self, x, *args):
        _x, _y = x
        res = 0.5 + (np.cos(np.sin(np.abs(
            _x**2 + _y**2))) ** 2 - 0.5) / (
                1 + 0.001 * (_x**2 + _y**2)) ** 2
        return res


class Schwefel(Benchmark):
    name = "Schwefel Function"
    latex_formula = r'f_{\text{Sphere}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-500, 500] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.ones(self.ndim) * 420.9687

    def evaluate(self, x, *args):
        d = x.shape[0]
        res = 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))
        return res


class Schwefel2_20(Benchmark):
    name = "Schwefel 2_20 Function"
    latex_formula = r'f_{\text{Sphere}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-100, 100] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        res = np.sum(np.abs(x))
        return res


class Schwefel2_22(Benchmark):
    name = "Schwefel 2_22 Function"
    latex_formula = r'f_{\text{Sphere}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-100, 100] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        res = np.sum(np.abs(x)) + np.prod(np.abs(x))
        return res


class Schwefel2_23(Benchmark):
    name = "Schwefel 2_23 Function"
    latex_formula = r'f_{\text{Sphere}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-100, 100] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        res = np.sum(x**10)
        return res


class Shekel(Benchmark):
    name = "Shekel Function"
    latex_formula = r'f_{\text{Sphere}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = False
        self.dim_default = 4
        self.check_ndim_and_bounds(ndim, bounds, np.array(
            [[-10, 10], [-10, 10], [-10, 10], [-10, 10]]))
        self.m = 10
        self.beta = 1 / 10 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
        self.C = np.array(
            [
                [4, 4, 4, 4],
                [1, 1, 1, 1],
                [8, 8, 8, 8],
                [6, 6, 6, 6],
                [3, 7, 3, 7],
                [2, 9, 2, 9],
                [5, 3, 5, 3],
                [8, 1, 8, 1],
                [6, 2, 6, 2],
                [7, 3.6, 7, 3.6],
            ]
        )
        self.f_global = 0.
        self.x_global = self.C[0]

    def evaluate(self, x, *args):
        x1, x2, x3, x4 = x
        res = -np.sum([[np.sum((x - self.C[i]) ** 2 + self.beta[i]) ** -1]
                      for i in range(self.m)])
        return res


class Shubert(Benchmark):
    name = "Shubert Function"
    latex_formula = r'f_{\text{Sphere}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10, 10] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        d = x.shape[0]
        for i in range(0, d):
            res = np.prod(np.sum([i * np.cos((j + 1) * x[i] + j)
                          for j in range(1, 5 + 1)]))
        return res


class ShubertN3(Benchmark):
    name = "Shubert N. 3 Function"
    latex_formula = r'f_{\text{Sphere}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10, 10] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.ones(self.ndim) * -7.4

    def evaluate(self, x, *args):
        res = np.sum(np.sum([j * np.sin((j + 1) * x + j)
                     for j in range(1, 5 + 1)]))
        return res


class ShubertN4(Benchmark):
    name = "Shubert N. 4 Function"
    latex_formula = r'f_{\text{Sphere}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10, 10] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.ones(self.ndim) * 4.85

    def evaluate(self, x, *args):
        res = np.sum(np.sum([j * np.cos((j + 1) * x + j)
                     for j in range(1, 5 + 1)]))
        return res


class StyblinskiTang(Benchmark):
    name = "Styblinski Tang Function"
    latex_formula = r'f_{\text{Sphere}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-5, 5] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.ones(self.ndim) * -2.903534

    def evaluate(self, x, *args):
        res = 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)
        return res


class SumSquares(Benchmark):
    name = "Sum Squares Function"
    latex_formula = r'f_{\text{Sphere}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10, 10.] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        d = x.shape[0]
        i = np.arange(1, d + 1)
        res = np.sum(i * x**2)
        return res


class Salomon(Benchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Salomon}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}


    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-100, 100]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for :math:`i = 1, ..., n`
    """
    name = "Qing Function"
    latex_formula = r'f_{\text{Salomon}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        u = np.sqrt(np.sum(x ** 2))
        return 1 - np.cos(2 * np.pi * u) + 0.1 * u


class Sphere(Benchmark):
    name = "Sphere Function"
    latex_formula = r'f_{\text{Sphere}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = True
    unimodal = True
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-5.12, 5.12] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        res = np.sum(x**2)
        return res
