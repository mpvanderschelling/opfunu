#!/usr/bin/env python
# Created by "Thieu" at 17:22, 22/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu
import autograd.numpy as np


print("====================Test FreudensteinRoth")
ndim = 2
problem = opfunu.name_based.FreudensteinRoth(ndim=ndim)
x = np.ones(ndim)
print(problem.evaluate(x))
print(problem.x_global)

print(problem.evaluate(problem.x_global))
print(problem.is_succeed(x))
print(problem.is_succeed(problem.x_global))
