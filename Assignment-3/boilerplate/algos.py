from typing import Callable, Literal

import numpy.typing as npt
import matplotlib.pyplot as plt
import numpy as np



def projected_gd(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    point: npt.NDArray[np.float64],
    constraint_type: Literal["linear", "l_2"],
    constraints: npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], np.float64]
) -> npt.NDArray[np.float64]:
    
    def projectionLinear(x):
        lb, ub = constraints
        temp = np.maximum(x, lb)
        return np.minimum(temp, ub)

    def projectionL2(x):
        c, r = constraints
        if np.linalg.norm(x - c) <= r:
            return x
        temp =  max(r, np.linalg.norm(x - c))
        return c + ((x - c) * r / temp)

    if constraint_type == "linear":
        projectionFunction = projectionLinear
    elif constraint_type == "l_2":
        projectionFunction = projectionL2

    def normG(M, x, projectionFunction):
        temp = x - projectionFunction(x - (d_f(x) / M))
        return M * temp

    def backtrackingLineSearch(x, projectionFunction):
        s = 1
        alpha = 0.001
        beta, tk,epsilon = 0.9, 1, 0.000001
        functionValue, gradValue = f(x), d_f(x)
        i = 0
        while i< 1000:
            projectedX = projectionFunction(x - tk * gradValue)
            normGradient = np.linalg.norm(normG(1 / tk, x, projectionFunction))
            condition_value = abs(functionValue - f(projectedX) - alpha * tk * normGradient**2)
            if condition_value < epsilon:
                break
            tk = tk * beta
            i +=1 
        return tk

    x = point
    maxIteration = 1000
    for n in range(maxIteration):
        g = d_f(x)
        tk = backtrackingLineSearch(x, projectionFunction)
        xNew = projectionFunction(x - tk * g)
        x = xNew
    return xNew

def dual_ascent(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    c: list[Callable[[npt.NDArray[np.float64]], np.float64 | float]],
    d_c: list[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
    initial_point: npt.NDArray[np.float64],
) -> tuple [npt.NDArray[np.float64]]:
    alpha, x = 0.001, initial_point
    numOfConstraints = len(c)
    lambdas = np.ones(numOfConstraints)
    for n in range(100000):
        temp = 0
        for i in range(numOfConstraints):
            temp += lambdas[i] * d_c[i](x)
        g = d_f(x) + temp
        xNew = x - alpha * g
        for i in range(numOfConstraints):
            lambdas[i] = max(lambdas[i] + alpha * c[i](xNew), 0)
        x = xNew
    return x, lambdas

