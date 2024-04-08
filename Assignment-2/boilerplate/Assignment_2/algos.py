from typing import Callable, Literal

from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np


def plotGraph(functionValue,legend,f,initialPoint, condition, type):
    iterations = (int)(1e4)
    xValues = list(range(1, iterations + 1))
    xResampled = np.linspace(min(xValues), max(xValues), len(functionValue))
    xLabel = "Iterations"
    if type == "vals":
        yLabel = "f(x)"
    elif type == "grad":
        yLabel = "f\'(x)"
    plt.plot(xResampled, functionValue, linestyle='-', color='blue')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend(legend)
    plt.title(f'Graph: {yLabel} vs {xLabel}')
    plt.savefig(f"plots/{f.__name__}_{np.array2string(initialPoint)}_{condition}_{type}.png")
    plt.close()

def plotContourGraph(condition, f, initialPoint, xValues, yValues):
    left = np.round(np.min([xValues, yValues])) - 2
    right = np.round(np.max([xValues, yValues])) + 2
    x = np.linspace(left, right, 100)
    y = np.linspace(left, right, 100)
    X, Y = np.meshgrid(x, y)
    Z = []
    for i in range(len(X)):
        row = []
        for j in range(len(Y)):
            row.append(f(np.array([X[i][j], Y[i][j]])))
        Z.append(row)
    Z = np.array(Z)

    plt.contour(X, Y, Z, levels=20)
    for i in range(len(xValues) - 1):  # Plotting arrows
        plt.arrow(xValues[i], yValues[i],
                  xValues[i + 1] - xValues[i], yValues[i + 1] - yValues[i],
                  head_width=0.2, head_length=0.2, fc='red', ec='r')
    plt.scatter(xValues[0], yValues[0], color='green')  # Mark the initial point with yellow dot
    plt.scatter(xValues[-1], yValues[-1], color='yellow')  # Mark the final point as green dot.
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title(f'Contour Plot with Update Arrows ({f.__name__})')
    plt.colorbar()  # Add colorbar
    plt.savefig(f"plots/{f.__name__}_{np.array2string(initialPoint)}_{condition}_cont.png")
    plt.close()


def hestenesStiefel(g_k1, g_k, d_k):
    temp =  g_k1 - g_k
    num = np.dot(g_k1.T, temp)
    denum = np.dot(d_k.T, temp)
    if denum == 0.0:
        denum = 0.000001
    beta_k = num / denum
    return beta_k

def polakRibiere(g_k1, g_k):
    temp =  g_k1 - g_k
    num = np.dot(g_k1.T, temp)
    denum = np.dot(g_k.T, g_k)
    if denum == 0.0:
        denum = 0.000001
    beta_k = num / denum
    return beta_k

def fletcherReeves(g_k1, g_k):
    num = np.dot(g_k1.T, g_k1)
    denum = np.dot(g_k.T, g_k)
    if denum == 0.0:
        denum = 0.000001
    beta_k = num / denum
    return beta_k

def bijectionMethod(initial_point, f, d_f, d_k):
    c1, c2, alpha, t, beta, k, epsilon = 0.001, 0.1, 0, 1, 1e6, 0, 1e-6
    while k < 10**4:
        if f(initial_point + (t * d_k)) > f(initial_point) + c1 * t * np.dot(d_f(initial_point).T, d_k):
            beta = t 
            t = 0.5 * (alpha + beta)
        elif np.dot(d_f(initial_point + (t * d_k)).T, d_k) < (c2 * np.dot(d_f(initial_point).T, d_k)):
            alpha = t
            t = 0.5 * (alpha + beta)
        else:
            break
        k = k + 1
    return t

def conjugate_descent(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    approach: Literal["Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves"],
) -> NDArray[np.float64]:
    
    k = 0
    epsilon = 1e-6
    g_k = d_f(inital_point)
    if np.all(g_k == 0): 
        return inital_point
    else:
        d_k = -g_k

    y_val = [f(inital_point)]
    y_dashVal = [np.linalg.norm(d_f(inital_point))]
    xValues, yValues = [], []
    initial_val = inital_point
    condition = approach
    
    while k < 10**4:
        alpha_k = bijectionMethod(inital_point, f, d_f, d_k)
        if inital_point.shape[0] == 2:
            xValues.append(inital_point[0])
            yValues.append(inital_point[1])
        inital_point = inital_point + (alpha_k * d_k)
        g_k1 = d_f(inital_point)
        y_val.append(f(inital_point))
        y_dashVal.append(np.linalg.norm(d_f(inital_point)))

        if np.linalg.norm(g_k1) < epsilon:
            break
        if approach == "Hestenes-Stiefel":
            beta_k = hestenesStiefel(g_k1, g_k, d_k)
        elif approach == "Polak-Ribiere":
            beta_k = polakRibiere(g_k1, g_k)
        elif approach == "Fletcher-Reeves":
            beta_k = fletcherReeves(g_k1, g_k)
        d_k = -g_k1 + (beta_k * d_k)
        g_k = g_k1
        k = k+1
    if len(inital_point) == 2:
        plotContourGraph(condition, f, initial_val, xValues, yValues)
    plotGraph(y_val, "ConjugateGradientMethod", f, initial_val, condition, "vals")
    plotGraph(y_dashVal, "ConjugateGradientMethod", d_f, initial_val, condition, "grad")
    return inital_point


def sr1(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    k = 0
    epsilon = 1e-6
    condition = "SR1"
    H_k = np.eye(len(inital_point))
    g_k = d_f(inital_point)
    y_val = [f(inital_point)]
    y_dashVal = [np.linalg.norm(d_f(inital_point))]
    xValues, yValues = [], []
    initial_val = inital_point
    while k < 10**4:
        if np.linalg.norm(g_k) < epsilon:
            break
        d_k = -np.dot(H_k, g_k)
        alpha_k = bijectionMethod(inital_point, f, d_f, d_k)
        if inital_point.shape[0] == 2:
            xValues.append(inital_point[0])
            yValues.append(inital_point[1])
        inital_point = inital_point + alpha_k * d_k
        g_k1 = d_f(inital_point)
        y_val.append(f(inital_point))
        y_dashVal.append(np.linalg.norm(d_f(inital_point)))
        diff_x = alpha_k * d_k
        diff_g = g_k1 - g_k
        temp = diff_x - np.dot(H_k, diff_g)
        num = np.dot(temp.reshape(-1,1),temp.reshape(-1,1).T)
        denum = np.dot(diff_g.T, temp)
        if denum == 0.0:
            H_k += (num/0.000001)
        else:
            H_k += num / denum
        g_k = g_k1
        k = k + 1
    
    if len(inital_point) == 2:
        plotContourGraph(condition, f, initial_val, xValues, yValues)
    plotGraph(y_val, "QuasiNewtonMethod", f, initial_val, condition, "vals")
    plotGraph(y_dashVal, "QuasiNewtonMethod", d_f, initial_val, condition, "grad")
    return inital_point

def dfp(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    k = 0
    epsilon = 1e-6
    H_k = np.eye(len(inital_point))
    g_k = d_f(inital_point)
    y_val = [f(inital_point)]
    y_dashVal = [np.linalg.norm(d_f(inital_point))]
    xValues, yValues = [], []
    initial_val = inital_point
    condition = "DFP"
    while k < 10**4:
        if np.linalg.norm(g_k) < epsilon:
            break
        d_k = -np.dot(H_k, g_k)
        if inital_point.shape[0] == 2:
            xValues.append(inital_point[0])
            yValues.append(inital_point[1])
        alpha_k = bijectionMethod(inital_point, f, d_f, d_k)
        inital_point = inital_point + alpha_k * d_k
        y_val.append(f(inital_point))
        y_dashVal.append(np.linalg.norm(d_f(inital_point)))
        g_k1 = d_f(inital_point)
        diff_x = alpha_k * d_k
        diff_g = g_k1 - g_k
        temp = np.dot(H_k, diff_g)
        n1 = np.dot(diff_x.reshape(-1,1), diff_x.reshape(-1,1).T)
        n2 = np.dot(temp.reshape(-1, 1), temp.reshape(-1,1).T)
        d1 = np.dot(diff_x.T, diff_g)
        d2 = np.dot(diff_g.T, temp)
        if np.abs(d1) == 0.0:
            d1 = 0.000001
        if np.abs(d2) == 0.0:
            d2 = 0.000001
        H_k += ((n1 / d1) - (n2 / d2))
        g_k = g_k1
        k = k + 1
    if len(inital_point) == 2:
        plotContourGraph(condition, f, initial_val, xValues, yValues)
    plotGraph(y_val, "QuasiNewtonMethod", f, initial_val, condition, "vals")
    plotGraph(y_dashVal, "QuasiNewtonMethod", d_f, initial_val, condition, "grad")
    return inital_point

def bfgs(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    k = 0
    H_k = np.eye(len(inital_point))
    epsilon = 1e-6
    g_k = d_f(inital_point)
    y_val = [f(inital_point)]
    y_dashVal = [np.linalg.norm(d_f(inital_point))]
    xValues, yValues = [], []
    initial_val = inital_point
    condition = "BFGS"
    while k < 10**4:
        if np.linalg.norm(g_k) < epsilon:
            break
        d_k = -np.dot(H_k, g_k)
        if inital_point.shape[0] == 2:
            xValues.append(inital_point[0])
            yValues.append(inital_point[1])
        alpha_k = bijectionMethod(inital_point, f, d_f, d_k)
        inital_point = inital_point + alpha_k * d_k
        y_val.append(f(inital_point))
        y_dashVal.append(np.linalg.norm(d_f(inital_point)))
        g_k1 = d_f(inital_point)
        diff_x = alpha_k * d_k
        diff_g = g_k1 - g_k
        n1 = np.dot(diff_g.T, np.dot(H_k, diff_g))
        n2 = np.outer(diff_x, diff_x)
        temp = np.outer(np.dot(H_k,diff_g), diff_x)
        n3 = temp + temp.T
        d1 = np.dot(diff_g, diff_x)
        d2 = np.dot(diff_x, diff_g)
        if np.abs(d1) == 0.0:
            d1 = 0.000001
        if np.abs(d2) == 0.0:
            d2 = 0.000001
        H_k += ((1 + (n1 / d1)) * (n2 / d2) - (n3 / d1))
        g_k = g_k1
        k = k + 1
    if len(inital_point) == 2:
        plotContourGraph(condition, f, initial_val, xValues, yValues)
    plotGraph(y_val, "QuasiNewtonMethod", f, initial_val, condition, "vals")
    plotGraph(y_dashVal, "QuasiNewtonMethod", d_f, initial_val, condition, "grad")
    return inital_point

