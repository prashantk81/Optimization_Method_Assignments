from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
def backtrackingWithArmijoCondition(f, d_f, initialPoint):
    alpha_0, rho, c, k, epsilon = 10.0, 0.75, 0.001, 0, 1e-6
    gradient = d_f(initialPoint)
    fValue=[]
    d_fValue=[]
    xValues=[]
    yValues=[]
    while k <= 1e4 and np.linalg.norm(gradient) > epsilon:
        alpha = alpha_0
        dk = - gradient
        if initialPoint.shape == (2,):
            xValues.append(initialPoint[0])
            yValues.append(initialPoint[1])
        while f(initialPoint+alpha*dk) > f(initialPoint) + c*alpha*np.dot(gradient,dk):
            alpha = rho*alpha
        fValue.append(f(initialPoint))
        d_fValue.append(np.linalg.norm(gradient))
        initialPoint = initialPoint + alpha*dk
        gradient = d_f(initialPoint)
        k = k+1
    return initialPoint, fValue, d_fValue, xValues, yValues

def bijectionMethodWithWolfeCondition(f, d_f, initialPoint):
    c1, c2, alpha_0, t, beta_0, k, epsilon = 0.001, 0.1, 0, 1, 1e6, 0, 1e-6
    gradient = d_f(initialPoint)
    fValue=[]
    d_fValue=[]
    xValues=[]
    yValues=[]
    while k <= 1e4 and np.linalg.norm(gradient) > epsilon:
        dk = -1*gradient
        alpha, beta = alpha_0, beta_0
        if initialPoint.shape == (2,):
            xValues.append(initialPoint[0])
            yValues.append(initialPoint[1])
        while 1:
            if f(initialPoint + t*dk) > f(initialPoint) + c1*t*np.dot(gradient,dk):
                beta = t
                t = 0.5*(alpha + beta)
            elif np.dot(d_f(initialPoint + t*dk),dk) < c2*np.dot(gradient,dk):
                alpha = t
                t = 0.5*(alpha + beta)
            else:
                break
        fValue.append(f(initialPoint))
        d_fValue.append(np.linalg.norm(gradient))
        initialPoint = initialPoint + t*dk
        gradient = d_f(initialPoint)
        k = k+1    
    return initialPoint, fValue, d_fValue, xValues, yValues

def plotGraph(functionValue,f,initialPoint, condition, type):
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

# Do not rename or delete this function
def steepest_descent(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    inital_point: npt.NDArray[np.float64],
    condition: Literal["Backtracking", "Bisection"],
) -> npt.NDArray[np.float64]:
    if(condition == "Backtracking"):
        finalValues, fValue, d_fValue, xValues, yValues = backtrackingWithArmijoCondition(f, d_f, inital_point)
        plotGraph(fValue,f,inital_point,condition,"vals")
        plotGraph(d_fValue,f,inital_point,condition,"grad")
        if len(inital_point) == 2:
            plotContourGraph(condition, f, inital_point, xValues, yValues)
        return finalValues
    
    elif(condition == "Bisection"):
        finalValues, fValue, d_fValue, xValues, yValues = bijectionMethodWithWolfeCondition(f, d_f, inital_point)
        plotGraph(fValue,f,inital_point,condition,"vals")
        plotGraph(d_fValue,d_f,inital_point,condition, "grad")
        if len(inital_point) == 2:
            plotContourGraph(condition, f, inital_point, xValues, yValues)
        return finalValues
    # Complete this function
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_vals.png" for plotting f(x) vs iters
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_grad.png" for plotting |f'(x)| vs iters
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_cont.png" for plotting the contour plot
    pass

def pureNewtonsMathod(f, d_f, d2_f, initialPoint):
    epsilon, k = 1e-6, 0
    fValue=[]
    d_fValue=[]
    xValues=[]
    yValues=[]
    while k <= 1e4 and np.linalg.norm(d_f(initialPoint)) > epsilon:
        gradient = -d_f(initialPoint)
        heassian = d2_f(initialPoint)
        dk = np.linalg.solve(heassian, gradient)
        if initialPoint.shape == (2,):
            xValues.append(initialPoint[0])
            yValues.append(initialPoint[1])
        fValue.append(f(initialPoint))
        d_fValue.append(np.linalg.norm(-gradient))
        initialPoint = initialPoint + dk
        k = k+1
    return initialPoint, fValue, d_fValue, xValues, yValues

def dampedNewtonMethod(f, d_f, d2_f, initialPoint):
    epsilon, k, alpha, beta = 1e-6, 0, 0.001, 0.75
    fValue=[]
    d_fValue=[]
    xValues=[]
    yValues=[]
    while k <= 1e4 and np.linalg.norm(d_f(initialPoint)) > epsilon:
        gradient = -d_f(initialPoint)
        heassian = d2_f(initialPoint)
        dk = np.linalg.solve(heassian, gradient)
        tk = 1
        while f(initialPoint) - f(initialPoint + (tk * dk)) < -alpha*tk*(np.dot(-gradient,dk)):
            tk = beta * tk
        if initialPoint.shape == (2,):
            xValues.append(initialPoint[0])
            yValues.append(initialPoint[1])
        fValue.append(f(initialPoint))
        d_fValue.append(np.linalg.norm(-gradient))        
        initialPoint = initialPoint + ( tk * dk )
        k = k+1
    return initialPoint, fValue, d_fValue, xValues, yValues

def levenbergMarquardtModification(f, d_f, d2_f, initialPoint):
    epsilon, k = 1e-6, 0
    gradient = d_f(initialPoint)
    fValue=[]
    d_fValue=[]
    xValues=[]
    yValues=[]
    while k <= 1e4 and np.linalg.norm(gradient) > epsilon:
        hessian = d2_f(initialPoint)
        lambdaMin = np.min(np.linalg.eigvals(hessian))
        if lambdaMin <= 0:
            uk = -1*lambdaMin + 0.1
            size = np.shape(hessian)[0]
            dk = -1*np.dot(np.linalg.inv(hessian + uk * np.eye(size)),gradient)
        else:
            dk = -1*np.dot(np.linalg.inv(hessian),gradient)
        if initialPoint.shape == (2,):
            xValues.append(initialPoint[0])
            yValues.append(initialPoint[1])
        fValue.append(f(initialPoint))
        d_fValue.append(np.linalg.norm(gradient)) 
        initialPoint = initialPoint + dk
        gradient = d_f(initialPoint)
        k = k+1
    return initialPoint, fValue, d_fValue, xValues, yValues

def combiningDampedNewtonMethodWithLMM(f, d_f, d2_f, initialPoint):
    alpha_0, rho, c, k, epsilon = 10.0, 0.75, 0.001, 0, 1e-6
    gradient = d_f(initialPoint)
    fValue=[]
    d_fValue=[]
    xValues=[]
    yValues=[]
    while k <= 1e4 and np.linalg.norm(gradient) > epsilon:
        alpha = alpha_0
        hessian = d2_f(initialPoint)
        lambdaMin = np.min(np.linalg.eigvals(hessian))
        if lambdaMin <= 0:
            uk = -1*lambdaMin + 0.1
            size = np.shape(hessian)[0]
            dk = -1*np.dot(np.linalg.inv(hessian + uk * np.identity(size)),gradient)
        else:
            dk = -1*np.dot(np.linalg.inv(hessian),gradient)
        
        while f(initialPoint+alpha*dk) > f(initialPoint) + c*alpha*np.dot(gradient,dk):
            alpha = rho*alpha
        if initialPoint.shape == (2,):
            xValues.append(initialPoint[0])
            yValues.append(initialPoint[1])
        fValue.append(f(initialPoint))
        d_fValue.append(np.linalg.norm(gradient)) 
        initialPoint = initialPoint + alpha*dk
        gradient = d_f(initialPoint)
        k = k+1
    return initialPoint, fValue, d_fValue, xValues, yValues

# Do not rename or delete this function
def newton_method(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    inital_point: npt.NDArray[np.float64],
    condition: Literal["Pure", "Damped", "Levenberg-Marquardt", "Combined"],
) -> npt.NDArray[np.float64]:
    
    if condition == "Pure":
        finalValues, fValue, d_fValue, xValues, yValues = pureNewtonsMathod(f, d_f, d2_f, inital_point)
        plotGraph(fValue,f,inital_point,condition,"vals")
        plotGraph(d_fValue,f,inital_point,condition,"grad")
        if len(inital_point) == 2:
            plotContourGraph(condition, f, inital_point, xValues, yValues)        
        return finalValues
    elif condition == "Damped":
        finalValues, fValue, d_fValue, xValues, yValues = dampedNewtonMethod(f, d_f, d2_f, inital_point)
        plotGraph(fValue,f,inital_point,condition,"vals")
        plotGraph(d_fValue,f,inital_point,condition,"grad")
        if len(inital_point) == 2:
            plotContourGraph(condition, f, inital_point, xValues, yValues)         
        return finalValues
    elif condition == "Levenberg-Marquardt":
        finalValues, fValue, d_fValue, xValues, yValues =  levenbergMarquardtModification(f, d_f, d2_f, inital_point)
        plotGraph(fValue,f,inital_point,condition,"vals")
        plotGraph(d_fValue,f,inital_point,condition,"grad")
        if len(inital_point) == 2:
            plotContourGraph(condition, f, inital_point, xValues, yValues)
        return finalValues
    elif condition == "Combined":
        finalValues, fValue, d_fValue, xValues, yValues = combiningDampedNewtonMethodWithLMM(f, d_f, d2_f, inital_point)
        plotGraph(fValue,f,inital_point,condition,"vals")
        plotGraph(d_fValue,f,inital_point,condition,"grad")
        if len(inital_point) == 2:
            plotContourGraph(condition, f, inital_point, xValues, yValues)
        return finalValues
    
    # Complete this function
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_vals.png" for plotting f(x) vs iters
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_grad.png" for plotting |f'(x)| vs iters
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_cont.png" for plotting the contour plot
