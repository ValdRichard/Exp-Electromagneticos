import matplotlib.pyplot as plt
import numpy as np
from scipy import odr
import pandas as pd
from scipy.optimize import curve_fit

# Ajuste lineal para ln(H) vs sqrt(w)
def ajuste_gráfico_ODR(x, y, sx, sy, xlabel, ylabel, titulo):
    def linear_model(B, x):
        return B[0] * x + B[1]
    modelo = odr.Model(linear_model)
    data = odr.RealData(x, y, sx=sx, sy=sy)
    ajuste = odr.ODR(data, modelo, beta0=[-1, 0])
    output = ajuste.run()
    pendiente, ordenada = output.beta
    err_pendiente, err_ordenada = output.sd_beta
    varianza_residual = output.res_var  
    r_squared = 1 - np.sum((y - linear_model(output.beta, x)) ** 2) / np.sum((y - np.mean(y)) ** 2)

    # Gráfico con el ajuste lineal
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, y, yerr=sy, xerr=sx, fmt='o', color='darkorange',  ecolor='darkorange', elinewidth=1.5, capsize=2.5, label='Datos experimentales')
    ax.plot(x, linear_model(output.beta, x), color='indigo', label=f'Ajuste lineal $ R^2 = {r_squared:.4f} $')
    ax.set_xlabel(f'{xlabel}', size=14)
    ax.set_ylabel(f'{ylabel}', size=14)
    ax.set_title(f'{titulo}', size=16)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    return pendiente, ordenada, err_pendiente, err_ordenada, r_squared, varianza_residual

def ajuste_gráfico_curvefit(x, y, sy, xlabel, ylabel, titulo):
# Ajuste lineal con curvefit
    def linear_model(x, m, b):
        return m * x + b
    modelo = linear_model
    popt, pcov = curve_fit(modelo, x, y, sigma=sy, absolute_sigma=True)
    pendiente, ordenada = popt          
    err_pendiente, err_ordenada = np.sqrt(np.diag(pcov))
    varianza_residual = np.sum((y - modelo(x, *popt)) ** 2) / (len(y) - len(popt))
    r_squared = 1 - np.sum((y - modelo(x, *popt)) ** 2) / np.sum((y - np.mean(y)) ** 2)
# Gráfico con el ajuste lineal
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, y, yerr=sy, fmt='o', color='darkorange',  ecolor='darkorange', elinewidth=1.5, capsize=2.5, label='Datos experimentales')
    ax.plot(x, linear_model(x, *popt), color='indigo', label=f'Ajuste lineal $ R^2 = {r_squared:.4f} $')
    ax.set_xlabel(f'{xlabel}', size=14)
    ax.set_ylabel(f'{ylabel}', size=14)
    ax.set_title(f'{titulo}', size=16)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    return pendiente, ordenada, err_pendiente, err_ordenada, r_squared, varianza_residual
