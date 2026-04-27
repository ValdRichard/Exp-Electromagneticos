import matplotlib.pyplot as plt
import numpy as np
from scipy import odr
import pandas as pd
from scipy.optimize import curve_fit

# Ajuste lineal para ln(H) vs sqrt(w)
def ajuste_gráfico_ODR(x, y, sx, sy, xlabel, ylabel, archivo):
    def linear_model(B, x):
        return B[0] * x + B[1]
    modelo = odr.Model(linear_model)
    data = odr.RealData(x, y, sx=sx, sy=sy)
    ajuste = odr.ODR(data, modelo, beta0=[-1, 0], ifixb=[1, 0])
    output = ajuste.run()
    pendiente, ordenada = output.beta
    err_pendiente, err_ordenada = output.sd_beta
    varianza_residual = output.res_var  
    r_squared = 1 - np.sum((y - linear_model(output.beta, x)) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    # Gráfico con el ajuste lineal
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, y, yerr=sy, xerr=sx, fmt='o', color='darkorange',  ecolor='darkorange', elinewidth=1.5, capsize=2.5, label='Datos experimentales')
    ax.plot(x, linear_model(output.beta, x), color='#00b2ed', label=f'Ajuste lineal: y = {pendiente:.3f}x \n' r'$\tau_{\varphi} =$ 5.37(77) x 10$^{-4}$ s' f'\n' f' $ R^2 = {r_squared:.4f} $')
    ax.set_xlabel(f'{xlabel}', size=14)
    ax.set_ylabel(f'{ylabel}', size=14)
    ax.legend(shadow=True, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'Figuras2/{archivo}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return pendiente, ordenada, err_pendiente, err_ordenada, r_squared, varianza_residual

def ajuste_gráfico_curvefit(x, y, sy, xlabel, ylabel, archivo):
# Ajuste lineal con curvefit
    def linear_model(x, m):
        return m * x 
    modelo = linear_model
    popt, pcov = curve_fit(modelo, x, y, sigma=sy, absolute_sigma=True)
    pendiente = popt [0]        
    err_pendiente = np.sqrt(np.diag(pcov))[0]
    varianza_residual = np.sum((y - modelo(x, *popt)) ** 2) / (len(y) - len(popt))
    r_squared = 1 - np.sum((y - modelo(x, *popt)) ** 2) / np.sum((y - np.mean(y)) ** 2)
# Gráfico con el ajuste lineal
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, y, yerr=sy, fmt='o', color='darkorange',  ecolor='darkorange', elinewidth=1.5, capsize=2.5, label='Datos experimentales')
    ax.plot(x, linear_model(x, *popt), color='indigo', label=f'Ajuste lineal: $y = {pendiente:.3f} x $\n' r'$\tau_{H} =$ 6.00(16) x 10$^{-4}$ s 'f'\n' f'$ R^2 = {r_squared:.4f} $')
    ax.set_xlabel(f'{xlabel}', size=14)
    ax.set_ylabel(f'{ylabel}', size=14)
    ax.legend(shadow=True, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'Figuras2/{archivo}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pendiente, err_pendiente, r_squared, varianza_residual
