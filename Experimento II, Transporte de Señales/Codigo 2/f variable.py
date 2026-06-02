
from xml.parsers.expat import model

import matplotlib.pyplot as plt
import numpy as np      
from scipy import odr
import pandas as pd
from funciones import ajuste_gráfico_ODR, ajuste_gráfico_curvefit
import os

# Datos experimentales

url = 'https://docs.google.com/spreadsheets/d/1UOEOFKQLhlWB2_Yebj6ubf3RC3mSYtyYfgYvnw7MLqc/export?format=csv&gid=701853377'

df = pd.read_csv(url)
Ve = df['Ve'].values
Vs = df['Vs'].values
logH = 20*np.log10(df['H'].values) 
w = 2 * np.pi * df['f'].values
raizw = np.sqrt(w)
deltat = df['delta t'].values
phi = w * deltat


# Incertezas

errf = 0.05 * df['f'].values
errw = 2 * np.pi * errf
errraizw = 0.5 * errw / raizw
errdeltat = np.array([0.005/10] * 2 + [0.0025/10] * 2 + [0.001/10] * 2 + [0.0005/10] * 5 + [0.00025/10] * 1)
errVe = 0.03 * Ve + 0.1 * 5 + 0.001 
errVs= 0.03 * Vs + 0.1 * np.array([0.5] * 2 + [0.1] * 1 + [0.05] * 3 + [0.02] * 3 + [0.01] * 1 + [0.005] * 2) + 0.001
errH = np.sqrt((errVs / Ve) ** 2 + (Vs * errVe / Ve ** 2) ** 2)
errlogH = 20 * errH / (Vs / Ve) / np.log(10)
errphi = np.sqrt((w * errdeltat) ** 2 + (deltat * errw) ** 2)


# Bode para H y gráfico de phi vs sqrt(w)
pendienteH, ordenadaH, err_pendienteH, err_ordenadaH, r_squaredH, varianza_residualH = ajuste_gráfico_ODR(raizw, logH, errraizw, errlogH, r'$\sqrt{\omega}$ [Hz$^{1/2}$]', r'$log_{10}(H)$ [1]')
pendientePhi, ordenadaPhi, err_pendientePhi, err_ordenadaPhi, r_squaredPhi, varianza_residualPhi = ajuste_gráfico_ODR(raizw, phi, errraizw, errphi, r'$\sqrt{\omega}$ [Hz$^{1/2}$]', r'$\varphi$ [rad]')

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.errorbar(x = raizw, y = logH, yerr=errlogH, xerr=errraizw,color = 'indigo', label=r'Datos experimentales de $log_{10}(H)$', fmt= 'o',ecolor= 'indigo', elinewidth=1.5, capsize=2.5)
ax1.plot(raizw, pendienteH * raizw + ordenadaH, color='indigo', label=f'Ajuste lineal:' r' $log_{10}(H)$' f'$= {pendienteH:.3f} \sqrt{{\omega}} + {ordenadaH:.3f}$ \n' f' $ R^2 = {r_squaredH:.4f} $')
ax1.set_xlabel(r'$\sqrt{\omega} $ [Hz$^{1/2}$]', size=14)
ax1.set_ylabel(r'$20log_{10}(H)$ [dB]', size=14)#, color='#a200ed')
ax2 = ax1.twinx()
ax2.errorbar(x = raizw, y = phi, yerr=errphi, xerr=errraizw, color='darkorange', label=r'Datos experimentales de $\varphi$', fmt= 'o', ecolor='darkorange', elinewidth=1.5, capsize=2.5)
ax2.plot(raizw, pendientePhi * raizw + ordenadaPhi, color='darkorange', label=f'Ajuste lineal: $\\varphi = {pendientePhi:.3f} \sqrt{{\omega}} + {ordenadaPhi:.3f}$ \n' f' $ R^2 = {r_squaredPhi:.4f} $')
ax2.set_ylabel(r'$\varphi$ [radianes]', size=14)#, color='darkorange')
ax1.grid(True, linestyle='--', alpha=0.7)
fig.legend(loc='upper right', bbox_to_anchor=(0.63, 0.89), shadow= True, fontsize=8.45)
plt.show()
# # Ajuste lineal para ln(H) vs sqrt(w)
# pendienteH, ordenadaH, err_pendienteH, err_ordenadaH, r_squaredH, varianza_residualH = ajuste_gráfico_ODR(raizw, lnH, errraizw, errlnH, r'$\sqrt{\omega}$ [Hz$^{1/2}$]', r'$ln(H)$ [1]')
# z = 7

# errz = 1
# tauH = ((pendienteH/z)**2 ) * 2
# errtauH = tauH * np.sqrt((2*err_pendienteH/pendienteH)**2 + (errz/z)**2)


# # Ajuste lineal para phi vs sqrt(w)
# pendientePhi, ordenadaPhi, err_pendientePhi, err_ordenadaPhi, r_squaredPhi, varianza_residualPhi = ajuste_gráfico_ODR(raizw, phi, errraizw, errphi, r'$\sqrt{\omega}$ [Hz$^{1/2}$]', r'$\varphi$ [rad]', 'phi_vs_raizw')
# z = 7
# errz = 1
# tauPhi = ((pendientePhi/z)**2 ) * 2
# errtauPhi = tauPhi * np.sqrt((2*err_pendientePhi/pendientePhi)**2 + (errz/z)**2)


# R = 2700
# C = 220e-9
# tauref = R*C
# errC = C*0.05
# errR = R*0.05
# errtauref = np.sqrt((R*errC)**2 + (C*errR)**2)
# print(f'RC_ref = {tauref:.2e} ± {errtauref:.2e} s')
# print(f'RC_phi = {tauPhi:.2e} ± {errtauPhi:.2e} s')
# print(f'RC_H = {tauH:.2e} ± {errtauH:.2e} s')
