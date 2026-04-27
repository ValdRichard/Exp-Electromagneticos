
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
lnH = np.log(df['H'].values)
w = 2 * np.pi * df['f'].values
raizw = np.sqrt(w)
deltat = df['delta t'].values
phi = w * deltat


# Incertezas

errf = 0.05 * df['f'].values
errw = 2 * np.pi * errf
errraizw = 0.5 * errw / raizw
errdeltat = np.array([0.005/5] * 2 + [0.0025/5] * 2 + [0.001/5] * 2 + [0.0005/5] * 5 + [0.00025/5] * 1)
errVe = 0.03 * Ve + 0.1 * 5 + 0.001 
errVs= 0.03 * Vs + 0.1 * np.array([0.5] * 2 + [0.1] * 1 + [0.05] * 3 + [0.02] * 3 + [0.01] * 1 + [0.005] * 2) + 0.001
errH = np.sqrt((errVs / Ve) ** 2 + (Vs * errVe / Ve ** 2) ** 2)
errlnH = errH / df['H'].values
errphi = np.sqrt((w * errdeltat) ** 2 + (deltat * errw) ** 2)


# Bode para H y gráfico de phi vs sqrt(w)
"""fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.errorbar(x = raizw, y = lnH, yerr=errlnH, xerr=errraizw,  color='#a200ed', label=r'Datos experimentales de ln(H)', fmt= 'o', ecolor="#a200ed", elinewidth=1.5, capsize=2.5)
ax1.set_xlabel(r'$\sqrt{\omega} $ [Hz$^{1/2}$]', size=14)
ax1.set_ylabel(r'$ln(H)$ [1]', size=14)#, color='#a200ed')
ax2 = ax1.twinx()
ax2.errorbar(x = raizw, y = phi, yerr=errphi, xerr=errraizw, color='darkorange', label=r'Datos experimentales de $\varphi$', fmt= 'o', ecolor='darkorange', elinewidth=1.5, capsize=2.5)
ax2.set_ylabel(r'$\varphi$ [radianes]', size=14)#, color='darkorange')
ax1.set_title(f'Diagrama de Bode de $H(\omega)$')
ax1.grid(True, linestyle='--', alpha=0.7)
fig.legend(loc='upper right', bbox_to_anchor=(0.89, 0.85), shadow= True, fontsize=12)
plt.savefig(f'Figuras2/Bode_H_phi(sincolores).png', dpi=300, bbox_inches='tight')
plt.show()"""

# Ajuste lineal para ln(H) vs sqrt(w)
"""pendienteH, ordenadaH, err_pendienteH, err_ordenadaH, r_squaredH, varianza_residualH = ajuste_gráfico_ODR(raizw, lnH, errraizw, errlnH, r'$\sqrt{\omega}$ [Hz$^{1/2}$]', r'$ln(H)$ [1]', 'lnH_vs_raizw', r'$\tau_H =$ 6.67(95) x 10$^{-4}$ s', 0.05, 0.15)
z = 7

errz = 1
tauH = ((pendienteH/z)**2 ) * 2
errtauH = tauH * np.sqrt((2*err_pendienteH/pendienteH)**2 + (errz/z)**2)"""


# Ajuste lineal para phi vs sqrt(w)
pendientePhi, ordenadaPhi, err_pendientePhi, err_ordenadaPhi, r_squaredPhi, varianza_residualPhi = ajuste_gráfico_ODR(raizw, phi, errraizw, errphi, r'$\sqrt{\omega}$ [Hz$^{1/2}$]', r'$\varphi$ [rad]', 'phi_vs_raizw')
z = 7
errz = 1
tauPhi = ((pendientePhi/z)**2 ) * 2
errtauPhi = tauPhi * np.sqrt((2*err_pendientePhi/pendientePhi)**2 + (errz/z)**2)


R = 2700
C = 220e-9
tauref = R*C
errC = C*0.05
errR = R*0.05
errtauref = np.sqrt((R*errC)**2 + (C*errR)**2)
print(f'RC_ref = {tauref:.2e} ± {errtauref:.2e} s')
print(f'RC_phi = {tauPhi:.2e} ± {errtauPhi:.2e} s')
#print(f'RC_H = {tauH:.2e} ± {errtauH:.2e} s')
