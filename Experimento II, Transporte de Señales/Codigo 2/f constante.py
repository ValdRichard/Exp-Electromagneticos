import matplotlib.pyplot as plt
import numpy as np      
from scipy import odr
import pandas as pd
from funciones import ajuste_gráfico_ODR, ajuste_gráfico_curvefit

# Datos experimentales
url = 'https://docs.google.com/spreadsheets/d/1UOEOFKQLhlWB2_Yebj6ubf3RC3mSYtyYfgYvnw7MLqc/export?format=csv'

df = pd.read_csv(url)

Ve = df['Ve'].values
Vs = df['Vs'].values
lnH = np.log(df['H'].values)
w = 2 * np.pi * 50.10
deltat = df['delta t'].values
phi = w * deltat
z = df['z'].values 

# Incertezas

errf = 0.05 * 50.10
errw = 2 * np.pi * errf
errdeltat = 0.0025/5
errVe = 0.03 * Ve + 0.1 * 5 + 0.001 
errVs= 0.03 * Vs + 0.1 * np.array([5] * 2 + [2] * 2 + [1] * 3 + [0.5] * 2 + [0.2] * 1 + [0.1] * 3 + [0.05] * 3) + 0.001
errH = np.sqrt((errVs / Ve) ** 2 + (Vs * errVe / Ve ** 2) ** 2)
errlnH = errH / df['H'].values
errphi = np.sqrt((w * errdeltat) ** 2 + (deltat * errw) ** 2)

#Ajuste lineal para ln(H) vs z
pendienteH, err_pendienteH, r_squaredH, varianza_residualH = ajuste_gráfico_curvefit(z, lnH, errlnH, r'$z$ [1]', r'$ln(H)$ [1]', 'ln(H) vs z')

tauH = 2/w * pendienteH**2
errtauH = tauH * 2 * err_pendienteH / np.abs(pendienteH)

"""#Ajuste lineal para phi vs z
pendientePhi, err_pendientePhi, r_squaredPhi, varianza_residualPhi = ajuste_gráfico_curvefit(z, phi, errphi, r'$z$ [1]', r'$\varphi$ [rad]', 'phi vs z')

tauPhi = 2/w * pendientePhi**2
errtauPhi = tauPhi * 2 * err_pendientePhi / pendientePhi"""

R = 2700
C = 220e-9
tauref = R*C
errC = C*0.05
errR = R*0.05
errtauref = np.sqrt((R*errC)**2 + (C*errR)**2)
print(f'RC_ref = {tauref:.2e} ± {errtauref:.2e} s')
#print(f'RC_phi = {tauPhi:.2e} ± {errtauPhi:.2e} s')
print(f'RC_H = {tauH:.2e} ± {errtauH:.2e} s')



