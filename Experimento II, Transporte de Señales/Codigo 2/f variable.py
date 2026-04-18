
import matplotlib.pyplot as plt
import numpy as np      
from scipy import odr
import pandas as pd


# Datos experimentales

url = 'https://docs.google.com/spreadsheets/d/1UOEOFKQLhlWB2_Yebj6ubf3RC3mSYtyYfgYvnw7MLqc/export?format=csv&gid=701853377'

df = pd.read_csv(url)
Ve = df['Ve'].values
Vs = df['Vs'].values
lnH = np.log(df['H'].values)
phi = df['phi'].values
w = 2 * np.pi * df['f'].values
raizw = np.sqrt(w)

# Incertezas

errf = 0.05 * df['f'].values
errraizw = np.sqrt((np.pi/2) * df['f'].values) * 0.05
errVe = 0.03 * Ve + 0.1 * 5 + 0.001 
errVs= 0.03 * Vs + 0.1 * np.array([0.5] * 2 + [0.1] * 1 + [0.05] * 3 + [0.02] * 3 + [0.01] * 1 + [0.005] * 2) + 0.001
errH = np.sqrt((errVs / Ve) ** 2 + (Vs * errVe / Ve ** 2) ** 2)
errlnH = errH / df['H'].values
errphi = 0.1


# Bode para H y gráfico de phi vs sqrt(w)
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.errorbar(x = raizw, y = lnH, yerr=errlnH, xerr=errraizw,  color='indigo', label=r'Datos experimentales de ln(H)', fmt= 'o', ecolor='indigo', elinewidth=1.5, capsize=2.5)
ax1.set_xlabel(r'$\sqrt{\omega} $ [Hz$^{1/2}$]', size=14)
ax1.set_ylabel(r'$ln(H)$ [1]', size=14)
ax2 = ax1.twinx()
ax2.errorbar(x = raizw, y = phi, yerr=errphi, xerr=errraizw, color='darkorange', label=r'Datos experimentales de $\varphi$', fmt= 'o', ecolor='darkorange', elinewidth=1.5, capsize=2.5)
ax2.set_ylabel(r'$\varphi$ [radianes]', size=14)
ax1.set_title(r'H vs $\sqrt{\omega}$')
ax1.grid(True, linestyle='--', alpha=0.7)
fig.legend(loc='upper right', bbox_to_anchor=(0.89, 0.85), shadow= True, fontsize=12)
plt.show()