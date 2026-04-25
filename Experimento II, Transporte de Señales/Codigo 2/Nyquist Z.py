import matplotlib.pyplot as plt
import numpy as np      
import pandas as pd

# Datos experimentales

url = 'https://docs.google.com/spreadsheets/d/1UOEOFKQLhlWB2_Yebj6ubf3RC3mSYtyYfgYvnw7MLqc/export?format=csv&gid=2070018110'
df = pd.read_csv(url)

R = 2700
errR = 0.05 * R
Vf = df['Vf'].values
Vr = df['Vr'].values
Z = R * Vf/Vr
deltat = df['Delta t'].values
w = 2 * np.pi * df['f'].values
phi = w * deltat
ReZ = Z * np.cos(phi)
ImZ = Z * np.sin(phi)

# Incertezas
errVf = 0.03 * Vf + 0.1 * 5 + 0.001
errVr = 0.03 * Vr + 0.1 * 2 + 0.001
errZ = Z * np.sqrt((errVf/Vf)**2 + (errVr/Vr)**2 + (errR/R)**2)
errZ2 = np.sqrt((errVf/Vf)**2 + (errVr/Vr)**2)
errdeltat = np.array([0.005/5] * 1 + [0.0025/5] * 1 + [0.001/5] * 2 + [0.0005/5] * 5 + [0.00025/5] * 3)
errf = 0.05 * df['f'].values
errw = 2 * np.pi * errf
errphi = phi * np.sqrt((errdeltat/deltat) ** 2 + (errw/w) ** 2)
errRez = Z * np.sqrt((np.sqrt((errVf/Vf)**2 + (errVr/Vr)**2 + (errR/R)**2) * np.cos(phi))**2 + (np.sin(phi) * errphi)**2)
errImZ = Z * np.sqrt((np.sqrt((errVf/Vf)**2 + (errVr/Vr)**2 + (errR/R)**2) * np.sin(phi))**2 + (np.cos(phi) * errphi)**2)


#print(f"lo que me esta rompiendo las bolas: {np.sqrt((w * errdeltat) ** 2)}")
#print(f"errphi: {errphi}")




# Gráfico de Nyquist de Z
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(ReZ, ImZ, xerr=errRez, yerr=errImZ, fmt='o', color='darkorange', ecolor='darkorange', elinewidth=1.5, capsize=2.5)
ax.set_xlabel(r'Re($Z$) [Ω]', size=14)
ax.set_ylabel(r'Im($Z$) [Ω]', size=14)
ax.set_title('Diagrama de Nyquist de $Z$', size=16)
ax.grid(True, linestyle='--', alpha=0.7)
plt.show()