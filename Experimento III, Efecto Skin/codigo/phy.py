import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Todos los datos
v_p = np.array([0.187, 0.0914, 0.0645, 0.0492, 0.0394, 0.0319, 0.0271, 0.0174, 0.0147, 0.012, 0.00931, 0.189, 0.129, 0.108, 0.0211])
v_s = np.array([5.17, 5.12, 3.89, 3, 2.45, 2.05, 1.76, 1.13, 0.987, 0.798, 0.639, 5.06, 5.92, 5.65, 1.98])
t = np.array([0.0158, 0.00364, 0.0021, 0.0015, 0.00116, 0.00094, 0.000788, 0.000492, 0.000406, 0.000346, 0.000276, 0.0165, 0.00784, 0.00536, 0.000604])
f = np.array([21.1, 121.4, 218.8, 319.8, 420.2, 521.3, 628.3, 1019, 1224, 1506, 1913, 20, 49.95, 80.65, 819.7])
mu = 4 * np.pi * 1e-7
es =  0.00156
des= 0.00001
dv_p = np.array([0.02661, 0.013742, 0.007935, 0.02076, 0.007182, 0.003957, 0.003813, 0.002522, 0.002441, 0.002236, 0.0022793, 0.02667, 0.01487, 0.01424, 0.003633])
dv_s = np.array([0.6561, 0.6546, 0.6177, 0.291, 0.2745, 0.2625, 0.2538, 0.0849, 0.08061, 0.07494, 0.07017, 0.6528, 0.6786, 0.6705, 0.1604])
df = np.array([1.055, 6.07, 10.94, 15.99, 21.01, 26.065, 31.415, 50.95, 61.2, 75.3, 95.65, 1, 2.4975, 4.0325, 40.985])
dt= np.array([0.005, 0.001, 0.0005, 0.0005, 0.00025, 0.00025, 0.0001, 0.0001, 0.00005, 0.00005, 0.00005, 0.0025, 0.001, 0.001, 0.0001])

x = np.sqrt(f*2*np.pi)
y = 2*np.pi*f*t+np.pi/2

dx = np.sqrt((np.pi/2)*((df**2)/f))
dy = 2*np.pi*np.sqrt((t*df)**2 +(f*dt)**2)

# modelo
def lineal(x, a, b):
    return a * x + b

popt, pcov = curve_fit(lineal, x, y, sigma=dy, absolute_sigma=True)
a, b = popt
da, db = np.sqrt(np.diag(pcov))

# ajuste
x_fit = np.linspace(min(x) - 0.5, max(x) + 0.5, 100)
y_fit = lineal(x_fit, a, b)

# Chi**2 red
y_ajustado = lineal(x, a, b)
chi2 = np.sum(((y - y_ajustado) / dy)**2)
chi2_red = chi2 / (len(x) - 2)

# R**2
ss_res = np.sum((y - y_ajustado)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - (ss_res / ss_tot)


# grafico
plt.figure(figsize=(8, 6))

plt.errorbar(x, y, xerr=dx, yerr=dy, fmt='o', color='blue',
             ecolor='gray', capsize=3, label='Datos experimentales')

# Recta ajustada
plt.plot(x_fit, y_fit, 'r-', label=f'Ajuste lineal: y = ({a:.3f} ± {da:.3f}) x + ({b:.3f} ± {db:.3f})')


plt.xlabel(r'$\sqrt{\omega}$', fontsize=14)
plt.ylabel(r'$\omega\Delta t$', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

print("Resultados del ajuste")
print(f"Pendiente (a) = {a:.4f} ± {da:.4f}")
print(f"Ordenada al origen (b) = {b:.4f} ± {db:.4f}")
print(f"Chi-cuadrado reducido = {chi2_red:.4f}")
print(f"R² = {r2:.4f}")

sigma = ((a/es)**2)*(2/mu)
ds= 4*np.sqrt((((a/es**2)*da)**2)/mu**2 +((((a**2)/es**3)*des)**2)/mu**2)

print(f"sigma = {sigma:.4e} ± {ds:.4e}")

#Sacando puntos feos a ver que onda (f=21.1, 20 y 49.95) #2.498e+07
#Si saco solo los de 20, 5.013e+07

v_p = np.array([0.0914, 0.0645, 0.0492, 0.0394, 0.0319, 0.0271, 0.0174, 0.0147, 0.012, 0.00931, 0.108, 0.0211])
v_s = np.array([5.12, 3.89, 3, 2.45, 2.05, 1.76, 1.13, 0.987, 0.798, 0.639, 5.65, 1.98, 5.90])
t = np.array([0.00364, 0.0021, 0.0015, 0.00116, 0.00094, 0.000788, 0.000492, 0.000406, 0.000346, 0.000276, 0.00536, 0.000604])
f = np.array([121.4, 218.8, 319.8, 420.2, 521.3, 628.3, 1019, 1224, 1506, 1913, 80.65, 819.7])
mu = 4 * np.pi * 1e-7
es =  0.00156
dv_p = np.array([0.013742, 0.007935, 0.02076, 0.007182, 0.003957, 0.003813, 0.002522, 0.002441, 0.002236, 0.0022793, 0.01424, 0.003633])
dv_s = np.array([0.6546, 0.6177, 0.291, 0.2745, 0.2625, 0.2538, 0.0849, 0.08061, 0.07494, 0.07017, 0.6705, 0.1604])
df = np.array([6.07, 10.94, 15.99, 21.01, 26.065, 31.415, 50.95, 61.2, 75.3, 95.65, 4.0325, 40.985])
dt= np.array([0.001, 0.0005, 0.0005, 0.00025, 0.00025, 0.0001, 0.0001, 0.00005, 0.00005, 0.00005, 0.001, 0.0001])
dt/=5

x = np.sqrt(f*2*np.pi)
y = 2*np.pi*f*t+np.pi/2

dx = np.sqrt((np.pi/2)*((df**2)/f))
dy = 2*np.pi*np.sqrt((t*df)**2 +(f*dt)**2)

# modelo
def lineal(x, m, b):
    return m * x + b

popt, pcov = curve_fit(lineal, x, y, sigma=dy, absolute_sigma=True)
m, b = popt
dm, db = np.sqrt(np.diag(pcov))

# ajuste
x_fit = np.linspace(min(x) - 0.5, max(x) + 0.5, 100)
y_fit = lineal(x_fit, m, b)

# Chi**2 red
y_ajustado = lineal(x, m, b)
chi2 = np.sum(((y - y_ajustado) / dy)**2)
chi2_red = chi2 / (len(x) - 2)

# R**2
ss_res = np.sum((y - y_ajustado)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - (ss_res / ss_tot)


# grafico
plt.figure(figsize=(8, 6))

plt.errorbar(x, y, xerr=dx, yerr=dy, fmt='o', color='blue',
             ecolor='gray', capsize=3, label='Datos experimentales')

# Recta ajustada
plt.plot(x_fit, y_fit, 'r-', label=f'Ajuste lineal: y = ({m:.3f} ± {dm:.3f}) x + ({b:.3f} ± {db:.3f})')


plt.xlabel(r'$\sqrt{\omega}$', fontsize=14)
plt.ylabel(r'$\omega\Delta t$', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

print("Resultados del ajuste")
print(f"Pendiente (a) = {m:.4f} ± {dm:.4f}")
print(f"Ordenada al origen (b) = {b:.4f} ± {db:.4f}")
print(f"Chi-cuadrado reducido = {chi2_red:.4f}")
print(f"R² = {r2:.4f}")

sigma = ((m/es)**2)*(2/mu)
ds= 4*np.sqrt((((m/es**2)*dm)**2)/mu**2 +((((m**2)/es**3)*des)**2)/mu**2)

print(f"sigma = {sigma:.4e} ± {ds:.4e}")