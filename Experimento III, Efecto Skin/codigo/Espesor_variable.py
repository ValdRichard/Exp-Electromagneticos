import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Datos tal cual los pibes
#espesor de la placa = np.array([0,1.54,1.56,1.56,1.64,1.63,1.69,1.60,1.55,1.58,1.56])   #Estos dos no importan 
#error en el espesor de la placa= np.array([0,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])
es = np.array([1.54,3.10,4.66,6.30,7.93,9.62,11.22,12.77,14.35,15.91])
des = np.array([0.01, 0.01414213562, 0.01732050808, 0.02, 0.02236067977, 0.02449489743, 0.02645751311, 0.02828427125, 0.03, 0.0316227766])
Vi = np.array([172,172,173,174,175,175,175,175,176,176])
dVi = np.array([1,1,1,1,1,1,1,1,1,1])
Vs = np.array([455,418,381,345,314,286,258,237,217,198])
dVs = np.array([1,2,2,2,1,1,1,1,1,1])
t = np.array([5.88,6.60,7.20,7.88,8.12,8.32,8.40,8.60,8.80,9.00])
dt = np.array([0.01,0.01,0.01,0.01,0.01,0.1,0.1,0.1,0.1,0.1])
phi = np.array([1.862034532, 2.090038761, 2.280042284, 2.495379611, 2.571381021, 2.634715528, 2.660049332, 2.72338384, 2.786718347, 2.850052855])
dphi = np.array([0.004865960927, 0.005217753198, 0.005522115665, 0.005877247338, 0.006100483201, 0.006111960232, 0.03210406429, 0.03212496242, 0.03214633819, 0.03216819067])
H = np.array([2.645348837, 2.430232558, 2.202312139, 1.982758621, 1.794285714, 1.634285714, 1.474285714, 1.354285714, 1.232954545, 1.125])
dH = np.array([0.01644215493, 0.01829874811, 0.01719609827, 0.01618541389, 0.01173790125, 0.01094832358, 0.01017964093, 0.009619860062, 0.009019923338, 0.008552268877])

#Grafico para ver que onda 

plt.figure(figsize=(8,6))

plt.errorbar(es, H, xerr=des, yerr=dH, fmt='bo', label=r'$|H|$')
plt.errorbar(es, phi, xerr=des, yerr=dphi, fmt='ro', label=r'$\phi$')

plt.xlabel('Espesor suma')
plt.ylabel('Y')

plt.legend()
plt.grid(True, alpha=0.4)

plt.show()

#Módulo

def modelo_H(x,A,B):
    return A*np.exp(-B*x)

popt_H, pcov_H = curve_fit(
    modelo_H,
    es,
    H,
    sigma=dH,
    absolute_sigma=True,
    bounds=([0,0],[np.inf,np.inf]))

A_H, B_H = popt_H
dA_H, dB_H = np.sqrt(np.diag(pcov_H))

H_aj = modelo_H(
    es,
    A_H,
    B_H)

chi2_H = np.sum(
    ((H-H_aj)/dH)**2)

gl_H = len(H)-2

chi2red_H = chi2_H/gl_H


print("AJUSTE |H|")
print(f"A = {A_H:.4e} ± {dA_H:.4e}")
print(f"B = {B_H:.4e} ± {dB_H:.4e}")
print(f"χ² reducido = {chi2red_H:.4f}")

x_fit = np.linspace(min(es), max(es), 500)

plt.figure(figsize=(8,6))
plt.errorbar(es, H, xerr=des, yerr=dH, fmt='bo', ecolor='gray', capsize=3, label='Datos experimentales')
plt.plot(x_fit, modelo_H(x_fit,A_H,B_H),'r-', linewidth=2, label='Ajuste exponencial')
plt.xlabel(r'$\sigma$ mm')
plt.ylabel(r'$|H|$')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

#Delta con el módulo
delta_H = 1/B_H

ddelta_H = dB_H/(B_H**2)

print(f"delta_H = {delta_H:.4e} ± {ddelta_H:.4e}")

#Phy

def modelo_phi(x,phi0,B):
    return phi0 + B*x

popt_phi, pcov_phi = curve_fit(
    modelo_phi,
    es,
    phi,
    sigma=dphi,
    absolute_sigma=True,
    bounds=([0,0],[np.inf,np.inf]))

phi0_fit, B_phi = popt_phi
dphi0, dB_phi = np.sqrt(np.diag(pcov_phi))

phi_aj = modelo_phi(
    es,
    phi0_fit,
    B_phi)

chi2_phi = np.sum(
    ((phi-phi_aj)/dphi)**2)

gl_phi = len(phi)-2
chi2red_phi = chi2_phi/gl_phi

print("AJUSTE φ")
print(f"phi0 = {phi0_fit:.4e} ± {dphi0:.4e}")
print(f"B = {B_phi:.4e} ± {dB_phi:.4e}")
print(f"χ² reducido = {chi2red_phi:.4f}")

x_fit = np.linspace(
    min(es),
    max(es),
    500)

plt.figure(figsize=(8,6))
plt.errorbar(es, phi, xerr=des, yerr=dphi, fmt='ro', ecolor='gray', capsize=3, label='Datos experimentales')
plt.plot(x_fit, modelo_phi(x_fit,phi0_fit,B_phi),'b-', linewidth=2, label='Ajuste lineal')
plt.xlabel(r'$\sigma$ mm')
plt.ylabel(r'$\phi$ (rad)')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

#Delta con phi

delta_phi = 1/B_phi

ddelta_phi = dB_phi/(B_phi**2)

print(f"delta_phi = {delta_phi:.4e} ± {ddelta_phi:.4e}")


# Gráfico combinado

x_fit = np.linspace(min(es), max(es), 500)

fig, ax1 = plt.subplots(figsize=(9,6))

ax1.errorbar(es, H, xerr=des, yerr=dH, fmt='o', color='maroon', alpha=0.9, capsize=3, label='Datos |H|')
ax1.plot(x_fit, modelo_H(x_fit, A_H, B_H), '-', color='maroon', linewidth=2, label='Ajuste exponencial |H|')
ax1.set_xlabel(r'$\sigma$ (mm)', fontsize=12)
ax1.set_ylabel(r'$|H|$', color='maroon')
ax1.tick_params(axis='y', labelcolor='maroon')

ax2 = ax1.twinx()
ax2.errorbar(es, phi, xerr=des, yerr=dphi, fmt='o', color='purple', alpha=0.5, capsize=3, label=r'Datos $\varphi$')

ax2.plot(x_fit, modelo_phi(x_fit, phi0_fit, B_phi), '-', color='purple', linewidth=2, label=r'Ajuste lineal $\varphi$')

ax2.set_ylabel(r'$\varphi$ (rad)', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.48,0.99))
ax1.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()