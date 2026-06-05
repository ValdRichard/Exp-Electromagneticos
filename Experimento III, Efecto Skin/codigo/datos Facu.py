"""
Análisis del efecto skin en un material conductor.

El dispositivo experimental consiste en una bobina primaria que genera un
campo magnético oscilante que atraviesa el material bajo estudio antes de
ser detectado por una bobina secundaria. A medida que la frecuencia aumenta,
el efecto skin atenúa el campo en el interior del material, reduciendo la
señal en la secundaria de forma exponencial en función de sqrt(omega).

Variables medidas:
    v_p : tensión pico en la bobina primaria [V]
    v_s : tensión pico en la bobina secundaria [V]
    t   : retardo temporal entre las señales [s]
    f   : frecuencia de la señal [Hz]
    es  : espesor del material [m]
    mu  : permeabilidad magnética del vacío [H/m]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES Y PARÁMETROS FIJOS
# ─────────────────────────────────────────────────────────────────────────────

mu  = 4 * np.pi * 1e-7   # Permeabilidad del vacío [H/m]
es  = 0.00156             # Espesor del material [m]
des = 0.00001             # Incerteza del espesor [m]

# Parámetros teóricos fijos (calculados externamente / de literatura)
A_fijo = (64 * np.pi) / 415035    # Valor teórico de la amplitud
B_fijo = 0.00759251               # Valor teórico del coeficiente de decaimiento

# Parámetros para el ajuste 2 (reparametrizacion fisica)
n = (64 * np.pi) / 415035         # Factor de escala geométrico para A
u = 0.00000123656                 # Factor de escala para B = sqrt(mu/2*sigma) * es


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def expo(x, A, B):
    """Modelo exponencial: A * exp(-B * x)."""
    return A * np.exp(-B * x)

def lineal(x, m, b):
    """Modelo lineal: m * x + b."""
    return m * x + b

def calc_sigma(B, dB, es, des, mu):
    """
    Calcula la conductividad eléctrica sigma y su incerteza a partir del
    coeficiente de decaimiento B del ajuste exponencial.

    De la teoría del efecto skin: B = es * sqrt(mu * sigma / 2)
    => sigma = (B / es)^2 * (2 / mu)

    La incerteza se obtiene por propagación de errores en B y es.
    """
    sigma = ((B / es)**2) * (2 / mu)
    ds = 4 * np.sqrt(
        (((B / es**2) * dB)**2) / mu**2 +
        ((((B**2) / es**3) * des)**2) / mu**2
    )
    return sigma, ds

def chi2_reducido(y_obs, y_mod, dy, n_params):
    """Calcula el chi-cuadrado reducido."""
    chi2 = np.sum(((y_obs - y_mod) / dy)**2)
    gl   = len(y_obs) - n_params
    return chi2 / gl

def R2(y_obs, y_mod):
    """Calcula el coeficiente de determinación R²."""
    ss_res = np.sum((y_obs - y_mod)**2)
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    return 1 - ss_res / ss_tot


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 1: TODOS LOS PUNTOS (incluye mediciones a ~20 Hz)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

f = np.array([
    10.28, 20.3, 50.5, 100.2, 176.4, 253.3,
    360, 453, 600, 802, 1000, 1256,
    1500, 2044, 2517, 3023
])

df = np.array([
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    1, 1, 1, 1, 1,
    10, 10, 10, 10, 10
])

t = np.array([
    0.02, 0.0084, 0.002, 0.00077, 0.00022, 0.00012,
    0.00008, 0.00004, 0.000032, 0.000008, 0.00002,
    0.00002, 0.000024, 0.000022, 0.00002, 0.000028
])

dt = np.array([
    0.01, 0.0001, 0.001, 0.00001, 0.00001, 0.00001,
    0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
    0.00001, 0.00001, 0.00001, 0.00001, 0.00001
])

v_p = np.array([
    0.000197, 0.000171, 0.00012, 0.0000929,
    0.0000691, 0.0000544, 0.0000414, 0.000034,
    0.0000264, 0.0000202, 0.0000162, 0.000013,
    0.0000112, 0.00000847, 0.00000686, 0.0000058
])

dv_p = np.array([
    1e-6, 1e-6, 1e-6, 1e-7, 1e-7, 1e-7,
    1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7,
    1e-7, 1e-7, 1e-7, 1e-7
])

v_s = np.array([
    3.22, 4.91, 5.62, 5.13, 4.09, 3.31,
    2.58, 2.15, 1.70, 1.32, 1.07,
    0.874, 0.736, 0.575, 0.461, 0.403
])

dv_s = np.array([
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    0.01, 0.01, 0.1, 0.1, 0.1,
    0.01, 0.01, 0.01, 0.01, 0.01
])

# Variables del ajuste del módulo: x = sqrt(omega), y = |H|/omega
x1   = np.sqrt(f * 2 * np.pi)
y1   = v_s / (v_p * 2 * np.pi * f)
dx1  = np.sqrt((np.pi / 2) * (df**2 / f))
dy1  = (1 / (2 * np.pi * v_p * f)) * np.sqrt(
    dv_s**2 + (v_s * dv_p / v_p)**2 + (df * v_s / f)**2
)

# ── Ajuste exponencial libre: A y C libres ──────────────────────────────────
#
#   Primer ajuste exploratorio con todos los datos, A y C completamente libres.
#   Sirve para tener una estimación inicial de sigma antes de filtrar puntos.

popt1_full, pcov1_full = curve_fit(
    expo, x1, y1, sigma=dy1, absolute_sigma=True,
    bounds=([0, 0], [np.inf, np.inf])
)
A1f, C1f  = popt1_full
dA1f, dC1f = np.sqrt(np.diag(pcov1_full))

x_fit1 = np.linspace(x1.min(), x1.max(), 500)
y_fit1 = expo(x_fit1, A1f, C1f)

chi2r_1f = chi2_reducido(y1, expo(x1, A1f, C1f), dy1, n_params=2)
sigma_1f, ds_1f = calc_sigma(C1f, dC1f, es, des, mu)

print("─── AJUSTE 1 (todos los puntos, A y C libres) ───")
print(f"  A = {A1f:.4e} ± {dA1f:.4e}")
print(f"  C = {C1f:.4e} ± {dC1f:.4e}")
print(f"  χ² reducido = {chi2r_1f:.4f}")
print(f"  sigma = {sigma_1f:.4e} ± {ds_1f:.4e} S/m")

plt.figure(figsize=(8, 6))
plt.errorbar(x1, y1, xerr=dx1, yerr=dy1,
             fmt='o', color='blue', ecolor='gray', capsize=3,
             label='Datos experimentales')
plt.plot(x_fit1, y_fit1, 'r-',
         label=fr'Ajuste: ${A1f:.2e}\,e^{{-{C1f:.2e}\,x}}$')
plt.xlabel(r'$\sqrt{\omega}$', fontsize=14)
plt.ylabel(r'$|H|/\omega$', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.title("Todos los puntos – ajuste libre")
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 2: SIN LOS PUNTOS DE ~20 Hz
# ─────────────────────────────────────────────────────────────────────────────
#
#   Se excluyen las mediciones a 20 y 21.1 Hz por considerar que están fuera
#   del régimen de validez del modelo (o presentan mayor incerteza sistemática).
#   ─────────────────────────────────────────────────────────────────────────────
#   BUG CORREGIDO: en la versión original comentada, v_s tenía 14 elementos
#   (terminaba en ..., 5.90, 5.92) mientras que v_p, t y f tenían 13.
#   El valor 5.92 corresponde al punto de 20 Hz del dataset 1 y no debe estar.
#   Se eliminó ese elemento espurio.
# ─────────────────────────────────────────────────────────────────────────────


f = np.array([
    10.28, 20.3, 50.5, 100.2, 176.4, 253.3,
    360, 453, 600, 802, 1000, 1256,
    1500, 2044, 2517, 3023
])

df = np.array([
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    1, 1, 1, 1, 1,
    10, 10, 10, 10, 10
])

t = np.array([
    0.02, 0.0084, 0.002, 0.00077, 0.00022, 0.00012,
    0.00008, 0.00004, 0.000032, 0.000008, 0.00002,
    0.00002, 0.000024, 0.000022, 0.00002, 0.000028
])

dt = np.array([
    0.01, 0.0001, 0.001, 0.00001, 0.00001, 0.00001,
    0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
    0.00001, 0.00001, 0.00001, 0.00001, 0.00001
])

v_p = np.array([
    0.000197, 0.000171, 0.00012, 0.0000929,
    0.0000691, 0.0000544, 0.0000414, 0.000034,
    0.0000264, 0.0000202, 0.0000162, 0.000013,
    0.0000112, 0.00000847, 0.00000686, 0.0000058
])

dv_p = np.array([
    1e-6, 1e-6, 1e-6, 1e-7, 1e-7, 1e-7,
    1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7,
    1e-7, 1e-7, 1e-7, 1e-7
])

v_s = np.array([
    3.22, 4.91, 5.62, 5.13, 4.09, 3.31,
    2.58, 2.15, 1.70, 1.32, 1.07,
    0.874, 0.736, 0.575, 0.461, 0.403
])

dv_s = np.array([
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    0.01, 0.01, 0.1, 0.1, 0.1,
    0.01, 0.01, 0.01, 0.01, 0.01
])

# Variables del ajuste del módulo
x    = np.sqrt(f * 2 * np.pi)
y    = v_s / (v_p * 2 * np.pi * f)
dx   = np.sqrt((np.pi / 2) * (df**2 / f))
dy   = (1 / (2 * np.pi * v_p * f)) * np.sqrt(
    dv_s**2 + (v_s * dv_p / v_p)**2 + (df * v_s / f)**2
)

# ── Ajuste exponencial libre (Dataset 2) ─────────────────────────────────────
#
#   Repetición del ajuste libre ahora sin los puntos problemáticos de ~20 Hz.
#   Este resultado es el que se usa como referencia principal del módulo.

popt_free, pcov_free = curve_fit(
    expo, x, y, sigma=dy, absolute_sigma=True,
    bounds=([0, 0], [np.inf, np.inf])
)
A_fit, B_fit = popt_free
dA_fit, dB_fit = np.sqrt(np.diag(pcov_free))

chi2r_free = chi2_reducido(y, expo(x, A_fit, B_fit), dy, n_params=2)
sigma_free, ds_free = calc_sigma(B_fit, dB_fit, es, des, mu)

print("\n─── AJUSTE libre Dataset 2 (sin ~20 Hz, A y B libres) ───")
print(f"  A = {A_fit:.4e} ± {dA_fit:.4e}")
print(f"  B = {B_fit:.4e} ± {dB_fit:.4e}")
print(f"  χ² reducido = {chi2r_free:.4f}")
print(f"  sigma = {sigma_free:.4e} ± {ds_free:.4e} S/m")

x_fit = np.linspace(x.min(), x.max(), 500)

plt.figure(figsize=(8, 6))
plt.errorbar(x, y, xerr=dx, yerr=dy,
             fmt='o', color='blue', ecolor='gray', capsize=3,
             label='Datos experimentales')
plt.plot(x_fit, expo(x_fit, A_fit, B_fit), 'r-',
         label=fr'Ajuste: ${A_fit:.2e}\,e^{{-{B_fit:.2e}\,x}}$')
plt.xlabel(r'$\sqrt{\omega}$', fontsize=14)
plt.ylabel(r'$|H|/\omega$', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.title("Sin puntos de ~20 Hz – ajuste libre")
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# COMPARACIÓN DE CUATRO VARIANTES DE AJUSTE (Dataset 2)
# ─────────────────────────────────────────────────────────────────────────────
#
#   Se comparan cuatro modelos para el módulo |H|/omega vs sqrt(omega):
#
#   1) A y B completamente libres: ajuste más general.
#   2) A = C*n  (C libre),  B = D*u  (D libre):
#         reparametrización física donde n fija la escala geométrica del
#         prefactor y u = sqrt(mu/(2*sigma_ref))*es fija la escala del
#         exponente. D² da directamente sigma en unidades de sigma_ref.
#   3) A = A_fijo (teórico), B libre:
#         se fija el prefactor al valor teórico y se ajusta solo el decaimiento.
#   4) A = A_fijo, B = B_fijo:
#         curva completamente teórica, sin parámetros libres.
#         El chi² reducido indica qué tan bien predice la teoría sin ajuste.

# ── Ajuste 1: A, B libres ────────────────────────────────────────────────────
popt_aj1, pcov_aj1 = curve_fit(
    expo, x, y, sigma=dy, absolute_sigma=True,
    bounds=([0, 0], [np.inf, np.inf])
)
A1, B1   = popt_aj1
dA1, dB1 = np.sqrt(np.diag(pcov_aj1))
chi2r_aj1 = chi2_reducido(y, expo(x, A1, B1), dy, n_params=2)

# ── Ajuste 2: A = C*n,  B = D*u (C, D libres) ───────────────────────────────
def expo_reparametrizado(x, C, D):
    """A = C*n,  B = D*u  =>  D² ∝ sigma."""
    return (C * n) * np.exp(-(D * u) * x)

popt_aj2, pcov_aj2 = curve_fit(
    expo_reparametrizado, x, y, sigma=dy, absolute_sigma=True,
    bounds=([0, 0], [np.inf, np.inf])
)
C2, D2   = popt_aj2
dC2, dD2 = np.sqrt(np.diag(pcov_aj2))
A2, B2   = C2 * n,   D2 * u
dA2, dB2 = dC2 * n,  dD2 * u
chi2r_aj2 = chi2_reducido(y, expo_reparametrizado(x, C2, D2), dy, n_params=2)
sigma_aj2 = D2**2   # en unidades de sigma_ref (ver definicion de u)

# ── Ajuste 3: A fijo (teórico), B libre ─────────────────────────────────────
def expo_A_fijo(x, B):
    return A_fijo * np.exp(-B * x)

popt_aj3, pcov_aj3 = curve_fit(
    expo_A_fijo, x, y, sigma=dy, absolute_sigma=True,
    bounds=([0], [np.inf])
)
B3  = popt_aj3[0]
dB3 = np.sqrt(np.diag(pcov_aj3))[0]
chi2r_aj3 = chi2_reducido(y, expo_A_fijo(x, B3), dy, n_params=1)

# ── Ajuste 4: A y B fijos (curva teórica pura, 0 parámetros libres) ─────────
y_teorico = expo(x, A_fijo, B_fijo)
chi2r_aj4 = chi2_reducido(y, y_teorico, dy, n_params=0)

# ── Resumen numérico ─────────────────────────────────────────────────────────
print("\n─── COMPARACIÓN DE AJUSTES (Dataset 2) ───")
print("Ajuste 1: A, B libres")
print(f"  A = {A1:.4e} ± {dA1:.4e}")
print(f"  B = {B1:.4e} ± {dB1:.4e}")
print(f"  χ² reducido = {chi2r_aj1:.4f}")

print("Ajuste 2: A = C·n, B = D·u  (reparametrización física)")
print(f"  C = {C2:.4e} ± {dC2:.4e}")
print(f"  D = {D2:.4e} ± {dD2:.4e}")
print(f"  A = {A2:.4e} ± {dA2:.4e}")
print(f"  B = {B2:.4e} ± {dB2:.4e}")
print(f"  sigma (D²) = {sigma_aj2:.6e}")
print(f"  χ² reducido = {chi2r_aj2:.4f}")

print("Ajuste 3: A fijo, B libre")
print(f"  A fijo = {A_fijo:.4e}")
print(f"  B = {B3:.4e} ± {dB3:.4e}")
print(f"  χ² reducido = {chi2r_aj3:.4f}")

print("Ajuste 4: A fijo, B fijo (curva teórica)")
print(f"  A fijo = {A_fijo:.4e}")
print(f"  B fijo = {B_fijo:.4e}")
print(f"  χ² reducido = {chi2r_aj4:.4f}")

# ── Gráfico comparativo de los cuatro ajustes ────────────────────────────────
x_fit = np.linspace(x.min(), x.max(), 500)

plt.figure(figsize=(8, 6))
plt.errorbar(x, y, xerr=dx, yerr=dy,
             fmt='o', color='blue', ecolor='gray', capsize=3,
             label='Datos experimentales')
plt.plot(x_fit, expo(x_fit, A1, B1),
         color='red',    linestyle='-',  linewidth=2, label='1) A, B libres')
plt.plot(x_fit, expo_reparametrizado(x_fit, C2, D2),
         color='orange', linestyle='--', linewidth=3, label='2) A=C·n, B=D·u')
plt.plot(x_fit, expo_A_fijo(x_fit, B3),
         color='black',  linestyle=':',  linewidth=4, label='3) A fijo, B libre')
plt.plot(x_fit, expo(x_fit, A_fijo, B_fijo),
         color='green',  linestyle='-.', linewidth=2, label='4) Teórico (A, B fijos)')
plt.xlabel(r'$\sqrt{\omega}$', fontsize=14)
plt.ylabel(r'$|H|/\omega$', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.title("Comparación de cuatro variantes de ajuste")
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO: AJUSTE vs CURVA TEÓRICA (escala lineal)
# ─────────────────────────────────────────────────────────────────────────────

plt.figure(figsize=(8, 6))
plt.errorbar(x, y, xerr=dx, yerr=dy,
             fmt='o', color='maroon', ecolor='maroon', capsize=3,
             label='Datos experimentales')
plt.plot(x_fit, expo_reparametrizado(x_fit, C2, D2),
         color='maroon', linestyle='-', linewidth=3, label='Ajuste exponencial')
plt.plot(x_fit, expo(x_fit, A_fijo, B_fijo),
         color='purple', linestyle='-.', linewidth=3, label='Curva teórica')
plt.xlabel(r'$\sqrt{\omega}$', fontsize=14)
plt.ylabel(r'$|H|/\omega$', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1), framealpha=0.9)
plt.title("Ajuste vs curva teórica (escala lineal)")
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO: AJUSTE vs CURVA TEÓRICA EN dB
# ─────────────────────────────────────────────────────────────────────────────

y_dB    = 20 * np.log10(y)
dy_dB   = 20 / (np.log(10) * y) * dy
R2_aj2  = R2(y, expo_reparametrizado(x, C2, D2))

plt.figure(figsize=(8, 6))
plt.errorbar(x, y_dB, xerr=dx, yerr=dy_dB,
             fmt='o', color='maroon', ecolor='maroon', capsize=3,
             label=r'Datos $|H|/\omega$ [dB]')
plt.plot(x_fit, 20 * np.log10(expo_reparametrizado(x_fit, C2, D2)),
         color='maroon', linestyle='-', linewidth=3,
         label=fr'Ajuste $|H|/\omega$ [dB]  ($R^2={R2_aj2:.4f}$)')
plt.plot(x_fit, 20 * np.log10(expo(x_fit, A_fijo, B_fijo)),
         color='purple', linestyle='--', linewidth=3,
         label='Curva teórica')
plt.xlabel(r'$\sqrt{\omega}\;[\mathrm{Hz}^{1/2}]$', fontsize=14)
plt.ylabel(r'$20\log_{10}(|H|/\omega)$ [dB]', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1), framealpha=0.9)
plt.title("Ajuste vs curva teórica (escala logarítmica)")
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CONDUCTIVIDADES SIGMA PARA CADA AJUSTE
# ─────────────────────────────────────────────────────────────────────────────

sigma1, ds1 = calc_sigma(B1,      dB1,      es, des, mu)
sigma2, ds2 = calc_sigma(B2,      dB2,      es, des, mu)
sigma3, ds3 = calc_sigma(B3,      dB3,      es, des, mu)
sigma4, ds4 = calc_sigma(B_fijo,  0,        es, des, mu)

print("\n─── CONDUCTIVIDAD ELÉCTRICA sigma [S/m] ───")
print(f"  Ajuste 1 (A, B libres)        : sigma = {sigma1:.4e} ± {ds1:.4e}")
print(f"  Ajuste 2 (reparametrizado)    : sigma = {sigma2:.4e} ± {ds2:.4e}")
print(f"  Ajuste 3 (A fijo, B libre)    : sigma = {sigma3:.4e} ± {ds3:.4e}")
print(f"  Ajuste 4 (teórico)            : sigma = {sigma4:.4e} ± {ds4:.4e}")

delta  = 1 / B2
ddelta = dB2 / B2**2
print(f"\n  Profundidad de penetración δ = 1/B2 = {delta:.4e} ± {ddelta:.4e} m")


# ─────────────────────────────────────────────────────────────────────────────
# AJUSTE DE LA FASE φ (Dataset 2)
# ─────────────────────────────────────────────────────────────────────────────
#
#   La fase relativa entre bobinas es φ = ω·Δt + π/2.
#   El modelo del efecto skin predice que φ crece linealmente con sqrt(omega):
#       φ = m * sqrt(omega) + b
#   donde la pendiente m = es * sqrt(mu*sigma/2) = B (igual coeficiente que
#   el módulo), lo que permite una verificación independiente de sigma.

y_phi  = 2 * np.pi * f * t + np.pi / 2
dy_phi = 2 * np.pi * np.sqrt((t * df)**2 + (f * dt)**2)

popt_phi, pcov_phi = curve_fit(
    lineal, x, y_phi, sigma=dy_phi, absolute_sigma=True
)
m_phi, b_phi   = popt_phi
dm_phi, db_phi = np.sqrt(np.diag(pcov_phi))

chi2r_phi = chi2_reducido(y_phi, lineal(x, m_phi, b_phi), dy_phi, n_params=2)
R2_phi    = R2(y_phi, lineal(x, m_phi, b_phi))

sigma_phi, ds_phi = calc_sigma(m_phi, dm_phi, es, des, mu)

print("\n─── AJUSTE LINEAL DE LA FASE (Dataset 2) ───")
print(f"  Pendiente m = {m_phi:.4f} ± {dm_phi:.4f}")
print(f"  Ordenada b  = {b_phi:.4f} ± {db_phi:.4f}")
print(f"  χ² reducido = {chi2r_phi:.4f}")
print(f"  R²          = {R2_phi:.4f}")
print(f"  sigma       = {sigma_phi:.4e} ± {ds_phi:.4e} S/m")

x_fit_phi = np.linspace(x.min() - 0.5, x.max() + 0.5, 100)

plt.figure(figsize=(8, 6))
plt.errorbar(x, y_phi, xerr=dx, yerr=dy_phi,
             fmt='o', color='blue', ecolor='gray', capsize=3,
             label='Datos experimentales')
plt.plot(x_fit_phi, lineal(x_fit_phi, m_phi, b_phi), 'r-',
         label=f'Ajuste: y = ({m_phi:.3f} ± {dm_phi:.3f}) x + ({b_phi:.3f} ± {db_phi:.3f})')
plt.xlabel(r'$\sqrt{\omega}$', fontsize=14)
plt.ylabel(r'$\varphi$ [rad]', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.title("Ajuste lineal de la fase")
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO COMBINADO: MÓDULO (dB) Y FASE EN EL MISMO PANEL
# ─────────────────────────────────────────────────────────────────────────────
#
#   Visualización final que superpone ambas magnitudes medidas con sus ajustes
#   usando dos ejes y (izquierdo para dB, derecho para radianes).
#   Permite comparar visualmente la consistencia entre los valores de sigma
#   obtenidos por las dos vías independientes (módulo y fase).

y_H      = v_s / (v_p * 2 * np.pi * f)
dy_H     = (1 / (2 * np.pi * v_p * f)) * np.sqrt(
    dv_s**2 + (v_s * dv_p / v_p)**2 + (df * v_s / f)**2
)
y_H_dB   = 20 * np.log10(y_H)
dy_H_dB  = 20 / (np.log(10) * y_H) * dy_H

R2_H   = R2(y_H, expo_reparametrizado(x, C2, D2))
R2_phi = R2(y_phi, lineal(x, m_phi, b_phi))

x_fit     = np.linspace(x.min(), x.max(), 500)
x_fit_phi = np.linspace(x.min() - 0.5, x.max() + 0.5, 500)

fig, ax1 = plt.subplots(figsize=(9, 6))

ax1.errorbar(x, y_H_dB, xerr=dx, yerr=dy_H_dB,
             fmt='o', color='maroon', capsize=3, markersize=7,
             label=r'Datos $(|H|/\omega)$ [dB]')
ax1.plot(x_fit, 20 * np.log10(expo_reparametrizado(x_fit, C2, D2)),
         '-', color='maroon', linewidth=2,
         label=fr'Ajuste $(|H|/\omega)$ [dB], $\sigma=8(1)\times10^8$ S/m ($R^2={R2_H:.4f}$)')
ax1.set_xlabel(r'$\sqrt{\omega}\;[\mathrm{Hz}^{1/2}]$', fontsize=14)
ax1.set_ylabel(r'$20\log_{10}(|H|/\omega)$ [dB]', color='maroon', fontsize=14)
ax1.tick_params(axis='x', labelsize=13)
ax1.tick_params(axis='y', labelsize=13, labelcolor='maroon')

ax2 = ax1.twinx()
ax2.errorbar(x, y_phi, xerr=dx, yerr=dy_phi,
             fmt='o', color='purple', alpha=0.6, capsize=1, markersize=6,
             label=r'Datos $\varphi$')
ax2.plot(x_fit_phi, lineal(x_fit_phi, m_phi, b_phi),
         '-', color='purple', linewidth=2,
         label=fr'Ajuste $\varphi$, $\sigma=4(2)\times10^7$ S/m ($R^2={R2_phi:.4f}$)')
ax2.set_ylabel(r'$\varphi$ [rad]', color='purple', fontsize=14)
ax2.tick_params(axis='y', labelsize=13, labelcolor='purple')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='upper center', bbox_to_anchor=(0.35, 0.23),
           ncol=1, fontsize=10, handlelength=3)

ax1.grid(True, linestyle='--', alpha=0.6)
plt.title("Módulo y fase – comparación de ajustes")
plt.tight_layout()
plt.show()
print(B2)
print(m_phi)



# ============================================================================
# TEST: ¿Existe una dependencia extra en frecuencia?
# Usamos B obtenido de la fase
# ============================================================================

B_fase = m_phi

def modelo_fase(x, A):
    return A * np.exp(-B_fase * x)

# Ajusto solamente A
popt_A, pcov_A = curve_fit(
    modelo_fase,
    x,
    y,
    sigma=dy,
    absolute_sigma=True,
    bounds=([0], [np.inf])
)

A_fase = popt_A[0]
dA_fase = np.sqrt(np.diag(pcov_A))[0]

print("\n--- Ajuste usando B obtenido de la fase ---")
print(f"A = {A_fase:.6e} ± {dA_fase:.6e}")
print(f"B (fijo) = {B_fase:.6e}")

# modelo esperado si el módulo obedeciera exactamente la teoría
y_modelo = modelo_fase(x, A_fase)

# cociente dato/modelo
R = y / y_modelo

# incertidumbre del cociente
dR = dy / y_modelo

# ============================================================================
# Ajuste potencia: R = C * omega^(-n)
# Tomamos logaritmos:
#
# ln(R) = ln(C) - n ln(omega)
# ============================================================================

omega = 2*np.pi*f

lnR = np.log(R)
dlnR = dR / R

lnw = np.log(omega)

popt_res, pcov_res = curve_fit(
    lineal,
    lnw,
    lnR,
    sigma=dlnR,
    absolute_sigma=True
)

m_res, b_res = popt_res
dm_res, db_res = np.sqrt(np.diag(pcov_res))

n_extra = -m_res
dn_extra = dm_res

print("\n--- Dependencia residual ---")
print(f"n = {n_extra:.4f} ± {dn_extra:.4f}")

# ============================================================================
# gráfico log-log de los residuos
# ============================================================================

xfit = np.linspace(lnw.min(), lnw.max(), 300)

plt.figure(figsize=(8,6))

plt.errorbar(
    lnw,
    lnR,
    yerr=dlnR,
    fmt='o',
    capsize=3,
    label='Datos'
)

plt.plot(
    xfit,
    lineal(xfit, m_res, b_res),
    '-',
    linewidth=2,
    label=fr'$\ln R = ({m_res:.3f})\ln\omega + ({b_res:.3f})$'
)

plt.xlabel(r'$\ln(\omega)$')
plt.ylabel(r'$\ln\left[\dfrac{\mathrm{dato}}{\mathrm{modelo}}\right]$')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================================
# gráfico directo de residuos
# ============================================================================

plt.figure(figsize=(8,6))

plt.errorbar(
    np.sqrt(omega),
    R,
    yerr=dR,
    fmt='o',
    capsize=3
)

plt.axhline(1, color='red', linestyle='--')

plt.xlabel(r'$\sqrt{\omega}$')
plt.ylabel(r'$\mathrm{dato}/\mathrm{modelo}$')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



# ============================================================================
# MODELO CORREGIDO:
#
# y = A * omega^{-n} * exp(-B*sqrt(omega))
#
# con B fijado al obtenido por la fase
# ============================================================================

B_fase = m_phi
n_fase = 0.6899

def modelo_corregido(f, A):
    omega = 2*np.pi*f
    return A * omega**(-n_fase) * np.exp(-B_fase*np.sqrt(omega))

popt_corr, pcov_corr = curve_fit(
    modelo_corregido,
    f,
    y,
    sigma=dy,
    absolute_sigma=True,
    bounds=([0],[np.inf])
)

A_corr = popt_corr[0]
dA_corr = np.sqrt(np.diag(pcov_corr))[0]

y_corr = modelo_corregido(f, A_corr)

chi2_corr = chi2_reducido(
    y,
    y_corr,
    dy,
    n_params=1
)

R2_corr = R2(y, y_corr)

print("\n--- MODELO CORREGIDO ---")
print(f"A = {A_corr:.6e} ± {dA_corr:.6e}")
print(f"B fijo = {B_fase:.6e}")
print(f"n fijo = {n_fase:.4f}")
print(f"χ² reducido = {chi2_corr:.4f}")
print(f"R² = {R2_corr:.6f}")

# gráfico

omega_fit = np.linspace(f.min(), f.max(), 500)

plt.figure(figsize=(8,6))

plt.errorbar(
    f,
    y,
    yerr=dy,
    fmt='o',
    capsize=3,
    label='Datos'
)

plt.plot(
    omega_fit,
    modelo_corregido(omega_fit, A_corr),
    '-',
    linewidth=2,
    label='Modelo corregido'
)

plt.xlabel('f [Hz]')
plt.ylabel(r'$V_s/(V_p\omega)$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

def modelo_general(f, A, n, B):
    omega = 2*np.pi*f
    return A * omega**(-n) * np.exp(-B*np.sqrt(omega))

popt, pcov = curve_fit(
    modelo_general,
    f,
    y,
    sigma=dy,
    absolute_sigma=True,
    bounds=([0,0,0],[np.inf,np.inf,np.inf])
)

A_fit, n_fit, B_fit = popt
dA_fit, dn_fit, dB_fit = np.sqrt(np.diag(pcov))

sigma_fit, dsigma_fit = calc_sigma(
    B_fit,
    dB_fit,
    es,
    des,
    mu
)

print("\n--- MODELO GENERAL ---")
print(f"A = {A_fit:.6e} ± {dA_fit:.6e}")
print(f"n = {n_fit:.4f} ± {dn_fit:.4f}")
print(f"B = {B_fit:.6e} ± {dB_fit:.6e}")
print(f"sigma = {sigma_fit:.4e} ± {dsigma_fit:.4e}")