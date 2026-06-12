import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ─── Parámetros y datos (Dataset 2) ──────────────────────────────────────────

mu  = 4 * np.pi * 1e-7
es  = 0.00156
des = 0.00001

v_p = np.array([0.0914, 0.0645, 0.0492, 0.0394, 0.0319, 0.0271,
                0.0174, 0.0147, 0.012, 0.00931, 0.108, 0.0211, 0.129])
v_s = np.array([5.12, 3.89, 3, 2.45, 2.05, 1.76,
                1.13, 0.987, 0.798, 0.639, 5.65, 1.98, 5.90])
f   = np.array([121.4, 218.8, 319.8, 420.2, 521.3, 628.3,
                1019, 1224, 1506, 1913, 80.65, 819.7, 49.95])
#dv_p = np.array([0.0001, 0.0001, 0.00001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.0000001, 0.000001, 0.00001, 0.000001, 0.0001])
#dv_s = np.array([0.0001, 0.0001, 0.001, 0.0001, 0.0001, 0.0001,  0.0001, 0.00001, 0.00001, 0.00001, 0.0001, 0.0001, 0.001])
#df   = np.array([0.1, 0.1, 0.1, 0.01, 0.001, 0.001, 0.01, 0.1, 0.1, 0.01, 0.0001, 0.001, 0.01])
#dt   = np.array([0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.0001, 0.000001, 0.00001])
t    = np.array([0.00364, 0.0021, 0.0015, 0.00116, 0.00094, 0.000788, 0.000492, 0.000406, 0.000346, 0.000276, 0.00536, 0.000604, 0.00784])
dv_p = np.array([0.013742, 0.007935, 0.02076, 0.007182, 0.003957, 0.003813, 0.002522, 0.002441, 0.002236, 0.0022793, 0.01424, 0.003633, 0.01487])
dv_s = np.array([0.6546, 0.6177, 0.291, 0.2745, 0.2625, 0.2538, 0.0849, 0.08061, 0.07494, 0.07017, 0.6705, 0.1604, 0.6786])
df = np.array([6.07, 10.94, 15.99, 21.01, 26.065, 31.415, 50.95, 61.2, 75.3, 95.65, 4.0325, 40.985, 2.4975])
dt= np.array([0.001, 0.0005, 0.0005, 0.00025, 0.00025, 0.0001, 0.0001, 0.00005, 0.00005, 0.00005, 0.001, 0.0001, 0.001])
dt/=5


n_geom = (64 * np.pi) / 415035
u      = 0.00000123656
A_fijo = n_geom
B_fijo = 0.00759251

mask = f != 819.7

v_p  = v_p[mask]
v_s  = v_s[mask]
f    = f[mask]
dv_p = dv_p[mask]
dv_s = dv_s[mask]
df   = df[mask]
dt = dt[mask]
t= t[mask]
x   = np.sqrt(f * 2 * np.pi)
y   = v_s / (v_p * 2 * np.pi * f)
dx  = np.sqrt((np.pi / 2) * (df**2 / f))
dy  = (1 / (2 * np.pi * v_p * f)) * np.sqrt(
    dv_s**2 + (v_s * dv_p / v_p)**2 + (df * v_s / f)**2
)

# ─── Ajuste 2 (reparametrizado) ───────────────────────────────────────────────

def expo_reparametrizado(x, C, D):
    return (C * n_geom) * np.exp(-(D * u) * x)

popt_aj2, _ = curve_fit(
    expo_reparametrizado, x, y, sigma=dy, absolute_sigma=True,
    bounds=([0, 0], [np.inf, np.inf])
)
C2, D2 = popt_aj2

# ─── Ajuste de fase para obtener B_fase ───────────────────────────────────────

def lineal(x, m, b):
    return m * x + b

y_phi  = 2 * np.pi * f * t + np.pi / 2
dy_phi = 2 * np.pi * np.sqrt((t * df)**2 + (f * dt)**2)

popt_phi, _ = curve_fit(lineal, x, y_phi, sigma=dy_phi, absolute_sigma=True)
m_phi, _ = popt_phi
B_fase = m_phi

# ─── Residuos ln(dato/modelo) vs ln(omega) ────────────────────────────────────

def modelo_fase_amp(x, A):
    return A * np.exp(-B_fase * x)

popt_A, _ = curve_fit(
    modelo_fase_amp, x, y, sigma=dy, absolute_sigma=True, bounds=([0], [np.inf])
)
A_fase = popt_A[0]

modelo_vals = modelo_fase_amp(x, A_fase)
R_res  = y / modelo_vals
dR_res = dy / modelo_vals
lnR    = np.log(R_res)
dlnR   = dR_res / R_res
omega  = 2 * np.pi * f
lnw    = np.log(omega)

popt_res, _ = curve_fit(lineal, lnw, lnR, sigma=dlnR, absolute_sigma=True)
m_res, b_res = popt_res
dm_res, db_res = np.sqrt(np.diag(
    curve_fit(lineal, lnw, lnR, sigma=dlnR, absolute_sigma=True)[1]
))

# ─── Figura con dos paneles ────────────────────────────────────────────────────

x_fit    = np.linspace(x.min(), x.max(), 500)
lnw_fit  = np.linspace(lnw.min(), lnw.max(), 300)

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

# ── Panel izquierdo: ajuste vs teórico en dB ──────────────────────────────────

#y_dB   = 20 * np.log10(y)
#dy_dB  = 20 / (np.log(10) * y) * dy

# Normalizar según el punto de menor frecuencia
idx0 = np.argmin(f)
x0 = x[idx0]

# Valores de referencia de cada curva
y0_exp = y[idx0]
y0_fit = expo_reparametrizado(x0, C2, D2)
y0_teo = A_fijo * np.exp(-B_fijo * x0)

# Datos experimentales normalizados
y_norm = y / y0_exp
dy_norm = dy / y0_exp

y_dB  = 20*np.log10(y_norm)
dy_dB = 20/(np.log(10)*y_norm) * dy_norm

# Curvas normalizadas
y_fit_norm = expo_reparametrizado(x_fit, C2, D2) / y0_fit
y_teo_norm = (A_fijo*np.exp(-B_fijo*x_fit)) / y0_teo

def R2(y_obs, y_mod):
    ss_res = np.sum((y_obs - y_mod)**2)
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    return 1 - ss_res / ss_tot

R2_aj2 = R2(y, expo_reparametrizado(x, C2, D2))

ax_left.errorbar(x, y_dB, xerr=dx, yerr=dy_dB,
                 fmt='o', color='maroon', ecolor='maroon', capsize=3,
                 label=r'$\left[\frac{|H|/\omega}{(|H|/\omega)_0}\right]$ [dB]')

ax_left.plot(
    x_fit,
    20*np.log10(y_fit_norm),
    '-',
    color='maroon',
    linewidth=2,
    label=fr'Ajuste ($R^2={R2_aj2:.4f}$)'
)

ax_left.plot(
    x_fit,
    20*np.log10(y_teo_norm),
    '--',
    color='purple',
    linewidth=2,
    label='Curva teórica'
)
ax_left.set_xlabel(r'$\sqrt{\omega}\;[\mathrm{Hz}^{1/2}]$', fontsize=13)
ax_left.set_ylabel(r'$20\log_{10}\!\left[\frac{|H|/\omega}{(|H|/\omega)_0}\right]$ [dB]', fontsize=13)
ax_left.grid(True, linestyle='--', alpha=0.6)
ax_left.legend(loc='upper right', fontsize=10)
#ax_left.set_title("Ajuste vs curva teórica (escala logarítmica)")

# ── Panel derecho: ln(dato/modelo) vs ln(omega) ───────────────────────────────

ax_right.errorbar(lnw, lnR, yerr=dlnR,
                  fmt='o', color='maroon', ecolor='maroon', capsize=3,
                  label=r'$\ln(\mathrm{Dato}/\mathrm{Modelo})$')
ax_right.plot(lnw_fit, lineal(lnw_fit, m_res, b_res),
              '--', color='purple', linewidth=2,
              label=fr'Ajuste ($n={-m_res:.3f}\pm{dm_res:.3f}$, $R^2={R2(lnR, lineal(lnw, m_res, b_res)):.4f}$)')
ax_right.axhline(0, color='gray', linestyle=':', linewidth=1)
ax_right.set_xlabel(r'$\ln(\omega)$', fontsize=13)
ax_right.set_ylabel(r'$\ln\left[\mathrm{dato}/\mathrm{modelo}\right]$', fontsize=13)
ax_right.grid(True, linestyle='--', alpha=0.6)
ax_right.legend(fontsize=10)
#ax_right.set_title(r"Dependencia residual en $\omega$")

plt.tight_layout()
plt.show()