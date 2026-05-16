
import numpy as np
import matplotlib.pyplot as plt

from scipy.odr import ODR, Model, RealData

from funciones import (
    modelo_rlc_tau_libre,
    modelo_rlc_tau_fijo,
    modelo_fase,
    calcular_H,
    calcular_phi,
    calcular_r2
)

# =========================================================
# DATOS
# =========================================================

C = 100e-9 # F
R = 1000   # ohm

tau = R*C

fr = np.array([
    105, 499, 1002, 1500, 2003, 2488, 3275, 3996, 5000,
    5995, 6990, 9009, 11010, 15000, 20000, 30100, 40000, 50000
]) # Hz

Ve = np.array([
    6.45, 6.25, 6.20, 6.06, 5.99, 6.10, 5.99, 6.05, 6.07,
    5.88, 6.05, 6.18, 6.20, 6.50, 6.38, 6.40, 6.50, 6.60
]) # V

Vr = np.array([
    0.409, 1.91, 3.43, 4.51, 5.14, 5.66, 5.75, 5.71, 5.45,
    5.01, 4.59, 3.99, 3.39, 2.60, 2.11, 1.45, 1.10, 0.88
]) # V

dt = np.array([
    -2.39, -0.416, -0.156, -0.081, -0.0404, -0.0216, 0,
    0.0076, 0.0144, 0.015, 0.016, 0.015, 0.0139,
    0.0118, 0.0097, 0.007, 0.0056, 0.0045
]) * 1e-3 # s

# =========================================================
# ERRORES
# =========================================================

errC = 10e-9
errR = 0.5

errfr = fr * 0.05
errVe = Ve * 0.05
errVr = Vr * 0.05
errdt = np.abs(dt) * 0.05

# =========================================================
# MAGNITUDES
# =========================================================

w = 2*np.pi*fr
errw = 2*np.pi*errfr

# ---- H y error ----
H, errH = calcular_H(
    Vr,
    Ve,
    errVr,
    errVe
)

# ---- phi y error ----
phi, errphi = calcular_phi(
    w,
    dt,
    errw,
    errdt
)

# =========================================================
# TAU EXPERIMENTAL RC
# =========================================================

tau_exp = 87.73878e-6
errtau_exp = 2.18809e-6

# usamos el tau obtenido del ajuste del módulo
tau_fase = tau_ef_fit

# modelo de fase con tau fijo
model_phi = Model(
    lambda p, x: modelo_fase(
        [p[0], tau_fase],
        x
    )
)

data_phi = RealData(
    w,
    phi,
    sx=errw,
    sy=errphi
)

# SOLO ajustamos w0
odr_phi = ODR(
    data_phi,
    model_phi,
    beta0=[w0_guess]
)

output_phi = odr_phi.run()

w0_phi = output_phi.beta[0]
err_w0_phi = output_phi.sd_beta[0]

# =========================================================
# R2
# =========================================================

# ---- tau libre ----
H_fit_libre = modelo_rlc_tau_libre(
    output_libre.beta,
    w
)

r2_libre = calcular_r2(
    H,
    H_fit_libre
)

# ---- tau fijo ----
H_fit_fijo = modelo_rlc_tau_fijo(
    output_fijo.beta,
    w,
    tau_exp
)

r2_fijo = calcular_r2(
    H,
    H_fit_fijo
)

# ---- fase ----
phi_fit = modelo_fase(
    [w0_phi, tau_fase],
    w
)

r2_phi = calcular_r2(
    phi,
    phi_fit
)

# =========================================================
# GRAFICO BODE COMPLETO
# =========================================================

fig, ax1 = plt.subplots(figsize=(10,7))

# =========================================================
# EJE H
# =========================================================

ax1.errorbar(
    w,
    H,
    xerr=errw,
    yerr=errH,
    fmt='ko',
    alpha=0.35,
    markersize=7,
    label='Datos módulo'
)

w_plot = np.linspace(
    min(w),
    max(w),
    2000
)

# =========================================================
# TAU FIJO
# =========================================================

ax1.plot(
    w_plot,
    modelo_rlc_tau_fijo(
        [w0_fijo, A_fijo],
        w_plot,
        tau_exp
    ),
    color='crimson',
    lw=2.5,
    label=(
        r'$H(\omega)$ $\tau_{exp}$'
        '\n'
        + rf'$\tau$ = ({tau_exp*1e4:.2f}$\pm${errtau_exp*1e4:.2f})$\times10^{{-4}}$ s'
        '\n'
        + rf'$\omega_0$ = ({w0_fijo:.0f}$\pm${err_w0_fijo:.0f}) rad/s'
        '\n'
        + rf'$R^2$ = {r2_fijo:.3f}'
    )
)

# =========================================================
# TAU LIBRE
# =========================================================

ax1.plot(
    w_plot,
    modelo_rlc_tau_libre(
        [w0_fit, A_fit, tau_ef_fit],
        w_plot
    ),
    color='darkslateblue',
    lw=2.5,
    label=(
        r'$H(\omega)$ $\tau_{ef}$'
        '\n'
        + rf'$\tau_{{ef}}$ = ({tau_ef_fit*1e4:.2f}$\pm${err_tau_ef*1e4:.2f})$\times10^{{-4}}$ s'
        '\n'
        + rf'$\omega_0$ = ({w0_fit:.0f}$\pm${err_w0:.0f}) rad/s'
        '\n'
        + rf'$R^2$ = {r2_libre:.3f}'
    )
)

ax1.set_xlabel(
    r'$\omega$ [rad/s]',
    fontsize=16
)

ax1.set_ylabel(
    r'$H(\omega)$',
    fontsize=16
)

ax1.tick_params(
    axis='both',
    labelsize=13
)

# =========================================================
# EJE FASE
# =========================================================

ax2 = ax1.twinx()

ax2.errorbar(
    w,
    phi,
    xerr=errw,
    yerr=errphi,
    fmt='o',
    color='forestgreen',
    alpha=0.45,
    markersize=6,
    label='Datos fase'
)

ax2.plot(
    w_plot,
    modelo_fase(
        [w0_phi, tau_fase],
        w_plot
    ),
    color='forestgreen',
    lw=2.5,
    ls='--',
    label=(
        r'$\phi(\omega)$'
        '\n'
        + rf'$\tau$ = ({tau_fase*1e4:.2f}$\pm${err_tau_ef*1e4:.2f})$\times10^{{-4}}$ s'
        '\n'
        + rf'$\omega_0$ = ({w0_phi:.0f}$\pm${err_w0_phi:.0f}) rad/s'
        '\n'
        + rf'$R^2$ = {r2_phi:.3f}'
    )
)

ax2.set_ylabel(
    r'$\phi$ [rad]',
    fontsize=16
)

ax2.tick_params(
    axis='y',
    labelsize=13
)

# =========================================================
# LEYENDA
# =========================================================

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    fontsize=11,
    loc='best',
    frameon=True,
    shadow=True
)

# =========================================================
# ESTETICA
# =========================================================

ax1.grid(True, which='both', alpha=0.3)

ax1.set_xscale('log')

plt.tight_layout()

plt.savefig(
    "bode_completo_rlc.png",
    dpi=500
)

plt.show()

# =========================================================
# REPORTES
# =========================================================

print("="*60)
print("AJUSTE H - TAU LIBRE")
print("="*60)

print(f"w0 = {w0_fit:.2f} ± {err_w0:.2f} rad/s")
print(f"tau_ef = {tau_ef_fit:.6e} ± {err_tau_ef:.6e} s")
print(f"A = {A_fit:.3f} ± {err_A:.3f}")
print(f"R² = {r2_libre:.5f}")

print()

print("="*60)
print("AJUSTE H - TAU FIJO")
print("="*60)

print(f"w0 = {w0_fijo:.2f} ± {err_w0_fijo:.2f} rad/s")
print(f"tau_exp = {tau_exp:.6e} ± {errtau_exp:.6e} s")
print(f"A = {A_fijo:.3f} ± {err_A_fijo:.3f}")
print(f"R² = {r2_fijo:.5f}")

print()

print("="*60)
print("AJUSTE FASE")
print("="*60)

print(f"w0 = {w0_phi:.2f} ± {err_w0_phi:.2f} rad/s")
print(f"tau fijo = {tau_fase:.6e} ± {err_tau_ef:.6e} s")
print(f"R² = {r2_phi:.5f}")