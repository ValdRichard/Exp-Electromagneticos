import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
from scipy.optimize import curve_fit

# ─── Constantes y datos (Dataset 2) ──────────────────────────────────────────

mu  = 4 * np.pi * 1e-7
es  = 0.00156
des = 0.00001

R_ohm = 40.1
N     = 3200
L     = 0.207

v_p = np.array([0.0914, 0.0645, 0.0492, 0.0394, 0.0319, 0.0271,
                0.0174, 0.0147, 0.012, 0.00931, 0.108, 0.0211, 0.129])
v_s = np.array([5.12, 3.89, 3, 2.45, 2.05, 1.76,
                1.13, 0.987, 0.798, 0.639, 5.65, 1.98, 5.90])
f   = np.array([121.4, 218.8, 319.8, 420.2, 521.3, 628.3,
                1019, 1224, 1506, 1913, 80.65, 819.7, 49.95])
#dv_p = np.array([0.0001, 0.0001, 0.00001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.0000001, 0.000001, 0.00001, 0.000001, 0.0001])
#dv_s = np.array([0.0001, 0.0001, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001, 0.00001, 0.0001, 0.0001, 0.001])
#df   = np.array([0.1, 0.1, 0.1, 0.01, 0.001, 0.001, 0.01, 0.1, 0.1, 0.01, 0.0001, 0.001, 0.01])

dv_p = np.array([0.013742, 0.007935, 0.02076, 0.007182, 0.003957, 0.003813, 0.002522, 0.002441, 0.002236, 0.0022793, 0.01424, 0.003633, 0.01487])
dv_s = np.array([0.6546, 0.6177, 0.291, 0.2745, 0.2625, 0.2538, 0.0849, 0.08061, 0.07494, 0.07017, 0.6705, 0.1604, 0.6786])
df = np.array([6.07, 10.94, 15.99, 21.01, 26.065, 31.415, 50.95, 61.2, 75.3, 95.65, 4.0325, 40.985, 2.4975])


mask = f != 819.7

v_p  = v_p[mask]
v_s  = v_s[mask]
f    = f[mask]
dv_p = dv_p[mask]
dv_s = dv_s[mask]
df   = df[mask]

x   = np.sqrt(f * 2 * np.pi)
y   = v_s / (v_p * 2 * np.pi * f)
dx  = np.sqrt((np.pi / 2) * (df**2 / f))
dy  = (1 / (2 * np.pi * v_p * f)) * np.sqrt(
    dv_s**2 + (v_s * dv_p / v_p)**2 + (df * v_s / f)**2
)


# ─── Auxiliares ──────────────────────────────────────────────────────────────

def R2(y_obs, y_mod):
    ss_res = np.sum((y_obs - y_mod)**2)
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    return 1 - ss_res / ss_tot

def chi2_reducido(y_obs, y_mod, dy, n_params):
    chi2 = np.sum(((y_obs - y_mod) / dy)**2)
    return chi2 / (len(y_obs) - n_params)

def calc_sigma(B, dB, es, des, mu):
    sigma = ((B / es)**2) * (2 / mu)
    ds = 4 * np.sqrt(
        (((B / es**2) * dB)**2) / mu**2 +
        ((((B**2) / es**3) * des)**2) / mu**2
    )
    return sigma, ds

# ─── Modelo general en función de sqrt(omega) ────────────────────────────────
#
#   x = sqrt(omega),  y = v_s / (v_p * omega)
#
#   y(x) = A * (mu*N / R*L) * x^(-2n) * exp(-sqrt(B*mu/2) * es * x)
#
#   porque omega = x^2  =>  omega^(-n) = x^(-2n)

def modelo_general_odr(beta, x):
    A, n, B = beta
    return A * ((mu * N) / R_ohm * L) * x**(-2*n) * np.exp(-np.sqrt((B * mu) / 2) * es * x)

def modelo_general_cf(x, A, n, B):
    return A * ((mu * N) / R_ohm * L) * x**(-2*n) * np.exp(-np.sqrt((B * mu) / 2) * es * x)

popt0, _ = curve_fit(
    modelo_general_cf, x, y, sigma=dy, absolute_sigma=True,
    bounds=([0, 0, 0], [np.inf, np.inf, np.inf])
)

# ─── ODR ─────────────────────────────────────────────────────────────────────

data  = RealData(x, y, sx=dx, sy=dy)
model = Model(modelo_general_odr)
odr   = ODR(data, model, beta0=popt0)
odr.set_job(fit_type=0)
out   = odr.run()

A_fit, n_fit, B_fit    = out.beta
dA_fit, dn_fit, dB_fit = out.sd_beta

y_fit    = modelo_general_odr(out.beta, x)
chi2_fit = chi2_reducido(y, y_fit, dy, n_params=3)
R2_fit   = R2(y, y_fit)
sigma_fit, dsigma_fit = calc_sigma(B_fit, dB_fit, es, des, mu)

print("─── MODELO GENERAL (ODR) ───")
print(f"A     = {A_fit:.6e} ± {dA_fit:.6e}")
print(f"n     = {n_fit:.4f} ± {dn_fit:.4f}")
print(f"B     = {B_fit:.6e} ± {dB_fit:.6e}")
print(f"sigma = {sigma_fit:.4e} ± {dsigma_fit:.4e} S/m")
print(f"χ² reducido = {chi2_fit:.4f}")
print(f"R²          = {R2_fit:.6f}")

# ─── Gráfico ─────────────────────────────────────────────────────────────────

x_fit   = np.linspace(x.min(), x.max(), 1000)
y_dB    = 20 * np.log10(y)
dy_dB   = 20 / (np.log(10) * y) * dy
yfit_dB = 20 * np.log10(modelo_general_odr(out.beta, x_fit))

plt.figure(figsize=(8, 6))

plt.errorbar(x, y_dB, xerr=dx, yerr=dy_dB,
             fmt='o', color='maroon', ecolor='maroon', capsize=3,
             label=r'Datos $|H|/\omega$ [dB]')
plt.plot(x_fit, yfit_dB,
         '-', color='maroon', linewidth=2,
         label=(fr'Ajuste general'
                '\n'
                fr'$n={n_fit:.3f}\pm{dn_fit:.3f}$'
                '\n'
                fr'$\sigma=({B_fit:.3e}\pm{dB_fit:.3e})$'
                '\n'
                fr'$R^2={R2_fit:.4f}$'))

plt.xlabel(r'$\sqrt{\omega}\;[\mathrm{Hz}^{1/2}]$', fontsize=13)
plt.ylabel(r'$20\log_{10}(|H|/\omega)$ [dB]', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', fontsize=10)
#plt.title("Modelo general – ajuste (ODR)")

plt.tight_layout()
plt.show()