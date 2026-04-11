import numpy as np

C = 100e-9 #F
R = 1000 #ohm
tau = R*C

fr = np.array([
    105, 499, 1002, 1500, 2003, 2488, 3275, 3996, 5000,
    5995, 6990, 9009, 11010, 15000, 20000, 30100, 40000, 50000
]) #Hz

Ve = np.array([
    6.45, 6.25, 6.20, 6.06, 5.99, 6.10, 5.99, 6.05, 6.07,
    5.88, 6.05, 6.18, 6.20, 6.50, 6.38, 6.40, 6.50, 6.60
]) #V

Vr = np.array([
    0.409, 1.91, 3.43, 4.51, 5.14, 5.66, 5.75, 5.71, 5.45,
    5.01, 4.59, 3.99, 3.39, 2.60, 2.11, 1.45, 1.10, 0.88
]) #V

dt = np.array([
    -2.39, -0.416, -0.156, -0.081, -0.0404, -0.0216, 0, 0.0076, 0.0144,
    0.015, 0.016, 0.015, 0.0139, 0.0118, 0.0097, 0.007, 0.0056, 0.0045
]) * 1e-3 #s

def error_ultima_cifra(arr):
    """Genera un error de 1 en la última cifra decimal para cada elemento de un array."""
    errores = []
    for x in arr:
        s = str(abs(x))
        if '.' in s:
            decimales = len(s.split('.')[1])
            errores.append(10**(-decimales))
        else:
            errores.append(1.0) # Si es entero, el error es 1
    return np.array(errores)

errC = 10e-9
errR = 0.5
errtau = np.sqrt((R*errC)**2 + (C*errR)**2)
errfr = fr*0.03
errVe = Ve*0.03
errVr = Vr*0.03
errdt= dt*0.03

w = 2 * np.pi * fr
errw = 2 * np.pi * errfr

H = Vr / Ve
errH = H * np.sqrt((errVr/Vr)**2 + (errVe/Ve)**2)

tau1 = 87.73878e-6
errtau1= 2.18809e-6

import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

# ======================
# MODELO RLC (tau libre)
# ======================
def modelo_rlc(p, x):
    w0, A, tau = p
    denominador = np.sqrt((x * tau)**2 + (1 - (x / w0)**2)**2)
    return A * (x * tau) / denominador

# ======================
# DATOS (asegurate de tener definidos w, H, errw, errH)
# ======================

# ======================
# ODR
# ======================
model_rlc = Model(modelo_rlc)
data_rlc = RealData(w, H, sx=errw, sy=errH)

# semillas iniciales
w0_guess = w[np.argmax(H)]
A_guess = np.max(H)
tau_guess = 1e-4  # semilla razonable (RC)

my_odr = ODR(data_rlc, model_rlc, beta0=[w0_guess, A_guess, tau_guess])
output = my_odr.run()

# ======================
# RESULTADOS
# ======================
w0_fit, A_fit, tau_fit = output.beta
err_w0, err_A, err_tau = output.sd_beta

# ======================
# MÉTRICAS
# ======================
y_fit = modelo_rlc(output.beta, w)

chi2_obs = np.sum(((H - y_fit) / errH)**2)
dof = len(w) - 3
chi2_red = chi2_obs / dof

residuos = H - y_fit
ss_res = np.sum(residuos**2)
ss_tot = np.sum((H - np.mean(H))**2)
r2_rlc = 1 - (ss_res / ss_tot)

# ======================
# (OPCIONAL) CÁLCULO DE L
# ======================
# Si tenés C conocido:
# L = 1 / (C * w0^2)
# errL por propagación

C = 100e-9  # ajustá si corresponde
L_fit = 1 / (C * w0_fit**2)
errL = L_fit * np.sqrt((err_w0 / w0_fit)**2 * 4)

# ======================
# GRÁFICO
# ======================
plt.figure(figsize=(9, 6.5))
plt.errorbar(w, H, xerr=errw, yerr=errH, fmt='ko', alpha=0.3, label='Datos')

w_curva = np.linspace(min(w), max(w), 1000)
plt.plot(w_curva, modelo_rlc([w0_fit, A_fit, tau_fit], w_curva),
         color='darkslateblue',
         label=f'Ajuste $R^2$={r2_rlc:.2f}')

plt.axvline(w0_fit, color='crimson', ls='--',
            label=f'$\omega_0$ = {w0_fit:.0f}({err_w0:.0f}) rad/s')

plt.xlabel("$\omega$ [rad/s]", fontsize=14)
plt.ylabel("H", fontsize=14)
plt.legend(fontsize=13)
plt.grid(True, alpha=0.3)
plt.show()

# ======================
# REPORTE
# ======================
print(f"--- RESULTADOS (TAU LIBRE) ---")
print(f"w0 ajustado: {w0_fit:.2f} ± {err_w0:.2f} rad/s")
print(f"Amplitud A:  {A_fit:.3f} ± {err_A:.3f}")
print(f"tau ajustado: {tau_fit:.6e} ± {err_tau:.6e} s")
print(f"Inductancia L: {L_fit:.6f} ± {errL:.6f} H")
print(f"R² del ajuste: {r2_rlc:.5f}")
print(f"Chi-cuadrado reducido: {chi2_red:.4f}")

err_cap = 4.90234188189095e-09
cap = 8.773877927430522e-08

rl = (tau_ef_fit - tau1)/cap
err_rl = np.sqrt( ((tau1*err_tau_ef_fit)/cap)**2 + ((tau_ef_fit*errtau1) / cap )**2 + (((tau_ef_fit - tau1)*err_cap)/cap**2)**2 )
print(f"Resistencia agregada por el inductor: {rl:.2f} ± {err_rl:.3f}")

# --- CONFIGURACIÓN DEL SEGUNDO AJUSTE (TAU ESTRICTAMENTE FIJO) ---
tau_fijo_val = 87.73878e-6  # El valor experimental (tau1) que mencionaste

def modelo_rlc_solo_w0_A(p, x):
    w0, A = p[0], p[1]
    # Usamos la constante global tau_fijo_val
    denominador = np.sqrt((x * tau_fijo_val)**2 + (1 - (x / w0)**2)**2)
    return A * (x * tau_fijo_val) / denominador

# 1. Ejecución del ODR para el nuevo modelo
model_fijo = Model(modelo_rlc_solo_w0_A)
data_fijo = RealData(w, H, sx=errw, sy=errH)

# Semillas iniciales
beta0_fijo = [w0_guess, A_guess]

odr_fijo = ODR(data_fijo, model_fijo, beta0=beta0_fijo)
output_fijo = odr_fijo.run()

# 2. Resultados del segundo ajuste
w0_f_fit, A_f_fit = output_fijo.beta
err_w0_f, err_A_f = output_fijo.sd_beta

# 3. Cálculo de R2 para el nuevo ajuste
y_fit_fijo = modelo_rlc_solo_w0_A(output_fijo.beta, w)
res_f = H - y_fit_fijo
r2_fijo = 1 - (np.sum(res_f**2) / np.sum((H - np.mean(H))**2))

# 4. Cálculo de Inductancia L (L = 1 / (C * w0^2))
# Asumiendo que tenés definidos 'cap' y 'err_cap' de antes
L_fijo = 1 / (cap * w0_f_fit**2)
errL_fijo = L_fijo * np.sqrt((err_cap/cap)**2 + (2 * err_w0_f / w0_f_fit)**2)

# --- GRÁFICO COMPARATIVO ---
plt.figure(figsize=(10, 7))
plt.errorbar(w, H, xerr=errw, yerr=errH, fmt='ko', alpha=0.2, label='Datos')

w_linea = np.linspace(min(w), max(w), 1000)

# Curva 1: Ajuste anterior (Tau libre/efectivo)
plt.plot(w_linea, modelo_rlc_tau_fijo([w0_fit, A_fit, tau_ef_fit], w_linea),
         color='darkslateblue', alpha=0.6, label=f'Ajuste $\\tau_{{ef}}$ ($R^2$={r2_rlc:.4f})')

# Curva 2: Ajuste nuevo (Tau fijo experimental)
plt.plot(w_linea, modelo_rlc_solo_w0_A([w0_f_fit, A_f_fit], w_linea),
         color='forestgreen', ls='--', lw=2, label=f'Ajuste $\\tau_{{fijo}}$ ($R^2$={r2_fijo:.4f})')

plt.axvline(w0_f_fit, color='forestgreen', alpha=0.5, ls=':', label=f'$\omega_0$ fijo')

plt.xlabel("$\omega$ [rad/s]", fontsize=14)
plt.ylabel("Transferencia H", fontsize=14)
plt.title("Comparación de modelos RLC (Frecuencia de Resonancia)", fontsize=15)
plt.legend(fontsize=11, frameon=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.show()

# --- REPORTES ---
print(f"{'='*50}")
print(f"   RESULTADOS AJUSTE TAU FIJO ({tau_fijo_val*1e6:.2f} µs)")
print(f"{'='*50}")
print(f"w0 ajustado:  {w0_f_fit:.2f} ± {err_w0_f:.2f} rad/s")
print(f"Amplitud A:   {A_f_fit:.3f} ± {err_A_f:.3f}")
print(f"Inductancia L: {L_fijo:.6f} ± {errL_fijo:.6f} H")
print(f"R² del ajuste: {r2_fijo:.6f}")
print(f"{'='*50}")