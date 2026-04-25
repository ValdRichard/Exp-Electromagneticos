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

errC = 10e-9
errR = 0.5
errtau = np.sqrt((R*errC)**2 + (C*errR)**2)
errfr = fr*0.05
errVe = Ve*0.05
errVr = Vr*0.05
errdt= dt*0.05

w = 2 * np.pi * fr
errw = 2 * np.pi * errfr

H = Vr / Ve
errH = H * np.sqrt((errVr/Vr)**2 + (errVe/Ve)**2)

phi = w*dt
errphi = np.sqrt((errw*dt)**2 + (w*errdt)**2)

Re = H*np.cos(phi)
Im = H*np.sin(phi)
errRe = np.sqrt((errH*np.cos(phi))**2 + (H*np.sin(phi)*errphi)**2)
errIm = np.sqrt((errH*np.sin(phi))**2 + (H*np.cos(phi)*errphi)**2)

tau1 = 87.73878e-6
errtau1= 2.18809e-6
print(errtau)
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

err_cap = 4.90234188189095e-09
cap = 8.773877927430522e-08
def modelo_rlc_tau_fijo(p, x):
    w0, A, tau_ef = p[0], p[1], p[2]

    denominador = np.sqrt((x * tau_ef)**2 + (1 - (x / w0)**2)**2)
    return A * (x * tau_ef) / denominador

# 2. CONFIGURACIÓN DEL ODR
model_rlc = Model(modelo_rlc_tau_fijo)
data_rlc = RealData(w, H, sx=errw, sy=errH)

# SEMILLAS (Solo para w0 y A)
w0_guess = w[np.argmax(H)]
A_guess = np.max(H)
tau_ef = 1e-4

# beta0 debe tener solo 2 valores ahora
my_odr = ODR(data_rlc, model_rlc, beta0=[w0_guess, A_guess, tau_ef])
output = my_odr.run()

# 3. RESULTADOS
w0_fit, A_fit, tau_ef_fit = output.beta
err_w0, err_A, err_tau_ef_fit = output.sd_beta

# 4. CÁLCULO DE CHI CUADRADO REDUCIDO (Ya lo tenías, lo mantengo)
y_fit = modelo_rlc_tau_fijo(output.beta, w)
chi2_obs = np.sum(((H - y_fit) / errH)**2)
dof = len(w) - 3
chi2_red = chi2_obs / dof

# --- NUEVO: CÁLCULO DE R2 ---
residuos = H - y_fit
ss_res = np.sum(residuos**2)
ss_tot = np.sum((H - np.mean(H))**2)
r2_rlc = 1 - (ss_res / ss_tot)

# --- CÁLCULO DE L Y SU INCERTEZA ---
# La relación es L = 1 / (C * w0^2)
L_fit = 1 / (cap * w0_fit**2)

# Propagación de error para L
# Usamos la fórmula de derivadas parciales o errores relativos
errL = L_fit * np.sqrt((err_cap / cap)**2 + (2 * err_w0 / w0_fit)**2)

# 6. GRÁFICO (Agregamos R2 a la leyenda del ajuste)
plt.figure(figsize=(9, 6.5))
plt.errorbar(w, H, xerr=errw, yerr=errH, fmt='ko', alpha=0.3, label='Datos')

w_curva = np.linspace(min(w), max(w), 1000)
plt.plot(w_curva, modelo_rlc_tau_fijo([w0_fit, A_fit, tau_ef_fit], w_curva), color='darkslateblue', ls='-',
         label=f'Ajuste $R^2$={r2_rlc:.2f}')
plt.axvline(w0_fit, color='crimson', ls='--', label=f'$\omega_0$ = {w0_fit:.0f}({err_w0:.0f}) rad/s')

plt.xlabel("$\omega$ [rad/s]", fontsize=14)
plt.ylabel("H", fontsize=14)
plt.legend(fontsize=13, frameon=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.show()

# REPORTES
print(f"--- RESULTADOS (TAU FIJO = {tau1:.2e} s) ---")
print(f"w0 ajustado: {w0_fit:.2f} ± {err_w0:.2f} rad/s")
print(f"Amplitud A:  {A_fit:.3f} ± {err_A:.3f}")
# print(f"Resistencia parasitaria Rl:  {Rl_fit:.3f} ± {err_Rl_fit:.3f}")
print(f"Inductancia L: {L_fit:.6f} ± {errL:.6f} H")
print(f"R² del ajuste: {r2_rlc:.5f}")
print(f"Chi-cuadrado reducido: {chi2_red:.4f}")



rl = (tau_ef_fit - tau1)/cap
err_rl = np.sqrt( ((tau1*err_tau_ef_fit)/cap)**2 + ((tau_ef_fit*errtau1) / cap )**2 + (((tau_ef_fit - tau1)*err_cap)/cap**2)**2 )
print(f"Resistencia agregada por el inductor: {rl:.2f} ± {err_rl:.3f}")


print(f"tau Fijo {tau1}")
print(f"error tau fijo {errtau1}")
print(f"tau ef fit {tau_ef_fit}")
print(f"err tau ef fit {err_tau_ef_fit}")


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

# --- GRÁFICO COMPARATIVO (ORODEN DE CAPAS INVERTIDO) ---
plt.figure(figsize=(9, 6.5))

# 1. Datos experimentales al frente (zorder alto)
plt.errorbar(w, H, xerr=errw, yerr=errH, fmt='ko', alpha=0.5, label='Datos', zorder=5)

w_linea = np.linspace(min(w), max(w), 1000)


plt.plot(w_linea, modelo_rlc_solo_w0_A([w0_f_fit, A_f_fit], w_linea),
         color='crimson', ls='-', lw=2.5, zorder=3,
         label=f'Ajuste $\\tau_{{exp}}$ = 0,88(2) $\\times10^{{-4}}$ s' + f'\n$R^2$= 0,99')

plt.plot(w_linea, modelo_rlc_tau_fijo([w0_fit, A_fit, tau_ef_fit], w_linea),
         color='darkslateblue', ls='-', lw=2.5, zorder=4,
         label=f'Ajuste $\\tau_{{ef}}$ = 1,0(1) $\\times10^{{-4}}$ s' + f'\n$R^2$= 0,99')

plt.axvline(w0_fit, color='darkslateblue', ls='--', zorder=4, lw=1.8,
            label=f'$\omega_r$ = 20,8(1) $\\times10^3$ rad/s')

#plt.axvline(w0_f_fit, color='forestgreen', ls=':', alpha=0.8, zorder=3,
            #label=f'$\omega_{{0, fijo}}$ = {w0_f_fit:.1f} rad/s')

# Configuración estética
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("$\omega$ [rad/s]", fontsize=14)
plt.ylabel("H", fontsize=14)
plt.title("", fontsize=15)
plt.legend(fontsize=12.5, loc='upper right', frameon=True, shadow=True)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tau_RLC.png', dpi=500)
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

# --- CONFIGURACIÓN DEL GRÁFICO HORIZONTAL ---
# (Ancho=12, Alto=6 para que sea bien apaisado)
plt.figure(figsize=(7, 7))

# Gráfico de Nyquist: Re en el eje X, Im en el eje Y
plt.errorbar(Re, Im, xerr=errRe, yerr=errIm,
             fmt='o', color='darkslateblue', capsize=3,
             label='Datos', markersize=5, alpha=0.8)

# --- ESTILO Y ESCALAS AGRANDADAS ---
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.xlabel(r'$Re(H)$', fontsize=14)
plt.ylabel(r'$Im(H)$', fontsize=14)
plt.title('', fontsize=17, pad=15)

# Ajuste de límites para que el semicírculo se vea centrado y claro
# La parte real va de 0 a 1, la imaginaria de -0.5 a 0
plt.xlim(-0.1, 1.1)
plt.ylim(-0.6, 0.6)

# Agregamos la leyenda
plt.legend(fontsize=13, loc='best', frameon=True, shadow=True)

# Guías visuales
plt.grid(True, linestyle='--', alpha=0.5)
plt.axhline(0, color='black', linewidth=1, alpha=0.5) # Eje Real
plt.axvline(0, color='black', linewidth=1, alpha=0.5) # Eje Imaginario

plt.tight_layout()

# Guardar en alta resolución para el informe
plt.savefig("nyquist RLC.png", dpi=500, bbox_inches='tight')

plt.show()


# Imprimir los w0 obtenidos en todos los ajustes y pasarlos a hz con su error 
print(f"Frecuencia de resonancia (tau libre): {w0_fit/(2*np.pi):.2f} ± {err_w0/(2*np.pi):.2f} Hz")
print(f"Frecuencia de resonancia (tau fijo): {w0_f_fit/(2*np.pi):.2f} ± {err_w0_f/(2*np.pi):.2f} Hz")

# Imprimir los w0 en rad/s con su error
print(f"Frecuencia de resonancia (tau libre): {w0_fit:.2f} ± {err_w0:.2f} rad/s")
print(f"Frecuencia de resonancia (tau fijo): {w0_f_fit:.2f} ± {err_w0_f:.2f} rad/s")