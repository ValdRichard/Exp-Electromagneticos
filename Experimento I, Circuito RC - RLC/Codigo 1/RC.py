import matplotlib.pyplot as plt
import math
from scipy import odr
import numpy as np
from funciones import analizar_bode, modelo_fase_r, modelo_fase_c, ajustar_fase

#RESISTENCIA
R = 1000
C = 100e-9
fref = 1/(2 * np.pi * R * C)
wref = 1/(R*C)
tau0 = R*C
errC = 10e-9
errR = R*0.05
errwref = wref * np.sqrt((errR/R)**2 + (errC/C)**2)
print(f"Frecuencia angular teorica: {wref:.2f} Hz")
print(f"Error de frecuencia angular: {errwref:.2f} Hz")
errtau = np.sqrt((R*errC)**2 + (C*errR)**2)


fr = np.array([100.4, 177.6, 328, 704, 1001, 1304, 1599.5, 2536, 3498, 5000, 10000, 14000, 19100, 35050]) #Hz
Afemr = np.array([6.44, 6.445, 6.46, 6.37, 6.35, 6.30, 6.41, 6.21, 6.27, 6.25, 6.19, 6.17, 6.18, 6.24]) #V
Ar = np.array([0.388, 0.680, 1.26, 2.50, 3.25, 3.88, 4.45, 5.17, 5.662, 5.9, 6.12, 6.14, 6.18, 6.265]) #V
deltatr = np.array([0.0024, 0.00136, 0.000650, 0.000260, 0.000160, 0.000108, 0.00008, 0.000036, 0.0000210, 0.0000120, 0.0000032, 0.0000016, 0.0000010, 0.000000200]) #s

errfr = np.array([0.3, 0.3, 3, 1, 1, 2, 1, 2, 3, 5, 10, 20, 50, 200]) #Hz
errAfemr = np.array([0.01, 0.005, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.03, 0.02, 0.03]) #V
errAr = np.array([0.002, 0.001, 0.01, 0.01, 0.05, 0.01, 0.02, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.005]) #V
errdeltatr = deltatr*0.05

Hr = Ar/Afemr
wr = 2*np.pi*fr
phir = wr * deltatr

# Propagación de errores Resistencia
err_Hr = np.sqrt((errAr / Afemr)**2 + ( (Ar * errAfemr) / Afemr**2)**2)
err_phir = phir * np.sqrt((errfr / fr)**2 + (errdeltatr / deltatr)**2)
err_wr = 2*np.pi*errfr

#CAPACITOR

fc = np.array([60.2, 111.8, 177.8, 298, 502.5, 708, 998, 1303, 1598, 3010, 5560, 7060, 7090, 13600, 19200, 35200]) #Hz
Afemc = np.array([6.41, 6.44, 6.44, 6.46, 6.44, 6.41, 6.37, 6.33, 6.29, 6.14, 6.22, 6.19, 6.09, 6.2, 6.17, 6.19]) #V
Ac = np.array([6.39, 6.41, 6.39, 6.44, 6.14, 5.88, 5.44, 4.95, 4.53, 2.96, 1.78, 1.41, 1.41, 0.76, 0.541, 0.292]) #V
deltatc = np.array([0.000072, 0.000088, 0.000088, 0.000076, 0.000100, 0.000088, 0.000092, 0.000084, 0.000080, 0.000057, 0.0000364, 0.000030, 0.000030, 0.0000168, 0.0000124, 0.00000690]) #s

errfc = np.array([0.1, 0.3, 0.2, 1, 1.5, 2, 4, 3, 4, 10, 10, 20, 10, 50, 100, 200]) #Hz
errAfemc = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.03]) #V
errAc = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.002, 0.002]) #V
errdeltatc = deltatc*0.05

Hc = Ac/Afemc
wc = 2*np.pi*fc
phic = - wc * deltatc
phic_ = - wc*deltatc

# Propagación de errores Capacitor
err_Hc = Hc * np.sqrt((errAfemc / Afemc)**2 + (errAc / Ac)**2)
err_phic = np.abs(phic) * np.sqrt((errfc / fc)**2 + (errdeltatc / deltatc)**2)
err_wc = 2*np.pi*errfc

#NYQUIST con cos-sen

Re_Hr = Hr * np.cos(phir)
Im_Hr = Hr * np.sin(phir)
err_Re_Hr = np.sqrt((np.cos(phir) * err_Hr)**2 + (Hr * np.sin(phir) * err_phir)**2)
err_Im_Hr = np.sqrt((np.sin(phir) * err_Hr)**2 + (Hr * np.cos(phir) * err_phir)**2)
Re_Hc = Hc * np.cos(phic_)
Im_Hc = Hc * np.sin(phic_)
err_Re_Hc = np.sqrt((np.cos(phic_) * err_Hc)**2 + (Hc * np.sin(phic_) * err_phic)**2)
err_Im_Hc = np.sqrt((np.sin(phic_) * err_Hc)**2 + (Hc * np.cos(phic_) * err_phic)**2)


# --- CREACIÓN DEL GRÁFICO ---
fig, ax = plt.subplots(figsize=(7, 7))

# Gráfico Resistencia con barras de error
ax.errorbar(Re_Hr, Im_Hr, xerr=err_Re_Hr, yerr=err_Im_Hr,
            fmt='o', color='darkorange', ecolor='darkorange',
            elinewidth=1.5, capsize=3, label='Pasa-altos', markersize=5)

# Gráfico Capacitor con barras de error
ax.errorbar(Re_Hc, Im_Hc, xerr=err_Re_Hc, yerr=err_Im_Hc,
            fmt='s', color='indigo', ecolor='indigo',
            elinewidth=1.5, capsize=3, label='Pasa-bajos', markersize=5)

# Configuraciones visuales
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_title("", fontsize=16)
ax.set_xlabel(r"$Re(H)$", fontsize=14)
ax.set_ylabel(r"$Im(H)$", fontsize=14)

ax.axhline(0, color='black', linewidth=1, alpha=0.5)
ax.axvline(0, color='black', linewidth=1, alpha=0.5)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=13, frameon=True, shadow=True)

plt.tight_layout()
plt.savefig("nyquist_RC.png", dpi=500)
#plt.show()


# --- CONFIGURACIÓN MANUAL DE LEYENDAS ---

# Pasa-altos (Resistencia)
lab_w0_r = r'$\omega_0 = 1,2(1) \times 10^4$ rad/s'
lab_m_r = r'Ajuste lineal $R^2$=0,99 (m libre)' # Recta de subida
lab_f_r = r'Ajuste lineal $R^2$=0,99 (m=0)'  # Recta horizontal

# Pasa-bajos (Capacitor)
lab_w0_c = r'$\omega_0 = 1,0(2) \times 10^4$ rad/s'
lab_m_c = r'Ajuste lineal $R^2$=0,99 (m libre)' # Recta de caída
lab_f_c = r'Ajuste lineal $R^2$=0,99 (m=0)'  # Recta horizontal

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))

# Ejecución Resistencia
w0_r, err_w0_r, r2_r1, r2_r2 = analizar_bode(wr, Hr, err_wr, err_Hr,
                                             r1=slice(0, 4), r2=slice(10, 14),
                                             m1_f=None, m2_f=0.0,
                                             titulo="Pasa-altos", ax=ax1,
                                             label_w0=lab_w0_r,
                                             label_m_libre=lab_m_r,
                                             label_m_fija=lab_f_r)

# Ejecución Capacitor
w0_c, err_w0_c, r2_c1, r2_c2 = analizar_bode(wc, Hc, err_wc, err_Hc,
                                             r1=slice(0, 5), r2=slice(11, 15),
                                             m1_f=0.0, m2_f=None,
                                             titulo="Pasa-bajos", ax=ax2,
                                             label_w0=lab_w0_c,
                                             label_m_libre=lab_m_c,
                                             label_m_fija=lab_f_c)

plt.tight_layout()
plt.savefig("rectas_RC.png", dpi=500)
#plt.show()

# Prints finales (calculando f_0 a partir de w_0)
f0_r, ef0_r = w0_r/(2*np.pi), err_w0_r/(2*np.pi)
f0_c, ef0_c = w0_c/(2*np.pi), err_w0_c/(2*np.pi)

print(f"\n{'='*40}\n           RESULTADOS FINALES\n{'='*40}")
print(f"Frecuencia referencia (RC): {fref:.2f} Hz")
print(f"Corte Resistencia: {f0_r:.2f} ± {ef0_r:.2f} Hz")
print(f"  - Tramo Subida (m libre): R² = {r2_r1:.6f}")
print(f"Corte Capacitor:   {f0_c:.2f} ± {ef0_c:.2f} Hz")
print(f"  - Tramo Caída  (m libre): R² = {r2_c2:.6f}")

print(f"w0c: {w0_c} ± {err_w0_c}")
print(f"w0r: {w0_r} ± {err_w0_r}")

# Promedio pesado por incertezas
# pesos = 1 / sigma^2
w_r = 1 / ef0_r**2
w_c = 1 / ef0_c**2

f0_prom = (f0_r * w_r + f0_c * w_c) / (w_r + w_c)

# Error del promedio pesado
ef0_prom = np.sqrt(1 / (w_r + w_c))



# --- CÁLCULO DE PROMEDIOS PESADOS ---

# 1. En Hertz (ya lo tenías, lo organizamos)
weights_f = 1 / ef0_r**2 + 1 / ef0_c**2
f0_prom = (f0_r / ef0_r**2 + f0_c / ef0_c**2) / weights_f
ef0_prom = np.sqrt(1 / weights_f)

# 2. En Radianes/segundo
# Usamos directamente w0_r y w0_c con sus errores err_w0_r y err_w0_r
weights_w = 1 / err_w0_r**2 + 1 / err_w0_c**2
w0_prom = (w0_r / err_w0_r**2 + w0_c / err_w0_c**2) / weights_w
ew0_prom = np.sqrt(1 / weights_w)

# --- PRINT DE CONSOLIDACIÓN ---
print(f"\n{'#'*40}")
print(f"{'RESUMEN DE PROMEDIOS PESADOS':^40}")
print(f"{'#'*40}")
print(f"Frecuencia (f0):  {f0_prom:.2f} ± {ef0_prom:.2f} Hz")
print(f"Angular (w0):     {w0_prom:.2f} ± {ew0_prom:.2f} rad/s")


# --- 3. EJECUCIÓN DE LOS AJUSTES ---
tau_r_fit, err_tau_r, res_r = ajustar_fase(wr, phir, err_wr, err_phir, tau0, 'resistencia')
tau_c_fit, err_tau_c, res_c = ajustar_fase(wc, phic, err_wc, err_phic, tau0, 'capacitor')

# --- 4. GRÁFICOS COMPARATIVOS ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))

# Configuración común de estilo
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel(r'$\omega$ [rad/s]', fontsize=14)
    ax.set_ylabel(r'$\phi$ [rad]', fontsize=14)

# Subplot 1: Resistencia (Pasa-Altos)
ax1.errorbar(wr, phir, xerr=err_wr, yerr=err_phir, fmt='o', color='black', markersize=7,
             alpha=0.3, label='Datos', capsize=3)
w_smooth_r = np.linspace(min(wr), max(wr), 500)
ax1.plot(w_smooth_r, modelo_fase_r([tau_r_fit], w_smooth_r), color='indigo', lw=2,
         label=fr'Ajuste $R^2$ = 0,99' + f'\n$\\tau$ = {tau_r_fit*1e6:.0f}({err_tau_r*1e6:.0f}) $\mu$s')
ax1.set_title("Pasa-altos", fontsize=16)
ax1.set_xscale('log')
ax1.legend(fontsize=13, frameon=True, shadow=True)

# Subplot 2: Capacitor (Pasa-Bajos)
ax2.errorbar(wc, phic, xerr=err_wc, yerr=err_phic, fmt='o', color='black', markersize=7,
             alpha=0.3, label='Datos', capsize=3)
w_smooth_c = np.linspace(min(wc), max(wc), 500)
ax2.plot(w_smooth_c, modelo_fase_c([tau_c_fit], w_smooth_c), color='darkorange', lw=2,
         label=fr'Ajuste $R^2$ = 0,99' + f'\n$\\tau$ = {tau_c_fit*1e6:.0f}({err_tau_c*1e6:.0f}) $\mu$s')
ax2.set_title("Pasa-bajos", fontsize=16)
ax2.set_xscale('log')
ax2.legend(fontsize=13, frameon=True, shadow=True)

plt.tight_layout()
plt.savefig("tau_RC.png", dpi=500)
#plt.show()

# --- 5. REPORTE DE RESULTADOS ---
print(f"Valor teórico de tau (R*C): {tau0*1e6:.2f} \u00b1 {errtau*1e6:.2f} \u03bcs")
print("-" * 50)
print(f"Tau ajustado (Resistencia): {tau_r_fit*1e6:.2f} \u00b1 {err_tau_r*1e6:.2f} \u03bcs")
print(f"Tau ajustado (Capacitor):   {tau_c_fit*1e6:.2f} \u00b1 {err_tau_c*1e6:.2f} \u03bcs")

#promedio pesado tau

w_r = 1 / (err_tau_r**2)
w_c = 1 / (err_tau_c**2)

# 2. Promedio ponderado
tau_prom = (tau_r_fit * w_r + tau_c_fit * w_c) / (w_r + w_c)

# 3. Error del promedio ponderado
err_tau_prom = np.sqrt(1 / (w_r + w_c))
cac = tau_prom / R
err_cac = cac * np.sqrt((err_tau_prom/tau_prom)**2 + (errR/R)**2)
print (cac)
print(err_cac)
print (tau_prom)
print (err_tau_prom)
print(f"Tau Promedio Ponderado: {tau_prom*1e6:.5f} ± {err_tau_prom*1e6:.5f} μs")

# --- CÁLCULO DE R^2 POR SEPARADO ---

def calcular_r2_manual(obs, pred):
    residuo = obs - pred
    ss_res = np.sum(residuo**2)
    ss_tot = np.sum((obs - np.mean(obs))**2)
    return 1 - (ss_res / ss_tot)

# 1. Predicciones teóricas usando los taus obtenidos
# Usamos las funciones de modelo que definiste antes
pred_r = modelo_fase_r([tau_r_fit], wr)
pred_c = modelo_fase_c([tau_c_fit], wc)

# 2. Cálculo de los coeficientes
r2_resistencia = calcular_r2_manual(phir, pred_r)
r2_capacitor = calcular_r2_manual(phic, pred_c)

# --- REPORTE ---
print(f"Coeficientes de determinación (R²):")
print(f"{'-'*35}")
print(f"Pasa altos (Resistencia): {r2_resistencia:.5f}")
print(f"Pasa bajos (Capacitor):   {r2_capacitor:.5f}")

#Promedio ponderado del método de ajuste con phi y de rectas
# Datos
x1 = 11397
s1 = 284

x2 = 11063
s2 = 919

# Pesos
w1 = 1 / s1**2
w2 = 1 / s2**2

# Promedio ponderado
x_bar = (w1 * x1 + w2 * x2) / (w1 + w2)

# Error del promedio
s_bar = np.sqrt(1 / (w1 + w2))

print("Promedio ponderado:", x_bar)
print("Incertidumbre:", s_bar)



# Imprimir todos los valores de las frecuencias obtenidas pasadas a Hz junto con su error que es multiplicar por 2*pi el error de w0
f0_r_hz = w0_r / (2 * np.pi)        
ef0_r_hz = err_w0_r / (2 * np.pi)
f0_c_hz = w0_c / (2 * np.pi)
ef0_c_hz = err_w0_c / (2 * np.pi)

print(f"\nFrecuencia de corte Resistencia: {f0_r_hz:.2f} ± {ef0_r_hz:.2f} Hz")
print(f"Frecuencia de corte Capacitor:   {f0_c_hz:.2f} ± {ef0_c_hz:.2f} Hz")

# Imprimir los resultados del w0 y su error en rad/s
print(f"\nFrecuencia de corte Resistencia: {w0_r:.2f} ± {err_w0_r:.2f} rad/s")      
print(f"Frecuencia de corte Capacitor:   {w0_c:.2f} ± {err_w0_c:.2f} rad/s")
 

# Imprimir el promedio ponderado de las frecuencias de corte en Hz y en rad/s
f0_prom_hz = x_bar / (2 * np.pi)  
ef0_prom_hz = s_bar / (2 * np.pi)
print(f"\nPromedio ponderado de frecuencia de corte: {f0_prom_hz:.2f} ± {ef0_prom_hz:.2f} Hz")
print(f"Promedio ponderado de frecuencia de corte: {x_bar:.2f} ± {s_bar:.2f} rad/s")


print(f"\nPromedio pesado: {f0_prom:.2f} ± {ef0_prom:.2f} Hz")
print(f"Promedio pesado: {w0_prom:.2f} ± {ew0_prom:.2f} rad/s")