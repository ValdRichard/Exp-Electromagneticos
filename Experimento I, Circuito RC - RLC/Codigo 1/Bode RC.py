import matplotlib.pyplot as plt
import math
from scipy import odr
import numpy as np
from funciones import analizar_bode_final, modelo_fase_r, modelo_fase_c, ajustar_fase

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


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 6.5), constrained_layout=True)

# --- EJECUCIÓN RESISTENCIA ---
# No le pases etiquetas, la función las genera sola con los R2 reales
res_r = analizar_bode_final(
    wr, Hr, err_wr, err_Hr, phir, err_phir, 
    r1=slice(0, 4), r2=slice(10, 14), tau0=tau0, 
    m1_f=None, m2_f=0.0, tipo='resistencia', titulo="", ax=ax1, loc_leyenda='center right'
)
w0m_r, dw0m_r, w0f_r, dw0f_r, r2_r1, r2_r2 = res_r

# --- EJECUCIÓN CAPACITOR ---
res_c = analizar_bode_final(
    wc, Hc, err_wc, err_Hc, phic, err_phic, 
    r1=slice(0, 5), r2=slice(10, 15), tau0=tau0, 
    m1_f=0.0, m2_f=None, tipo='capacitor', titulo="", ax=ax2, loc_leyenda='center left'
)
w0m_c, dw0m_c, w0f_c, dw0f_c, r2_c1, r2_c2 = res_c


# --- BLOQUE DE PRINT (Esto es lo que vas a ver en la terminal) ---
print("\n" + "="*50)
print(f"{'RESULTADOS DEL EXPERIMENTO RC':^50}")
print("="*50)

# Reporte Resistencia
print(f"\n>>> FILTRO PASA-ALTOS (R)")
print(f"  - w0 Módulo:  ({w0m_r:.2f} +/- {dw0m_r:.2f}) rad/s  | R2_r1: {r2_r1:.4f}, R2_r2: {r2_r2:.4f}")
print(f"  - w0 Fase:    ({w0f_r:.2f} +/- {dw0f_r:.2f}) rad/s")

# Reporte Capacitor
print(f"\n>>> FILTRO PASA-BAJOS (C)")
print(f"  - w0 Módulo:  ({w0m_c:.2f} +/- {dw0m_c:.2f}) rad/s  | R2_c1: {r2_c1:.4f}, R2_c2: {r2_c2:.4f}")
print(f"  - w0 Fase:    ({w0f_c:.2f} +/- {dw0f_c:.2f}) rad/s")

print("\n" + "="*50)

plt.show()