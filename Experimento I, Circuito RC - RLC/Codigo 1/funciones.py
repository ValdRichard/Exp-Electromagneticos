
from scipy import odr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def ajustar_recta(x, y, err_x, err_y, r, fix_m=None):
    x_r = x[r]
    y_r = y[r]
    sx_r = err_x[r]
    sy_r = err_y[r]

    def f(B, x):
        return B[0]*x + B[1]

    modelo = odr.Model(f)
    datos = odr.RealData(x_r, y_r, sx=sx_r, sy=sy_r)

    if fix_m is None:
        beta0 = [1.0, 0.0]
        odr_inst = odr.ODR(datos, modelo, beta0=beta0)
    else:
        beta0 = [fix_m, 0.0]
        odr_inst = odr.ODR(datos, modelo, beta0=beta0)
        odr_inst.set_job(fit_type=0)
        odr_inst.ifixb = [0, 1]

    out = odr_inst.run()
    B = out.beta
    errB = out.sd_beta

    # Cálculo de R2
    y_pred = B[0] * x_r + B[1]
    residuos = y_r - y_pred
    ss_res = np.sum(residuos**2)
    ss_tot = np.sum((y_r - np.mean(y_r))**2)
    r2 = 1 - (ss_res / ss_tot)

    return B, errB, r2



def analizar_bode(w, H, err_w, err_H, r1, r2, m1_f=None, m2_f=None, titulo="Bode", ax=None,
                  label_w0=None, label_m_libre=None, label_m_fija=None):
    
    # 1. Preparación de datos y logaritmos
    x_log, y_log = np.log10(w), np.log10(H)
    err_x = (1 / (w * np.log(10))) * err_w
    err_y = (1 / (H * np.log(10))) * err_H

    # 2. Ajustes ODR
    b1, eb1, r2_1 = ajustar_recta(x_log, y_log, err_x, err_y, r1, fix_m=m1_f)
    b2, eb2, r2_2 = ajustar_recta(x_log, y_log, err_x, err_y, r2, fix_m=m2_f)

    # 3. Intersección y w0
    x_int = (b2[1] - b1[1]) / (b1[0] - b2[0])
    w_0 = 10**x_int

    # --- LÓGICA DE INCERTIDUMBRE (Tu código original) ---
    if m2_f == 0 or m2_f is not None: # Si el tramo 2 es el fijo
        m, errm, B1, errB1, B2, errB2 = b1[0], eb1[0], b1[1], eb1[1], b2[1], eb2[1]
    else: # Si el tramo 1 es el fijo
        m, errm, B1, errB1, B2, errB2 = b2[0], eb2[0], b2[1], eb2[1], b1[1], eb1[1]
    
    err_x_int = np.sqrt((errB2/m)**2 + (errB1/m)**2 + ((B2-B1)*errm/m**2)**2)
    err_w0 = w_0 * np.log(10) * err_x_int

    # ---------------------------------------------------------
    # IDENTIFICACIÓN EXPLÍCITA PARA LA LEYENDA
    # ---------------------------------------------------------
    # Colores fijos: Indigo para libre, Orange para fija
    c_libre, c_fija = 'indigo', 'darkorange'

    # Determinamos qué resultado (b1 o b2) es el libre para sacar su R2
    if m1_f is None: # El tramo 1 es libre
        res_libre, res_fijo = {'b': b1, 'r2': r2_1}, {'b': b2, 'r2': r2_2}
        col1, col2 = c_libre, c_fija
    else: # El tramo 2 es libre (o m1_f está fijo)
        res_libre, res_fijo = {'b': b2, 'r2': r2_2}, {'b': b1, 'r2': r2_1}
        col1, col2 = c_fija, c_libre

    # ---------------------------------------------------------
    # GRÁFICO
    # ---------------------------------------------------------
    if ax is None: fig, ax = plt.subplots(figsize=(12, 7))

# --- Lógica de puntos corregida ---
    indices = np.arange(len(x_log))
    # indices[r1] funciona perfecto con slice, mask booleana o lista de ints.
    idx_r1 = indices[r1]
    idx_r2 = indices[r2]
    
    idx_usados = np.union1d(idx_r1, idx_r2)
    idx_resto = np.setdiff1d(indices, idx_usados)

    # Graficamos el "resto" en gris
    if len(idx_resto) > 0:
        ax.errorbar(x_log[idx_resto], y_log[idx_resto], xerr=err_x[idx_resto], yerr=err_y[idx_resto],
                    fmt='o', color='gray', markersize=8, alpha=0.4, label='_nolegend_')

    # Graficamos puntos con label='_nolegend_' para que no interfieran
    ax.errorbar(x_log[r1], y_log[r1], xerr=err_x[r1], yerr=err_y[r1], fmt='o', color=col1, alpha=0.8, markersize=8, label='_nolegend_')
    ax.errorbar(x_log[r2], y_log[r2], xerr=err_x[r2], yerr=err_y[r2], fmt='o', color=col2, alpha=0.8, markersize=8, label='_nolegend_')

    # Graficamos rectas
    x_f = np.linspace(min(x_log), max(x_log), 100)
    ax.plot(x_f, b1[0]*x_f + b1[1], ls='--', lw=2.2, color=col1, label='_nolegend_')
    ax.plot(x_f, b2[0]*x_f + b2[1], ls='--', lw=2.2, color=col2, label='_nolegend_')
    
    # Punto w0
    ax.plot(x_int, b1[0]*x_int + b1[1], 's', color='crimson', markersize=10, zorder=5, label='_nolegend_')

    # ---------------------------------------------------------
    # CONSTRUCCIÓN DE LA LEYENDA MANUAL (ORDEN FORZADO)
    # ---------------------------------------------------------
    txt_libre = label_m_libre if label_m_libre else f"Ajuste libre ($R^2$={res_libre['r2']:.3f})"
    txt_fija = label_m_fija if label_m_fija else "Pendiente fija"
    txt_w0 = label_w0 if label_w0 else rf"$\omega_0 = {w_0:.2f} \pm {err_w0:.2f}$"

    # Aquí definimos el orden: LIBRE primero, FIJA segundo
    handles_manuales = [
        Line2D([0], [0], color=c_libre, lw=2.2, ls='--', label=txt_libre),
        Line2D([0], [0], color=c_fija, lw=2.2, ls='--', label=txt_fija),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='crimson', markersize=10, label=txt_w0)
    ]

    # Aplicamos la leyenda ignorando cualquier otra cosa graficada
    ax.legend(handles=handles_manuales, loc='best', fontsize=16, frameon=True, shadow=True)

    # Estética final
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim(-1.3, 0.1)
    #ax.set_title(titulo, fontsize=16)
    ax.set_xlabel(r"$\log_{10}(\omega)$", fontsize=16)
    ax.set_ylabel(r"$\log_{10}(H)$", fontsize=16)
    ax.grid(True, alpha=0.2)

    return w_0, err_w0, r2_1, r2_2

    # --- 1. MODELOS TEÓRICOS PARA ODR ---
def modelo_fase_r(p, w):
    # p[0] es tau
    return np.arctan(1 / (w * p[0]))

def modelo_fase_c(p, w):
    # p[0] es tau
    return np.arctan(-w * p[0])

# --- 2. FUNCION PARA AJUSTE ODR ---
def ajustar_fase(w, phi, err_w, err_phi, tau0, tipo='resistencia'):
    # Creamos el modelo según el componente
    func = modelo_fase_r if tipo == 'resistencia' else modelo_fase_c
    mdl = odr.Model(func)

    # Preparamos los datos
    data = odr.RealData(w, phi, sx=err_w, sy=err_phi)

    # Inicializamos con tu valor de tau calculado (R*C)
    my_odr = odr.ODR(data, mdl, beta0=[tau0])
    out = my_odr.run()

    return out.beta[0], out.sd_beta[0], out


