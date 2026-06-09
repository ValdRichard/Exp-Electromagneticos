import numpy as np
import matplotlib.pyplot as plt
from odrpack import odr_fit
def graficarDatos(serie, titulo, mostrar=False, color_ch1="tab:blue", color_ch2="tab:orange"):
    """
    Grafica Ch1 [V] y Ch2 [mV] en función de T [°C].
    Parámetros
    ----------
    serie   : DataFrame con columnas T, Ch1, Ch2
    titulo  : string para el título del gráfico
    mostrar : si True llama plt.show() al final
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.suptitle(titulo, fontsize=13)

    ax1.set_xlabel("T [°C]")
    ax1.set_ylabel("Ch1 [V]", color=color_ch1)
    ax1.plot(serie["T"], serie["Ch1"], "o-", color=color_ch1,
             markersize=4, linewidth=1.2, label="Ch1")
    ax1.tick_params(axis="y", labelcolor=color_ch1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Ch2 [mV]", color=color_ch2)
    ax2.plot(serie["T"], serie["Ch2"], "s-", color=color_ch2,
             markersize=4, linewidth=1.2, label="Ch2")
    ax2.tick_params(axis="y", labelcolor=color_ch2)

    lineas1, etiq1 = ax1.get_legend_handles_labels()
    lineas2, etiq2 = ax2.get_legend_handles_labels()
    ax1.legend(lineas1 + lineas2, etiq1 + etiq2, loc="upper right")

    ax1.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if mostrar:
        plt.show()

    return fig


def graficarAjuste(serie, resultado, titulo, mostrar=False):
    """
    Grafica Ch2 con barras de error y la curva ajustada M(T)/M0 = A*(1-T/Tc)^beta.
    Parámetros
    ----------
    serie     : DataFrame con columnas T, Ch2
    resultado : OdrResult devuelto por ajustarMagnetizacion
    titulo    : string para el título
    mostrar   : si True llama plt.show() al final
    """
    A, Tc, beta = resultado.beta
    sA, sTc, sbeta = resultado.sd_beta

    T   = serie["T"].values
    Ch2 = serie["Ch2"].values
    sx  = 0.05 * T
    sy  = 0.05 * Ch2

    T_curva = np.linspace(T.min(), Tc * 0.9999, 300)
    y_curva = A * (1 - T_curva / Tc) ** beta

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(titulo, fontsize=13)

    ax.errorbar(T, Ch2, xerr=sx, yerr=sy,
                fmt="o", color="tab:blue", markersize=4,
                ecolor="gray", elinewidth=1, capsize=3,
                label="Datos (Ch2)")

    label_ajuste = (
        rf"Ajuste ODR"  "\n"
        rf"$A = {A:.2f} \pm {sA:.2f}$"  "\n"
        rf"$T_c = {Tc:.2f} \pm {sTc:.2f}$ °C"  "\n"
        rf"$\beta = {beta:.4f} \pm {sbeta:.4f}$"
    )
    ax.plot(T_curva, y_curva, color="tab:red", linewidth=1.8, label=label_ajuste)

    ax.set_xlabel("T [°C]")
    ax.set_ylabel("Ch2 [mV]  (∝ M)")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if mostrar:
        plt.show()

    return fig


# ── Ajuste ODR ───────────────────────────────────────────────────────────────
 
def ajustarMagnetizacionTc(serie, beta0=(75.0, 150.0, 0.3)):
    """
    Ajusta M(T)/M0 = A * (1 - T/Tc)^beta sobre Ch2 usando ODR.
    Errores: 5% en T y 5% en Ch2.
    Parámetros
    ----------
    serie : DataFrame con columnas T, Ch2
    beta0 : valores iniciales (A, Tc, beta)
    Retorna
    -------
    OdrResult con .beta = [A, Tc, beta] y .sd_beta = errores estándar
    """
    T   = serie["T"].values
    Ch2 = serie["Ch2"].values
 
    def modelo(x, params):
        A, Tc, b = params
        arg = 1 - x / Tc
        
        return A * arg**b
 
    sx = 0.05 * T
    sy = 0.05 * Ch2
 
    resultado = odr_fit(
        modelo, T, Ch2,
        beta0=list(beta0),
        weight_x=1.0 / sx**2,
        weight_y=1.0 / sy**2,
    )
    return resultado

def ajustarMagnetizacionBlochFijo(serie, beta0=(75.0, 150.0)):
    """
    Ajuste de Bloch:
        M(T) = A * (1 - (T/Tc)^(3/2))
    Ajusta A y Tc.
    """
    T = serie["T"].values
    Ch2 = serie["Ch2"].values

    def modelo(x, params):
        A, Tc = params
        return A * (1 - (x / Tc)**(3/2))

    sx = 0.05 * T
    sy = 0.05 * Ch2

    resultado = odr_fit(
        modelo,
        T,
        Ch2,
        beta0=list(beta0),
        weight_x=1.0 / sx**2,
        weight_y=1.0 / sy**2,
    )

    return resultado

def ajustarMagnetizacionBlochLibre(serie, beta0=(75.0, 150.0, 1.5)):
    """
    Ajuste tipo Bloch con exponente libre:
        M(T) = A * (1 - (T/Tc)^n)
    Ajusta A, Tc y n.
    """
    T = serie["T"].values
    Ch2 = serie["Ch2"].values

    def modelo(x, params):
        A, Tc, n = params
        return A * (1 - (x / Tc)**n)

    sx = 0.05 * T
    sy = 0.05 * Ch2

    resultado = odr_fit(
        modelo,
        T,
        Ch2,
        beta0=list(beta0),
        weight_x=1.0 / sx**2,
        weight_y=1.0 / sy**2,
    )

    return resultado
