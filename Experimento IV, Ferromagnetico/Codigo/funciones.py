import numpy as np
import matplotlib.pyplot as plt
from odrpack import odr_fit


# =============================================================================
# GRAFICOS DE DATOS
# =============================================================================

def graficarDatos(
    serie,
    titulo,
    mostrar=False,
    color_ch1="tab:blue",
    color_ch2="tab:orange"
):

    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.suptitle(titulo, fontsize=13)

    ax1.set_xlabel("T [°C]")
    ax1.set_ylabel("Ch1 [V]", color=color_ch1)

    ax1.plot(
        serie["T"],
        serie["Ch1"],
        "o-",
        color=color_ch1,
        markersize=4,
        linewidth=1.2,
        label="Ch1",
    )

    ax1.tick_params(axis="y", labelcolor=color_ch1)

    ax2 = ax1.twinx()

    ax2.set_ylabel("Ch2 [mV]", color=color_ch2)

    ax2.plot(
        serie["T"],
        serie["Ch2"],
        "s-",
        color=color_ch2,
        markersize=4,
        linewidth=1.2,
        label="Ch2",
    )

    ax2.tick_params(axis="y", labelcolor=color_ch2)

    l1, e1 = ax1.get_legend_handles_labels()
    l2, e2 = ax2.get_legend_handles_labels()

    ax1.legend(l1 + l2, e1 + e2)

    ax1.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    if mostrar:
        plt.show()

    return fig


# =============================================================================
# AJUSTE CRITICO
# =============================================================================

def ajustarMagnetizacionTc(
    serie,
    beta0=(75.0, 150.0, 0.3)
):

    T = serie["T"].values
    Ch2 = serie["Ch2"].values

    def modelo(x, p):
        A, Tc, beta = p
        return A * (1 - x / Tc) ** beta

    sx = 0.05 * T
    sy = 0.05 * Ch2

    resultado = odr_fit(
        modelo,
        T,
        Ch2,
        beta0=list(beta0),
        weight_x=1 / sx**2,
        weight_y=1 / sy**2,
    )

    return resultado


def graficarAjusteTc(
    serie,
    resultado,
    nombre_serie,
    ax=None
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    A, Tc, beta = resultado.beta
    sA, sTc, sBeta = resultado.sd_beta

    T = serie["T"].values
    Ch2 = serie["Ch2"].values

    sx = 0.05 * T
    sy = 0.05 * Ch2

    Tcurva = np.linspace(T.min(), Tc * 0.999, 400)
    ycurva = A * (1 - Tcurva / Tc) ** beta

    eb = ax.errorbar(
        T,
        Ch2,
        xerr=sx,
        yerr=sy,
        fmt="o",
        ms=5,
        color="0.25",
        ecolor="0.25",
        elinewidth=1,
        capsize=3,
        label="Puntos experimentales"
    )

    for barra in eb[2]:
        barra.set_alpha(0.4)

    for cap in eb[1]:
        cap.set_alpha(0.4)

    label = (
        f"Ajuste {nombre_serie}\n"
        rf"$T_c={Tc:.2f}\pm{sTc:.2f}$"
        "\n"
        rf"$\beta={beta:.4f}\pm{sBeta:.4f}$"
    )

    ax.plot(
        Tcurva,
        ycurva,
        lw=2.5,
        color="#ff7700",
        label=label
    )

    ax.set_xlabel(r"$T$ [°C]")
    ax.set_ylabel(r"$M \propto V$ [mV]")

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    return fig, ax

# =============================================================================
# BLOCH FIJO
# =============================================================================

def ajustarMagnetizacionBlochFijo(
    serie,
    beta0=(75.0, 150.0)
):

    T = serie["T"].values
    Ch2 = serie["Ch2"].values

    def modelo(x, p):
        A, Tc = p
        return A * (1 - (x / Tc) ** (3 / 2))

    sx = 0.05 * T
    sy = 0.05 * Ch2

    resultado = odr_fit(
        modelo,
        T,
        Ch2,
        beta0=list(beta0),
        weight_x=1 / sx**2,
        weight_y=1 / sy**2,
    )

    return resultado


def graficarAjusteBlochFijo(
    serie,
    resultado,
    titulo,
    mostrar=False
):

    A, Tc = resultado.beta
    sA, sTc = resultado.sd_beta

    T = serie["T"].values
    Ch2 = serie["Ch2"].values

    sx = 0.05 * T
    sy = 0.05 * Ch2

    Tcurva = np.linspace(T.min(), T.max(), 400)

    ycurva = A * (1 - (Tcurva / Tc) ** (3 / 2))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        T,
        Ch2,
        xerr=sx,
        yerr=sy,
        fmt="o",
        capsize=3,
    )

    label = (
        rf"$A={A:.2f}\pm{sA:.2f}$"
        "\n"
        rf"$T_c={Tc:.2f}\pm{sTc:.2f}$"
        "\n"
        rf"$n=1.5$ fijo"
    )

    ax.plot(Tcurva, ycurva, lw=2, label=label)

    ax.set_xlabel("T [°C]")
    ax.set_ylabel("Ch2 [mV]")

    ax.legend()
    ax.grid(True)

    fig.suptitle(titulo)

    if mostrar:
        plt.show()

    return fig


# =============================================================================
# BLOCH LIBRE
# =============================================================================

def ajustarMagnetizacionBlochLibre(
    serie,
    beta0=(75.0, 150.0, 1.5)
):

    T = serie["T"].values
    Ch2 = serie["Ch2"].values

    def modelo(x, p):
        A, Tc, n = p
        return A * (1 - (x / Tc) ** n)

    sx = 0.05 * T
    sy = 0.05 * Ch2

    resultado = odr_fit(
        modelo,
        T,
        Ch2,
        beta0=list(beta0),
        weight_x=1 / sx**2,
        weight_y=1 / sy**2,
    )

    return resultado


def graficarAjusteBlochLibre(
    serie,
    resultado,
    titulo,
    mostrar=False
):

    A, Tc, n = resultado.beta
    sA, sTc, sn = resultado.sd_beta

    T = serie["T"].values
    Ch2 = serie["Ch2"].values

    sx = 0.05 * T
    sy = 0.05 * Ch2

    Tcurva = np.linspace(T.min(), T.max(), 400)

    ycurva = A * (1 - (Tcurva / Tc) ** n)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        T,
        Ch2,
        xerr=sx,
        yerr=sy,
        fmt="o",
        capsize=3,
    )

    label = (
        rf"$A={A:.2f}\pm{sA:.2f}$"
        "\n"
        rf"$T_c={Tc:.2f}\pm{sTc:.2f}$"
        "\n"
        rf"$n={n:.4f}\pm{sn:.4f}$"
    )

    ax.plot(Tcurva, ycurva, lw=2, label=label)

    ax.set_xlabel("T [°C]")
    ax.set_ylabel("Ch2 [mV]")

    ax.legend()
    ax.grid(True)

    fig.suptitle(titulo)

    if mostrar:
        plt.show()

    return fig

def analizar_histeresis(
    temperatura,
    frecuencia,
    sampling=100e3,
    error_relativo=0.05,
    mostrar=True
):
    """
    Procesa una medición de histéresis.

    Parámetros
    ----------
    temperatura : int o float
        Temperatura que aparece en el nombre del archivo.
        Ej: 21 -> histeresis_21C_100ks.txt

    frecuencia : float
        Frecuencia de excitación [Hz].

    sampling : float
        Frecuencia de muestreo [Hz].

    error_relativo : float
        Error instrumental relativo.

    mostrar : bool
        Si True grafica el ciclo promedio.

    Retorna
    -------
    dict
    """

    # ==========================
    # Lectura
    # ==========================

    archivo = (
        rf".\Experimento IV, Ferromagnetico\Codigo\Datos"
        rf"\histeresis_{temperatura}C_100ks.txt"
    )

    data = np.loadtxt(archivo, skiprows=4)

    # ==========================
    # Canales
    # ==========================

    ch1 = data[:, 1]
    ch2 = data[:, 2]

    # ==========================
    # Período
    # ==========================

    N_periodo = round(sampling / frecuencia)

    N_total = len(ch1)
    N_ciclos = N_total // N_periodo
    N_util = N_ciclos * N_periodo

    print(f"\n===== {temperatura} °C =====")
    print(f"Frecuencia = {frecuencia:.3f} Hz")
    print(f"Muestras por período = {N_periodo}")
    print(f"Ciclos completos = {N_ciclos}")

    # ==========================
    # Recorte
    # ==========================

    ch1 = ch1[:N_util]
    ch2 = ch2[:N_util]

    # ==========================
    # Separación en ciclos
    # ==========================

    ciclos_ch1 = ch1.reshape(N_ciclos, N_periodo)
    ciclos_ch2 = ch2.reshape(N_ciclos, N_periodo)

    # ==========================
    # Promedio
    # ==========================

    ch1_prom = np.mean(ciclos_ch1, axis=0)
    ch2_prom = np.mean(ciclos_ch2, axis=0)

    # ==========================
    # Dispersión entre ciclos
    # ==========================

    err_ch1_stat = np.std(ciclos_ch1, axis=0, ddof=1)
    err_ch2_stat = np.std(ciclos_ch2, axis=0, ddof=1)

    # ==========================
    # Error del promedio
    # ==========================

    err_ch1_prom = err_ch1_stat / np.sqrt(N_ciclos)
    err_ch2_prom = err_ch2_stat / np.sqrt(N_ciclos)

    # ==========================
    # Error instrumental
    # ==========================

    err_ch1_inst = error_relativo * np.abs(ch1_prom)
    err_ch2_inst = error_relativo * np.abs(ch2_prom)

    # ==========================
    # Error total
    # ==========================

    err_ch1 = np.sqrt(err_ch1_prom**2 + err_ch1_inst**2)
    err_ch2 = np.sqrt(err_ch2_prom**2 + err_ch2_inst**2)

    # ==========================
    # Información del primer punto
    # ==========================

    print("\nPrimer punto:")

    print("Valores usados:")
    print(ciclos_ch2[:, 0])

    print(f"Promedio           = {ch2_prom[0]:.4f}")
    print(f"Desviación         = {err_ch2_stat[0]:.4f}")
    print(f"Error del promedio = {err_ch2_prom[0]:.4f}")

    # ==========================
    # Gráfico
    # ==========================

    if mostrar:

        plt.figure(figsize=(7, 5))

        plt.errorbar(
            ch1_prom,
            ch2_prom,
            xerr=err_ch1,
            yerr=err_ch2,
            fmt="o",
            ms=3,
            capsize=2,
            label=f"{N_ciclos} ciclos promediados"
        )

        plt.title(f"Histéresis promedio - {temperatura} °C")
        plt.xlabel("CH1")
        plt.ylabel("CH2")
        plt.grid(True)
        plt.legend()

        plt.show()

    return {
        "temperatura": temperatura,
        "frecuencia": frecuencia,
        "N_periodo": N_periodo,
        "N_ciclos": N_ciclos,
        "ch1_prom": ch1_prom,
        "ch2_prom": ch2_prom,
        "err_ch1": err_ch1,
        "err_ch2": err_ch2,
        "ciclos_ch1": ciclos_ch1,
        "ciclos_ch2": ciclos_ch2,
    }

def graficar_histeresis(*resultados):

    plt.figure(figsize=(8, 6))

    for res in resultados:

        plt.scatter(
            res["ch1_prom"],
            res["ch2_prom"],
            s=10,
            label=f'{res["temperatura"]} °C'
        )

    plt.xlabel("CH1 (∝ H)")
    plt.ylabel("CH2 (∝ M)")
    plt.title("Ciclos de histéresis promediados")
    plt.grid(True)
    plt.legend()

    plt.show()

def graficar_histeresis_error(*resultados):

    plt.figure(figsize=(8, 6))

    for res in resultados:

        plt.errorbar(
            res["ch1_prom"],
            res["ch2_prom"],
            xerr=res["err_ch1"],
            yerr=res["err_ch2"],
            fmt='o',
            ms=3,
            capsize=2,
            label=f'{res["temperatura"]} °C'
        )

    plt.xlabel("CH1 (∝ H)")
    plt.ylabel("CH2 (∝ M)")
    plt.title("Ciclos de histéresis promediados")
    plt.grid(True)
    plt.legend()

    plt.show()

def graficar_histeresis_no_scatter(*resultados):

    plt.figure(figsize=(8,6))

    for res in sorted(resultados,
                      key=lambda x: x["temperatura"]):

        plt.plot(
            res["ch1_prom"],
            res["ch2_prom"],
            'o-',
            ms=2,
            linewidth=1,
            label=f'{res["temperatura"]} °C'
        )

    plt.xlabel("CH1 (∝ H)")
    plt.ylabel("CH2 (∝ M)")
    plt.title("Comparación de ciclos de histéresis")
    plt.grid(True)
    plt.legend(title="Temperatura")

    plt.show()


def graficar_ciclo_individual(res, ciclo=0):

    plt.figure(figsize=(7,5))

    plt.plot(
        res["ciclos_ch1"][ciclo],
        res["ciclos_ch2"][ciclo],
        "o-",
        ms=3
    )

    plt.xlabel("CH1 (∝ H)")
    plt.ylabel("CH2 (∝ M)")
    plt.title(
        f'{res["temperatura"]} °C - ciclo {ciclo}'
    )

    plt.grid(True)
    plt.show()


def graficar_todos_los_ciclos(res):

    plt.figure(figsize=(7,5))

    for i in range(res["N_ciclos"]):

        plt.plot(
            res["ciclos_ch1"][i],
            res["ciclos_ch2"][i],
            alpha=0.4,
            lw=1
        )

    plt.xlabel("CH1 (∝ H)")
    plt.ylabel("CH2 (∝ M)")
    plt.title(
        f'{res["temperatura"]} °C - todos los ciclos'
    )

    plt.grid(True)

    plt.show()

def verificar_periodo(res):

    plt.figure(figsize=(8,5))

    for i in range(min(5, res["N_ciclos"])):

        plt.plot(
            res["ciclos_ch1"][i],
            label=f"Ciclo {i}"
        )

    plt.title(
        f'{res["temperatura"]} °C - CH1 de los primeros ciclos'
    )

    plt.xlabel("Índice dentro del período")
    plt.ylabel("CH1")

    plt.legend()
    plt.grid(True)

    plt.show()

def verificar_ciclos(res):

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8,6),
        sharex=True
    )

    for i in range(min(5, res["N_ciclos"])):

        ax1.plot(
            res["ciclos_ch1"][i],
            label=f"Ciclo {i}"
        )

        ax2.plot(
            res["ciclos_ch2"][i],
            label=f"Ciclo {i}"
        )

    ax1.set_title("CH1")
    ax2.set_title("CH2")

    ax1.grid()
    ax2.grid()

    ax1.legend()

    plt.show()

def ajuste_recta(x, y, sy):
    """
    Ajuste lineal ponderado:
        y = a + b x

    Parámetros
    ----------
    x, y : arrays
    sy   : errores de y

    Retorna
    -------
    a, sa, b, sb
    """

    w = 1 / sy**2

    S    = np.sum(w)
    Sx   = np.sum(w * x)
    Sy   = np.sum(w * y)
    Sxx  = np.sum(w * x**2)
    Sxy  = np.sum(w * x * y)

    Delta = S * Sxx - Sx**2

    a = (Sxx * Sy - Sx * Sxy) / Delta
    b = (S * Sxy - Sx * Sy) / Delta

    sa = np.sqrt(Sxx / Delta)
    sb = np.sqrt(S / Delta)

    return a, sa, b, sb

def extraer_coercividad(ch1, ch2, n_puntos=3):

    # ==========================
    # Tomar solo media histéresis
    # ==========================

    mitad = len(ch1) // 2

    H = ch1[mitad:]
    M = ch2[mitad:]

    # ==========================
    # Solo H positivos
    # ==========================

    mask = H > 0

    H = H[mask]
    M = M[mask]

    # ==========================
    # Puntos más cercanos a M=0
    # ==========================

    idx = np.argsort(np.abs(M))[:n_puntos]

    H_fit = H[idx]
    M_fit = M[idx]

    # ==========================
    # Ajuste lineal
    # ==========================

    sy = 0.05 * np.abs(M_fit)
    sy[sy == 0] = 1

    a, sa, b, sb = ajuste_recta(H_fit, M_fit, sy)

    Hc = -a / b

    return Hc, H_fit, M_fit

def graficar_coercividad(ch1, ch2, n_puntos=3):

    mitad = len(ch1) // 2

    H = ch1[mitad:]
    M = ch2[mitad:]

    mask = H > 0

    H = H[mask]
    M = M[mask]

    idx = np.argsort(np.abs(M))[:n_puntos]

    H_fit = H[idx]
    M_fit = M[idx]

    sy = 0.05 * np.abs(M_fit)
    sy[sy == 0] = 1

    a, sa, b, sb = ajuste_recta(H_fit, M_fit, sy)

    Hc = -a / b

    H_recta = np.linspace(
        H_fit.min() - 5,
        H_fit.max() + 5,
        100
    )

    M_recta = a + b * H_recta

    plt.figure(figsize=(7,5))

    # Histéresis completa
    plt.scatter(
        ch1,
        ch2,
        s=10,
        alpha=0.3,
        label="Histéresis completa"
    )

    # Media histéresis utilizada
    plt.scatter(
        H,
        M,
        s=15,
        label="Datos usados"
    )

    # Puntos del ajuste
    plt.scatter(
        H_fit,
        M_fit,
        s=100,
        label="Puntos seleccionados"
    )

    plt.plot(
        H_recta,
        M_recta,
        linewidth=2,
        label="Ajuste lineal"
    )

    plt.axvline(
        Hc,
        linestyle="--",
        label=fr"$H_c={Hc:.2f}$"
    )

    plt.axhline(
        0,
        color="black",
        linewidth=0.8
    )

    plt.xlabel("CH1 ∝ H")
    plt.ylabel("CH2 ∝ M")

    plt.legend()
    plt.grid()

    plt.show()

    return Hc