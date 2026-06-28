import numpy as np
from scipy.odr import ODR, RealData, Model
import matplotlib.pyplot as plt
import pandas as pd

def filtrar_puntos(
    df,
    ignorar=None,
    col_fecha="fecha",
    col_idx="idx_dia"
):
    df = df.copy()

    if ignorar is None:
        return df

    if len(ignorar) == 0:
        return df

    # Caso varios días:
    # ignorar = [("23/06", 4), ("25/06", 7)]
    if isinstance(ignorar[0], tuple):
        mask = pd.Series(True, index=df.index)

        for fecha, idx in ignorar:
            mask &= ~(
                (df[col_fecha] == fecha) &
                (df[col_idx] == idx)
            )

        return df.loc[mask].copy()

    # Caso un solo día:
    # ignorar = [4, 7, 12]
    return df.loc[~df[col_idx].isin(ignorar)].copy()


# =========================================================================
# ASIGNAR ERRORES
# =========================================================================

def asignar_errores(df, configs, col_fecha="fecha", col_f="f"):
    df = df.copy()

    df["err_f"] = np.nan
    df["err_Ve"] = np.nan
    df["err_Vs"] = np.nan

    for fecha, cfg in configs.items():
        mask_fecha = df[col_fecha] == fecha

        limites = cfg["limites"]

        err_f_rel = cfg["err_f"]
        err_Ve_rel = cfg["err_Ve"]
        err_Vs_rel = cfg["err_Vs"]

        bins = [0] + limites + [np.inf]
        labels = range(len(bins) - 1)

        rangos = pd.cut(
            df.loc[mask_fecha, col_f],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True
        )

        porcentaje_f = rangos.map(dict(zip(labels, err_f_rel))).astype(float)
        porcentaje_Ve = rangos.map(dict(zip(labels, err_Ve_rel))).astype(float)
        porcentaje_Vs = rangos.map(dict(zip(labels, err_Vs_rel))).astype(float)

        df.loc[mask_fecha, "err_f"] = porcentaje_f * df.loc[mask_fecha, "f"]
        df.loc[mask_fecha, "err_Ve"] = porcentaje_Ve * df.loc[mask_fecha, "Ve"]
        df.loc[mask_fecha, "err_Vs"] = porcentaje_Vs * df.loc[mask_fecha, "Vs"]

    df["idx_dia"] = df.groupby(col_fecha).cumcount()

    return df


# =========================================================================
# GRAFICAR DATOS
# =========================================================================

def graficar_datos(
    df,
    x,
    y,
    xerr=None,
    yerr=None,
    col_fecha="fecha",
    col_idx="idx_dia",
    anotar=True,
    escala_x=None,
    escala_y=None,
    xlabel=None,
    ylabel=None,
    titulo=None,
    figsize=(8, 5),
    capsize=3,
    ms=5,
    ignorar=None
):
    df = filtrar_puntos(
        df,
        ignorar=ignorar,
        col_fecha=col_fecha,
        col_idx=col_idx
    )

    fig, ax = plt.subplots(figsize=figsize)

    for fecha, grupo in df.groupby(col_fecha):

        ax.errorbar(
            grupo[x],
            grupo[y],
            xerr=grupo[xerr] if xerr is not None else None,
            yerr=grupo[yerr] if yerr is not None else None,
            fmt="o",
            ms=ms,
            capsize=capsize,
            linestyle="none"
        )

        if anotar:
            for _, fila in grupo.iterrows():
                etiqueta = f"{int(fila[col_idx])}"

                ax.annotate(
                    etiqueta,
                    (fila[x], fila[y]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8
                )

    if escala_x is not None:
        ax.set_xscale(escala_x)

    if escala_y is not None:
        ax.set_yscale(escala_y)

    ax.set_xlabel(xlabel if xlabel is not None else x)
    ax.set_ylabel(ylabel if ylabel is not None else y)

    if titulo is not None:
        ax.set_title(titulo)

    ax.grid(True, which="both", alpha=0.4)

    plt.tight_layout()
    plt.show()

    return fig, ax


# =========================================================================
# MODELOS CPE
# =========================================================================

def modelo_cpe(beta, re_z):
    n = beta[0]
    return re_z * np.tan(np.pi*n/2)


def modelo_cpe_ordenada(beta, re_z):
    n, b = beta
    return re_z * np.tan(np.pi*n/2) + b


# =========================================================================
# AJUSTES ODR
# =========================================================================

def ajustar_n_cpe(
    re_z,
    im_z,
    sre_z=None,
    sim_z=None,
    n0=0.8
):
    re_z = np.asarray(re_z, dtype=float)
    im_z = np.asarray(im_z, dtype=float)

    if sre_z is None:
        sre_z = np.ones_like(re_z)
    else:
        sre_z = np.asarray(sre_z, dtype=float)

    if sim_z is None:
        sim_z = np.ones_like(im_z)
    else:
        sim_z = np.asarray(sim_z, dtype=float)

    data = RealData(
        x=re_z,
        y=im_z,
        sx=sre_z,
        sy=sim_z
    )

    model = Model(modelo_cpe)

    odr = ODR(
        data,
        model,
        beta0=[n0]
    )

    return odr.run()


def ajustar_n_cpe_ordenada(
    re_z,
    im_z,
    sre_z=None,
    sim_z=None,
    n0=0.8,
    b0=0.0
):
    re_z = np.asarray(re_z, dtype=float)
    im_z = np.asarray(im_z, dtype=float)

    if sre_z is None:
        sre_z = np.ones_like(re_z)
    else:
        sre_z = np.asarray(sre_z, dtype=float)

    if sim_z is None:
        sim_z = np.ones_like(im_z)
    else:
        sim_z = np.asarray(sim_z, dtype=float)

    data = RealData(
        x=re_z,
        y=im_z,
        sx=sre_z,
        sy=sim_z
    )

    model = Model(modelo_cpe_ordenada)

    odr = ODR(
        data,
        model,
        beta0=[n0, b0]
    )

    return odr.run()


# =========================================================================
# AJUSTE CPE DESDE DATAFRAME
# =========================================================================

def ajustar_cpe_df(
    df,
    re_min=1000,
    re_max=2500,
    col_re="ReZ",
    col_im="ImZ",
    col_sre="err_ReZ",
    col_sim="err_ImZ",
    col_fecha="fecha",
    col_idx="idx_dia",
    con_ordenada=True,
    n0=0.8,
    b0=0.0,
    anotar=True,
    mostrar_todos=True,
    ignorar=None,
    ax=None
):
    df_original = df.copy()

    df = filtrar_puntos(
        df,
        ignorar=ignorar,
        col_fecha=col_fecha,
        col_idx=col_idx
    )

    mask_rango = (
        (df[col_re] >= re_min) &
        (df[col_re] <= re_max)
    )

    df_fit = df.loc[mask_rango].copy()

    columnas = [col_re, col_im, col_sre, col_sim]

    mask_finitos = np.ones(len(df_fit), dtype=bool)

    for col in columnas:
        mask_finitos &= np.isfinite(df_fit[col].to_numpy())

    mask_finitos &= df_fit[col_sre].to_numpy() > 0
    mask_finitos &= df_fit[col_sim].to_numpy() > 0

    df_fit = df_fit.loc[mask_finitos].copy()

    if len(df_fit) < 2:
        raise ValueError("Hay menos de 2 puntos en el rango elegido. No se puede ajustar.")

    re_z = df_fit[col_re].to_numpy()
    im_z = df_fit[col_im].to_numpy()
    sre_z = df_fit[col_sre].to_numpy()
    sim_z = df_fit[col_sim].to_numpy()

    if con_ordenada:
        resultado = ajustar_n_cpe_ordenada(
            re_z,
            im_z,
            sre_z=sre_z,
            sim_z=sim_z,
            n0=n0,
            b0=b0
        )

        n, b = resultado.beta
        sn, sb = resultado.sd_beta

    else:
        resultado = ajustar_n_cpe(
            re_z,
            im_z,
            sre_z=sre_z,
            sim_z=sim_z,
            n0=n0
        )

        n = resultado.beta[0]
        sn = resultado.sd_beta[0]
        b = 0.0
        sb = 0.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    if mostrar_todos:
        ax.errorbar(
            df_original[col_re],
            df_original[col_im],
            xerr=df_original[col_sre],
            yerr=df_original[col_sim],
            fmt="o",
            ms=4,
            capsize=2,
            alpha=0.20,
            linestyle="none",
            label="Todos los datos"
        )

    ax.errorbar(
        re_z,
        im_z,
        xerr=sre_z,
        yerr=sim_z,
        fmt="o",
        ms=6,
        capsize=3,
        linestyle="none",
        label=rf"Datos ajustados: {re_min} $\leq$ ReZ $\leq$ {re_max}"
    )

    if anotar:
        for _, fila in df_fit.iterrows():
            etiqueta = f"{int(fila[col_idx])}"

            ax.annotate(
                etiqueta,
                (fila[col_re], fila[col_im]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8
            )

    xfit = np.linspace(np.min(re_z), np.max(re_z), 500)

    if con_ordenada:
        yfit = modelo_cpe_ordenada([n, b], xfit)

        label_fit = (
            rf"Ajuste CPE + b" "\n"
            rf"$n = {n:.4f} \pm {sn:.4f}$" "\n"
            rf"$b = {b:.3e} \pm {sb:.3e}$" "\n"
            rf"$\chi^2_\nu = {resultado.res_var:.3f}$"
        )

    else:
        yfit = modelo_cpe([n], xfit)

        label_fit = (
            rf"Ajuste CPE" "\n"
            rf"$n = {n:.4f} \pm {sn:.4f}$" "\n"
            rf"$\chi^2_\nu = {resultado.res_var:.3f}$"
        )

    ax.plot(
        xfit,
        yfit,
        "-",
        label=label_fit
    )

    ax.set_xlabel(r"$\mathrm{Re}(Z)$ [$\Omega$]")
    ax.set_ylabel(r"$\mathrm{-Im}(Z)$ [$\Omega$]")
    ax.set_title("Ajuste de elemento de fase constante")
    ax.grid(True, alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.show()

    print("===================================")
    print("Ajuste CPE")
    print("===================================")
    print(f"Rango usado: {re_min} <= ReZ <= {re_max}")
    print(f"Número de puntos usados: {len(df_fit)}")
    print(f"n = {n:.6f} ± {sn:.6f}")

    if con_ordenada:
        print(f"b = {b:.6e} ± {sb:.6e}")

    print(f"chi2_red = {resultado.res_var:.6f}")
    print("===================================")

    print("Puntos usados:")
    print(df_fit[[col_fecha, col_idx, col_re, col_im, col_sre, col_sim]])

    return resultado, df_fit

def graficar_bode_impedancia(df):
    """
    Grafica el Diagrama de Bode modificado: 
    Eje Y izquierdo: Impedancia Z (Escala Logarítmica)
    Eje Y derecho: Fase phi (Escala Lineal)
    Eje X: Frecuencia angular omega (Escala Logarítmica)
    Incluye barras de error para todas las variables.
    """
    # Creamos la figura y el primer eje (Izquierdo - Impedancia)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # --- GRÁFICO DE IMPEDANCIA (Z) ---
    color_Z = 'tab:blue'
    ax1.set_xlabel(r'$\omega$ [rad/s]', fontsize=12)
    ax1.set_ylabel(r'Impedancia $|Z|$ [$\Omega$]', color=color_Z, fontsize=12)
    
    # errorbar grafica los puntos y sus incertezas al mismo tiempo
    ax1.errorbar(
        df["w"], df["Z"], 
        xerr=df["err_w"], yerr=df["err_Z"], 
        fmt='o', color=color_Z, edgecolor='black', markersize=5,
        capsize=3, elinewidth=1.2, label=r'$|Z|$'
    )
    ax1.tick_params(axis='y', labelcolor=color_Z)
    ax1.set_xscale('log')
    ax1.set_yscale('log') # Impedancia en escala logarítmica
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # --- GRÁFICO DE FASE (phi) ---
    # twinx() crea un segundo eje Y que comparte el mismo eje X logarítmico
    ax2 = ax1.twinx()  
    
    color_phi = 'tab:red'
    ax2.set_ylabel(r'Fase $\phi$ [rad]', color=color_phi, fontsize=12)
    
    ax2.errorbar(
        df["w"], df["phi"], 
        xerr=df["err_w"], yerr=df["err_phi"], 
        fmt='s', color=color_phi, edgecolor='black', markersize=5,
        capsize=3, elinewidth=1.2, label=r'$\phi$'
    )
    ax2.tick_params(axis='y', labelcolor=color_phi)
    # La fase queda en escala lineal por defecto

    # --- DETALLES FINALES ---
    plt.title('Diagrama de Bode: Impedancia y Fase en función de $\omega$', fontsize=14, pad=15)
    fig.tight_layout() # Evita que se solapen las etiquetas de los ejes
    
    # Mostramos el gráfico
    plt.show()