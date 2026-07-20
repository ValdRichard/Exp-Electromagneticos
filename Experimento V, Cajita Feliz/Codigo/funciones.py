import numpy as np
from scipy.odr import ODR, RealData, Model
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import least_squares
def armar_df_medicion(fecha, f, Ve, Vs, dt, err_dt, R, configs):
    df = pd.DataFrame({
        "fecha": fecha,
        "f": f,
        "Ve": Ve,
        "Vs": Vs,
        "dt": dt,
        "err_dt": err_dt
    })

    df = asignar_errores(df, configs)

    df["w"] = 2*np.pi*df["f"]
    df["H"] = df["Vs"] / df["Ve"]
    df["Z"] = (df["Ve"] / df["Vs"]) * R
    df["phi"] = -df["w"] * df["dt"]

    df["ReZ"] = df["Z"] * np.cos(df["phi"])
    df["ImZ"] = df["Z"] * np.sin(df["phi"])
    df["-ImZ"] = -df["ImZ"]

    df["err_w"] = 2*np.pi*df["err_f"]

    df["err_H"] = np.abs(df["H"]) * np.sqrt(
        (df["err_Ve"]/df["Ve"])**2 +
        (df["err_Vs"]/df["Vs"])**2
    )

    df["err_Z"] = np.abs(df["Z"]) * np.sqrt(
        (df["err_Ve"]/df["Ve"])**2 +
        (df["err_Vs"]/df["Vs"])**2
    )

    df["err_phi"] = np.abs(df["phi"]) * np.sqrt(
        (df["err_w"]/df["w"])**2 +
        (df["err_dt"]/df["dt"])**2
    )

    df["err_ReZ"] = np.sqrt(
        (np.cos(df["phi"]) * df["err_Z"])**2 +
        (df["Z"] * np.sin(df["phi"]) * df["err_phi"])**2
    )

    df["err_ImZ"] = np.sqrt(
        (np.sin(df["phi"]) * df["err_Z"])**2 +
        (df["Z"] * np.cos(df["phi"]) * df["err_phi"])**2
    )

    return df

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
        fmt='o', color=color_Z, markeredgecolor=None, markersize=5,
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
        fmt='s', color=color_phi, markeredgecolor=None, markersize=5,
        capsize=3, elinewidth=1.2, label=r'$\phi$'
    )
    ax2.tick_params(axis='y', labelcolor=color_phi)
    # La fase queda en escala lineal por defecto

    # --- DETALLES FINALES ---
    plt.title('Diagrama de Bode: Impedancia y Fase en función de $\omega$', fontsize=14, pad=15)
    fig.tight_layout() # Evita que se solapen las etiquetas de los ejes
    ax1.set_xlim(39, 150000)
    
    # Mostramos el gráfico
    plt.show()
def ajustar_cpe_nyquist_omega(
    df,
    w_min,
    w_max,
    col_w="w",
    col_re="ReZ",
    col_im="-ImZ",
    col_sre="err_ReZ",
    col_sim="err_ImZ",
    col_fecha="fecha",
    col_idx="idx_dia",
    con_ordenada=True,
    n0=0.8,
    b0=0.0,
    ignorar=None,
    anotar=True,
    separar_por_fecha=True
):
    """
    Ajusta un CPE desde Nyquist usando solo los puntos con:

        w_min <= omega <= w_max

    Si separar_por_fecha=True, grafica los puntos usados separados por fecha.
    Si separar_por_fecha=False, grafica todos los puntos juntos.
    """

    df = filtrar_puntos(
        df,
        ignorar=ignorar,
        col_fecha=col_fecha,
        col_idx=col_idx
    )

    mask_rango = (
        (df[col_w] >= w_min) &
        (df[col_w] <= w_max)
    )

    df_fit = df.loc[mask_rango].copy()

    columnas = [col_w, col_re, col_im, col_sre, col_sim]

    mask_finitos = np.ones(len(df_fit), dtype=bool)

    for col in columnas:
        mask_finitos &= np.isfinite(df_fit[col].to_numpy())

    mask_finitos &= df_fit[col_sre].to_numpy() > 0
    mask_finitos &= df_fit[col_sim].to_numpy() > 0

    df_fit = df_fit.loc[mask_finitos].copy()

    if len(df_fit) < 2:
        raise ValueError("Hay menos de 2 puntos en el rango de omega elegido.")

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

    fig, ax = plt.subplots(figsize=(7, 6))

    # =========================================================================
    # GRAFICAR DATOS USADOS EN EL AJUSTE
    # =========================================================================

    if separar_por_fecha:
        for fecha, grupo in df_fit.groupby(col_fecha):
            ax.errorbar(
                grupo[col_re],
                grupo[col_im],
                xerr=grupo[col_sre],
                yerr=grupo[col_sim],
                fmt="o",
                ms=6,
                capsize=3,
                linestyle="none",
                label=f"Datos {fecha}"
            )
    else:
        ax.errorbar(
            re_z,
            im_z,
            xerr=sre_z,
            yerr=sim_z,
            fmt="o",
            ms=6,
            capsize=3,
            linestyle="none",
            label="Datos ajustados"
        )

    if anotar:
        for _, fila in df_fit.iterrows():
            ax.annotate(
                f"{int(fila[col_idx])}",
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
    ax.set_ylabel(r"$-\mathrm{Im}(Z)$ [$\Omega$]")
    ax.set_title(
        rf"Ajuste CPE desde Nyquist: ${w_min} \leq \omega \leq {w_max}$"
    )
    ax.grid(True, alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.show()

    print("===================================")
    print("Ajuste CPE desde Nyquist")
    print("===================================")
    print(f"Rango usado: {w_min} <= omega <= {w_max}")
    print(f"Número de puntos usados: {len(df_fit)}")
    print(f"n = {n:.6f} ± {sn:.6f}")

    if con_ordenada:
        print(f"b = {b:.6e} ± {sb:.6e}")

    print(f"chi2_red = {resultado.res_var:.6f}")
    print("===================================")

    print("Puntos usados:")
    print(df_fit[[col_fecha, col_idx, col_w, col_re, col_im, col_sre, col_sim]])

    return resultado, df_fit
def ajustar_cpe_fase_omega(
    df,
    w_min,
    w_max,
    col_w="w",
    col_phi="phi",
    col_sphi="err_phi",
    col_fecha="fecha",
    col_idx="idx_dia",
    ignorar=None,
    anotar=True,
    separar_por_fecha=True
):
    """
    Ajusta la fase en un rango de omega suponiendo CPE:

        phi(omega) = phi0

    Luego calcula:

        n = -2*phi0/pi

    Si separar_por_fecha=True, grafica los puntos usados separados por fecha.
    Si separar_por_fecha=False, grafica todos los puntos juntos.
    """

    df = filtrar_puntos(
        df,
        ignorar=ignorar,
        col_fecha=col_fecha,
        col_idx=col_idx
    )

    mask_rango = (
        (df[col_w] >= w_min) &
        (df[col_w] <= w_max)
    )

    df_fit = df.loc[mask_rango].copy()

    columnas = [col_w, col_phi, col_sphi]

    mask_finitos = np.ones(len(df_fit), dtype=bool)

    for col in columnas:
        mask_finitos &= np.isfinite(df_fit[col].to_numpy())

    mask_finitos &= df_fit[col_sphi].to_numpy() > 0

    df_fit = df_fit.loc[mask_finitos].copy()

    if len(df_fit) < 2:
        raise ValueError("Hay menos de 2 puntos en el rango de omega elegido.")

    w = df_fit[col_w].to_numpy()
    phi = df_fit[col_phi].to_numpy()
    sphi = df_fit[col_sphi].to_numpy()

    pesos = 1 / sphi**2

    phi0 = np.sum(pesos * phi) / np.sum(pesos)
    err_phi0 = np.sqrt(1 / np.sum(pesos))

    residuos = phi - phi0
    chi2 = np.sum((residuos / sphi)**2)
    chi2_red = chi2 / (len(phi) - 1)

    n = -2 * phi0 / np.pi
    err_n = 2 * err_phi0 / np.pi

    fig, ax = plt.subplots(figsize=(8, 5))

    # =========================================================================
    # GRAFICAR DATOS USADOS EN EL AJUSTE
    # =========================================================================

    if separar_por_fecha:
        for fecha, grupo in df_fit.groupby(col_fecha):
            ax.errorbar(
                grupo[col_w],
                grupo[col_phi],
                xerr=grupo["err_w"] if "err_w" in grupo.columns else None,
                yerr=grupo[col_sphi],
                fmt="o",
                ms=6,
                capsize=3,
                linestyle="none",
                label=f"Datos {fecha}"
            )
    else:
        ax.errorbar(
            w,
            phi,
            xerr=df_fit["err_w"] if "err_w" in df_fit.columns else None,
            yerr=sphi,
            fmt="o",
            ms=6,
            capsize=3,
            linestyle="none",
            label="Datos ajustados"
        )

    if anotar:
        for _, fila in df_fit.iterrows():
            ax.annotate(
                f"{int(fila[col_idx])}",
                (fila[col_w], fila[col_phi]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8
            )

    wfit = np.linspace(np.min(w), np.max(w), 500)
    phifit = np.full_like(wfit, phi0)

    label_fit = (
        rf"Ajuste $\phi = \phi_0$" "\n"
        rf"$\phi_0 = {phi0:.4f} \pm {err_phi0:.4f}$ rad" "\n"
        rf"$n = {n:.4f} \pm {err_n:.4f}$" "\n"
        rf"$\chi^2_\nu = {chi2_red:.3f}$"
    )

    ax.plot(
        wfit,
        phifit,
        "-",
        label=label_fit
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel(r"$\phi$ [rad]")
    ax.set_title(
        rf"Ajuste de fase CPE: ${w_min} \leq \omega \leq {w_max}$"
    )
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.show()

    print("===================================")
    print("Ajuste CPE desde fase")
    print("===================================")
    print(f"Rango usado: {w_min} <= omega <= {w_max}")
    print(f"Número de puntos usados: {len(df_fit)}")
    print(f"phi0 = {phi0:.6f} ± {err_phi0:.6f} rad")
    print(f"n = {n:.6f} ± {err_n:.6f}")
    print(f"chi2_red = {chi2_red:.6f}")
    print("===================================")

    print("Puntos usados:")
    print(df_fit[[col_fecha, col_idx, col_w, col_phi, col_sphi]])

    return {
        "phi0": phi0,
        "err_phi0": err_phi0,
        "n": n,
        "err_n": err_n,
        "chi2_red": chi2_red,
        "df_fit": df_fit
    }


# =========================================================================
# MODELO: Rs + (Rp || Cp) + CPE + Cs
# =========================================================================

def impedancia_modelo_rc_cpe_cs(w, Rs, Rp, Cp, Q, n, Cs):
    """
    Modelo:

        Rs - (Rp || Cp) - CPE - Cs

    Todo en serie.

    Parámetros:
        Rs : resistencia serie [ohm]
        Rp : resistencia del paralelo [ohm]
        Cp : capacitor del paralelo [F]
        Q  : constante del CPE [s^n / ohm]
        n  : exponente del CPE
        Cs : capacitor en serie [F]
    """

    w = np.asarray(w, dtype=float)
    jw = 1j * w

    Z_Rs = Rs

    Z_RpCp = 1 / (1/Rp + jw*Cp)

    Z_CPE = 1 / (Q * jw**n)

    Z_Cs = 1 / (jw * Cs)

    Z_total = Z_Rs + Z_RpCp + Z_CPE + Z_Cs

    return Z_total

def ajustar_modelo_rc_cpe_cs(
    df,
    w_min,
    w_max,
    col_w="w",
    col_re="ReZ",
    col_im="-ImZ",
    col_sre="err_ReZ",
    col_sim="err_ImZ",
    col_fecha="fecha",
    col_idx="idx_dia",
    ignorar=None,
    p0=None,
    n_fijo=None,
    Q_fijo=None,
    anotar=True,
    graficar=True
):
    """
    Ajusta el modelo:

        Z = Rs + (Rp || Cp) + CPE + Cs

    contra los datos de Nyquist:

        x = ReZ
        y = -ImZ

    usando el rango:

        w_min <= omega <= w_max

    Si n_fijo no es None, deja fijo n.
    Si Q_fijo no es None, deja fijo Q.
    """

    df = filtrar_puntos(
        df,
        ignorar=ignorar,
        col_fecha=col_fecha,
        col_idx=col_idx
    )

    mask_rango = (
        (df[col_w] >= w_min) &
        (df[col_w] <= w_max)
    )

    df_fit = df.loc[mask_rango].copy()

    columnas = [col_w, col_re, col_im, col_sre, col_sim]

    mask_finitos = np.ones(len(df_fit), dtype=bool)

    for col in columnas:
        mask_finitos &= np.isfinite(df_fit[col].to_numpy())

    mask_finitos &= df_fit[col_sre].to_numpy() > 0
    mask_finitos &= df_fit[col_sim].to_numpy() > 0

    df_fit = df_fit.loc[mask_finitos].copy()

    if len(df_fit) < 4:
        raise ValueError("Hay pocos puntos para ajustar este modelo.")

    w = df_fit[col_w].to_numpy()
    re_exp = df_fit[col_re].to_numpy()
    mim_exp = df_fit[col_im].to_numpy()      # esto es -ImZ

    sre = df_fit[col_sre].to_numpy()
    sim = df_fit[col_sim].to_numpy()

    # =========================================================
    # Estimaciones iniciales razonables
    # =========================================================

    w_med = np.exp(np.mean(np.log(w)))
    Zabs_med = np.median(np.sqrt(re_exp**2 + mim_exp**2))

    Rs0 = max(1e-6, np.min(re_exp))
    Rp0 = max(1e-6, np.max(re_exp) - Rs0)

    if Rp0 <= 0:
        Rp0 = max(1.0, Zabs_med)

    Cp0 = 1 / (w_med * Rp0)
    n0 = 0.8
    Q0 = 1 / (Zabs_med * w_med**n0)
    Cs0 = 1 / (w_med * Zabs_med)

    valores_iniciales = {
        "Rs": Rs0,
        "Rp": Rp0,
        "Cp": Cp0,
        "Q": Q0,
        "n": n0,
        "Cs": Cs0
    }

    if p0 is not None:
        valores_iniciales.update(p0)

    if n_fijo is not None:
        valores_iniciales["n"] = n_fijo

    if Q_fijo is not None:
        valores_iniciales["Q"] = Q_fijo

    # Parámetros positivos que conviene ajustar en logaritmo
    nombres_positivos = ["Rs", "Rp", "Cp", "Q", "Cs"]

    nombres_libres = []

    for nombre in nombres_positivos:
        if nombre == "Q" and Q_fijo is not None:
            continue
        nombres_libres.append(nombre)

    if n_fijo is None:
        nombres_libres.append("n")

    # =========================================================
    # Conversión entre vector de ajuste y diccionario físico
    # =========================================================

    def diccionario_a_vector(params):
        u = []

        for nombre in nombres_libres:
            valor = params[nombre]

            if nombre in nombres_positivos:
                u.append(np.log(valor))
            else:
                u.append(valor)

        return np.array(u, dtype=float)


    def vector_a_diccionario(u):
        params = valores_iniciales.copy()

        for valor, nombre in zip(u, nombres_libres):
            if nombre in nombres_positivos:
                params[nombre] = np.exp(valor)
            else:
                params[nombre] = valor

        if n_fijo is not None:
            params["n"] = n_fijo

        if Q_fijo is not None:
            params["Q"] = Q_fijo

        return params


    u0 = diccionario_a_vector(valores_iniciales)

    # =========================================================
    # Límites
    # =========================================================

    lower = []
    upper = []

    for nombre in nombres_libres:
        if nombre == "Rs":
            lower.append(np.log(1e-12))
            upper.append(np.log(1e12))

        elif nombre == "Rp":
            lower.append(np.log(1e-12))
            upper.append(np.log(1e12))

        elif nombre == "Cp":
            lower.append(np.log(1e-15))
            upper.append(np.log(1e3))

        elif nombre == "Q":
            lower.append(np.log(1e-15))
            upper.append(np.log(1e3))

        elif nombre == "Cs":
            lower.append(np.log(1e-15))
            upper.append(np.log(1e3))

        elif nombre == "n":
            lower.append(0.0)
            upper.append(1.0)

    lower = np.array(lower, dtype=float)
    upper = np.array(upper, dtype=float)

    # =========================================================
    # Residuales: ajustamos ReZ y -ImZ al mismo tiempo
    # =========================================================

    def residuales(u):
        params = vector_a_diccionario(u)

        Z_mod = impedancia_modelo_rc_cpe_cs(
            w,
            Rs=params["Rs"],
            Rp=params["Rp"],
            Cp=params["Cp"],
            Q=params["Q"],
            n=params["n"],
            Cs=params["Cs"]
        )

        re_mod = np.real(Z_mod)
        mim_mod = -np.imag(Z_mod)

        r_re = (re_mod - re_exp) / sre
        r_im = (mim_mod - mim_exp) / sim

        return np.concatenate([r_re, r_im])


    resultado = least_squares(
        residuales,
        u0,
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=20000
    )

    params_fit = vector_a_diccionario(resultado.x)

    residuos_finales = residuales(resultado.x)
    chi2 = np.sum(residuos_finales**2)
    dof = len(residuos_finales) - len(resultado.x)
    chi2_red = chi2 / dof if dof > 0 else np.nan

    # =========================================================
    # Estimación de errores
    # =========================================================

    errores = {}

    try:
        J = resultado.jac
        cov_u = np.linalg.pinv(J.T @ J) * chi2_red
        err_u = np.sqrt(np.diag(cov_u))

        for nombre, err_param_u in zip(nombres_libres, err_u):
            if nombre in nombres_positivos:
                errores[nombre] = params_fit[nombre] * err_param_u
            else:
                errores[nombre] = err_param_u

    except Exception:
        for nombre in nombres_libres:
            errores[nombre] = np.nan

    if n_fijo is not None:
        errores["n"] = 0.0

    if Q_fijo is not None:
        errores["Q"] = 0.0

    # =========================================================
    # Gráficos
    # =========================================================

    if graficar:
        graficar_ajuste_modelo_rc_cpe_cs(
            df_fit,
            params_fit,
            errores,
            chi2_red,
            w_min=w_min,
            w_max=w_max,
            col_w=col_w,
            col_re=col_re,
            col_im=col_im,
            col_sre=col_sre,
            col_sim=col_sim,
            col_fecha=col_fecha,
            col_idx=col_idx,
            anotar=anotar
        )

    # =========================================================
    # Salida por pantalla
    # =========================================================

    print("===================================")
    print("Ajuste modelo Rs + (Rp || Cp) + CPE + Cs")
    print("===================================")
    print(f"Rango usado: {w_min} <= omega <= {w_max}")
    print(f"Número de puntos usados: {len(df_fit)}")
    print(f"chi2_red = {chi2_red:.6f}")
    print("-----------------------------------")

    for nombre in ["Rs", "Rp", "Cp", "Q", "n", "Cs"]:
        valor = params_fit[nombre]
        error = errores.get(nombre, np.nan)

        if nombre in ["Rs", "Rp"]:
            print(f"{nombre} = {valor:.6e} ± {error:.6e} ohm")
        elif nombre in ["Cp", "Cs"]:
            print(f"{nombre} = {valor:.6e} ± {error:.6e} F")
        elif nombre == "Q":
            print(f"{nombre} = {valor:.6e} ± {error:.6e}")
        else:
            print(f"{nombre} = {valor:.6f} ± {error:.6f}")

    print("===================================")

    print("Puntos usados:")
    print(df_fit[[col_fecha, col_idx, col_w, col_re, col_im, col_sre, col_sim]])

    return {
        "params": params_fit,
        "errores": errores,
        "chi2_red": chi2_red,
        "resultado": resultado,
        "df_fit": df_fit,
        "nombres_libres": nombres_libres
    }

def graficar_ajuste_modelo_rc_cpe_cs(
    df_fit,
    params,
    errores=None,
    chi2_red=None,
    w_min=None,
    w_max=None,
    col_w="w",
    col_re="ReZ",
    col_im="-ImZ",
    col_sre="err_ReZ",
    col_sim="err_ImZ",
    col_fecha="fecha",
    col_idx="idx_dia",
    anotar=True
):
    """
    Grafica solamente los puntos usados en el ajuste
    y el modelo ajustado.

    Hace:
        1) Nyquist: ReZ vs -ImZ
        2) Bode módulo: |Z| vs omega
        3) Bode fase: phi vs omega
    """

    w = df_fit[col_w].to_numpy()

    w_curva = np.logspace(
        np.log10(np.min(w)),
        np.log10(np.max(w)),
        800
    )

    Z_curva = impedancia_modelo_rc_cpe_cs(
        w_curva,
        Rs=params["Rs"],
        Rp=params["Rp"],
        Cp=params["Cp"],
        Q=params["Q"],
        n=params["n"],
        Cs=params["Cs"]
    )

    Re_curva = np.real(Z_curva)
    mIm_curva = -np.imag(Z_curva)
    Zabs_curva = np.abs(Z_curva)
    phi_curva = np.angle(Z_curva)

    # =========================================================================
    # NYQUIST
    # =========================================================================

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.errorbar(
        df_fit[col_re],
        df_fit[col_im],
        xerr=df_fit[col_sre],
        yerr=df_fit[col_sim],
        fmt="o",
        ms=6,
        capsize=3,
        linestyle="none",
        label="Datos ajustados"
    )

    ax.plot(
        Re_curva,
        mIm_curva,
        "-",
        label="Modelo ajustado"
    )

    if anotar:
        for _, fila in df_fit.iterrows():
            ax.annotate(
                f"{int(fila[col_idx])}",
                (fila[col_re], fila[col_im]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8
            )

    titulo = "Ajuste en plano complejo"

    if w_min is not None and w_max is not None:
        titulo += rf": ${w_min} \leq \omega \leq {w_max}$"

    ax.set_xlabel(r"$\mathrm{Re}(Z)$ [$\Omega$]")
    ax.set_ylabel(r"$-\mathrm{Im}(Z)$ [$\Omega$]")
    ax.set_title(titulo)
    ax.grid(True, alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # =========================================================================
    # BODE: MÓDULO
    # =========================================================================

    fig, ax = plt.subplots(figsize=(8, 5))

    Zabs_exp = np.sqrt(
        df_fit[col_re].to_numpy()**2 +
        df_fit[col_im].to_numpy()**2
    )

    ax.plot(
        w_curva,
        Zabs_curva,
        "-",
        label="Modelo ajustado"
    )

    ax.errorbar(
        df_fit[col_w],
        Zabs_exp,
        xerr=df_fit["err_w"] if "err_w" in df_fit.columns else None,
        fmt="o",
        ms=6,
        capsize=3,
        linestyle="none",
        label="Datos ajustados"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel(r"$|Z|$ [$\Omega$]")
    ax.set_title("Bode: módulo de impedancia")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # =========================================================================
    # BODE: FASE
    # =========================================================================

    if "phi" in df_fit.columns and "err_phi" in df_fit.columns:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            w_curva,
            phi_curva,
            "-",
            label="Modelo ajustado"
        )

        ax.errorbar(
            df_fit[col_w],
            df_fit["phi"],
            xerr=df_fit["err_w"] if "err_w" in df_fit.columns else None,
            yerr=df_fit["err_phi"],
            fmt="o",
            ms=6,
            capsize=3,
            linestyle="none",
            label="Datos ajustados"
        )

        if anotar:
            for _, fila in df_fit.iterrows():
                ax.annotate(
                    f"{int(fila[col_idx])}",
                    (fila[col_w], fila["phi"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8
                )

        ax.set_xscale("log")
        ax.set_xlabel(r"$\omega$ [rad/s]")
        ax.set_ylabel(r"$\phi$ [rad]")
        ax.set_title("Bode: fase")
        ax.grid(True, which="both", alpha=0.4)
        ax.legend()

        plt.tight_layout()
        plt.show()
# =========================================================================
# MODELO: Rs + (Rp || Cp) + CPE + Cs
# =========================================================================

def impedancia_modelo_R_RpCp_CPE_Cs(w, Rs, Rp, Cp, Q, n, Cs):
    """
    Modelo de circuito:

        Rs -- (Rp || Cp) -- CPE -- Cs

    Todo en serie.

    Parámetros:
        Rs : resistencia serie [ohm]
        Rp : resistencia del paralelo [ohm]
        Cp : capacitor del paralelo [F]
        Q  : constante del CPE
        n  : exponente del CPE, entre 0 y 1
        Cs : capacitor en serie [F]
    """

    w = np.asarray(w, dtype=float)
    jw = 1j * w

    Z_Rs = Rs

    Z_RpCp = 1 / (1/Rp + jw*Cp)

    Z_CPE = 1 / (Q * jw**n)

    Z_Cs = 1 / (jw * Cs)

    Z_total = Z_Rs + Z_RpCp + Z_CPE + Z_Cs

    return Z_total

def ajustar_modelo_R_RpCp_CPE_Cs_ReIm(
    df,
    w_min,
    w_max,
    col_w="w",
    col_re="ReZ",
    col_im="ImZ",
    col_sre="err_ReZ",
    col_sim="err_ImZ",
    col_fecha="fecha",
    col_idx="idx_dia",
    ignorar=None,
    p0=None,
    n_fijo=None,
    graficar=True,
    separar_por_fecha=True,
    anotar=True
):
    """
    Ajusta el modelo:

        Rs -- (Rp || Cp) -- CPE -- Cs

    usando simultáneamente ReZ e ImZ.

    El ajuste minimiza:

        (ReZ_modelo - ReZ_exp)/err_ReZ
        (ImZ_modelo - ImZ_exp)/err_ImZ

    en el rango:

        w_min <= omega <= w_max

    IMPORTANTE:
    - El ajuste se hace con ImZ.
    - El gráfico se muestra como ReZ vs -ImZ para verlo estilo Nyquist.
    """

    # ---------------------------------------------------------------------
    # 1) Filtrado por puntos ignorados
    # ---------------------------------------------------------------------
    df = filtrar_puntos(
        df,
        ignorar=ignorar,
        col_fecha=col_fecha,
        col_idx=col_idx
    )

    # ---------------------------------------------------------------------
    # 2) Filtrado por omega
    # ---------------------------------------------------------------------
    mask_rango = (
        (df[col_w] >= w_min) &
        (df[col_w] <= w_max)
    )

    df_fit = df.loc[mask_rango].copy()

    # ---------------------------------------------------------------------
    # 3) Sacar NaN, inf y errores nulos
    # ---------------------------------------------------------------------
    columnas = [col_w, col_re, col_im, col_sre, col_sim]

    mask_finitos = np.ones(len(df_fit), dtype=bool)

    for col in columnas:
        mask_finitos &= np.isfinite(df_fit[col].to_numpy())

    mask_finitos &= df_fit[col_sre].to_numpy() > 0
    mask_finitos &= df_fit[col_sim].to_numpy() > 0

    df_fit = df_fit.loc[mask_finitos].copy()

    if len(df_fit) < 5:
        raise ValueError("Hay pocos puntos para ajustar este modelo.")

    # ---------------------------------------------------------------------
    # 4) Arrays experimentales
    # ---------------------------------------------------------------------
    w = df_fit[col_w].to_numpy()

    ReZ_exp = df_fit[col_re].to_numpy()
    ImZ_exp = df_fit[col_im].to_numpy()

    err_ReZ = df_fit[col_sre].to_numpy()
    err_ImZ = df_fit[col_sim].to_numpy()

    # ---------------------------------------------------------------------
    # 5) Valores iniciales automáticos
    # ---------------------------------------------------------------------
    w_med = np.exp(np.mean(np.log(w)))

    Z_abs = np.sqrt(ReZ_exp**2 + ImZ_exp**2)
    Z_med = np.median(Z_abs)

    Rs0 = max(1e-9, np.min(ReZ_exp))
    Rp0 = max(1e-9, np.max(ReZ_exp) - Rs0)

    if Rp0 <= 0:
        Rp0 = max(1.0, Z_med)

    Cp0 = 1 / (w_med * Rp0)

    if n_fijo is None:
        n0 = 0.8
    else:
        n0 = n_fijo

    Q0 = 1 / (Z_med * w_med**n0)
    Cs0 = 1 / (w_med * Z_med)

    params0 = {
        "Rs": Rs0,
        "Rp": Rp0,
        "Cp": Cp0,
        "Q": Q0,
        "n": n0,
        "Cs": Cs0
    }

    if p0 is not None:
        params0.update(p0)

    if n_fijo is not None:
        params0["n"] = n_fijo

    # ---------------------------------------------------------------------
    # 6) Parámetros libres
    # ---------------------------------------------------------------------
    parametros_positivos = ["Rs", "Rp", "Cp", "Q", "Cs"]
    parametros_libres = ["Rs", "Rp", "Cp", "Q", "Cs"]

    if n_fijo is None:
        parametros_libres.append("n")

    def params_a_vector(params):
        u = []

        for nombre in parametros_libres:
            if nombre in parametros_positivos:
                u.append(np.log(params[nombre]))
            else:
                u.append(params[nombre])

        return np.array(u, dtype=float)

    def vector_a_params(u):
        params = params0.copy()

        for valor, nombre in zip(u, parametros_libres):
            if nombre in parametros_positivos:
                params[nombre] = np.exp(valor)
            else:
                params[nombre] = valor

        if n_fijo is not None:
            params["n"] = n_fijo

        return params

    u0 = params_a_vector(params0)

    # ---------------------------------------------------------------------
    # 7) Límites del ajuste
    # ---------------------------------------------------------------------
    lower = []
    upper = []

    for nombre in parametros_libres:

        if nombre == "Rs":
            lower.append(np.log(1e-12))
            upper.append(np.log(1e12))

        elif nombre == "Rp":
            lower.append(np.log(1e-12))
            upper.append(np.log(1e12))

        elif nombre == "Cp":
            lower.append(np.log(1e-15))
            upper.append(np.log(1e1))

        elif nombre == "Q":
            lower.append(np.log(1e-15))
            upper.append(np.log(1e1))

        elif nombre == "Cs":
            lower.append(np.log(1e-15))
            upper.append(np.log(1e1))

        elif nombre == "n":
            lower.append(0.0)
            upper.append(1.0)

    lower = np.array(lower, dtype=float)
    upper = np.array(upper, dtype=float)

    # ---------------------------------------------------------------------
    # 8) Residuales complejos: ReZ e ImZ
    # ---------------------------------------------------------------------
    def residuales(u):
        params = vector_a_params(u)

        Z_modelo = impedancia_modelo_R_RpCp_CPE_Cs(
            w,
            Rs=params["Rs"],
            Rp=params["Rp"],
            Cp=params["Cp"],
            Q=params["Q"],
            n=params["n"],
            Cs=params["Cs"]
        )

        ReZ_modelo = np.real(Z_modelo)
        ImZ_modelo = np.imag(Z_modelo)

        r_re = (ReZ_modelo - ReZ_exp) / err_ReZ
        r_im = (ImZ_modelo - ImZ_exp) / err_ImZ

        return np.concatenate([r_re, r_im])

    # ---------------------------------------------------------------------
    # 9) Ajuste
    # ---------------------------------------------------------------------
    resultado = least_squares(
        residuales,
        u0,
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=20000
    )

    params_fit = vector_a_params(resultado.x)

    residuos = residuales(resultado.x)

    chi2 = np.sum(residuos**2)
    dof = len(residuos) - len(resultado.x)
    chi2_red = chi2 / dof if dof > 0 else np.nan

    # ---------------------------------------------------------------------
    # 10) Errores aproximados
    # ---------------------------------------------------------------------
    errores = {}

    try:
        J = resultado.jac
        cov_u = np.linalg.pinv(J.T @ J) * chi2_red
        err_u = np.sqrt(np.diag(cov_u))

        for nombre, err_nombre_u in zip(parametros_libres, err_u):
            if nombre in parametros_positivos:
                errores[nombre] = params_fit[nombre] * err_nombre_u
            else:
                errores[nombre] = err_nombre_u

    except Exception:
        for nombre in parametros_libres:
            errores[nombre] = np.nan

    if n_fijo is not None:
        errores["n"] = 0.0

    # ---------------------------------------------------------------------
    # 11) Gráfico del ajuste
    # ---------------------------------------------------------------------
    if graficar:
        w_curva = np.logspace(np.log10(np.min(w)), np.log10(np.max(w)), 800)

        Z_curva = impedancia_modelo_R_RpCp_CPE_Cs(
            w_curva,
            Rs=params_fit["Rs"],
            Rp=params_fit["Rp"],
            Cp=params_fit["Cp"],
            Q=params_fit["Q"],
            n=params_fit["n"],
            Cs=params_fit["Cs"]
        )

        Re_curva = np.real(Z_curva)
        mIm_curva = -np.imag(Z_curva)

        fig, ax = plt.subplots(figsize=(7, 6))

        if separar_por_fecha:
            for fecha, grupo in df_fit.groupby(col_fecha):
                ax.errorbar(
                    grupo[col_re],
                    -grupo[col_im],
                    xerr=grupo[col_sre],
                    yerr=grupo[col_sim],
                    fmt="o",
                    ms=6,
                    capsize=3,
                    linestyle="none",
                    label=f"Datos {fecha}"
                )
        else:
            ax.errorbar(
                df_fit[col_re],
                -df_fit[col_im],
                xerr=df_fit[col_sre],
                yerr=df_fit[col_sim],
                fmt="o",
                ms=6,
                capsize=3,
                linestyle="none",
                label="Datos ajustados"
            )

        ax.plot(
            Re_curva,
            mIm_curva,
            "-",
            label=(
                "Modelo ajustado\n"
                rf"$\chi^2_\nu = {chi2_red:.3f}$"
            )
        )

        if anotar:
            for _, fila in df_fit.iterrows():
                ax.annotate(
                    f"{int(fila[col_idx])}",
                    (fila[col_re], -fila[col_im]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8
                )

        ax.set_xlabel(r"$\mathrm{Re}(Z)$ [$\Omega$]")
        ax.set_ylabel(r"$-\mathrm{Im}(Z)$ [$\Omega$]")
        ax.set_title(
            rf"Ajuste modelo: ${w_min} \leq \omega \leq {w_max}$"
        )
        ax.grid(True, alpha=0.4)
        ax.legend()

        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------
    # 12) Imprimir resultados
    # ---------------------------------------------------------------------
    print("===================================")
    print("Ajuste modelo Rs + (Rp || Cp) + CPE + Cs")
    print("===================================")
    print(f"Rango usado: {w_min} <= omega <= {w_max}")
    print(f"Número de puntos usados: {len(df_fit)}")
    print(f"chi2_red = {chi2_red:.6f}")
    print("-----------------------------------")

    print(f"Rs = {params_fit['Rs']:.6e} ± {errores.get('Rs', np.nan):.6e} ohm")
    print(f"Rp = {params_fit['Rp']:.6e} ± {errores.get('Rp', np.nan):.6e} ohm")
    print(f"Cp = {params_fit['Cp']:.6e} ± {errores.get('Cp', np.nan):.6e} F")
    print(f"Q  = {params_fit['Q']:.6e} ± {errores.get('Q', np.nan):.6e}")
    print(f"n  = {params_fit['n']:.6f} ± {errores.get('n', np.nan):.6f}")
    print(f"Cs = {params_fit['Cs']:.6e} ± {errores.get('Cs', np.nan):.6e} F")

    print("===================================")

    print("Puntos usados:")
    print(df_fit[[col_fecha, col_idx, col_w, col_re, col_im, col_sre, col_sim]])

    return {
        "params": params_fit,
        "errores": errores,
        "chi2_red": chi2_red,
        "resultado": resultado,
        "df_fit": df_fit,
        "parametros_libres": parametros_libres
    }

def impedancia_modelo_R_RpCp_CPEparCs(w, Rs, Rp, Cp, Q, n, Cs):
    w = np.asarray(w, dtype=float)
    jw = 1j * w

    Z_Rs = Rs

    Z_RpCp = 1 / (1/Rp + jw*Cp)

    Z_CPE = 1 / (Q * jw**n)
    Z_Cs = 1 / (jw * Cs)

    Z_CPEparCs = 1 / (1/Z_CPE + 1/Z_Cs)

    Z_total = Z_Rs + Z_RpCp + Z_CPEparCs

    return Z_total

def ajustar_modelo_R_RpCp_CPEparCs_ReIm(
    df,
    w_min,
    w_max,
    col_w="w",
    col_re="ReZ",
    col_im="ImZ",
    col_sre="err_ReZ",
    col_sim="err_ImZ",
    col_fecha="fecha",
    col_idx="idx_dia",
    ignorar=None,
    p0=None,
    n_fijo=None,
    graficar=True,
    separar_por_fecha=True,
    anotar=True
):
    df = filtrar_puntos(
        df,
        ignorar=ignorar,
        col_fecha=col_fecha,
        col_idx=col_idx
    )

    mask_rango = (df[col_w] >= w_min) & (df[col_w] <= w_max)
    df_fit = df.loc[mask_rango].copy()

    columnas = [col_w, col_re, col_im, col_sre, col_sim]

    mask_finitos = np.ones(len(df_fit), dtype=bool)

    for col in columnas:
        mask_finitos &= np.isfinite(df_fit[col].to_numpy())

    mask_finitos &= df_fit[col_sre].to_numpy() > 0
    mask_finitos &= df_fit[col_sim].to_numpy() > 0

    df_fit = df_fit.loc[mask_finitos].copy()

    if len(df_fit) < 5:
        raise ValueError("Hay pocos puntos para ajustar este modelo.")

    w = df_fit[col_w].to_numpy()

    ReZ_exp = df_fit[col_re].to_numpy()
    ImZ_exp = df_fit[col_im].to_numpy()

    err_ReZ = df_fit[col_sre].to_numpy()
    err_ImZ = df_fit[col_sim].to_numpy()

    w_med = np.exp(np.mean(np.log(w)))
    Z_abs = np.sqrt(ReZ_exp**2 + ImZ_exp**2)
    Z_med = np.median(Z_abs)

    Rs0 = max(1e-9, np.min(ReZ_exp))
    Rp0 = max(1e-9, np.max(ReZ_exp) - Rs0)

    if Rp0 <= 0:
        Rp0 = max(1.0, Z_med)

    Cp0 = 1 / (w_med * Rp0)

    if n_fijo is None:
        n0 = 0.8
    else:
        n0 = n_fijo

    Q0 = 1 / (Z_med * w_med**n0)
    Cs0 = 1 / (w_med * Z_med)

    params0 = {
        "Rs": Rs0,
        "Rp": Rp0,
        "Cp": Cp0,
        "Q": Q0,
        "n": n0,
        "Cs": Cs0
    }

    if p0 is not None:
        params0.update(p0)

    if n_fijo is not None:
        params0["n"] = n_fijo

    parametros_positivos = ["Rs", "Rp", "Cp", "Q", "Cs"]
    parametros_libres = ["Rs", "Rp", "Cp", "Q", "Cs"]

    if n_fijo is None:
        parametros_libres.append("n")

    def params_a_vector(params):
        u = []

        for nombre in parametros_libres:
            if nombre in parametros_positivos:
                u.append(np.log(params[nombre]))
            else:
                u.append(params[nombre])

        return np.array(u, dtype=float)

    def vector_a_params(u):
        params = params0.copy()

        for valor, nombre in zip(u, parametros_libres):
            if nombre in parametros_positivos:
                params[nombre] = np.exp(valor)
            else:
                params[nombre] = valor

        if n_fijo is not None:
            params["n"] = n_fijo

        return params

    u0 = params_a_vector(params0)

    lower = []
    upper = []

    for nombre in parametros_libres:

        if nombre in ["Rs", "Rp"]:
            lower.append(np.log(1e-12))
            upper.append(np.log(1e12))

        elif nombre in ["Cp", "Q", "Cs"]:
            lower.append(np.log(1e-15))
            upper.append(np.log(1e1))

        elif nombre == "n":
            lower.append(0.0)
            upper.append(1.0)

    lower = np.array(lower, dtype=float)
    upper = np.array(upper, dtype=float)

    def residuales(u):
        params = vector_a_params(u)

        Z_modelo = impedancia_modelo_R_RpCp_CPEparCs(
            w,
            Rs=params["Rs"],
            Rp=params["Rp"],
            Cp=params["Cp"],
            Q=params["Q"],
            n=params["n"],
            Cs=params["Cs"]
        )

        ReZ_modelo = np.real(Z_modelo)
        ImZ_modelo = np.imag(Z_modelo)

        r_re = (ReZ_modelo - ReZ_exp) / err_ReZ
        r_im = (ImZ_modelo - ImZ_exp) / err_ImZ

        return np.concatenate([r_re, r_im])

    resultado = least_squares(
        residuales,
        u0,
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=20000
    )

    params_fit = vector_a_params(resultado.x)

    residuos = residuales(resultado.x)
    chi2 = np.sum(residuos**2)
    dof = len(residuos) - len(resultado.x)
    chi2_red = chi2 / dof if dof > 0 else np.nan

    errores = {}

    try:
        J = resultado.jac
        cov_u = np.linalg.pinv(J.T @ J) * chi2_red
        err_u = np.sqrt(np.diag(cov_u))

        for nombre, err_nombre_u in zip(parametros_libres, err_u):
            if nombre in parametros_positivos:
                errores[nombre] = params_fit[nombre] * err_nombre_u
            else:
                errores[nombre] = err_nombre_u

    except Exception:
        for nombre in parametros_libres:
            errores[nombre] = np.nan

    if n_fijo is not None:
        errores["n"] = 0.0

    if graficar:
        w_curva = np.logspace(np.log10(np.min(w)), np.log10(np.max(w)), 800)

        Z_curva = impedancia_modelo_R_RpCp_CPEparCs(
            w_curva,
            Rs=params_fit["Rs"],
            Rp=params_fit["Rp"],
            Cp=params_fit["Cp"],
            Q=params_fit["Q"],
            n=params_fit["n"],
            Cs=params_fit["Cs"]
        )

        Re_curva = np.real(Z_curva)
        mIm_curva = -np.imag(Z_curva)

        fig, ax = plt.subplots(figsize=(7, 6))

        if separar_por_fecha:
            for fecha, grupo in df_fit.groupby(col_fecha):
                ax.errorbar(
                    grupo[col_re],
                    -grupo[col_im],
                    xerr=grupo[col_sre],
                    yerr=grupo[col_sim],
                    fmt="o",
                    ms=6,
                    capsize=3,
                    linestyle="none",
                    label=f"Datos {fecha}"
                )
        else:
            ax.errorbar(
                df_fit[col_re],
                -df_fit[col_im],
                xerr=df_fit[col_sre],
                yerr=df_fit[col_sim],
                fmt="o",
                ms=6,
                capsize=3,
                linestyle="none",
                label="Datos ajustados"
            )

        ax.plot(
            Re_curva,
            mIm_curva,
            "-",
            label=rf"Modelo ajustado, $\chi^2_\nu = {chi2_red:.3f}$"
        )

        if anotar:
            for _, fila in df_fit.iterrows():
                ax.annotate(
                    f"{int(fila[col_idx])}",
                    (fila[col_re], -fila[col_im]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8
                )

        ax.set_xlabel(r"$\mathrm{Re}(Z)$ [$\Omega$]")
        ax.set_ylabel(r"$-\mathrm{Im}(Z)$ [$\Omega$]")
        ax.set_title(
            rf"Ajuste modelo: $R_s + (R_p||C_p) + (\mathrm{{CPE}}||C_s)$"
        )
        ax.grid(True, alpha=0.4)
        ax.legend()

        plt.tight_layout()
        plt.show()

    print("===================================")
    print("Ajuste modelo Rs + (Rp || Cp) + (CPE || Cs)")
    print("===================================")
    print(f"Rango usado: {w_min} <= omega <= {w_max}")
    print(f"Número de puntos usados: {len(df_fit)}")
    print(f"chi2_red = {chi2_red:.6f}")
    print("-----------------------------------")

    print(f"Rs = {params_fit['Rs']:.6e} ± {errores.get('Rs', np.nan):.6e} ohm")
    print(f"Rp = {params_fit['Rp']:.6e} ± {errores.get('Rp', np.nan):.6e} ohm")
    print(f"Cp = {params_fit['Cp']:.6e} ± {errores.get('Cp', np.nan):.6e} F")
    print(f"Q  = {params_fit['Q']:.6e} ± {errores.get('Q', np.nan):.6e}")
    print(f"n  = {params_fit['n']:.6f} ± {errores.get('n', np.nan):.6f}")
    print(f"Cs = {params_fit['Cs']:.6e} ± {errores.get('Cs', np.nan):.6e} F")
    print("===================================")

    return {
        "params": params_fit,
        "errores": errores,
        "chi2_red": chi2_red,
        "resultado": resultado,
        "df_fit": df_fit,
        "parametros_libres": parametros_libres
    }