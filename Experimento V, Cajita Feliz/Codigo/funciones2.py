import numpy as np
from scipy.odr import ODR, RealData, Model
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import least_squares
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.ticker as ticker

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


def graficar_datos(
    df,
    x,
    y,
    xerr=None,
    yerr=None,
    col_fecha="fecha",
    col_idx="idx_dia",
    anotar=False,
    color_puntos="steelblue",
    escala_x=None,
    escala_y=None,
    xlabel=None,
    ylabel=None,
    fontsize_labels=11,
    fontsize_ticks=11,
    titulo=None,
    figsize=(8, 5),
    capsize=3,
    ms=5,
    ignorar=None,
    # --- Parámetros de Zoom ---
    hacer_zoom=False,
    zoom_xlim=None,
    zoom_ylim=None,
    zoom_loc="lower right",
    zoom_bounds=None,
    step_x=500,  # Paso / equiespaciado para el eje X del zoom
    step_y=500   # Paso / equiespaciado para el eje Y del zoom
):
    df = filtrar_puntos(
        df,
        ignorar=ignorar,
        col_fecha=col_fecha,
        col_idx=col_idx
    )

    fig, ax = plt.subplots(figsize=figsize)

    # --- 1. Gráfico Principal ---
    ax.errorbar(
        df[x],
        df[y],
        xerr=df[xerr] if xerr is not None else None,
        yerr=df[yerr] if yerr is not None else None,
        fmt="o",
        ms=ms,
        capsize=capsize,
        color=color_puntos,
        ecolor=color_puntos,
        linestyle="none",
        label="Datos experimentales"
    )

    if anotar:
        for _, fila in df.iterrows():
            etiqueta = f"{int(fila[col_idx])}"
            ax.annotate(
                etiqueta,
                (fila[x], fila[y]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8
            )

    # --- 2. Subgráfico Zoom con Ejes Espaciados ---
    if hacer_zoom:
        if zoom_bounds is not None:
            axins = ax.inset_axes(zoom_bounds)
        else:
            axins = inset_axes(ax, width="40%", height="40%", loc=zoom_loc)

        axins.errorbar(
            df[x],
            df[y],
            xerr=df[xerr] if xerr is not None else None,
            yerr=df[yerr] if yerr is not None else None,
            fmt="o",
            ms=ms*0.8,
            capsize=capsize*0.8,
            color=color_puntos,
            ecolor=color_puntos,
            linestyle="none"
        )

        if zoom_xlim is not None:
            axins.set_xlim(zoom_xlim)
        if zoom_ylim is not None:
            axins.set_ylim(zoom_ylim)

        # Grilla interna
        axins.grid(True, alpha=0.3)

        # Define el paso/equiespaciado exacto entre marcas
        if step_x is not None:
            axins.xaxis.set_major_locator(ticker.MultipleLocator(step_x))
        if step_y is not None:
            axins.yaxis.set_major_locator(ticker.MultipleLocator(step_y))

        # Tipografía chica en los ejes del zoom para mayor nitidez
        axins.tick_params(axis='both', which='major', labelsize=7)

    # --- Estilos de Ejes Principales ---
    if escala_x is not None:
        ax.set_xscale(escala_x)
    if escala_y is not None:
        ax.set_yscale(escala_y)

    ax.set_xlabel(xlabel if xlabel is not None else x, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel if ylabel is not None else y, fontsize=fontsize_labels)

    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

    if titulo is not None:
        ax.set_title(titulo)

    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper right", fontsize=10)

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



import matplotlib.pyplot as plt

def graficar_bode_impedancia(
    df,
    fontsize_labels=13,
    fontsize_ticks=11,
    fontsize_legend=11
):
    # Creamos la figura y el primer eje (Izquierdo - Impedancia)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # --- GRÁFICO DE IMPEDANCIA (Z) ---
    color_Z = 'steelblue'
    ax1.set_xlabel(r'$\omega$ [rad/s]', fontsize=fontsize_labels)
    ax1.set_ylabel(r'Impedancia $|Z|$ [$\Omega$]', color=color_Z, fontsize=fontsize_labels)
    
    line1 = ax1.errorbar(
        df["w"], df["Z"], 
        xerr=df["err_w"], yerr=df["err_Z"], 
        fmt='o', color=color_Z, markeredgecolor=None, markersize=5,
        capsize=3, elinewidth=1.2, 
        label=r'Datos de Impedancia $|Z|$'
    )
    
    # Correcto: labelsize en lugar de fontsize
    ax1.tick_params(axis='both', labelcolor=color_Z, labelsize=fontsize_ticks)
    ax1.tick_params(axis='x', labelcolor='black', labelsize=fontsize_ticks)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # --- GRÁFICO DE FASE (phi) ---
    ax2 = ax1.twinx()  
    
    color_phi = 'firebrick'
    ax2.set_ylabel(r'Fase $\phi$ [rad]', color=color_phi, fontsize=fontsize_labels)
    
    line2 = ax2.errorbar(
        df["w"], df["phi"], 
        xerr=df["err_w"], yerr=df["err_phi"], 
        fmt='s', color=color_phi, markeredgecolor=None, markersize=5,
        capsize=3, elinewidth=1.2, 
        label=r'Datos de Fase $\phi$'
    )
    
    # Correcto: labelsize en lugar de fontsize
    ax2.tick_params(axis='y', labelcolor=color_phi, labelsize=fontsize_ticks)

    # --- LEYENDA UNIFICADA ---
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=fontsize_legend)

    # --- DETALLES FINALES ---
    #plt.title(r'Diagrama de Bode: Impedancia y Fase en función de $\omega$', fontsize=14, pad=15)
    ax1.set_xlim(39, 150000)
    
    fig.tight_layout()
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
                color="steelblue",
                ecolor="steelblue",
                zorder=1,
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
            color="steelblue",
            ecolor="steelblue",
            zorder=1,
            label="Datos experimentales"
        )

    if anotar:
        for _, fila in df_fit.iterrows():
            ax.annotate(
                f"{int(fila[col_idx])}",
                (fila[col_re], fila[col_im]),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=8
            )

    xfit = np.linspace(np.min(re_z), np.max(re_z), 500)

    if con_ordenada:
        yfit = modelo_cpe_ordenada([n, b], xfit)

        label_fit = (
            rf"Ajuste" "\n"
            rf"$n$ = 3,59(6) $\times 10^{{-1}}$" "\n"
            rf"$b$ = -2,0(2) $\times 10^{2}$ $\Omega$" "\n"
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
        color="firebrick",
        linestyle="-",
        linewidth=2,
        zorder=2,
        label=label_fit
    )

    ax.set_xlabel(r"$\mathrm{Re}(Z)$ [$\Omega$]", fontsize=13)
    ax.set_ylabel(r"$-\mathrm{Im}(Z)$ [$\Omega$]", fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=12)
  
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=11)


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
                    color="steelblue",
                    ecolor="steelblue",
                    zorder=1,
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
                color="steelblue",
                ecolor="steelblue",
                zorder=1,
                label="Datos experimentales"
            )

        ax.plot(
            Re_curva,
            mIm_curva,
            color="firebrick",
            linestyle="-",
            linewidth=2,
            zorder=2,
            label=(
                "Ajuste del modelo\n"
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

        ax.set_xlabel(r"$\mathrm{Re}(Z)$ [$\Omega$]", fontsize=12)
        ax.set_ylabel(r"$-\mathrm{Im}(Z)$ [$\Omega$]", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=11)

        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=11, loc="lower right")

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