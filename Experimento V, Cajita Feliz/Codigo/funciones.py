import numpy as np
from scipy.odr import ODR, RealData, Model
import matplotlib.pyplot as plt
import pandas as pd

def asignar_errores(df, configs, col_fecha="fecha", col_f="f"):
    df = df.copy()

    df["err_f"] = np.nan
    df["err_Ve"] = np.nan
    df["err_Vs"] = np.nan

    for fecha, cfg in configs.items():
        mask_fecha = df[col_fecha] == fecha

        limites = cfg["limites"]
        err_f = cfg["err_f"]
        err_Ve = cfg["err_Ve"]
        err_Vs = cfg["err_Vs"]

        # Armo bins tipo: [0, lim1], (lim1, lim2], ..., (lim6, inf)
        bins = [0] + limites + [np.inf]
        labels = range(len(bins) - 1)

        rangos = pd.cut(
            df.loc[mask_fecha, col_f],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True
        )

        df.loc[mask_fecha, "err_f"] = rangos.map(dict(zip(labels, err_f))).astype(float)
        df.loc[mask_fecha, "err_Ve"] = rangos.map(dict(zip(labels, err_Ve))).astype(float)
        df.loc[mask_fecha, "err_Vs"] = rangos.map(dict(zip(labels, err_Vs))).astype(float)

    return df


def cargar_impedancia_csv(
    archivo,
    col_re="ReZ",
    col_im="ImZ",
    col_sre="sReZ",
    col_sim="sImZ"
):

    df = pd.read_csv(archivo)

    re_z = df[col_re].to_numpy()
    im_z = df[col_im].to_numpy()

    sre_z = df[col_sre].to_numpy()
    sim_z = df[col_sim].to_numpy()

    return re_z, im_z, sre_z, sim_z

def modelo_cpe(beta, re_z):

    n = beta[0]

    return -re_z * np.tan(np.pi*n/2)

def modelo_cpe_ordenada(beta, re_z):

    n, b = beta

    return (
        -re_z * np.tan(np.pi*n/2)
        + b
    )

def ajustar_n_cpe_ordenada(
    re_z,
    im_z,
    sre_z=None,
    sim_z=None,
    n0=0.8,
    b0=0.0
):

    re_z = np.asarray(re_z)
    im_z = np.asarray(im_z)

    if sre_z is None:
        sre_z = np.ones_like(re_z)

    if sim_z is None:
        sim_z = np.ones_like(im_z)

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
        beta0=[n0, b0]
    )

    return odr.run()

def ajustar_n_cpe(
    re_z,
    im_z,
    sre_z=None,
    sim_z=None,
    n0=0.8
):

    re_z = np.asarray(re_z)
    im_z = np.asarray(im_z)

    if sre_z is None:
        sre_z = np.ones_like(re_z)

    if sim_z is None:
        sim_z = np.ones_like(im_z)

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

    out = odr.run()

    return out

def graficar_ajuste_cpe_ordenada(
    re_z,
    im_z,
    sre_z=None,
    sim_z=None,
    n0=0.8,
    b0=0.0,
    ax=None
):

    resultado = ajustar_n_cpe(
        re_z,
        im_z,
        sre_z,
        sim_z,
        n0,
        b0
    )

    n, b = resultado.beta
    sn, sb = resultado.sd_beta

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    ax.errorbar(
        re_z,
        im_z,
        xerr=sre_z,
        yerr=sim_z,
        fmt='o'
    )

    xfit = np.linspace(
        np.min(re_z),
        np.max(re_z),
        500
    )

    yfit = modelo_cpe(
        [n, b],
        xfit
    )

    ax.plot(
        xfit,
        yfit,
        '-',
        label=(
            rf"$n={n:.4f}\pm{sn:.4f}$" "\n"
            rf"$b={b:.3e}\pm{sb:.3e}$"
        )
    )

    ax.set_xlabel(r"$\Re(Z)$")
    ax.set_ylabel(r"$\Im(Z)$")
    ax.legend()
    ax.grid(True)

    print(f"n = {n:.6f} ± {sn:.6f}")
    print(f"b = {b:.6e} ± {sb:.6e}")
    print(f"χ²_red = {resultado.res_var:.4f}")

    return resultado

def graficar_ajuste_cpe(
    re_z,
    im_z,
    sre_z=None,
    sim_z=None,
    n0=0.8,
    ax=None
):

    resultado = ajustar_n_cpe(
        re_z,
        im_z,
        sre_z,
        sim_z,
        n0
    )

    n = resultado.beta[0]
    sn = resultado.sd_beta[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    ax.errorbar(
        re_z,
        im_z,
        xerr=sre_z,
        yerr=sim_z,
        fmt='o',
        capsize=2,
        label='Datos'
    )

    xfit = np.linspace(
        np.min(re_z),
        np.max(re_z),
        500
    )

    yfit = modelo_cpe(
        [n],
        xfit
    )

    ax.plot(
        xfit,
        yfit,
        '-',
        label=rf'Ajuste ($n={n:.4f}\pm{sn:.4f}$)'
    )

    ax.set_xlabel(r'$\mathrm{Re}(Z)$')
    ax.set_ylabel(r'$\mathrm{Im}(Z)$')

    ax.legend()
    ax.grid(True)

    print(f"n = {n:.6f} ± {sn:.6f}")
    print(f"χ²_red = {resultado.res_var:.4f}")

    return resultado