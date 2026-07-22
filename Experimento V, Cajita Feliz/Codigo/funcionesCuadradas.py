from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


# =============================================================================
# LECTURA DEL ARCHIVO
# =============================================================================

def cargar_osciloscopio(ruta, fs):
    """
    Lee exactamente el formato de tus archivos:

        indice    CH1    CH2

    Devuelve CH1 y CH2 en mV, y el tiempo en segundos.
    """
    patron = re.compile(
        r"^\s*(\d+)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)"
    )

    filas = []

    with open(ruta, "r", encoding="utf-8", errors="replace") as archivo:
        for linea in archivo:
            coincidencia = patron.match(linea)

            if coincidencia:
                filas.append(
                    (
                        int(coincidencia.group(1)),
                        float(coincidencia.group(2)),
                        float(coincidencia.group(3)),
                    )
                )

    if not filas:
        raise ValueError(
            f"No se encontraron filas numéricas en el archivo:\n{ruta}"
        )

    datos = pd.DataFrame(
        filas,
        columns=["muestra", "CH1_mV", "CH2_mV"],
    )

    datos["t_s"] = (
        datos["muestra"] - datos["muestra"].iloc[0]
    ) / fs

    return datos


# =============================================================================
# FRECUENCIA DE MUESTREO DESDE EL NOMBRE
# =============================================================================

def extraer_fs_desde_nombre(nombre):
    """
    Lee la frecuencia de muestreo directamente desde nombres como:

        444Hz___100KSs(1)
        1_132kHz___500KSs(1)
        6_703kHz___2_5MSs(1)

    Devuelve fs en muestras por segundo.
    """
    nombre = Path(nombre).stem

    coincidencia = re.search(
        r"___([0-9_]+)([kKmM]?)Ss\(\d+\)$",
        nombre,
    )

    if coincidencia is None:
        raise ValueError(
            f"No pude leer el sampleo desde el nombre: {nombre}"
        )

    valor = float(coincidencia.group(1).replace("_", "."))
    prefijo = coincidencia.group(2).lower()

    if prefijo == "k":
        valor *= 1e3
    elif prefijo == "m":
        valor *= 1e6

    return valor


def extraer_frecuencia_excitacion(nombre):
    """
    Lee la frecuencia de la onda cuadrada desde el comienzo del nombre.
    """
    nombre = Path(nombre).stem
    texto = nombre.split("___", maxsplit=1)[0]
    texto_minuscula = texto.lower()

    if texto_minuscula.endswith("khz"):
        return float(texto[:-3].replace("_", ".")) * 1e3

    if texto_minuscula.endswith("hz"):
        return float(texto[:-2].replace("_", "."))

    raise ValueError(
        f"No pude leer la frecuencia de excitación desde: {nombre}"
    )


# =============================================================================
# BÚSQUEDA DE ARCHIVOS
# =============================================================================

def buscar_archivo(carpeta_codigo, nombre_sin_extension):
    """
    Busca el archivo exacto dentro de Codigo y todas sus subcarpetas.

    Prueba:
        nombre.txt
        nombre.dat
        nombre sin extensión
    """
    carpeta_codigo = Path(carpeta_codigo).resolve()

    nombres_posibles = {
        nombre_sin_extension,
        f"{nombre_sin_extension}.txt",
        f"{nombre_sin_extension}.dat",
    }

    # Primero busca directamente en Codigo.
    for nombre in nombres_posibles:
        ruta = carpeta_codigo / nombre

        if ruta.is_file():
            return ruta

    # Después busca en todas las subcarpetas.
    for ruta in carpeta_codigo.rglob("*"):
        if ruta.is_file() and ruta.name in nombres_posibles:
            return ruta

    return None


# =============================================================================
# DETECCIÓN DE FLANCOS
# =============================================================================

def detectar_flancos(ch1, umbral=None):
    """
    Detecta los saltos de la onda cuadrada de CH1.

    Esta es la misma lógica que funcionó para:
        1_132kHz___500KSs(1).txt
    """
    diferencias = np.diff(np.asarray(ch1, dtype=float))

    if diferencias.size == 0:
        return np.array([], dtype=int)

    if umbral is None:
        umbral = 0.5 * np.max(np.abs(diferencias))

    indices = np.where(
        np.abs(diferencias) >= umbral
    )[0] + 1

    return indices


# =============================================================================
# AJUSTE
# =============================================================================

def ajustar_tau_compartida(datos, indices_flancos, fs):
    """
    Ajusta cada semiperíodo completo con:

        y(t) = C_j + A_j exp(-t/tau)

    usando una única tau compartida para todas las ramas.

    Esta mantiene la misma lógica del código que sí te funcionó.
    """
    segmentos = []

    for inicio, fin in zip(
        indices_flancos[:-1],
        indices_flancos[1:],
    ):
        y = datos["CH2_mV"].to_numpy()[inicio:fin]
        t = np.arange(len(y), dtype=float) / fs

        if len(y) > 0:
            segmentos.append((inicio, fin, t, y))

    if not segmentos:
        raise ValueError(
            "No hay suficientes flancos para formar un segmento completo."
        )

    tau_inicial = 80e-6
    parametros_iniciales = [tau_inicial]

    for _, _, _, y in segmentos:
        cantidad_final = min(30, len(y))
        C0 = float(np.median(y[-cantidad_final:]))
        A0 = float(y[0] - C0)

        parametros_iniciales.extend([C0, A0])

    def residuos(parametros):
        tau = parametros[0]
        resultado = []

        for j, (_, _, t, y) in enumerate(segmentos):
            C = parametros[1 + 2 * j]
            A = parametros[2 + 2 * j]

            y_modelo = C + A * np.exp(-t / tau)
            resultado.append(y - y_modelo)

        return np.concatenate(resultado)

    limite_inferior = [1e-9]
    limite_superior = [10e-3]

    for _ in segmentos:
        limite_inferior.extend([-np.inf, -np.inf])
        limite_superior.extend([np.inf, np.inf])

    ajuste = least_squares(
        residuos,
        parametros_iniciales,
        bounds=(limite_inferior, limite_superior),
    )

    parametros = ajuste.x
    tau = float(parametros[0])

    numero_datos = ajuste.fun.size
    numero_parametros = parametros.size
    grados_libertad = numero_datos - numero_parametros

    if grados_libertad > 0:
        varianza_residual = (
            np.sum(ajuste.fun**2) / grados_libertad
        )

        matriz = ajuste.jac.T @ ajuste.jac

        # Se conserva el cálculo original que sí funcionó.
        # La pseudoinversa se usa solamente si la inversa no puede calcularse.
        try:
            inversa = np.linalg.inv(matriz)
        except np.linalg.LinAlgError:
            inversa = np.linalg.pinv(matriz)

        covarianza = inversa * varianza_residual

        error_tau = float(
            np.sqrt(max(covarianza[0, 0], 0))
        )
    else:
        error_tau = np.nan

    resultados_segmentos = []

    for j, (inicio, fin, t, y) in enumerate(segmentos):
        C = float(parametros[1 + 2 * j])
        A = float(parametros[2 + 2 * j])

        y_modelo = C + A * np.exp(-t / tau)

        resultados_segmentos.append(
            {
                "inicio": int(inicio),
                "fin": int(fin),
                "t_rel_s": t,
                "y_mV": y,
                "modelo_mV": y_modelo,
                "C_mV": C,
                "A_mV": A,
            }
        )

    rmse = float(
        np.sqrt(np.mean(ajuste.fun**2))
    )

    return tau, error_tau, rmse, resultados_segmentos

def ajustar_dos_tau_compartidas(datos, indices_flancos, fs):
    """
    Ajusta todos los semiperíodos con:

        y(t) = C_j
             + A1_j exp(-t/tau1)
             + A2_j exp(-t/tau2)

    tau1 y tau2 son compartidas entre todas las ramas.
    C_j, A1_j y A2_j cambian para cada rama.
    """

    segmentos = []

    for inicio, fin in zip(
        indices_flancos[:-1],
        indices_flancos[1:],
    ):
        y = datos["CH2_mV"].to_numpy()[inicio:fin]
        t = np.arange(len(y), dtype=float) / fs

        if len(y) > 3:
            segmentos.append((inicio, fin, t, y))

    if not segmentos:
        raise ValueError(
            "No hay suficientes flancos para formar un segmento completo."
        )

    # Valores iniciales de las dos escalas temporales.
    tau1_inicial = 20e-6
    tau2_inicial = 150e-6

    parametros_iniciales = [
        tau1_inicial,
        tau2_inicial,
    ]

    for _, _, _, y in segmentos:
        cantidad_final = min(30, len(y))

        C0 = float(np.median(y[-cantidad_final:]))
        amplitud_total = float(y[0] - C0)

        # Repartimos inicialmente la amplitud entre ambas exponenciales.
        A1_0 = 0.5 * amplitud_total
        A2_0 = 0.5 * amplitud_total

        parametros_iniciales.extend(
            [C0, A1_0, A2_0]
        )

    def residuos(parametros):
        tau1 = parametros[0]
        tau2 = parametros[1]

        resultado = []

        for j, (_, _, t, y) in enumerate(segmentos):
            C = parametros[2 + 3 * j]
            A1 = parametros[3 + 3 * j]
            A2 = parametros[4 + 3 * j]

            y_modelo = (
                C
                + A1 * np.exp(-t / tau1)
                + A2 * np.exp(-t / tau2)
            )

            resultado.append(y - y_modelo)

        return np.concatenate(resultado)

    limite_inferior = [
        1e-9,   # tau1
        1e-9,   # tau2
    ]

    limite_superior = [
        10e-3,  # tau1
        10e-3,  # tau2
    ]

    for _ in segmentos:
        limite_inferior.extend(
            [-np.inf, -np.inf, -np.inf]
        )
        limite_superior.extend(
            [np.inf, np.inf, np.inf]
        )

    ajuste = least_squares(
        residuos,
        parametros_iniciales,
        bounds=(
            limite_inferior,
            limite_superior,
        ),
        max_nfev=20000,
    )

    parametros = ajuste.x

    tau1 = float(parametros[0])
    tau2 = float(parametros[1])

    numero_datos = ajuste.fun.size
    numero_parametros = parametros.size
    grados_libertad = numero_datos - numero_parametros

    error_tau1 = np.nan
    error_tau2 = np.nan

    if grados_libertad > 0:
        varianza_residual = (
            np.sum(ajuste.fun**2)
            / grados_libertad
        )

        matriz = ajuste.jac.T @ ajuste.jac

        try:
            inversa = np.linalg.inv(matriz)
        except np.linalg.LinAlgError:
            inversa = np.linalg.pinv(matriz)

        covarianza = inversa * varianza_residual

        error_tau1 = float(
            np.sqrt(max(covarianza[0, 0], 0))
        )

        error_tau2 = float(
            np.sqrt(max(covarianza[1, 1], 0))
        )

    resultados_segmentos = []

    for j, (inicio, fin, t, y) in enumerate(segmentos):
        C = float(parametros[2 + 3 * j])
        A1 = float(parametros[3 + 3 * j])
        A2 = float(parametros[4 + 3 * j])

        y_modelo = (
            C
            + A1 * np.exp(-t / tau1)
            + A2 * np.exp(-t / tau2)
        )

        resultados_segmentos.append(
            {
                "inicio": int(inicio),
                "fin": int(fin),
                "t_rel_s": t,
                "y_mV": y,
                "modelo_mV": y_modelo,
                "C_mV": C,
                "A1_mV": A1,
                "A2_mV": A2,
            }
        )

    rmse = float(
        np.sqrt(np.mean(ajuste.fun**2))
    )

    # Ordenamos para que tau1 sea siempre la menor.
    if tau1 > tau2:
        tau1, tau2 = tau2, tau1
        error_tau1, error_tau2 = (
            error_tau2,
            error_tau1,
        )

    return (
        tau1,
        error_tau1,
        tau2,
        error_tau2,
        rmse,
        resultados_segmentos,
    )
# =============================================================================
# GRÁFICO
# =============================================================================

def graficar_ajuste(
    datos,
    segmentos,
    nombre_archivo,
    tau,
    error_tau,
    salida,
    tau2=None,
    error_tau2=None,
):
    """
    Guarda la imagen del ajuste.

    Si tau2 es None, muestra un ajuste monoexponencial.
    Si tau2 tiene un valor, muestra las dos constantes de tiempo.
    """

    salida = Path(salida)
    salida.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figura, eje = plt.subplots(
        figsize=(10, 5)
    )

    eje.plot(
        datos["t_s"] * 1e3,
        datos["CH2_mV"] / 1000,
        ".",
        markersize=3,
        label="CH2",
    )

    for j, segmento in enumerate(segmentos):
        inicio = segmento["inicio"]

        t_absoluto = (
            datos["t_s"].iloc[inicio]
            + segmento["t_rel_s"]
        )

        etiqueta = (
            "Ajuste exponencial"
            if j == 0
            else None
        )

        eje.plot(
            t_absoluto * 1e3,
            segmento["modelo_mV"] / 1000,
            linewidth=2,
            label=etiqueta,
        )

    eje.set_xlabel("Tiempo (ms)")
    eje.set_ylabel("CH2 (V)")

    if tau2 is None:
        titulo_tau = (
            f"tau = ({tau * 1e6:.3f} ± "
            f"{error_tau * 1e6:.3f}) us"
        )
    else:
        titulo_tau = (
            f"tau1 = ({tau * 1e6:.3f} ± "
            f"{error_tau * 1e6:.3f}) us\n"
            f"tau2 = ({tau2 * 1e6:.3f} ± "
            f"{error_tau2 * 1e6:.3f}) us"
        )

    eje.set_title(
        f"{nombre_archivo}\n"
        f"{titulo_tau}"
    )

    eje.grid(True)
    eje.legend()

    figura.tight_layout()

    figura.savefig(
        salida,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figura)

    if not salida.is_file():
        raise OSError(
            f"No se pudo guardar la imagen:\n"
            f"{salida}"
        )

# =============================================================================
# PROCESAR UN ARCHIVO
# =============================================================================

def procesar_archivo(ruta, carpeta_salida):
    """
    Lee, detecta flancos, ajusta y guarda el gráfico de un archivo.
    """
    ruta = Path(ruta)
    carpeta_salida = Path(carpeta_salida)

    fs = extraer_fs_desde_nombre(ruta.name)
    frecuencia_hz = extraer_frecuencia_excitacion(ruta.name)

    datos = cargar_osciloscopio(
        ruta,
        fs=fs,
    )

    flancos = detectar_flancos(
        datos["CH1_mV"]
    )

    frecuencias_dos_tau = {
        95.0,
        143.0,
        444.0,
    }

    usar_dos_tau = any(
        np.isclose(
            frecuencia_hz,
            frecuencia_objetivo,
            rtol=0,
            atol=0.1,
        )
        for frecuencia_objetivo
        in frecuencias_dos_tau
    )

    if usar_dos_tau:
        (
            tau,
            error_tau,
            tau2,
            error_tau2,
            rmse,
            segmentos,
        ) = ajustar_dos_tau_compartidas(
            datos,
            flancos,
            fs=fs,
        )
    else:
        (
            tau,
            error_tau,
            rmse,
            segmentos,
        ) = ajustar_tau_compartida(
            datos,
            flancos,
            fs=fs,
        )

        tau2 = np.nan
        error_tau2 = np.nan
    ruta_imagen = (
        carpeta_salida
        / f"{ruta.stem}_ajuste.png"
    )

    graficar_ajuste(
        datos=datos,
        segmentos=segmentos,
        nombre_archivo=ruta.name,
        tau=tau,
        error_tau=error_tau,
        tau2=(
            tau2
            if usar_dos_tau
            else None
        ),
        error_tau2=(
            error_tau2
            if usar_dos_tau
            else None
        ),
        salida=ruta_imagen,
    )

    return {
        "archivo": ruta.name,
        "ruta": str(ruta),
        "frecuencia_Hz": frecuencia_hz,
        "fs_Ss": fs,
        "dt_us": 1e6 / fs,
        "cantidad_muestras": len(datos),
        "flancos": flancos.tolist(),
        "cantidad_ramas": len(segmentos),
        "tau_s": tau,
        "tau_us": tau * 1e6,
        "error_tau_s": error_tau,
        "error_tau_us": error_tau * 1e6,
        "rmse_mV": rmse,
        "imagen": str(ruta_imagen),
        "modelo": (
        "doble exponencial"
        if usar_dos_tau
        else "exponencial simple"
        ),
        "tau2_s": (
            tau2
            if usar_dos_tau
            else np.nan
        ),
        "tau2_us": (
            tau2 * 1e6
            if usar_dos_tau
            else np.nan
        ),
        "error_tau2_s": (
            error_tau2
            if usar_dos_tau
            else np.nan
        ),
        "error_tau2_us": (
            error_tau2 * 1e6
            if usar_dos_tau
            else np.nan
        ),
    }


def guardar_resumen_txt(resultados, ruta_salida):
    """
    Guarda un resumen legible con solamente:

        archivo
        tau
        error de tau
        RMSE
    """
    ruta_salida = Path(ruta_salida)

    with open(
        ruta_salida,
        "w",
        encoding="utf-8",
    ) as archivo_txt:

        for resultado in resultados:
            archivo_txt.write(
                f'Archivo: "{resultado["archivo"]}"\n'
            )
            archivo_txt.write(
                f'tau: {resultado["tau_us"]:.3f} us\n'
            )
            archivo_txt.write(
                f'err_tau: '
                f'{resultado["error_tau_us"]:.3f} us\n'
            )
            archivo_txt.write(
                f'rmse_mV: '
                f'{resultado["rmse_mV"]:.3f} mV\n'
            )
            archivo_txt.write("\n")

def promediar_por_frecuencia(resultados):
    """
    Agrupa las mediciones que tienen la misma frecuencia y calcula:

        - promedio de tau
        - desviación estándar entre repeticiones
        - error estándar del promedio
        - propagación de los errores individuales

    Para los ajustes biexponenciales hace lo mismo con tau2.
    """

    df = pd.DataFrame(resultados).copy()

    filas_promedio = []

    for frecuencia, grupo in df.groupby("frecuencia_Hz"):
        cantidad = len(grupo)

        # =============================================================
        # TAU 1
        # =============================================================

        tau = grupo["tau_us"].dropna()
        errores_tau = grupo.loc[
            tau.index,
            "error_tau_us"
        ]

        tau_promedio = tau.mean()

        if len(tau) > 1:
            tau_desviacion = tau.std(ddof=1)
            tau_error_media = (
                tau_desviacion / np.sqrt(len(tau))
            )
        else:
            tau_desviacion = np.nan
            tau_error_media = np.nan

        # Propagación de los errores individuales en un promedio:
        #
        # sigma_prom = sqrt(sum(sigma_i^2)) / N

        tau_error_propagado = (
            np.sqrt(np.sum(errores_tau**2))
            / len(errores_tau)
        )

        # =============================================================
        # TAU 2
        # =============================================================

        tau2 = grupo["tau2_us"].dropna()

        if len(tau2) > 0:
            errores_tau2 = grupo.loc[
                tau2.index,
                "error_tau2_us"
            ]

            tau2_promedio = tau2.mean()

            if len(tau2) > 1:
                tau2_desviacion = tau2.std(ddof=1)
                tau2_error_media = (
                    tau2_desviacion
                    / np.sqrt(len(tau2))
                )
            else:
                tau2_desviacion = np.nan
                tau2_error_media = np.nan

            tau2_error_propagado = (
                np.sqrt(np.sum(errores_tau2**2))
                / len(errores_tau2)
            )

        else:
            tau2_promedio = np.nan
            tau2_desviacion = np.nan
            tau2_error_media = np.nan
            tau2_error_propagado = np.nan

        # =============================================================
        # RMSE
        # =============================================================

        rmse_promedio = grupo["rmse_mV"].mean()

        filas_promedio.append(
            {
                "frecuencia_Hz": frecuencia,
                "cantidad_mediciones": cantidad,
                "modelo": grupo["modelo"].iloc[0],

                "tau_promedio_us": tau_promedio,
                "tau_desviacion_us": tau_desviacion,
                "tau_error_media_us": tau_error_media,
                "tau_error_propagado_us": tau_error_propagado,

                "tau2_promedio_us": tau2_promedio,
                "tau2_desviacion_us": tau2_desviacion,
                "tau2_error_media_us": tau2_error_media,
                "tau2_error_propagado_us": (
                    tau2_error_propagado
                ),

                "rmse_promedio_mV": rmse_promedio,
            }
        )

    resumen_promedios = pd.DataFrame(
        filas_promedio
    )

    resumen_promedios = resumen_promedios.sort_values(
        "frecuencia_Hz"
    ).reset_index(drop=True)

    return resumen_promedios

from matplotlib.lines import Line2D

# =============================================================================
# GRÁFICO PARA PRESENTACIÓN: CH1 Y CH2
# =============================================================================

def graficar_ajuste_presentacion(
    datos,
    segmentos,
    nombre_archivo,
    salida,
    tau,
    error_tau=None,
    tau2=None,
    error_tau2=None,
):
    """
    Figura para presentación:

    - CH2 arriba, con ajuste y recuadro del modelo.
    - CH1 abajo.
    - Líneas punteadas verticales para separar los tramos ajustados.
    - Todo el ajuste con un solo color.
    """

    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)

    t = datos["t_s"].to_numpy()
    ch1_v = datos["CH1_mV"].to_numpy() / 1000.0
    ch2_v = datos["CH2_mV"].to_numpy() / 1000.0

    # -------------------------------------------------------------------------
    # Recorte automático: desde el primer tramo ajustado hasta el último
    # -------------------------------------------------------------------------
    inicio_global = segmentos[0]["inicio"]
    fin_global = segmentos[-1]["fin"] - 1

    if len(t) > 1:
        dt = float(np.median(np.diff(t)))
    else:
        dt = 0.0

    margen = 3 * dt

    t_min = max(t[0], t[inicio_global] - margen)
    t_max = min(t[-1], t[fin_global] + margen)

    mascara = (t >= t_min) & (t <= t_max)

    # -------------------------------------------------------------------------
    # Estilo visual
    # -------------------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
    })

    figura, (ax_ch2, ax_ch1) = plt.subplots(
        2, 1,
        figsize=(10.8, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0]}
    )

    # =========================================================================
    # CH2 ARRIBA
    # =========================================================================

    ax_ch2.plot(
        t[mascara] * 1e3,
        ch2_v[mascara],
        ".",
        markersize=3.2,
        color="tab:blue",
        alpha=0.95,
        label="CH2",
    )

    # Todo el ajuste en un mismo color
    color_ajuste = "tab:orange"

    for j, segmento in enumerate(segmentos):
        inicio = segmento["inicio"]

        t_absoluto = (
            datos["t_s"].iloc[inicio]
            + segmento["t_rel_s"]
        )

        etiqueta = "Ajuste" if j == 0 else None

        ax_ch2.plot(
            t_absoluto * 1e3,
            segmento["modelo_mV"] / 1000.0,
            linewidth=2.2,
            color=color_ajuste,
            label=etiqueta,
        )

    # -------------------------------------------------------------------------
    # Líneas verticales punteadas para marcar cada tramo ajustado
    # -------------------------------------------------------------------------
    limites_segmentos_ms = []

    for segmento in segmentos:
        t_inicio = datos["t_s"].iloc[segmento["inicio"]] * 1e3
        limites_segmentos_ms.append(t_inicio)

    # Agregamos también el final del último segmento
    t_fin_ultimo = datos["t_s"].iloc[segmentos[-1]["fin"] - 1] * 1e3
    limites_segmentos_ms.append(t_fin_ultimo)

    for x in limites_segmentos_ms:
        ax_ch2.axvline(
            x=x,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
        )

    # -------------------------------------------------------------------------
    # Texto con el modelo
    # -------------------------------------------------------------------------
    if tau2 is None:
        if error_tau is None or np.isnan(error_tau):
            texto_tau = (
                rf"$\tau = {tau * 1e6:.2f}\,\mu s$"
            )
        else:
            texto_tau = (
                rf"$\tau = ({tau * 1e6:.2f} \pm {error_tau * 1e6:.2f})\,\mu s$"
            )

        texto_modelo = (
            "Modelo ajustado:\n"
            r"$V_i(t)=A_i + B_i \exp\left[-\frac{t-t_{0,i}}{\tau}\right]$" "\n"
            + texto_tau
        )

    else:
        if error_tau is None or np.isnan(error_tau):
            texto_tau1 = rf"$\tau_1 = {tau * 1e6:.2f}\,\mu s$"
        else:
            texto_tau1 = (
                rf"$\tau_1 = ({tau * 1e6:.2f} \pm {error_tau * 1e6:.2f})\,\mu s$"
            )

        if error_tau2 is None or np.isnan(error_tau2):
            texto_tau2 = rf"$\tau_2 = {tau2 * 1e6:.2f}\,\mu s$"
        else:
            texto_tau2 = (
                rf"$\tau_2 = ({tau2 * 1e6:.2f} \pm {error_tau2 * 1e6:.2f})\,\mu s$"
            )

        texto_modelo = (
            "Modelo ajustado:\n"
            r"$V_i(t)=A_i + B_{1,i}\exp\left[-\frac{t-t_{0,i}}{\tau_1}\right]$" "\n"
            r"$\qquad\quad + B_{2,i}\exp\left[-\frac{t-t_{0,i}}{\tau_2}\right]$" "\n"
            + texto_tau1 + "\n" + texto_tau2
        )

    ax_ch2.text(
        0.80,
        0.95,
        texto_modelo,
        transform=ax_ch2.transAxes,
        fontsize=18,
        va="top",
        ha="center",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="0.75",
            alpha=0.95,
        ),
    )

    leyenda_personalizada = [
        Line2D(
            [0], [0],
            marker=".",
            linestyle="None",
            markersize=8,
            color="tab:blue",
            label="CH2",
        ),
        Line2D(
            [0], [0],
            linestyle="-",
            linewidth=2.2,
            color="tab:orange",
            label="Ajuste",
        ),
    ]

    ax_ch2.set_ylabel("CH2 (V)")
    ax_ch2.set_title(nombre_archivo)
    ax_ch2.grid(True, alpha=0.30)
    ax_ch2.legend(
        handles=leyenda_personalizada,
        loc="lower left",
        frameon=True,
    )

    # =========================================================================
    # CH1 ABAJO
    # =========================================================================

    ax_ch1.plot(
        t[mascara] * 1e3,
        ch1_v[mascara],
        color="tab:blue",
        linewidth=1.6,
    )

    for x in limites_segmentos_ms:
        ax_ch1.axvline(
            x=x,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )

    ax_ch1.set_xlim(t_min * 1e3, t_max * 1e3)
    ax_ch1.set_xlabel("Tiempo (ms)")
    ax_ch1.set_ylabel("CH1 (V)")
    ax_ch1.grid(True, alpha=0.30)

    figura.tight_layout()

    figura.savefig(
        salida,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(figura)

    if not salida.is_file():
        raise OSError(
            f"No se pudo guardar la imagen:\n{salida}"
        )
# =============================================================================
# PROCESAR ARCHIVO SOLO PARA PRESENTACIÓN
# =============================================================================

def procesar_archivo_presentacion(ruta, carpeta_salida):
    """
    Procesa un archivo y guarda solamente la imagen de presentación.
    """

    ruta = Path(ruta)
    carpeta_salida = Path(carpeta_salida)

    fs = extraer_fs_desde_nombre(ruta.name)
    frecuencia_hz = extraer_frecuencia_excitacion(ruta.name)

    datos = cargar_osciloscopio(
        ruta,
        fs=fs,
    )

    flancos = detectar_flancos(
        datos["CH1_mV"]
    )

    frecuencias_dos_tau = {
        95.0,
        143.0,
        444.0,
    }

    usar_dos_tau = any(
        np.isclose(
            frecuencia_hz,
            frecuencia_objetivo,
            rtol=0,
            atol=0.1,
        )
        for frecuencia_objetivo in frecuencias_dos_tau
    )

    if usar_dos_tau:
        (
            tau1,
            error_tau1,
            tau2,
            error_tau2,
            rmse,
            segmentos,
        ) = ajustar_dos_tau_compartidas(
            datos,
            flancos,
            fs=fs,
        )

        ruta_imagen = (
            carpeta_salida
            / f"{ruta.stem}_presentacion.png"
        )

        graficar_ajuste_presentacion(
            datos=datos,
            segmentos=segmentos,
            nombre_archivo=ruta.name,
            salida=ruta_imagen,
            tau=tau1,
            error_tau=error_tau1,
            tau2=tau2,
            error_tau2=error_tau2,
        )

        return {
            "archivo": ruta.name,
            "ruta": str(ruta),
            "frecuencia_Hz": frecuencia_hz,
            "modelo": "doble exponencial",
            "tau1_us": tau1 * 1e6,
            "error_tau1_us": error_tau1 * 1e6,
            "tau2_us": tau2 * 1e6,
            "error_tau2_us": error_tau2 * 1e6,
            "imagen": str(ruta_imagen),
            "rmse_mV": rmse,
        }

    else:
        (
            tau,
            error_tau,
            rmse,
            segmentos,
        ) = ajustar_tau_compartida(
            datos,
            flancos,
            fs=fs,
        )

        ruta_imagen = (
            carpeta_salida
            / f"{ruta.stem}_presentacion.png"
        )

        graficar_ajuste_presentacion(
            datos=datos,
            segmentos=segmentos,
            nombre_archivo=ruta.name,
            salida=ruta_imagen,
            tau=tau,
            error_tau=error_tau,
            tau2=None,
            error_tau2=None,
        )

        return {
            "archivo": ruta.name,
            "ruta": str(ruta),
            "frecuencia_Hz": frecuencia_hz,
            "modelo": "exponencial simple",
            "tau_us": tau * 1e6,
            "error_tau_us": error_tau * 1e6,
            "imagen": str(ruta_imagen),
            "rmse_mV": rmse,
        }