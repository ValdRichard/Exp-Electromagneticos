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
):
    """
    Guarda la imagen del ajuste. No abre una ventana por cada archivo.
    """
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)

    figura, eje = plt.subplots(figsize=(10, 5))

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
    eje.set_title(
        f"{nombre_archivo}\n"
        f"tau = ({tau * 1e6:.3f} ± "
        f"{error_tau * 1e6:.3f}) us"
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
            f"No se pudo guardar la imagen:\n{salida}"
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

    tau, error_tau, rmse, segmentos = ajustar_tau_compartida(
        datos,
        flancos,
        fs=fs,
    )

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