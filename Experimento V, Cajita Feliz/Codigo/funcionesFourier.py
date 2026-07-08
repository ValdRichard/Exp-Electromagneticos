import numpy as np
from scipy.odr import ODR, RealData, Model
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import least_squares
import numpy as np

def cargar_y_hacer_fft(archivo, sampling=100e3, skiprows=4):
    # =========================================================================
    # CARGAR DATOS
    # =========================================================================

    data = np.loadtxt(
        archivo,
        skiprows=skiprows
    )

    # =========================================================================
    # SEPARAR COLUMNAS
    # =========================================================================

    indice = data[:, 0]
    ch1 = data[:, 1]
    ch2 = data[:, 2]

    # =========================================================================
    # EJE TEMPORAL
    # =========================================================================

    dt = 1 / sampling
    t = (indice - 1) * dt

    senal = np.column_stack((t, ch1, ch2))

    # =========================================================================
    # FFT CH1
    # =========================================================================

    y1 = ch1 - np.mean(ch1)

    N1 = len(y1)

    Y1 = np.fft.rfft(y1)

    freq1 = np.fft.rfftfreq(N1, d=dt)

    amp1 = np.abs(Y1) / N1
    amp1[1:] = 2 * amp1[1:]

    # =========================================================================
    # FFT CH2
    # =========================================================================

    y2 = ch2 - np.mean(ch2)

    N2 = len(y2)

    Y2 = np.fft.rfft(y2)

    freq2 = np.fft.rfftfreq(N2, d=dt)

    amp2 = np.abs(Y2) / N2
    amp2[1:] = 2 * amp2[1:]

    # =========================================================================
    # GUARDAR RESULTADOS
    # =========================================================================

    resultado = {
        "archivo": archivo,
        "sampling": sampling,
        "dt": dt,
        "senal": senal,
        "t": t,
        "ch1": ch1,
        "ch2": ch2,
        "freq_ch1": freq1,
        "fft_ch1": Y1,
        "amp_ch1": amp1,
        "freq_ch2": freq2,
        "fft_ch2": Y2,
        "amp_ch2": amp2
    }

    return resultado

def graficar_fft(
    resultados,
    numero_archivo,
    canal="ch1",
    f_min=0,
    f_max=None,
    escala_y="linear"
):
    """
    Grafica la FFT de un archivo y canal elegido.

    Parámetros:
        resultados      : diccionario con los 60 resultados
        numero_archivo  : número del archivo, por ejemplo 1, 2, ..., 60
        canal           : "ch1" o "ch2"
        f_min           : frecuencia mínima a mostrar [Hz]
        f_max           : frecuencia máxima a mostrar [Hz]
        escala_y        : "linear" o "log"
    """

    resultado = resultados[numero_archivo]

    if canal == "ch1":
        freq = resultado["freq_ch1"]
        amp = resultado["amp_ch1"]

    elif canal == "ch2":
        freq = resultado["freq_ch2"]
        amp = resultado["amp_ch2"]

    else:
        raise ValueError("canal debe ser 'ch1' o 'ch2'.")

    if f_max is None:
        f_max = freq.max()

    mask = (freq >= f_min) & (freq <= f_max)

    plt.figure(figsize=(8, 5))

    plt.plot(freq[mask], amp[mask])

    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud [mV]")
    plt.title(f"FFT archivo {numero_archivo}.txt - {canal.upper()}")

    plt.yscale(escala_y)

    plt.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.show()

def promediar_espectros_desde_fft(resultados, canal="ch2", archivos=None):
    """
    Promedia espectros usando las FFT ya calculadas en resultados.

    Para ruido, no promedia las señales temporales.
    Promedia el espectro de potencia de cada medición.

    resultados:
        diccionario con resultados[i] generado por cargar_y_hacer_fft

    canal:
        "ch1" o "ch2"

    archivos:
        lista de archivos a usar. Si es None, usa todos.
    """

    if archivos is None:
        archivos = sorted(resultados.keys())

    if canal == "ch1":
        key_fft = "fft_ch1"
        key_freq = "freq_ch1"
        key_ch = "ch1"
    elif canal == "ch2":
        key_fft = "fft_ch2"
        key_freq = "freq_ch2"
        key_ch = "ch2"
    else:
        raise ValueError("canal debe ser 'ch1' o 'ch2'.")

    psd_lista = []

    archivo_ref = archivos[0]
    freq_ref = resultados[archivo_ref][key_freq]

    for i in archivos:
        Y = resultados[i][key_fft]
        freq = resultados[i][key_freq]

        sampling = resultados[i]["sampling"]
        N = len(resultados[i][key_ch])

        if len(freq) != len(freq_ref):
            raise ValueError(f"El archivo {i} tiene distinta cantidad de frecuencias.")

        # Espectro de potencia unilateral
        # Unidades aproximadas: mV^2/Hz, si la señal está en mV
        psd = np.abs(Y)**2 / (sampling * N)

        # Como usamos rfft, pasamos a espectro unilateral.
        # No duplicamos la frecuencia cero.
        if N % 2 == 0:
            psd[1:-1] = 2 * psd[1:-1]
        else:
            psd[1:] = 2 * psd[1:]

        psd_lista.append(psd)

    psd_array = np.array(psd_lista)

    psd_prom = np.mean(psd_array, axis=0)
    psd_std = np.std(psd_array, axis=0, ddof=1)

    resultado_prom = {
        "canal": canal,
        "archivos": archivos,
        "cantidad_archivos": len(archivos),
        "freq": freq_ref,
        "psd_array": psd_array,
        "psd_prom": psd_prom,
        "psd_std": psd_std
    }

    return resultado_prom
def promediar_amplitudes_desde_fft(resultados, canal="ch2", archivos=None):
    if archivos is None:
        archivos = sorted(resultados.keys())

    if canal == "ch1":
        key_amp = "amp_ch1"
        key_freq = "freq_ch1"
    elif canal == "ch2":
        key_amp = "amp_ch2"
        key_freq = "freq_ch2"
    else:
        raise ValueError("canal debe ser 'ch1' o 'ch2'.")

    amp_lista = []

    archivo_ref = archivos[0]
    freq_ref = resultados[archivo_ref][key_freq]

    for i in archivos:
        amp = resultados[i][key_amp]
        freq = resultados[i][key_freq]

        if len(freq) != len(freq_ref):
            raise ValueError(f"El archivo {i} tiene distinta cantidad de frecuencias.")

        amp_lista.append(amp)

    amp_array = np.array(amp_lista)

    amp_prom = np.mean(amp_array, axis=0)
    amp_std = np.std(amp_array, axis=0, ddof=1)

    resultado_prom = {
        "canal": canal,
        "archivos": archivos,
        "cantidad_archivos": len(archivos),
        "freq": freq_ref,
        "amp_array": amp_array,
        "amp_prom": amp_prom,
        "amp_std": amp_std
    }

    return resultado_prom

########### DE ACA ES PRUEBA 

def promediar_cociente_amplitudes_desde_fft(
    resultados,
    entrada="ch1",
    salida="ch2",
    archivos=None,
    eps=1e-12
):
    if archivos is None:
        archivos = sorted(resultados.keys())

    if entrada == "ch1":
        key_amp_entrada = "amp_ch1"
        key_freq = "freq_ch1"
    elif entrada == "ch2":
        key_amp_entrada = "amp_ch2"
        key_freq = "freq_ch2"
    else:
        raise ValueError("entrada debe ser 'ch1' o 'ch2'.")

    if salida == "ch1":
        key_amp_salida = "amp_ch1"
    elif salida == "ch2":
        key_amp_salida = "amp_ch2"
    else:
        raise ValueError("salida debe ser 'ch1' o 'ch2'.")

    archivo_ref = archivos[0]
    freq_ref = resultados[archivo_ref][key_freq]

    cocientes = []

    for i in archivos:
        amp_entrada = resultados[i][key_amp_entrada]
        amp_salida = resultados[i][key_amp_salida]
        freq = resultados[i][key_freq]

        if len(freq) != len(freq_ref):
            raise ValueError(f"El archivo {i} tiene distinta cantidad de frecuencias.")

        cociente_i = np.divide(
            amp_salida,
            amp_entrada,
            out=np.full_like(amp_salida, np.nan, dtype=float),
            where=amp_entrada > eps
        )

        cocientes.append(cociente_i)

    cocientes = np.array(cocientes)

    cociente_prom = np.nanmean(cocientes, axis=0)

    if len(archivos) > 1:
        cociente_std = np.nanstd(cocientes, axis=0, ddof=1)
    else:
        cociente_std = np.zeros_like(cociente_prom)

    resultado = {
        "entrada": entrada,
        "salida": salida,
        "archivos": archivos,
        "cantidad_archivos": len(archivos),
        "freq": freq_ref,
        "cocientes": cocientes,
        "cociente_prom": cociente_prom,
        "cociente_std": cociente_std
    }

    return resultado