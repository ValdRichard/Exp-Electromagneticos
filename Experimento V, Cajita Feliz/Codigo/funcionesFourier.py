import numpy as np


def amplitud_unilateral(Y, N):
    """
    Convierte una FFT obtenida con rfft en un espectro
    de amplitud unilateral.

    Parámetros
    ----------
    Y : array complejo
        FFT de la señal.

    N : int
        Cantidad de muestras temporales.

    Retorna
    -------
    amp : array
        Amplitud asociada a cada frecuencia positiva.
    """

    amp = np.abs(Y) / N

    # rfft conserva solo las frecuencias no negativas.
    # Se duplica la amplitud de las frecuencias positivas,
    # excepto frecuencia cero y Nyquist.
    if N % 2 == 0:
        amp[1:-1] *= 2
    else:
        amp[1:] *= 2

    return amp


def cargar_y_hacer_fft(archivo, sampling=100e3, skiprows=4):
    """
    Carga un archivo con las columnas:
        índice, CH1, CH2

    y calcula la FFT de ambos canales.
    """

    # ============================================================
    # CARGAR DATOS
    # ============================================================

    data = np.loadtxt(
        archivo,
        skiprows=skiprows
    )

    indice = data[:, 0]
    ch1 = data[:, 1]
    ch2 = data[:, 2]

    if len(ch1) != len(ch2):
        raise ValueError(
            f"Los canales del archivo {archivo} "
            "tienen distinta cantidad de muestras."
        )

    # ============================================================
    # EJE TEMPORAL
    # ============================================================

    dt = 1 / sampling
    t = (indice - 1) * dt

    # ============================================================
    # QUITAR COMPONENTE CONTINUA
    # ============================================================

    y1 = ch1 - np.mean(ch1)
    y2 = ch2 - np.mean(ch2)

    N = len(ch1)

    # ============================================================
    # FFT
    # ============================================================

    Y1 = np.fft.rfft(y1)
    Y2 = np.fft.rfft(y2)

    freq = np.fft.rfftfreq(N, d=dt)

    # ============================================================
    # AMPLITUD UNILATERAL
    # ============================================================

    amp1 = amplitud_unilateral(Y1, N)
    amp2 = amplitud_unilateral(Y2, N)

    return {
        "archivo": archivo,
        "sampling": sampling,
        "dt": dt,
        "N": N,
        "t": t,
        "ch1": ch1,
        "ch2": ch2,
        "freq_ch1": freq,
        "freq_ch2": freq,
        "fft_ch1": Y1,
        "fft_ch2": Y2,
        "amp_ch1": amp1,
        "amp_ch2": amp2
    }


def analizar_cocientes_amplitud(
    resultados,
    entrada="ch1",
    salida="ch2",
    archivos=None,
    eps=1e-12
):
    """
    Calcula las dos formas de promediar la respuesta:

    1. Cociente de amplitudes promedio:

           <A_salida> / <A_entrada>

    2. Promedio de los cocientes individuales:

           <A_salida / A_entrada>

    Todos los cálculos se hacen frecuencia por frecuencia.
    """

    if archivos is None:
        archivos = sorted(resultados.keys())
    else:
        archivos = list(archivos)

    if len(archivos) == 0:
        raise ValueError("La lista de archivos está vacía.")

    if entrada not in ("ch1", "ch2"):
        raise ValueError("entrada debe ser 'ch1' o 'ch2'.")

    if salida not in ("ch1", "ch2"):
        raise ValueError("salida debe ser 'ch1' o 'ch2'.")

    key_amp_entrada = f"amp_{entrada}"
    key_amp_salida = f"amp_{salida}"
    key_freq = f"freq_{entrada}"

    archivo_ref = archivos[0]

    freq_ref = resultados[archivo_ref][key_freq]
    N_ref = resultados[archivo_ref]["N"]
    sampling_ref = resultados[archivo_ref]["sampling"]

    amplitudes_entrada = []
    amplitudes_salida = []

    # ============================================================
    # REUNIR LOS ESPECTROS DE TODOS LOS ARCHIVOS
    # ============================================================

    for numero_archivo in archivos:

        resultado = resultados[numero_archivo]

        freq = resultado[key_freq]
        N = resultado["N"]
        sampling = resultado["sampling"]

        if N != N_ref:
            raise ValueError(
                f"El archivo {numero_archivo} tiene N={N}, "
                f"pero el archivo de referencia tiene N={N_ref}."
            )

        if not np.isclose(sampling, sampling_ref):
            raise ValueError(
                f"El archivo {numero_archivo} tiene sampling={sampling}, "
                f"pero el archivo de referencia tiene "
                f"sampling={sampling_ref}."
            )

        if not np.allclose(freq, freq_ref):
            raise ValueError(
                f"El archivo {numero_archivo} tiene una "
                "grilla de frecuencias diferente."
            )

        amplitudes_entrada.append(
            resultado[key_amp_entrada]
        )

        amplitudes_salida.append(
            resultado[key_amp_salida]
        )

    amplitudes_entrada = np.asarray(amplitudes_entrada)
    amplitudes_salida = np.asarray(amplitudes_salida)

    # Las matrices tienen la forma:
    #
    # filas    -> archivos
    # columnas -> frecuencias

    # ============================================================
    # AMPLITUD PROMEDIO DE CADA CANAL
    # ============================================================

    amplitud_entrada_prom = np.mean(
        amplitudes_entrada,
        axis=0
    )

    amplitud_salida_prom = np.mean(
        amplitudes_salida,
        axis=0
    )

    amplitud_entrada_std = np.std(
        amplitudes_entrada,
        axis=0,
        ddof=1
    )

    amplitud_salida_std = np.std(
        amplitudes_salida,
        axis=0,
        ddof=1
    )

    # ============================================================
    # MÉTODO 1:
    # COCIENTE DE LOS PROMEDIOS
    #
    #     <A_salida> / <A_entrada>
    # ============================================================

    cociente_de_promedios = np.divide(
        amplitud_salida_prom,
        amplitud_entrada_prom,
        out=np.full_like(
            amplitud_salida_prom,
            np.nan,
            dtype=float
        ),
        where=amplitud_entrada_prom > eps
    )

    # ============================================================
    # MÉTODO 2:
    # COCIENTE DE CADA ARCHIVO Y LUEGO PROMEDIO
    #
    #     <A_salida / A_entrada>
    # ============================================================

    cocientes_individuales = np.divide(
        amplitudes_salida,
        amplitudes_entrada,
        out=np.full_like(
            amplitudes_salida,
            np.nan,
            dtype=float
        ),
        where=amplitudes_entrada > eps
    )

    promedio_de_cocientes = np.nanmean(
        cocientes_individuales,
        axis=0
    )

    std_de_cocientes = np.nanstd(
        cocientes_individuales,
        axis=0,
        ddof=1
    )

    cantidad_validos = np.sum(
        np.isfinite(cocientes_individuales),
        axis=0
    )

    error_promedio_de_cocientes = np.divide(
        std_de_cocientes,
        np.sqrt(cantidad_validos),
        out=np.full_like(
            std_de_cocientes,
            np.nan,
            dtype=float
        ),
        where=cantidad_validos > 0
    )

    return {
        "entrada": entrada,
        "salida": salida,
        "archivos": archivos,
        "cantidad_archivos": len(archivos),
        "freq": freq_ref,

        # Amplitudes de todos los archivos
        "amplitudes_entrada": amplitudes_entrada,
        "amplitudes_salida": amplitudes_salida,

        # Promedio de cada canal
        "amplitud_entrada_prom": amplitud_entrada_prom,
        "amplitud_salida_prom": amplitud_salida_prom,
        "amplitud_entrada_std": amplitud_entrada_std,
        "amplitud_salida_std": amplitud_salida_std,

        # Método 1
        "cociente_de_promedios": cociente_de_promedios,

        # Método 2
        "cocientes_individuales": cocientes_individuales,
        "promedio_de_cocientes": promedio_de_cocientes,
        "std_de_cocientes": std_de_cocientes,
        "error_promedio_de_cocientes":
            error_promedio_de_cocientes
    }