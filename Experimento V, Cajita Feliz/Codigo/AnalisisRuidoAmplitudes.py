import os
import numpy as np
import matplotlib.pyplot as plt

from funcionesFourier import (
    cargar_y_hacer_fft,
    analizar_cocientes_amplitud
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

carpeta = r"Experimento V, Cajita Feliz\Codigo\Ruido_100KSs"

sampling = 100e3
cantidad_archivos = 60

archivos = list(range(1, cantidad_archivos + 1))


# ============================================================
# CARGAR ARCHIVOS Y HACER LAS FFT
# ============================================================

resultados = {}

for numero_archivo in archivos:

    ruta = os.path.join(
        carpeta,
        f"{numero_archivo}.txt"
    )

    resultados[numero_archivo] = cargar_y_hacer_fft(
        ruta,
        sampling=sampling,
        skiprows=4
    )


# ============================================================
# VERIFICAR LAS FRECUENCIAS DE TODOS LOS ARCHIVOS
# ============================================================

archivo_ref = archivos[0]
freq_ref = resultados[archivo_ref]["freq_ch1"]

print("\nGrilla de frecuencias de referencia:")
print(f"Archivo de referencia: {archivo_ref}")
print(f"Cantidad de frecuencias: {len(freq_ref)}")
print(f"Primeras 10 frecuencias: {freq_ref[:10]}")
print(f"Últimas 10 frecuencias: {freq_ref[-10:]}")
print(f"Delta f: {freq_ref[1] - freq_ref[0]} Hz")
print()

for numero_archivo in archivos:

    freq_i = resultados[numero_archivo]["freq_ch1"]

    misma_longitud = len(freq_i) == len(freq_ref)

    if misma_longitud:
        iguales_exactamente = np.array_equal(freq_i, freq_ref)
        iguales_tolerancia = np.allclose(freq_i, freq_ref)
        diferencia_maxima = np.max(np.abs(freq_i - freq_ref))
    else:
        iguales_exactamente = False
        iguales_tolerancia = False
        diferencia_maxima = np.nan

    print(
        f"Archivo {numero_archivo:2d}: "
        f"bins={len(freq_i)}, "
        f"df={freq_i[1] - freq_i[0]:.10f} Hz, "
        f"exactamente iguales={iguales_exactamente}, "
        f"allclose={iguales_tolerancia}, "
        f"diferencia máxima={diferencia_maxima}"
    )
# ============================================================
# CALCULAR LOS DOS TIPOS DE PROMEDIO
# ============================================================

respuesta = analizar_cocientes_amplitud(
    resultados,
    entrada="ch2",
    salida="ch1",
    archivos=archivos,
    eps=1e-12
)

freq = respuesta["freq"]

amp_ch1_prom = respuesta["amplitud_entrada_prom"]
amp_ch2_prom = respuesta["amplitud_salida_prom"]

cociente_de_promedios = respuesta[
    "cociente_de_promedios"
]

promedio_de_cocientes = respuesta[
    "promedio_de_cocientes"
]

error_promedio_de_cocientes = respuesta[
    "error_promedio_de_cocientes"
]


# ============================================================
# GRAFICAR AMPLITUD PROMEDIO DE LOS DOS CANALES
# ============================================================

mask_amp = (
    (freq > 0)
    & np.isfinite(amp_ch1_prom)
    & np.isfinite(amp_ch2_prom)
)

plt.figure(figsize=(9, 5))

plt.plot(
    freq[mask_amp],
    amp_ch1_prom[mask_amp],
    label="Amplitud promedio CH1"
)

plt.plot(
    freq[mask_amp],
    amp_ch2_prom[mask_amp],
    label="Amplitud promedio CH2"
)

plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [mV]")
plt.title("Espectros de amplitud promedio")
plt.grid(True, which="both", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# COMPARAR LOS DOS MÉTODOS
# ============================================================

mask_respuesta = (
    (freq > 0)
    & np.isfinite(cociente_de_promedios)
    & np.isfinite(promedio_de_cocientes)
)

plt.figure(figsize=(9, 5))

plt.plot(
    freq[mask_respuesta],
    cociente_de_promedios[mask_respuesta],
    label="Cociente de promedios"
)

plt.plot(
    freq[mask_respuesta],
    promedio_de_cocientes[mask_respuesta],
    label="Promedio de cocientes"
)

plt.xlabel("Frecuencia [Hz]")
plt.ylabel("CH2 / CH1")
plt.title("Comparación de respuestas en frecuencia")
plt.grid(True, which="both", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()