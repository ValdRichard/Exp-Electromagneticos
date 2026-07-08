import numpy as np
import os
import matplotlib.pyplot as plt
from funcionesFourier import *

carpeta = r"Experimento V, Cajita Feliz\Codigo\Ruido_100KSs"

resultados = {}

for i in range(1, 61):
    archivo = os.path.join(carpeta, f"{i}.txt")

    resultados[i] = cargar_y_hacer_fft(
        archivo,
        sampling=100e3,
        skiprows=4
    )

rango = range(1,61)

espectro_ch1 = promediar_amplitudes_desde_fft(
    resultados,
    canal="ch1",
    archivos=rango
)

espectro_ch2 = promediar_amplitudes_desde_fft(
    resultados,
    canal="ch2",
    archivos=rango
)

freq_ch1 = espectro_ch1["freq"]
amp_ch1_prom = espectro_ch1["amp_prom"]

freq_ch2 = espectro_ch2["freq"]
amp_ch2_prom = espectro_ch2["amp_prom"]

plt.figure(figsize=(8, 5))
plt.plot(freq_ch1, amp_ch1_prom)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [mV]")
plt.title("Amplitud promedio - CH1")
plt.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(freq_ch2, amp_ch2_prom)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [mV]")
plt.title("Amplitud promedio - CH2")
plt.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()

eps = 1e-12

respuesta_amp = np.divide(
    amp_ch2_prom,
    amp_ch1_prom,
    out=np.zeros_like(amp_ch2_prom),
    where=amp_ch1_prom > eps
)

mask = freq_ch1 > 0

plt.figure(figsize=(8, 5))
plt.plot(freq_ch1[mask], respuesta_amp[mask])
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("CH2 / CH1")
plt.title("Respuesta en frecuencia usando amplitud")
plt.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()

respuesta_amp = promediar_cociente_amplitudes_desde_fft(
    resultados,
    entrada="ch1",
    salida="ch2",
    archivos=rango
)

freq = respuesta_amp["freq"]
cociente_prom = respuesta_amp["cociente_prom"]

mask = freq > 0

plt.figure(figsize=(8, 5))
plt.plot(freq[mask], cociente_prom[mask])
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("CH2 / CH1")
plt.title("Respuesta en frecuencia usando amplitud")
plt.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()