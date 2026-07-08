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


espectro_ch1 = promediar_espectros_desde_fft(
    resultados,
    canal="ch1",
    archivos=range(1, 61)
)

espectro_ch2 = promediar_espectros_desde_fft(
    resultados,
    canal="ch2",
    archivos=range(1, 61)
)

freq_ch1 = espectro_ch1["freq"]
psd_ch1_prom = espectro_ch1["psd_prom"]

freq_ch2 = espectro_ch2["freq"]
psd_ch2_prom = espectro_ch2["psd_prom"]

plt.figure(figsize=(8, 5))
plt.plot(freq_ch1, psd_ch1_prom)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [mV$^2$/Hz]")
plt.title("Espectro de potencia promedio - CH1")
plt.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(freq_ch2, psd_ch2_prom)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [mV$^2$/Hz]")
plt.title("Espectro de potencia promedio - CH2")
plt.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()

eps = 1e-12

respuesta_pot = np.divide(
    psd_ch2_prom,
    psd_ch1_prom,
    out=np.zeros_like(psd_ch2_prom),
    where=psd_ch1_prom > eps
)

mask = freq_ch1 > 0

plt.figure(figsize=(8, 5))
plt.plot(freq_ch1[mask], respuesta_pot[mask])
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("CH2 / CH1")
plt.title("Respuesta en frecuencia usando potencia")
plt.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()