import pandas as pd
import matplotlib.pyplot as plt

from funciones import *


# =============================================================================
# LECTURA
# =============================================================================

serie1 = pd.read_csv(
    r".\Experimento IV, Ferromagnetico\Codigo\Datos\datos1a.dat",
    names=["T", "Ch1", "Ch2"],
    skiprows=1,
)

serie2 = pd.read_csv(
    r".\Experimento IV, Ferromagnetico\Codigo\Datos\datos2a.dat",
    names=["T", "Ch1", "Ch2"],
    skiprows=1,
)


# =============================================================================
# GRAFICOS CRUDOS
# =============================================================================
# Quito los puntos que se hicieron dejando enfriar el material
serie1 = serie1[:-5]
# graficarDatos(
#     serie1,
#     "Serie 1"
# )
# plt.show()
# graficarDatos(
#     serie2,
#     "Serie 2"
# )
# plt.show()



# =============================================================================
# BLOCH FIJO
# =============================================================================

serie1Bloch = serie1.iloc[:-12]
serie2Bloch = serie2.iloc[:-12]

resBlochFijo1 = ajustarMagnetizacionBlochFijo(
    serie1Bloch
)

resBlochFijo2 = ajustarMagnetizacionBlochFijo(
    serie2Bloch
)


print("\n===== BLOCH FIJO =====\n")

print("Serie 1")
print(f"A  = {resBlochFijo1.beta[0]:.4f} ± {resBlochFijo1.sd_beta[0]:.4f}")
print(f"Tc = {resBlochFijo1.beta[1]:.4f} ± {resBlochFijo1.sd_beta[1]:.4f}")

print()

print("Serie 2")
print(f"A  = {resBlochFijo2.beta[0]:.4f} ± {resBlochFijo2.sd_beta[0]:.4f}")
print(f"Tc = {resBlochFijo2.beta[1]:.4f} ± {resBlochFijo2.sd_beta[1]:.4f}")

graficarAjusteBlochFijo(
    serie1Bloch,
    resBlochFijo1,
    "Serie 1 - Bloch fijo"
)
graficarAjusteBlochFijo(
    serie2Bloch,
    resBlochFijo2,
    "Serie 2 - Bloch fijo"
)
# plt.show()

# =============================================================================
# BLOCH LIBRE
# =============================================================================

resBlochLibre1 = ajustarMagnetizacionBlochLibre(
    serie1Bloch
)

resBlochLibre2 = ajustarMagnetizacionBlochLibre(
    serie2Bloch
)

print("\n===== BLOCH LIBRE =====\n")

print("Serie 1")
print(f"A  = {resBlochLibre1.beta[0]:.4f} ± {resBlochLibre1.sd_beta[0]:.4f}")
print(f"Tc = {resBlochLibre1.beta[1]:.4f} ± {resBlochLibre1.sd_beta[1]:.4f}")
print(f"n  = {resBlochLibre1.beta[2]:.4f} ± {resBlochLibre1.sd_beta[2]:.4f}")

print()

print("Serie 2")
print(f"A  = {resBlochLibre2.beta[0]:.4f} ± {resBlochLibre2.sd_beta[0]:.4f}")
print(f"Tc = {resBlochLibre2.beta[1]:.4f} ± {resBlochLibre2.sd_beta[1]:.4f}")
print(f"n  = {resBlochLibre2.beta[2]:.4f} ± {resBlochLibre2.sd_beta[2]:.4f}")

graficarAjusteBlochLibre(
    serie1Bloch,
    resBlochLibre1,
    "Serie 1 - Bloch libre"
)
# plt.show()
graficarAjusteBlochLibre(
    serie2Bloch,
    resBlochLibre2,
    "Serie 2 - Bloch libre"
)

plt.show()