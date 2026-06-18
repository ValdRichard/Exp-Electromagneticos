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
serie2Bloch = serie2.iloc[:-13]

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

fig, axs = plt.subplots(
    1, 2,
    figsize=(12, 5),
    sharey=True
)

graficarAjusteBlochFijo(
    serie1Bloch,
    resBlochFijo1,
    "Serie 1",
    ax=axs[0]
)

graficarAjusteBlochFijo(
    serie2Bloch,
    resBlochFijo2,
    "Serie 2",
    ax=axs[1]
)

plt.tight_layout()
plt.show()