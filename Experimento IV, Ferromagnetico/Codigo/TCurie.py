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
# AJUSTE CRITICO
# =============================================================================

serie1Tc = serie1.iloc[9:-1]
serie2Tc = serie2.iloc[6:-5]

resTc1 = ajustarMagnetizacionTc(serie1Tc)
resTc2 = ajustarMagnetizacionTc(serie2Tc)

print("\n===== AJUSTE CRITICO =====\n")

print("Serie 1")
print(f"A     = {resTc1.beta[0]:.4f} ± {resTc1.sd_beta[0]:.4f}")
print(f"Tc    = {resTc1.beta[1]:.4f} ± {resTc1.sd_beta[1]:.4f}")
print(f"beta  = {resTc1.beta[2]:.4f} ± {resTc1.sd_beta[2]:.4f}")

print()

print("Serie 2")
print(f"A     = {resTc2.beta[0]:.4f} ± {resTc2.sd_beta[0]:.4f}")
print(f"Tc    = {resTc2.beta[1]:.4f} ± {resTc2.sd_beta[1]:.4f}")
print(f"beta  = {resTc2.beta[2]:.4f} ± {resTc2.sd_beta[2]:.4f}")
graficarAjusteTc(
    serie1Tc,
    resTc1,
    "Serie 1 - Ajuste crítico"
)
plt.show()
graficarAjusteTc(
    serie2Tc,
    resTc2,
    "Serie 2 - Ajuste crítico"
)
plt.show()
