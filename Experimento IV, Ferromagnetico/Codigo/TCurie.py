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


x = np.array([0, 1])

y = np.array([45, 300])

sy = 0.05 * y

a, sa, b, sb = ajuste_recta(x, y, sy)

print(f"a = {a:.2f} ± {sa:.2f}")
print(f"b = {b:.2f} ± {sb:.2f}")

# =============================================================================
# COMPOSICION Mn/Zn A PARTIR DE Tc
# =============================================================================

def composicion_mn(tc, stc, a, sa, b, sb):

    x = (tc - a) / b

    sx = np.sqrt(
        (stc / b)**2 +
        (sa  / b)**2 +
        (x * sb / b)**2
    )

    return x, sx


print("\n===== COMPOSICION ESTIMADA =====\n")

x1, sx1 = composicion_mn(
    resTc1.beta[1],
    resTc1.sd_beta[1],
    a, sa,
    b, sb
)

x2, sx2 = composicion_mn(
    resTc2.beta[1],
    resTc2.sd_beta[1],
    a, sa,
    b, sb
)

print("Serie 1")
print(f"Mn = {x1:.5f} ± {sx1:.5f}")
print(f"Zn = {1-x1:.5f} ± {sx1:.5f}")

print()

print("Serie 2")
print(f"Mn = {x2:.5f} ± {sx2:.5f}")
print(f"Zn = {1-x2:.5f} ± {sx2:.5f}")

ratio1 = x1 / (1 - x1)
ratio2 = x2 / (1 - x2)

print(f"Mn:Zn ≈ {ratio1:.5f}:1")
print(f"Mn:Zn ≈ {ratio2:.5f}:1")