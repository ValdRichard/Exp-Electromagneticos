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

# =============================================================================
# CALIBRACION Tc vs COMPOSICION
# =============================================================================

x = np.array([0, 1])

y = np.array([45, 300])

sy = 0.05 * y

a, sa, b, sb = ajuste_recta(x, y, sy)

print("\n===== CALIBRACION =====\n")
print(f"a = {a:.2f} ± {sa:.2f}")
print(f"b = {b:.2f} ± {sb:.2f}")


# =============================================================================
# COMPOSICION A PARTIR DE Tc
# =============================================================================

def composicion_mn(tc, stc, a, sa, b, sb):

    x = (tc - a) / b

    sx = np.sqrt(
        (stc / b)**2 +
        (sa / b)**2 +
        (x * sb / b)**2
    )

    return x, sx


# =============================================================================
# PROMEDIO PESADO DE Tc
# =============================================================================

Tc_vals = np.array([
    resBlochFijo1.beta[1],
    resBlochFijo2.beta[1]
])

sTc_vals = np.array([
    resBlochFijo1.sd_beta[1],
    resBlochFijo2.sd_beta[1]
])

pesos = 1 / sTc_vals**2

Tc_prom = np.sum(
    pesos * Tc_vals
) / np.sum(pesos)

sTc_prom = np.sqrt(
    1 / np.sum(pesos)
)

print("\n===== Tc PROMEDIO (BLOCH) =====\n")
print(f"Tc = {Tc_prom:.2f} ± {sTc_prom:.2f} °C")


# =============================================================================
# COMPOSICION FINAL
# =============================================================================

x_prom, sx_prom = composicion_mn(
    Tc_prom,
    sTc_prom,
    a, sa,
    b, sb
)

print("\n===== COMPOSICION ESTIMADA =====\n")

print(f"Mn = {x_prom:.5f} ± {sx_prom:.5f}")
print(f"Zn = {1 - x_prom:.5f} ± {sx_prom:.5f}")

ratio = x_prom / (1 - x_prom)

print(f"Mn:Zn ≈ {ratio:.5f}:1")


# =============================================================================
# GRAFICO CALIBRACION
# =============================================================================

from matplotlib.lines import Line2D

x_recta = np.linspace(-0.1, 1.1, 200)
y_recta = a + b * x_recta

fig, ax = plt.subplots(figsize=(8, 6))

# Datos de calibración
eb = ax.errorbar(
    x,
    y,
    yerr=sy,
    fmt='o',
    ms=6,
    color='0.25',
    ecolor='0.25',
    capsize=3,
    label='Datos de calibración'
)

for barra in eb[2]:
    barra.set_alpha(0.4)

for cap in eb[1]:
    cap.set_alpha(0.4)

# Recta de calibración
ax.plot(
    x_recta,
    y_recta,
    lw=2.5,
    color='#355C7D',
    label=r'$T_c(x)=45+255x$'
)

# Resultado final
ax.errorbar(
    x_prom,
    Tc_prom,
    xerr=sx_prom,
    yerr=sTc_prom,
    fmt='D',
    ms=9,
    color='#1f4e79',
    ecolor='#1f4e79',
    capsize=4,
    label='Temperatura de Curie promedio'
)

extra_labels = [
    Line2D(
        [],
        [],
        linestyle='None',
        label=rf'Porcentaje Mn = ({100*x_prom:.1f} ± {100*sx_prom:.1f})%'
    )
]
handles, labels = ax.get_legend_handles_labels()
handles.extend(extra_labels)

ax.set_xlabel('Fracción de Mn')
ax.set_ylabel(r'$T_c$ [°C]')

ax.grid(True, alpha=0.3)
ax.legend(handles=handles)

plt.tight_layout()
plt.show()