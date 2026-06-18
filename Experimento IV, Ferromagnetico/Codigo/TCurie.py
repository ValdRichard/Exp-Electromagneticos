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

fig, axs = plt.subplots(
    1, 2,
    figsize=(12, 5),
    sharey=True
)

graficarAjusteTc(
    serie1Tc,
    resTc1,
    "serie 1",
    ax=axs[0]
)

graficarAjusteTc(
    serie2Tc,
    resTc2,
    "serie 2",
    ax=axs[1]
)

plt.tight_layout()
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

# =============================================================================
# COMPOSICION Mn/Zn A PARTIR DE Tc
# =============================================================================

from matplotlib.lines import Line2D

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

# =============================================================================
# GRAFICO CALIBRACION Tc vs COMPOSICION
# =============================================================================

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

# Barras de error más transparentes
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

# Temperatura de Curie - Serie 1
ax.errorbar(
    x1,
    resTc1.beta[1],
    xerr=sx1,
    yerr=resTc1.sd_beta[1],
    fmt='s',
    ms=8,
    color='#1f4e79',
    ecolor='#1f4e79',
    capsize=3,
    label='Temperatura de Curie, serie 1'
)

# Temperatura de Curie - Serie 2
ax.errorbar(
    x2,
    resTc2.beta[1],
    xerr=sx2,
    yerr=resTc2.sd_beta[1],
    fmt='^',
    ms=8,
    color='#4f81bd',
    ecolor='#4f81bd',
    capsize=3,
    label='Temperatura de Curie, serie 2'
)

# Líneas de texto extra para la leyenda
extra_labels = [
    Line2D(
        [], [],
        linestyle='None',
        label=rf'Porcentaje serie 1: Mn = {100*x1:.1f} ± {100*sx1:.1f}%'
    ),
    Line2D(
        [], [],
        linestyle='None',
        label=rf'Porcentaje serie 2: Mn = {100*x2:.1f} ± {100*sx2:.1f}%'
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

# =============================================================================
# RESULTADOS
# =============================================================================

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