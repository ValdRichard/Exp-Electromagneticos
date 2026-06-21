import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==========================================================
# PEGAR ACA LOS Tc PROMEDIO
# ==========================================================

Tc_heis  = 129.57
sTc_heis = 0.33

Tc_bloch  = 167.63
sTc_bloch = 0.9


# ==========================================================
# CALIBRACION
# ==========================================================

x_cal = np.array([0, 1])

Tc_cal = np.array([45, 300])

sTc_cal = 0.05 * Tc_cal


# ==========================================================
# AJUSTE LINEAL
# ==========================================================

b = (Tc_cal[1] - Tc_cal[0])/(x_cal[1] - x_cal[0])
a = Tc_cal[0]

# errores (como venías haciendo)
sa = sTc_cal[0]
sb = np.sqrt(sTc_cal[0]**2 + sTc_cal[1]**2)


# ==========================================================
# FUNCION DE COMPOSICION
# ==========================================================

def composicion_mn(tc, stc, a, sa, b, sb):

    x = (tc - a)/b

    sx = np.sqrt(
        (stc/b)**2 +
        (sa/b)**2 +
        (x*sb/b)**2
    )

    return x, sx


# ==========================================================
# COMPOSICIONES
# ==========================================================

x_heis, sx_heis = composicion_mn(
    Tc_heis, sTc_heis,
    a, sa,
    b, sb
)

x_bloch, sx_bloch = composicion_mn(
    Tc_bloch, sTc_bloch,
    a, sa,
    b, sb
)


# ==========================================================
# GRAFICO
# ==========================================================

x_recta = np.linspace(-0.1, 1.1, 200)
y_recta = a + b*x_recta

fig, ax = plt.subplots(figsize=(8,6))

# calibración
ax.errorbar(
    x_cal,
    Tc_cal,
    yerr=sTc_cal,
    fmt='o',
    color='0.25',
    ecolor='0.25',
    capsize=3,
    label='Datos de calibración'
)

# recta
ax.plot(
    x_recta,
    y_recta,
    color='#355C7D',
    lw=2.5,
    label=r'$T_c(x)=45+255x$'
)

# Heisenberg
ax.errorbar(
    x_heis,
    Tc_heis,
    xerr=sx_heis,
    yerr=sTc_heis,
    fmt='o',
    ms=8,
    capsize=4,
    color='#1f4e79',
    label='Heisenberg'
)

# Bloch
ax.errorbar(
    x_bloch,
    Tc_bloch,
    xerr=sx_bloch,
    yerr=sTc_bloch,
    fmt='s',
    ms=8,
    capsize=4,
    color='#6C5B7B',
    label='Bloch'
)

# líneas verticales
ax.axvline(
    x_heis,
    color='#1f4e79',
    linestyle='--',
    alpha=0.8
)

ax.axvline(
    x_bloch,
    color='#6C5B7B',
    linestyle='--',
    alpha=0.8
)

extra_labels = [
    Line2D(
        [], [],
        linestyle='None',
        label=rf'Heisenberg: ({100*x_heis:.1f} ± {100*sx_heis:.1f}) % Mn'
    ),
    Line2D(
        [], [],
        linestyle='None',
        label=rf'Bloch: ({100*x_bloch:.1f} ± {100*sx_bloch:.1f}) % Mn'
    )
]

handles, labels = ax.get_legend_handles_labels()
handles.extend(extra_labels)

ax.legend(handles=handles)

ax.set_xlabel('Fracción de Mn')
ax.set_ylabel(r'$T_c$ [°C]')

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ==========================================================
# RESULTADOS
# ==========================================================

print("\nHeisenberg")
print(f"Mn = {x_heis:.5f} ± {sx_heis:.5f}")

print("\nBloch")
print(f"Mn = {x_bloch:.5f} ± {sx_bloch:.5f}")

# ==========================================
# RELACION Mn:Zn Y PROPAGACION DE ERRORES
# ==========================================

def relacion_mn_zn(x, sx):
    R = x/(1 - x)
    sR = sx/(1 - x)**2
    return R, sR

R_heis, sR_heis = relacion_mn_zn(x_heis, sx_heis)
R_bloch, sR_bloch = relacion_mn_zn(x_bloch, sx_bloch)

print("\nRelación Mn:Zn")
print(f"Heisenberg = {R_heis:.3f} ± {sR_heis:.3f}")
print(f"Bloch      = {R_bloch:.3f} ± {sR_bloch:.3f}")