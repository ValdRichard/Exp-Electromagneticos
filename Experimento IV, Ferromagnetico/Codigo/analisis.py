import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from funciones import graficarDatos, graficarAjuste

# ── Lectura de datos ─────────────────────────────────────────────────────────

serie1 = pd.read_csv(
    r".\Experimento IV, Ferromagnetico\Codigo\Datos\datos1a.dat",
    names=["T", "Ch1", "Ch2"],
    skiprows=1,
)
serie1 = serie1.apply(pd.to_numeric, errors="coerce").dropna()

serie2 = pd.read_csv(
    r".\Experimento IV, Ferromagnetico\Codigo\Datos\datos2a.dat",
    names=["T", "Ch1", "Ch2"],
    skiprows=1,
)
serie2 = serie2.apply(pd.to_numeric, errors="coerce").dropna()

# ── Graficado Inicial ──────────────

# fig1 = graficarDatos(serie1, "1ª serie de medidas", mostrar= True)
# fig2 = graficarDatos(serie2, "2ª serie de medidas",  mostrar= True)

# ── Análisis cerca de TC ───────────────────────────────────────────────────────────────
serie1Tc = serie1.iloc[7:] 
serie2Tc = serie2.iloc[7:] 
res1 = ajustarMagnetizacionTc(serie1Tc)
res2 = ajustarMagnetizacionTc(serie2Tc)
 
print("── Serie 1 ──────────────────────────────")
print(f"  A    = {res1.beta[0]:.4f} ± {res1.sd_beta[0]:.4f}")
print(f"  Tc   = {res1.beta[1]:.4f} ± {res1.sd_beta[1]:.4f} °C")
print(f"  beta = {res1.beta[2]:.4f} ± {res1.sd_beta[2]:.4f}")
 
print("── Serie 2 ──────────────────────────────")
print(f"  A    = {res2.beta[0]:.4f} ± {res2.sd_beta[0]:.4f}")
print(f"  Tc   = {res2.beta[1]:.4f} ± {res2.sd_beta[1]:.4f} °C")
print(f"  beta = {res2.beta[2]:.4f} ± {res2.sd_beta[2]:.4f}")
 
# ── Graficado con ajuste ─────────────────────────────────────────────────────
 
fig3 = graficarAjuste(serie1Tc, res1, "1ª serie — ajuste ODR")
fig4 = graficarAjuste(serie2Tc, res2, "2ª serie — ajuste ODR")
 
plt.show()

# ── Análisis Bloch (Lejos de Tc) fijo ───────────────────────────────────────────────────────────────
serie1BlochFijo = serie1.iloc[:-10] 
serie2BlochFijo = serie2.iloc[:-10] 
res1 = ajustarMagnetizacionBlochFijo(serie1BlochFijo)
res2 = ajustarMagnetizacionBlochLibre(serie2BlochFijo)
 
print("── Serie 1 ──────────────────────────────")
print(f"  A    = {res1.beta[0]:.4f} ± {res1.sd_beta[0]:.4f}")
print(f"  Tc   = {res1.beta[1]:.4f} ± {res1.sd_beta[1]:.4f} °C")
print(f"  beta = {res1.beta[2]:.4f} ± {res1.sd_beta[2]:.4f}")
 
print("── Serie 2 ──────────────────────────────")
print(f"  A    = {res2.beta[0]:.4f} ± {res2.sd_beta[0]:.4f}")
print(f"  Tc   = {res2.beta[1]:.4f} ± {res2.sd_beta[1]:.4f} °C")
print(f"  beta = {res2.beta[2]:.4f} ± {res2.sd_beta[2]:.4f}")
 
# ── Graficado con ajuste ─────────────────────────────────────────────────────
 
fig3 = graficarAjuste(serie1BlochFijo, res1, "1ª serie — ajuste ODR")
fig4 = graficarAjuste(serie2BlochFijo, res2, "2ª serie — ajuste ODR")
 
plt.show()