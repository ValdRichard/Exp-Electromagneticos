from funciones import *

re_z, im_z, sre_z, sim_z = cargar_impedancia_csv(
    "datos.csv"
)
resultado = graficar_ajuste_cpe_ordenada(
    re_z,
    im_z,
    sre_z,
    sim_z
)


resultado = graficar_ajuste_cpe(
    re_z,
    im_z,
    sre_z,
    sim_z,
    n0=0.9
)