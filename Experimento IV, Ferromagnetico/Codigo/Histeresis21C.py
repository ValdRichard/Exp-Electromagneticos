from funciones import *
# Código que sirve para mostrar si los cortes de periodos estaban bien
# graficar_ciclo_individual(res21, 0)
# graficar_todos_los_ciclos(res21)
# verificar_periodo(res21)
# verificar_ciclos(res21)

res21 = analizar_histeresis(
    temperatura=21,
    frecuencia=700.040
)


res61 = analizar_histeresis(
    temperatura=61,
    frecuencia=700.046
)

res81 = analizar_histeresis(
    temperatura=81,
    frecuencia=700.007
)

res101 = analizar_histeresis(
    temperatura=101,
    frecuencia= 699.996  
)

res121 = analizar_histeresis(
    temperatura=121,
    frecuencia= 700.007
)

res141 = analizar_histeresis(
    temperatura=141,
    frecuencia= 699.998
)




graficar_histeresis(
    res21,
    res61,
    res81,
    res101,
    res121,
    res141
)

graficar_histeresis_error(
    res21,
    res61,
    res81,
    res101,
    res121,
    res141
)