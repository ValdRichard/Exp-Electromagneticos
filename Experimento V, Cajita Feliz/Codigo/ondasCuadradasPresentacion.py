from pathlib import Path
import pandas as pd

from funcionesCuadradas import (
    buscar_archivo,
    procesar_archivo_presentacion,
)

# =============================================================================
# RUTAS
# =============================================================================

CARPETA_CODIGO = Path(__file__).resolve().parent
CARPETA_DATOS = CARPETA_CODIGO

CARPETA_SALIDA = (
    CARPETA_CODIGO
    / "ImagenesPresentaciónOndasCuadradas"
)

# =============================================================================
# ARCHIVOS EXACTOS A PROCESAR
# =============================================================================

NOMBRES = [
    "444Hz___100KSs(1)",
    "444Hz___100KSs(2)",
    "444Hz___100KSs(3)",

    "1_132kHz___500KSs(1)",
    "1_132kHz___500KSs(2)",
    "1_132kHz___500KSs(3)",

    "3_170kHz___1MSs(1)",
    "3_170kHz___1MSs(2)",
    "3_170kHz___1MSs(3)",

    "6_703kHz___2_5MSs(1)",
    "6_703kHz___2_5MSs(2)",
    "6_703kHz___2_5MSs(3)",

    "12Hz___2_5KSs(1)",
    "12Hz___2_5KSs(2)",
    "12Hz___2_5KSs(3)",

    "13_700kHz___5MSs(1)",
    "13_700kHz___5MSs(2)",
    "13_700kHz___5MSs(3)",

    "25Hz___5KSs(1)",
    "25Hz___5KSs(2)",
    "25Hz___5KSs(3)",

    "38kHz___10MSs(1)",
    "38kHz___10MSs(2)",
    "38kHz___10MSs(3)",

    "95Hz___25KSs(1)",
    "95Hz___25KSs(2)",
    "95Hz___25KSs(3)",

    "143Hz___50KSs(1)",
    "143Hz___50KSs(2)",
    "143Hz___50KSs(3)",
]

# =============================================================================
# PROCESAMIENTO
# =============================================================================

def main():
    CARPETA_SALIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    resultados = []
    errores = []

    print(f"Buscando datos dentro de:\n{CARPETA_DATOS}")
    print(f"\nGuardando imágenes de presentación en:\n{CARPETA_SALIDA}")

    for nombre in NOMBRES:
        ruta = buscar_archivo(
            CARPETA_DATOS,
            nombre,
        )

        if ruta is None:
            mensaje = (
                f"No se encontró: {nombre}.txt "
                f"ni {nombre}.dat"
            )

            print(f"\nERROR: {mensaje}")

            errores.append(
                {
                    "archivo": nombre,
                    "error": mensaje,
                }
            )
            continue

        print(f"\nProcesando:\n{ruta}")

        try:
            resultado = procesar_archivo_presentacion(
                ruta,
                CARPETA_SALIDA,
            )

            resultados.append(resultado)

            print(
                f"  frecuencia = "
                f"{resultado['frecuencia_Hz']} Hz"
            )
            print(
                f"  modelo = "
                f"{resultado['modelo']}"
            )
            print(
                f"  imagen = "
                f"{resultado['imagen']}"
            )

        except Exception as error:
            mensaje = str(error)

            print(
                f"\nERROR procesando {ruta.name}:\n"
                f"{mensaje}"
            )

            errores.append(
                {
                    "archivo": ruta.name,
                    "error": mensaje,
                }
            )

    if resultados:
        resumen = pd.DataFrame(resultados)

        ruta_csv = (
            CARPETA_SALIDA
            / "resumen_presentacion.csv"
        )

        resumen.to_csv(
            ruta_csv,
            index=False,
        )

        print(
            "\nResumen de imágenes guardado en:\n"
            f"{ruta_csv}"
        )

    if errores:
        pd.DataFrame(errores).to_csv(
            CARPETA_SALIDA
            / "errores_presentacion.csv",
            index=False,
        )

        print(
            "\nErrores guardados en:\n"
            f"{CARPETA_SALIDA / 'errores_presentacion.csv'}"
        )


if __name__ == "__main__":
    main()