import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------
# Rutina para importación y limpieza de datos 
# version 0.0
# SOFIA SALAZAR SUAZA 
# 10/11/2024
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# Obtención y limpieza de datos
# ---------------------------------------------------------------

current_dir = os.getcwd()
file_path = os.path.join(current_dir, "Data", "DATOS2023.xlsx")
file_path1 = os.path.join(current_dir, "Data", "DATOS2024.xlsx")

df_2023 = pd.read_excel(file_path)
df_2024 = pd.read_excel(file_path1)

df_2023 = df_2023.rename(columns={"Unnamed: 0": "DÍA"})
df_2024 = df_2024.rename(columns={"Unnamed: 0": "DÍA"})

df_total = pd.concat([df_2023, df_2024], ignore_index=True)
df_total.columns = df_total.columns.str.strip().str.upper()

df_total.replace('ND', 0, inplace=True)
df_total.replace(['X', 'S'], pd.NA, inplace=True)
df_total['MATERIA ORGANICA MG/L BN'].fillna(df_total['COT  MG/L  BN'], inplace=True)
df_total = df_total.apply(pd.to_numeric, errors='coerce')
df_total.fillna(df_total.mean(), inplace=True)
df_total['FECHA'] = pd.to_datetime(df_total['FECHA'], errors='coerce')
df_total = df_total.loc[:, ~df_total.columns.duplicated()]
df_total.drop(columns=['MES'], inplace=True)
df_total.columns = df_total.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)

# ---------------------------------------------------------------
# Selección de variables importantes sin duplicados
# ---------------------------------------------------------------

variables_importantes_dos = ['AL_SO_PPM', 'AL_SO_SOLIDO_PPM', 'PAC_PPM', 'FECL3_PPM']

variables_importantes_captación = list(map(str, set([
    'TEMPERATURA_C_BN', 'OXIGENO_DISUELTO_MG_L_O2_BN', 'TURBIEDAD_UNT_BN',
    'COLOR_UPC_BN', 'CONDUCTIVIDAD_US_CM_BN', 'PH_BN', 'MATERIA_ORGANICA_MG_L_BN',
    'NITROGENO_AMONIACAL_G_L_BN', 'MANGANESOS_MG_L_MN_BN', 'ALCALINIDAD_TOTAL_MG_L_CACO3_BN',
    'CLORUROS_MG_L_CL_BN', 'DUREZA_TOTAL_MG_L_CACO3_BN', 'DUREZA_CALCICA_MG_L_CACO3_BN',
    'HIERRO_TOTAL_MG_L_FE_3_BN', 'ALUMINIO_RESIDUAL_MG_L_AL_BN', 'POTENCIAL_REDOX_MV_BN',
    'NITRITOS_MG_L_NO2_BN', 'FOSFATOS_MG_L_BN', 'NITRATOS_MG_L_NO3_BN', 'SULFATOS_MG_L_SO4_BN', 'COT_MG_L_BN'
])))

variables_importantes_cruda = list(map(str, set([
    'TEMPERATURA_C_CRU', 'OXIGENO_DISUELTO_MG_L_O2_CRU', 'TURBIEDAD_UNT_CRU', 
    'COLOR_UPC_CRU', 'CONDUCTIVIDAD_US_CM_CRU', 'PH_CRU', 'MATERIA_ORGANICA_MG_L_CRU',
    'NITROGENO_AMONIACAL_G_L_CRU', 'MANGANESOS_MG_L_MN_CRU', 'ALCALINIDAD_TOTAL_MG_L_CACO3_CRU',
    'CLORUROS_MG_L_CL_CRU', 'DUREZA_TOTAL_MG_L_CACO3_CRU', 'DUREZA_CALCICA_MG_L_CACO3_CRU', 
    'HIERRO_TOTAL_MG_L_FE_3_CRU', 'ALUMINIO_RESIDUAL_MG_L_AL_CRU', 'POTENCIAL_REDOX_MV_CRU',
    'NITRITOS_MG_L_NO2_CRU', 'NITRATOS_MG_L_NO3_CRU', 'FOSFATOS_MG_L_CRU', 'SULFATOS_MG_L_SO4_CRU', 
    'COT_MG_L_CRU', 'SOLIDOS_SUSPENDIDOS_MG_L_CRU'
])))

variables_importantes_mez = list(map(str, set([
    'OXIGENO_DISUELTO_MG_L_O2_MEZ', 'TEMPERATURA_C_MEZ', 'PH_MEZ', 
    'CLORO_LIBRE_MG_L_CL2_MEZ', 'CLORO_COMBINADO_MG_L_CL2_MEZ', 'CLORO_TOTAL_MG_L_CL2_MEZ'
])))

# Crear una lista única de todas las variables
todas_las_variables = list(map(str, set(variables_importantes_captación + variables_importantes_cruda + variables_importantes_mez)))

# ---------------------------------------------------------------
# Estandarización de las variables
# ---------------------------------------------------------------

scaler = StandardScaler()
df_standardized_global = df_total.copy()
df_standardized_global[todas_las_variables] = scaler.fit_transform(df_total[todas_las_variables])

# Definir X e y después de la estandarización
X = df_standardized_global[todas_las_variables]
df_standardized_global['DOSIS_TOTAL'] = (
    df_standardized_global['PAC_PPM'] + df_standardized_global['AL_SO_PPM'] + 
    df_standardized_global['AL_SO_SOLIDO_PPM'] + df_standardized_global['FECL3_PPM']
)
y = df_standardized_global['DOSIS_TOTAL']

# ---------------------------------------------------------------
# Rango de valores mínimo, máximo y promedio para cada variable
# ---------------------------------------------------------------

variable_ranges = {
    str(col): {  # Convertir clave a cadena explícitamente
        "min": float(df_standardized_global[col].min().item()),
        "max": float(df_standardized_global[col].max().item()),
        "mean": float(df_standardized_global[col].mean().item())
    }
    for col in todas_las_variables
}
