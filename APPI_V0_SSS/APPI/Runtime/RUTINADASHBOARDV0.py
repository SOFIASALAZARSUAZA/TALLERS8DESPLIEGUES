import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import pickle
import dash
from dash import dcc, html, Input, Output
import numpy as np
import pickle
import plotly.express as px
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------
# Rutina para DASHBOARD Control Dosis Coagulante
# version 0.0
# SOFIA SALAZAR SUAZA 
# 10/11/2024
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# # Variables, modelos y rangos predefinidos aquí...
# ---------------------------------------------------------------

# Datos manuales para el diccionario de rangos de variables
variable_ranges = {
    'MANGANESOS_MG_L_MN_CRU': {'min': 0.0, 'max': 2.25, 'mean': 0.0994644380952381},
    'HIERRO_TOTAL_MG_L_FE_3_CRU': {'min': 0.3, 'max': 7.2736, 'mean': 1.323865996343693},
    'OXIGENO_DISUELTO_MG_L_O2_CRU': {'min': 0.01, 'max': 6.61, 'mean': 1.5854296160877515},
    'CLORO_LIBRE_MG_L_CL2_MEZ': {'min': 0.0, 'max': 1.18, 'mean': 0.2500027100271003},
    'CONDUCTIVIDAD_US_CM_BN': {'min': 1.32, 'max': 380.0, 'mean': 141.76025594149908},
    'POTENCIAL_REDOX_MV_BN': {'min': 30, 'max': 385, 'mean': 296.672760511883},
    'TEMPERATURA_C_BN': {'min': 2.14, 'max': 22.8, 'mean': 17.03410256410256},
    'ALCALINIDAD_TOTAL_MG_L_CACO3_CRU': {'min': 0.04, 'max': 50.76, 'mean': 32.70703839122486},
    'DUREZA_CALCICA_MG_L_CACO3_CRU': {'min': 10.11, 'max': 39.19, 'mean': 20.340822669104206},
    'DUREZA_TOTAL_MG_L_CACO3_BN': {'min': 0.0, 'max': 690.0, 'mean': 40.34882998171846},
    'OXIGENO_DISUELTO_MG_L_O2_BN': {'min': 0.01, 'max': 6.61, 'mean': 2.236725274725275},
    'POTENCIAL_REDOX_MV_CRU': {'min': 30, 'max': 3085, 'mean': 303.57769652650825},
    'CONDUCTIVIDAD_US_CM_CRU': {'min': 1.32, 'max': 380.0, 'mean': 140.85585009140766},
    'CLORUROS_MG_L_CL_CRU': {'min': 1.44, 'max': 45.75, 'mean': 13.6485},
    'MANGANESOS_MG_L_MN_BN': {'min': 0.0, 'max': 2.71, 'mean': 0.12875274809160306},
    'PH_CRU': {'min': 0.04, 'max': 14.00, 'mean': 6.7286197440585},
    'SULFATOS_MG_L_SO4_CRU': {'min': 0.0, 'max': 22.68, 'mean': 5.155056524719227},
    'DUREZA_CALCICA_MG_L_CACO3_BN': {'min': 7.51, 'max': 45.92, 'mean': 20.060877513711155},
    'TURBIEDAD_UNT_BN': {'min': 3.5, 'max': 255.0, 'mean': 20.32965265082267},
    'NITRATOS_MG_L_NO3_BN' : {'min': 0.0, 'max': 38.16, 'mean': 13.80},
    'ALCALINIDAD_TOTAL_MG_L_CACO3_BN': {'min': 0.04, 'max': 54.13, 'mean': 32.2354113345521},
    'FOSFATOS_MG_L_CRU': {'min': 0.3, 'max': 2.2, 'mean': 0.6430389610389611},
    'DUREZA_TOTAL_MG_L_CACO3_CRU': {'min': 16.0, 'max': 690.0, 'mean': 40.76815356489945},
    'SULFATOS_MG_L_SO4_BN': {'min': 0.0, 'max': 15.73, 'mean': 1.8505722732459569},
    'PH_MEZ': {'min': 5.11, 'max': 14.00, 'mean': 6.183948811700183},
    'ALUMINIO_RESIDUAL_MG_L_AL_BN': {'min': 0.0, 'max': 0.228, 'mean': 0.014946983546617916},
    'SOLIDOS_SUSPENDIDOS_MG_L_CRU': {'min': 0.0, 'max': 112.5, 'mean': 8.943680998613036},
    'PH_BN': {'min': 0.04, 'max': 7.32, 'mean': 6.813510054844606},
    'COLOR_UPC_BN': {'min': 20, 'max': 2236, 'mean': 236.22669104204752},
    'FOSFATOS_MG_L_BN': {'min': 0.32, 'max': 3.59, 'mean': 0.816064935064935},
    'NITRATOS_MG_L_NO3_CRU': {'min': 1.0, 'max': 828.9, 'mean': 45.61994881170018},
    'TEMPERATURA_C_CRU': {'min': 2.82, 'max': 22.5, 'mean': 17.271297989031076},
    'HIERRO_TOTAL_MG_L_FE_3_BN': {'min': 0.27, 'max': 3.5102, 'mean': 1.485192138939671},
    'OXIGENO_DISUELTO_MG_L_O2_MEZ': {'min': 0.56, 'max': 20.1, 'mean': 4.253705667276051},
    'CLORO_TOTAL_MG_L_CL2_MEZ': {'min': 0.0, 'max': 1.83, 'mean': 0.7836585365853659},
    'NITRATOS_MG_L_NO3_BN': {'min': 0.0, 'max': 38.16, 'mean': 13.807461538461537},
    'TEMPERATURA_C_MEZ': {'min': 2.9, 'max': 165.0, 'mean': 17.438299817184642},
    'NITROGENO_AMONIACAL_G_L_BN': {'min': 207.23, 'max': 2620.5, 'mean': 1083.3401846435102},
    'COT_MG_L_CRU': {'min': 0.59, 'max': 18.25, 'mean': 5.645344827586207},
    'CLORO_COMBINADO_MG_L_CL2_MEZ': {'min': -0.02, 'max': 1.54, 'mean': 0.5298346883468834},
    'ALUMINIO_RESIDUAL_MG_L_AL_CRU': {'min': 0.0, 'max': 0.062, 'mean': 0.007049360146252286},
    'COT_MG_L_BN': {'min': 0.57, 'max': 18.25, 'mean': 5.450909090909091},
    'COLOR_UPC_CRU': {'min': 25, 'max': 1490, 'mean': 169.07312614259598},
    'NITRITOS_MG_L_NO2_CRU': {'min': 4.5, 'max': 2118.16, 'mean': 244.81729561243145},
    'NITROGENO_AMONIACAL_G_L_CRU': {'min': 66.32, 'max': 2153.7, 'mean': 910.1335831809872},
    'TURBIEDAD_UNT_CRU': {'min': 1.78, 'max': 166.0, 'mean': 13.58516453382084},
    'MATERIA_ORGANICA_MG_L_CRU': {'min': 1.44, 'max': 14.664015904572562, 'mean': 6.162650187858172},
    'CLORUROS_MG_L_CL_BN': {'min': 1.44, 'max': 32.22, 'mean': 12.912764378478663},
    'NITRITOS_MG_L_NO2_BN': {'min': 6.32, 'max': 919.27, 'mean': 383.65799817184643},
    'MATERIA_ORGANICA_MG_L_BN': {'min': 1.4, 'max': 29.0, 'mean': 6.563508996442993}
    }

# Listas de variables agrupadas
variables_importantes_captación = [
    'TEMPERATURA_C_BN', 'OXIGENO_DISUELTO_MG_L_O2_BN', 'TURBIEDAD_UNT_BN',
    'COLOR_UPC_BN', 'CONDUCTIVIDAD_US_CM_BN', 'PH_BN', 'MATERIA_ORGANICA_MG_L_BN',
    'NITROGENO_AMONIACAL_G_L_BN', 'MANGANESOS_MG_L_MN_BN', 'ALCALINIDAD_TOTAL_MG_L_CACO3_BN',
    'CLORUROS_MG_L_CL_BN', 'DUREZA_TOTAL_MG_L_CACO3_BN', 'DUREZA_CALCICA_MG_L_CACO3_BN',
    'HIERRO_TOTAL_MG_L_FE_3_BN', 'ALUMINIO_RESIDUAL_MG_L_AL_BN', 'POTENCIAL_REDOX_MV_BN',
    'NITRITOS_MG_L_NO2_BN', 'FOSFATOS_MG_L_BN', 'NITRATOS_MG_L_NO3_BN', 'SULFATOS_MG_L_SO4_BN', 'COT_MG_L_BN'
]

variables_importantes_cruda = [
    'TEMPERATURA_C_CRU', 'OXIGENO_DISUELTO_MG_L_O2_CRU', 'TURBIEDAD_UNT_CRU', 
    'COLOR_UPC_CRU', 'CONDUCTIVIDAD_US_CM_CRU', 'PH_CRU', 'MATERIA_ORGANICA_MG_L_CRU',
    'NITROGENO_AMONIACAL_G_L_CRU', 'MANGANESOS_MG_L_MN_CRU', 'ALCALINIDAD_TOTAL_MG_L_CACO3_CRU',
    'CLORUROS_MG_L_CL_CRU', 'DUREZA_TOTAL_MG_L_CACO3_CRU', 'DUREZA_CALCICA_MG_L_CACO3_CRU', 
    'HIERRO_TOTAL_MG_L_FE_3_CRU', 'ALUMINIO_RESIDUAL_MG_L_AL_CRU', 'POTENCIAL_REDOX_MV_CRU',
    'NITRITOS_MG_L_NO2_CRU', 'NITRATOS_MG_L_NO3_CRU', 'FOSFATOS_MG_L_CRU', 'SULFATOS_MG_L_SO4_CRU', 
    'COT_MG_L_CRU', 'SOLIDOS_SUSPENDIDOS_MG_L_CRU'
]

variables_importantes_mez = [
    'OXIGENO_DISUELTO_MG_L_O2_MEZ', 'TEMPERATURA_C_MEZ', 'PH_MEZ', 
    'CLORO_LIBRE_MG_L_CL2_MEZ', 'CLORO_COMBINADO_MG_L_CL2_MEZ', 'CLORO_TOTAL_MG_L_CL2_MEZ'
]

# Cargar los modelos guardados manualmente
with open("modelo_kmeans.pkl", "rb") as f:
    kmeans_model = pickle.load(f)
with open("modelo_random_forest_cluster_0.pkl", "rb") as f:
    model_cluster_0 = pickle.load(f)
with open("modelo_random_forest_cluster_1.pkl", "rb") as f:
    model_cluster_1 = pickle.load(f)

# ---------------------------------------------------------------
# Diseño del DASHBOARD
# ---------------------------------------------------------------
# Inicializar la app de Dash
app = dash.Dash(__name__)

# Variables para acumular predicciones
historico_predicciones_0 = []
historico_predicciones_1 = []

# Configurar el layout de la aplicación
app.layout = html.Div([
    # Encabezado de la aplicación, *recordar adicionar logo en imagen*
    html.Div([
        html.H1("Dashboard de Predicción de Dosis Total", style={"text-align": "center", "color": "blue"}),
        html.Div("Logo Empresa", style={"text-align": "right", "font-size": "24px", "margin-right": "10px"})
    ], style={"display": "flex", "justify-content": "space-between", "padding": "10px", "background-color": "#e0e0e0"}),

    # Sección principal
    html.Div([
        # Columna de control de variables
        html.Div([
            html.H3("Control de Variables", style={"text-align": "center"}),
            dcc.Tabs(id="tabs", children=[
                dcc.Tab(label="Variables BocToma (BN)", children=[
                    html.Div([html.Label("Seleccione los valores de las variables")] + [
                        html.Div([
                            html.Label(variable),
                            dcc.Slider(
                                id=f'bocatoma_{i}',
                                min=variable_ranges[variable]["min"],
                                max=variable_ranges[variable]["max"],
                                value=variable_ranges[variable]["mean"],
                                marks={
                                    round(variable_ranges[variable]["min"], 2): str(round(variable_ranges[variable]["min"], 2)),
                                    round(variable_ranges[variable]["max"], 2): str(round(variable_ranges[variable]["max"], 2))
                                },
                                tooltip={"placement": "bottom"}
                            )
                        ], style={"margin-bottom": "10px"}) for i, variable in enumerate(variables_importantes_captación)
                    ])
                ]),
                dcc.Tab(label="Variables Cruda (CRU)", children=[
                    html.Div([html.Label("Seleccione los valores de las variables")] + [
                        html.Div([
                            html.Label(variable),
                            dcc.Slider(
                                id=f'cruda_{i}',
                                min=variable_ranges[variable]["min"],
                                max=variable_ranges[variable]["max"],
                                value=variable_ranges[variable]["mean"],
                                marks={
                                    round(variable_ranges[variable]["min"], 2): str(round(variable_ranges[variable]["min"], 2)),
                                    round(variable_ranges[variable]["max"], 2): str(round(variable_ranges[variable]["max"], 2))
                                },
                                tooltip={"placement": "bottom"}
                            )
                        ], style={"margin-bottom": "10px"}) for i, variable in enumerate(variables_importantes_cruda)
                    ])
                ]),
                dcc.Tab(label="Variables Mezclada (Mez)", children=[
                    html.Div([html.Label("Seleccione los valores de las variables")] + [
                        html.Div([
                            html.Label(variable),
                            dcc.Slider(
                                id=f'mezclada_{i}',
                                min=variable_ranges[variable]["min"],
                                max=variable_ranges[variable]["max"],
                                value=variable_ranges[variable]["mean"],
                                marks={
                                    round(variable_ranges[variable]["min"], 2): str(round(variable_ranges[variable]["min"], 2)),
                                    round(variable_ranges[variable]["max"], 2): str(round(variable_ranges[variable]["max"], 2))
                                },
                                tooltip={"placement": "bottom"}
                            )
                        ], style={"margin-bottom": "10px"}) for i, variable in enumerate(variables_importantes_mez)
                    ])
                ])
            ])
        ], style={"width": "60%", "padding": "10px", "background-color": "#f5f5f5", "border-right": "1px solid #ddd"}),

        # Columna de gráficos de predicción
        html.Div([
            # Gráfico y predicción para Clúster 0
            html.Div([
                html.H3("Dosis Clúster 0", style={"text-align": "center", "color": "blue"}),
                html.Div(id="prediccion_cluster_0", style={"font-size": "24px", "text-align": "center", "margin-bottom": "10px", "color": "blue"}),
                dcc.Graph(id="grafico_prediccion_cluster_0")
            ], style={"padding": "10px"}),

            # Gráfico y predicción para Clúster 1
            html.Div([
                html.H3("Dosis Clúster 1", style={"text-align": "center", "color": "blue"}),
                html.Div(id="prediccion_cluster_1", style={"font-size": "24px", "text-align": "center", "margin-bottom": "10px", "color": "blue"}),
                dcc.Graph(id="grafico_prediccion_cluster_1")
            ], style={"padding": "10px"})
        ], style={"width": "40%", "padding": "10px", "background-color": "#f9f9f9"})
    ], style={"display": "flex", "flex-direction": "row"}),

    # Botón de actualización
    html.Div([
        html.Button("Actualizar Predicción", id="actualizar_button", n_clicks=0, style={"margin-top": "10px"})
    ], style={"text-align": "center", "padding": "10px"})
])

# ---------------------------------------------------------------
# Funciones del DASHBOARD
# ---------------------------------------------------------------
# Callback para actualizar la predicción y los gráficos
@app.callback(
    [
        Output("prediccion_cluster_0", "children"),
        Output("grafico_prediccion_cluster_0", "figure"),
        Output("prediccion_cluster_1", "children"),
        Output("grafico_prediccion_cluster_1", "figure")
    ],
    [Input("actualizar_button", "n_clicks")] +
    [Input(f'bocatoma_{i}', "value") for i in range(len(variables_importantes_captación))] +
    [Input(f'cruda_{i}', "value") for i in range(len(variables_importantes_cruda))] +
    [Input(f'mezclada_{i}', "value") for i in range(len(variables_importantes_mez))]
)
def actualizar_prediccion(n_clicks, *valores):
    todas_las_variables = [
        'ALUMINIO_RESIDUAL_MG_L_AL_BN', 'CLORUROS_MG_L_CL_BN', 'SULFATOS_MG_L_SO4_BN',
        'CONDUCTIVIDAD_US_CM_BN', 'OXIGENO_DISUELTO_MG_L_O2_BN', 'HIERRO_TOTAL_MG_L_FE_3_BN',
        'TEMPERATURA_C_BN', 'NITRATOS_MG_L_NO3_BN', 'NITROGENO_AMONIACAL_G_L_BN', 
        'NITRITOS_MG_L_NO2_BN', 'MANGANESOS_MG_L_MN_BN', 'COT_MG_L_BN',
        'ALCALINIDAD_TOTAL_MG_L_CACO3_BN', 'COLOR_UPC_BN', 'DUREZA_TOTAL_MG_L_CACO3_BN',
        'MATERIA_ORGANICA_MG_L_BN', 'TURBIEDAD_UNT_BN', 'PH_BN', 'POTENCIAL_REDOX_MV_BN',
        'DUREZA_CALCICA_MG_L_CACO3_BN', 'FOSFATOS_MG_L_BN', 'POTENCIAL_REDOX_MV_CRU',
        'SULFATOS_MG_L_SO4_CRU', 'SOLIDOS_SUSPENDIDOS_MG_L_CRU', 'COT_MG_L_CRU',
        'ALCALINIDAD_TOTAL_MG_L_CACO3_CRU', 'DUREZA_CALCICA_MG_L_CACO3_CRU', 
        'MANGANESOS_MG_L_MN_CRU', 'OXIGENO_DISUELTO_MG_L_O2_CRU', 'NITRATOS_MG_L_NO3_CRU',
        'MATERIA_ORGANICA_MG_L_CRU', 'CONDUCTIVIDAD_US_CM_CRU', 'NITROGENO_AMONIACAL_G_L_CRU',
        'TEMPERATURA_C_CRU', 'TURBIEDAD_UNT_CRU', 'NITRITOS_MG_L_NO2_CRU',
        'FOSFATOS_MG_L_CRU', 'DUREZA_TOTAL_MG_L_CACO3_CRU', 'COLOR_UPC_CRU', 
        'CLORUROS_MG_L_CL_CRU', 'ALUMINIO_RESIDUAL_MG_L_AL_CRU', 'HIERRO_TOTAL_MG_L_FE_3_CRU',
        'PH_CRU', 'CLORO_LIBRE_MG_L_CL2_MEZ', 'TEMPERATURA_C_MEZ', 'PH_MEZ', 
        'CLORO_COMBINADO_MG_L_CL2_MEZ', 'CLORO_TOTAL_MG_L_CL2_MEZ', 'OXIGENO_DISUELTO_MG_L_O2_MEZ'
    ]
    
    # Crear un DataFrame de los valores de entrada en el orden correcto
    valores_dict = {col: valor for col, valor in zip(todas_las_variables, valores)}
    valores_df = pd.DataFrame([valores_dict], columns=todas_las_variables)

    # Predecir el clúster usando el modelo de KMeans
    cluster = kmeans_model.predict(valores_df)[0]
    
    # Actualizar predicciones y gráficos según el clúster
    if cluster == 0:
        valores_df_0 = valores_df[model_cluster_0.feature_names_in_]
        prediccion_0 = model_cluster_0.predict(valores_df_0)[0]
        historico_predicciones_0.append(prediccion_0)  # Acumular predicción
        fig_0 = px.line(x=list(range(1, len(historico_predicciones_0) + 1)), y=historico_predicciones_0,
                        title="Predicción de Dosis Total - Clúster 0 (Historial Acumulado)",
                        labels={'x': 'Número de Predicción', 'y': 'Dosis Total'},
                        color_discrete_sequence=["blue"])
        prediccion_texto_0 = f"Predicción de Dosis Total para el Clúster 0: {prediccion_0:.2f}"
        prediccion_texto_1 = "N/A"
        fig_1 = px.scatter()  # Gráfico en blanco
    else:
        valores_df_1 = valores_df[model_cluster_1.feature_names_in_]
        prediccion_1 = model_cluster_1.predict(valores_df_1)[0]
        historico_predicciones_1.append(prediccion_1)  # Acumular predicción
        fig_1 = px.line(x=list(range(1, len(historico_predicciones_1) + 1)), y=historico_predicciones_1,
                        title="Predicción de Dosis Total - Clúster 1 (Historial Acumulado)",
                        labels={'x': 'Número de Predicción', 'y': 'Dosis Total'},
                        color_discrete_sequence=["blue"])
        prediccion_texto_1 = f"Predicción de Dosis Total para el Clúster 1: {prediccion_1:.2f}"
        prediccion_texto_0 = "N/A"
        fig_0 = px.scatter()  # Gráfico en blanco

    return prediccion_texto_0, fig_0, prediccion_texto_1, fig_1

# Ejecutar la aplicación * Recordar se usa ip local *
if __name__ == "__main__":
    app.run_server(debug=True, host="192.168.1.4", port=8050)  # Permite acceso desde cualquier dispositivo en la red
