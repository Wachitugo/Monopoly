import pandas as pd
from sklearn.impute import KNNImputer
from scipy.stats import zscore
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

print('0')

# Cargar el dataset
df = pd.read_excel('data/Base_clientes_Monopoly-0.xlsx', sheet_name=0, header=1)
print('1')
# Diccionario para renombrar las columnas
nombres_columnas = {
    'TxsCN_T12': 'tx_compras_nac',
    'TxsCI_T12': 'tx_compras_int',
    'TxsAN_T12': 'tx_avances_nac',
    'TxsAI_T12': 'tx_avances_int',
    'FacCN_T12': 'monto_compras_nac',
    'FacCI_T12': 'monto_compras_int',
    'FacAN_T12': 'monto_avances_nac',
    'FacAI_T12': 'monto_avances_int',
    'FlgAct_T12': 'flag_actividad',
    'FlgActCN_T12': 'flag_compras_nac',
    'FlgActCI_T12': 'flag_compras_int',
    'FlgActAN_T12': 'flag_avances_nac',
    'ColL1TE_T12': 'morosidad',
    'PagoNac_T12': 'pagos_nacional'
}
print('2')

# Seleccionar y renombrar columnas
df_analisis = df[nombres_columnas.keys()].copy()
df_analisis.rename(columns=nombres_columnas, inplace=True)

# Corregir transacciones negativas a 0
df_analisis['tx_compras_nac'] = df_analisis['tx_compras_nac'].clip(lower=0)

print('3')

# Inicializar el imputador
imputer_montos = KNNImputer(n_neighbors=5)
print('4')

columnas_montos = [
    'monto_compras_nac', 'monto_compras_int',
    'monto_avances_nac', 'monto_avances_int',
    'morosidad', 'pagos_nacional'
]

# Imputar valores faltantes
df_montos_imputado = pd.DataFrame(
    imputer_montos.fit_transform(df_analisis[columnas_montos]),
    columns=columnas_montos
)
print('5')

def tratar_atipicos_zscore(df, columnas_numericas, umbral=3):
    df_tratado = df.copy()
    
    for columna in columnas_numericas:
        # Calcular z-scores
        z_scores = zscore(df[columna])
        abs_z_scores = np.abs(z_scores)
        
        # Identificar y reemplazar valores atípicos con la mediana
        mask = abs_z_scores > umbral
        df_tratado.loc[mask, columna] = df[columna].median()
        
        # Mostrar información sobre los cambios
        n_outliers = mask.sum()
    
    return df_tratado

scaler = StandardScaler()

# Aplicar tratamiento de valores atípicos
df_outliers_morosidad = tratar_atipicos_zscore(df_montos_imputado, columnas_montos)
print('6')

# Convertir el resultado en un DataFrame
df_outliers_scaled = pd.DataFrame(df_outliers_morosidad, columns=df_outliers_morosidad.columns)

# Guardar los datos transformados en un archivo pickle
data_to_save = {'df_outliers_morosidad': df_outliers_scaled}
with open('checkpoints/datos.pkl', 'wb') as file:
    pickle.dump(data_to_save, file)