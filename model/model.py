import pickle
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

pickle_file_path = 'checkpoints/datos.pkl'

# Cargar el archivo pickle
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

# Acceder a las variables almacenadas en el archivo pickle
df_outliers_morosidad = data['df_outliers_morosidad']

variables_predictoras_gbc = [
    'monto_compras_nac', 'monto_compras_int',
    'monto_avances_nac', 'monto_avances_int',
    'pagos_nacional'
]
variables_predictoras = [
    # Comportamiento de consumo
    'monto_compras_nac', 'monto_compras_int',
    'monto_avances_nac', 'monto_avances_int',
    'pagos_nacional'
]

# Crear variable objetivo binaria para morosidad
df_outliers_morosidad['cliente_moroso'] = (df_outliers_morosidad['morosidad'] > df_outliers_morosidad['morosidad'].median()).astype(int)

X = df_outliers_morosidad[variables_predictoras_gbc]
y = df_outliers_morosidad['cliente_moroso']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Aplicar SMOTE al conjunto de entrenamiento
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train_balanced, y_train_balanced)

# Validación cruzada
cv_scores_gbc = cross_val_score(gb_model, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')

# Guardar el modelo entrenado en un archivo pickle
model_file_path = 'checkpoints/gb_model.pkl'
with open(model_file_path, 'wb') as file:
    pickle.dump(gb_model, file)

def realizar_prediccion(datos):
    # Cargar el modelo entrenado
    model_file_path = 'checkpoints/gb_model.pkl'
    with open(model_file_path, 'rb') as file:
        gb_model = pickle.load(file)
    
    # Realizar la predicción
    prediccion = gb_model.predict([datos])
    return prediccion[0]

