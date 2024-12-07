# Monopoly

## Instrucciones para Iniciar

### Clonar el Repositorio

Abre la terminal o el símbolo del sistema y navega al directorio donde deseas clonar el repositorio. Luego, ejecuta el siguiente comando para clonar el repositorio desde Git:

```bash
git clone https://github.com/Wachitugo/Monopoly.git
```

El proyecto se estructura de la siguiente manera:

```
Monopoly
├── checkpoints
│   ├── datos.pkl
│   ├── gb_model.pkl
├── data
│   ├── Base_clientes_Monopoly-0.xlsx
├── model
│   ├── data.py
│   ├── model.py
├── notebook
│   ├── monopoly.ipynb
├── templates
│   ├── index.html
├── .gitignore
├── README.md
├── requirements.txt
└── script.py
```

### Crear Entorno Virtual

Navega al directorio del proyecto en la terminal. Crea un entorno virtual ejecutando:

```bash
python -m venv venv
```

Esto creará un nuevo directorio `venv` en tu proyecto que contendrá el entorno virtual.

### Instalar Dependencias

Activa el entorno virtual:

En Windows:

```bash
.\env\Scripts\Activate.ps1
```

En macOS y Linux:

```bash
source env/bin/activate
```

Instala las dependencias del proyecto utilizando el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Ejecutar el Proyecto

Ejecuta el script Python principal con:

```bash
python script.py
```