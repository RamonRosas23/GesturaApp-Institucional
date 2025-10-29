
**GesturaApp** es una aplicación innovadora diseñada para interpretar el lenguaje de señas en tiempo real utilizando una cámara **Ultra Leap** y **MediaPipe**. La aplicación recopila las posiciones de las articulaciones de las manos y, mediante el uso de redes neuronales profundas, predice las señas realizadas por el usuario. El objetivo es proporcionar una herramienta eficiente y precisa para facilitar la comunicación entre personas con discapacidad auditiva y quienes no dominan el lenguaje de señas.**GesturaApp** es una aplicación innovadora diseñada para interpretar el lenguaje de señas en tiempo real utilizando una cámara **Ultra Leap** y **MediaPipe**. La aplicación recopila las posiciones de las articulaciones de las manos y, mediante el uso de redes neuronales profundas, predice las señas realizadas por el usuario. El objetivo es proporcionar una herramienta eficiente y precisa para facilitar la comunicación entre personas con discapacidad auditiva y quienes no dominan el lenguaje de señas.



##  Instalación Rápida##  Instalación Rápida



### Opción 1: Script Automático (Recomendado)### Opción 1: Script Automático (Recomendado)

```powershell```bash

# Ejecutar el script de instalación# Ejecutar el script de instalación

.\install.ps1.\install.ps1

``````



### Opción 2: Instalación Manual### Opción 2: Instalación Manual

```bash```bash

# Crear entorno virtual# Crear entorno virtual

python -m venv venvpython -m venv venv

venv\Scripts\activatevenv\Scripts\activate



# Instalar dependencias# Instalar dependencias

pip install -r requirements.txtpip install -r requirements.txt



# Construir e instalar Leap Motion bindings# Construir e instalar Leap Motion bindings

cd leapc-cfficd leapc-cffi

python -m buildpython -m build

cd ..cd ..

pip install leapc-cffi/dist/leapc_cffi-0.0.1.tar.gzpip install leapc-cffi/dist/leapc_cffi-0.0.1.tar.gz

pip install -e leapc-python-apipip install -e leapc-python-api

``````



##  Configuración Completa del Sistema##  Verificar Instalación

```bash

### 1.  Requisitos del Sistema Previos# Verificar que todo está instalado correctamente

Antes de instalar GesturaApp, asegúrate de tener:.\verify_install.ps1

```

#### **Software Base:**

- **Windows 10/11** (64-bit)##  Ejecutar la Aplicación

- **Python 3.8-3.11** - [Descargar](https://www.python.org/downloads/)```bash

- **MySQL Server 8.0+** - [Descargar](https://dev.mysql.com/downloads/mysql/)# Opción 1: Script de ejecución

- **Git** - [Descargar](https://git-scm.com/downloads).\run_app.ps1



#### **Hardware Requerido:**# Opción 2: Comando directo

- **Cámara web** (para MediaPipe)venv\Scripts\activate

- **Ultra Leap Controller** (opcional, para mayor precisión)python Aplicacion\GesturaV4.py

- **Micrófono** (para transcripción de voz)```

- **Mínimo 8GB RAM** (para modelos de ML)

##  Funcionalidades principales:

### 2.  Configuración de Base de Datos- **Captura Ultra Leap**: Captura en tiempo real de las posiciones de las articulaciones de las manos.

- **Reconocimiento MediaPipe**: Detección adicional usando MediaPipe para mayor precisión.

#### **Paso 1: Instalar MySQL**- **Redes Neuronales**: CNN, LSTM y modelos híbridos para reconocimiento de gestos.

1. Descargar e instalar MySQL Server- **Transcripción de voz**: Integración con Google Cloud Speech-to-Text.

2. Durante la instalación, configurar:- **Interfaz moderna**: PyQt6 con estilos personalizados y animaciones.

   - **Puerto:** 3306 (predeterminado)- **Base de datos**: Sistema de usuarios con MySQL.

   - **Usuario root:** Crear contraseña segura- **Inteligencia artificial**: Integración con Google Gemini AI.

   - **Método de autenticación:** Use Strong Password Encryption

##  Tecnologías utilizadas:

#### **Paso 2: Crear Base de Datos**- **Python 3.8+**: Lenguaje principal del proyecto

```sql- **PyQt6**: Interfaz gráfica de usuario moderna

-- Ejecutar en MySQL Workbench o línea de comandos- **TensorFlow/Keras**: Redes neuronales para predicción de gestos  

CREATE DATABASE login_app CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;- **OpenCV**: Procesamiento de imágenes y video

- **MediaPipe**: Detección de manos y poses

-- Importar estructura desde el archivo SQL- **Ultra Leap SDK**: Captura de datos de manos en 3D

-- mysql -u root -p login_app < login_app.sql- **scikit-learn**: Machine learning y preprocesamiento

```- **MySQL**: Base de datos para usuarios

- **Flask**: Servidor web para autenticación

### 3.  Configuración de Variables de Entorno- **Google Cloud**: Speech-to-Text y Gemini AI

- **Pandas/NumPy**: Procesamiento de datos

#### **Paso 1: Crear archivo .env**

```bash##  Requisitos del Sistema:

# Copiar el archivo de ejemplo- Windows 10/11

copy .env.example .env- Python 3.8 o superior

```- Ultra Leap Controller (opcional)

- Cámara web

#### **Paso 2: Configurar variables críticas**- MySQL Server

Edita el archivo `.env` con tus valores reales:- Micrófono (para transcripción de voz)



```bash##  Ejecutar Ejemplos:

# ==============================================```bash

# BASE DE DATOS (OBLIGATORIO)# Probar tracking de Leap Motion

# ==============================================python examples\tracking_event_example.py

DB_HOST=127.0.0.1

DB_USER=root# Probar modelos de redes neuronales

DB_PASSWORD=tu_password_mysqlpython RedesNeuronales\ProbarModelos.py

DB_NAME=login_app```

DB_PORT=3306

##  Estructura del Proyecto:

# ==============================================```

# SEGURIDAD (OBLIGATORIO)GesturaApp/

# ==============================================├── Aplicacion/           # Aplicación principal

# Generar con: python -c "import secrets; print(secrets.token_hex(32))"│   ├── GesturaV4.py     # Archivo principal

SECRET_KEY=tu_clave_secreta_muy_larga_y_aleatoria_de_64_caracteres_minimo│   ├── transcripcion.py # Módulo de transcripción de voz

```│   └── TranscriptionWorker.py

├── Login/               # Sistema de autenticación

### 4.  Configuración de Speech-to-Text (Google Cloud)│   ├── Login.py        # Interfaz de login

│   └── servidor_flask.py # Servidor de autenticación

#### **Paso 1: Crear Proyecto en Google Cloud**├── RedesNeuronales/     # Modelos de IA v1

1. Ir a [Google Cloud Console](https://console.cloud.google.com/)├── RedesNeuronalesV2/   # Modelos de IA v2 (mejorados)

2. Crear nuevo proyecto o seleccionar existente├── leapc-cffi/         # Bindings de Leap Motion (C)

3. Habilitar **Speech-to-Text API**├── leapc-python-api/   # API de Python para Leap Motion

├── assets/             # Recursos (estilos, animaciones)

#### **Paso 2: Crear Credenciales de Servicio**├── .env               # Variables de entorno

1. Ir a **IAM & Admin > Service Accounts**├── requirements.txt   # Dependencias de Python

2. Crear nueva cuenta de servicio:└── install.ps1       # Script de instalación automática

   - **Nombre:** GesturaApp Speech Service```

   - **Rol:** Cloud Speech Client

3. Crear clave JSON y descargar##  Configuración del Entorno:



#### **Paso 3: Instalar Credenciales**### 1. Configurar variables de entorno:

```bash```bash

# Colocar el archivo JSON descargado en:# Copiar el archivo de ejemplo

credentials/google-cloud-speech.jsoncopy .env.example .env



# O configurar ruta personalizada en .env:# Editar el archivo .env con tus credenciales reales

GOOGLE_APPLICATION_CREDENTIALS=credentials/google-cloud-speech.jsonnotepad .env

``````



### 5.  Configuración de IA (Google Gemini)### 2. Variables importantes a configurar:

```bash

#### **Obtener API Key de Gemini:**# Base de datos MySQL

1. Ir a [Google AI Studio](https://aistudio.google.com/)DB_HOST=127.0.0.1

2. Crear API KeyDB_USER=tu_usuario_mysql

3. Agregar al `.env`:DB_PASSWORD=tu_password_mysql

```bashDB_NAME=login_app

GOOGLE_GEMINI_API_KEY=tu_gemini_api_key_aqui

```# APIs externas (obtener de las respectivas plataformas)

GOOGLE_CLIENT_ID=tu_google_client_id

### 6.  Configuración de Autenticación OAuthGOOGLE_CLIENT_SECRET=tu_google_client_secret

OPENAI_API_KEY=tu_openai_api_key

#### **Google OAuth (Para login social):**

1. Ir a [Google Developers Console](https://console.developers.google.com/)# Email para notificaciones

2. Crear credenciales OAuth 2.0:EMAIL_USER=tu_email@gmail.com

   - **Tipo:** Web ApplicationEMAIL_PASSWORD=tu_app_password_gmail

   - **URIs autorizados:** `http://localhost:5000`

   - **Redirect URIs:** `http://localhost:5000/callback`# Clave secreta para Flask (generar una nueva)

SECRET_KEY=tu_clave_secreta_muy_larga_y_aleatoria

```bash```

# Agregar al .env:

GOOGLE_CLIENT_ID=tu_google_client_id.apps.googleusercontent.com### 3. Guías para obtener API Keys:

GOOGLE_CLIENT_SECRET=tu_google_client_secret- **Google OAuth**: https://console.developers.google.com/

```- **OpenAI API**: https://platform.openai.com/api-keys

- **Gmail App Password**: https://support.google.com/accounts/answer/185833

#### **Facebook OAuth (Opcional):**

1. Ir a [Facebook Developers](https://developers.facebook.com/)##  Uso de la Aplicación:

2. Crear aplicación Facebook1. **Conecta** la cámara Ultra Leap a tu computadora

3. Configurar Facebook Login2. **Ejecuta** la aplicación con `.\run_app.ps1`

3. **Inicia sesión** o crea una cuenta nueva

```bash4. **Calibra** la cámara y comienza a hacer gestos

# Agregar al .env:5. **Observa** las predicciones en tiempo real

FACEBOOK_APP_ID=tu_facebook_app_id6. **Usa** la transcripción de voz para comandos adicionales

FACEBOOK_APP_SECRET=tu_facebook_app_secret

```##  Contribuir:

GesturaApp es un proyecto de inclusión tecnológica. Las contribuciones son bienvenidas para mejorar la precisión del reconocimiento y agregar nuevas funcionalidades.

### 7.  Configuración de Pagos (Stripe)

##  Licencia:

#### **Para funcionalidad de pagos:**Consulta el archivo `LICENSE.md` para más detalles.

1. Crear cuenta en [Stripe](https://dashboard.stripe.com/)

2. Obtener claves de test/producción---

3. Crear productos en Stripe Dashboard*GesturaApp representa un paso adelante en la inclusión tecnológica, ofreciendo una herramienta accesible y precisa para la interpretación del lenguaje de señas.*


```bash
# Agregar al .env:
STRIPE_PUBLISHABLE_KEY=pk_test_tu_stripe_publishable_key
STRIPE_SECRET_KEY=sk_test_tu_stripe_secret_key
STRIPE_PRODUCT_ID=prod_tu_producto_id
```

### 8.  Configuración de Email (Gmail)

#### **Para envío de emails:**
1. Habilitar autenticación de 2 factores en Gmail
2. Crear App Password: [Guía](https://support.google.com/accounts/answer/185833)

```bash
# Agregar al .env:
EMAIL_USER=tu_email@gmail.com
EMAIL_PASSWORD=tu_app_password_gmail_de_16_caracteres
```

##  Verificar Instalación
```powershell
# Verificar que todo está instalado correctamente
.\verify_install.ps1
```

##  Ejecutar la Aplicación
```powershell
# Opción 1: Script de ejecución
.\run_app.ps1

# Opción 2: Comando directo
venv\Scripts\activate
python Aplicacion\GesturaV4.py
```

##  Probar Módulos Individuales

### **Probar Speech-to-Text:**
```python
# Verificar que Google Cloud está configurado
python -c "
from google.cloud import speech
client = speech.SpeechClient()
print('✓ Google Cloud Speech configurado correctamente')
"
```

### **Probar Leap Motion:**
```powershell
# Conectar Ultra Leap Controller y ejecutar:
python examples\tracking_event_example.py
```

### **Probar Modelos de ML:**
```powershell
python RedesNeuronales\ProbarModelos.py
```

##  Solución de Problemas Comunes

### **Error: MySQL Connection Failed**
```bash
# Verificar que MySQL está ejecutándose
net start mysql80

# Verificar conexión
mysql -u root -p -e "SHOW DATABASES;"
```

### **Error: Google Cloud Credentials**
```bash
# Verificar que el archivo existe
ls credentials/google-cloud-speech.json

# Verificar variable de entorno
echo $GOOGLE_APPLICATION_CREDENTIALS
```

### **Error: ModuleNotFoundError**
```bash
# Activar entorno virtual
venv\Scripts\activate

# Reinstalar dependencias
pip install -r requirements.txt
```

##  Funcionalidades principales:
- **Captura Ultra Leap**: Captura en tiempo real de las posiciones de las articulaciones de las manos.
- **Reconocimiento MediaPipe**: Detección adicional usando MediaPipe para mayor precisión.
- **Redes Neuronales**: CNN, LSTM y modelos híbridos para reconocimiento de gestos.
- **Transcripción de voz**: Integración con Google Cloud Speech-to-Text.
- **Interfaz moderna**: PyQt6 con estilos personalizados y animaciones.
- **Base de datos**: Sistema de usuarios con MySQL.
- **Inteligencia artificial**: Integración con Google Gemini AI.

##  Tecnologías utilizadas:
- **Python 3.8+**: Lenguaje principal del proyecto
- **PyQt6**: Interfaz gráfica de usuario moderna
- **TensorFlow/Keras**: Redes neuronales para predicción de gestos  
- **OpenCV**: Procesamiento de imágenes y video
- **MediaPipe**: Detección de manos y poses
- **Ultra Leap SDK**: Captura de datos de manos en 3D
- **scikit-learn**: Machine learning y preprocesamiento
- **MySQL**: Base de datos para usuarios
- **Flask**: Servidor web para autenticación
- **Google Cloud**: Speech-to-Text y Gemini AI
- **Pandas/NumPy**: Procesamiento de datos

##  Estructura del Proyecto:
```
GesturaApp/
├── Aplicacion/           # Aplicación principal
│   ├── GesturaV4.py     # Archivo principal
│   ├── transcripcion.py # Módulo de transcripción de voz
│   └── TranscriptionWorker.py
├── Login/               # Sistema de autenticación
│   ├── Login.py        # Interfaz de login
│   └── servidor_flask.py # Servidor de autenticación
├── RedesNeuronales/     # Modelos de IA v1
├── RedesNeuronalesV2/   # Modelos de IA v2 (mejorados)
├── leapc-cffi/         # Bindings de Leap Motion (C)
├── leapc-python-api/   # API de Python para Leap Motion
├── credentials/        # Credenciales de servicios
├── assets/             # Recursos (estilos, animaciones)
├── .env               # Variables de entorno
├── .env.example       # Plantilla de configuración
├── requirements.txt   # Dependencias de Python
└── install.ps1       # Script de instalación automática
```

##  Uso de la Aplicación:
1. **Conecta** la cámara Ultra Leap a tu computadora
2. **Ejecuta** la aplicación con `.\run_app.ps1`
3. **Inicia sesión** o crea una cuenta nueva
4. **Calibra** la cámara y comienza a hacer gestos
5. **Observa** las predicciones en tiempo real
6. **Usa** la transcripción de voz para comandos adicionales

##  Contribuir:
GesturaApp es un proyecto de inclusión tecnológica. Las contribuciones son bienvenidas para mejorar la precisión del reconocimiento y agregar nuevas funcionalidades.

## 📄 Licencia:
Consulta el archivo `LICENSE.md` para más detalles.

---
*GesturaApp representa un paso adelante en la inclusión tecnológica, ofreciendo una herramienta accesible y precisa para la interpretación del lenguaje de señas.*
