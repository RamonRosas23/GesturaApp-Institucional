# Librerías estándar de Python
import base64
import os  # Para manejar rutas y nombres de archivos
import re
import sys
import tempfile  # Para crear archivos temporales
from urllib.parse import unquote, unquote_plus

# Librerías externas
import mysql.connector
import pywinstyles
import requests
from dotenv import load_dotenv
from requests_oauthlib import OAuth2Session

# PyQt6 - Widgets
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QMessageBox, QFileDialog, QStackedWidget, QComboBox,
    QDateEdit, QHBoxLayout, QGraphicsDropShadowEffect,
    QFrame, QToolTip, QDialog, QStyleFactory, QGraphicsScene,
    QGraphicsView, QGraphicsPixmapItem, QGraphicsTextItem,
    QProgressBar, QSizePolicy, QCalendarWidget, QToolButton
)

# PyQt6 - Core
from PyQt6.QtCore import (
    QDate, Qt, QUrl, pyqtSlot, QRect, QPoint, QSize,
    QIODevice, QBuffer, QByteArray, QThread, QObject, pyqtSignal, QLocale, QPointF
)

# PyQt6 - Gui
from PyQt6.QtGui import (
    QPixmap, QFont, QColor, QIcon, QPainter, QPen, QBrush,
    QLinearGradient, QMovie, QAction
)

# PyQt6 - Web Engine
from PyQt6.QtWebEngineCore import QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView

# PyQt6 - Multimedia
from PyQt6.QtMultimedia import QCamera, QImageCapture, QMediaCaptureSession
from PyQt6.QtMultimediaWidgets import QVideoWidget




# Cargar variables de entorno
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)
secret_key = os.getenv('SECRET_KEY')  # Cargar la clave secreta del .env

# Carpeta de recursos
carpeta_recursos = os.path.dirname(os.path.dirname(__file__)) + '/assets/'

# Conectar con MySQL
def connect_db():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', '127.0.0.1'),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', ''),
        database=os.getenv('DB_NAME'),
        port=os.getenv('DB_PORT', '3306'),
        use_pure=True
    )


class AlterLoginWidget(QWidget):
    def __init__(self, authorization_url, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window  # Para acceder al main_window
        self.setWindowTitle("Iniciar sesión con Google/Facebook")
        self.setGeometry(100, 100, 800, 600)

        self.browser = QWebEngineView()

        # Conectar el evento urlChanged para manejar la redirección de éxito
        self.browser.urlChanged.connect(self.on_url_changed)

        # Convertir la URL de autenticación en un objeto QUrl
        self.browser.setUrl(QUrl(authorization_url))

        # Layout para la ventana
        layout = QVBoxLayout()
        layout.addWidget(self.browser)
        self.setLayout(layout)
    
    @pyqtSlot(QUrl)
    def on_url_changed(self, url):
        url_str = url.toString()
        # O, preferiblemente:
        url_str = str(url)


        # Manejar redirección de error
        if "google_error" in url_str or "facebook_error" in url_str:
            self.close()
            self.main_window.stack.setCurrentWidget(self.main_window.login_widget)  # Redirigir al login
            fail_sesion_box = CustomMessageBox("Error de autenticación", "El inicio de sesión fue cancelado o falló..", message_type="error")
            fail_sesion_box.exec()

        # Manejar éxito de inicio de sesión en Facebook
        elif "facebook_success" in url_str or "google_success" in url_str:
            # Primero decodificar toda la URL para evitar caracteres especiales mal interpretados
            full_query = unquote_plus(url.query())

            # Extraer el valor completo del avatar hasta el próximo parámetro
            avatar_key = "avatar="
            avatar_start = full_query.find(avatar_key) + len(avatar_key)
            avatar_end = full_query.find("&first_name")
            avatar_url = full_query[avatar_start:avatar_end]

            # Decodificar avatar y otros parámetros
            avatar_url = unquote(avatar_url)

            query_params = dict(param.split('=', 1) for param in full_query.replace(avatar_url, "AVATAR_PLACEHOLDER").split('&'))
            query_params['avatar'] = avatar_url
            print(str(query_params.get('user_id')))

            profile_data = {
                'user_id': query_params.get('user_id'),
                'username': query_params.get('username'),
                'email': query_params.get('email'),
                'nombres': query_params.get('first_name'),
                'apellidos': query_params.get('last_name'),
                'avatar': query_params.get('avatar'),
                'telefono': 'No disponible',
                'direccion': query_params.get('direccion') or 'No disponible',
                'genero': query_params.get('gender') or 'No disponible',
                'fecha_nacimiento': query_params.get('birthday') or 'No disponible',
                'customer_id': query_params.get('customer_id')
            }
            
            # Cerrar la ventana de autenticación
            self.close()

            # Remover el AlterLoginWidget del stack
            index = self.main_window.stack.indexOf(self)
            if index != -1:
                self.main_window.stack.removeWidget(self)

            # Llamar a handle_login_success en la ventana principal
            self.main_window.handle_login_success(profile_data)
        
        # Manejar restablecimiento de contraseña
        elif "reset_password" in url_str:
            # Solo abre la URL en el QWebEngineView
            self.open_reset_password_link(url_str)


class UserProfileWindow(QWidget):
    def __init__(self, user_data, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle(f"Perfil de {user_data['username']}")
        self.setGeometry(100, 100, 800, 600)

        self.user_data = user_data
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Mostrar avatar si está disponible
        avatar_label = QLabel(self)
        try:
            if self.user_data['avatar']:
                avatar_data = self.user_data['avatar']

                if isinstance(avatar_data, bytes):
                    # Caso: el avatar es un archivo de imagen binario cargado desde la computadora
                    image = QPixmap()
                    image.loadFromData(avatar_data)
                    avatar_label.setPixmap(image.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

                if isinstance(avatar_data, str):
                    # Verificar si el avatar es una URL
                    if avatar_data.startswith("http"):
                        try:
                            response = requests.get(avatar_data)
                            if response.status_code == 200:
                                avatar_bytes = response.content
                                image = QPixmap()
                                if image.loadFromData(avatar_bytes):
                                    avatar_label.setPixmap(image.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                                else:
                                    layout.addWidget(QLabel("Error al cargar el avatar desde los datos descargados"))
                            elif response.status_code == 404:
                                layout.addWidget(QLabel("Error 404: El enlace del avatar no es válido o ha expirado."))
                            else:
                                layout.addWidget(QLabel(f"No se pudo descargar el avatar desde la URL. Código de estado: {response.status_code}"))
                        except Exception as e:
                            layout.addWidget(QLabel(f"Error al descargar el avatar desde la URL: {str(e)}"))
                    else:
                        # Decodificar la cadena base64 si no es una URL
                        try:
                            avatar_bytes = base64.b64decode(avatar_data)
                            image = QPixmap()
                            if image.loadFromData(avatar_bytes):
                                avatar_label.setPixmap(image.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                            else:
                                layout.addWidget(QLabel("Error al cargar el avatar desde los datos base64"))
                        except Exception as e:
                            layout.addWidget(QLabel(f"Error al decodificar el avatar en base64: {str(e)}"))
                else:
                    layout.addWidget(QLabel("Formato de avatar no soportado"))
            else:
                layout.addWidget(QLabel("No hay avatar disponible"))
        except Exception as e:
            layout.addWidget(QLabel(f"Error al cargar el avatar: {str(e)}"))

        layout.addWidget(avatar_label)

        # Mostrar todos los datos del usuario
        layout.addWidget(QLabel(f"Nombre de usuario: {self.user_data['username']}"))
        layout.addWidget(QLabel(f"Correo electrónico: {self.user_data['email']}"))
        layout.addWidget(QLabel(f"Nombres: {self.user_data['nombres']}"))
        layout.addWidget(QLabel(f"Apellidos: {self.user_data['apellidos']}"))
        layout.addWidget(QLabel(f"Teléfono: {self.user_data['telefono']}"))
        layout.addWidget(QLabel(f"Dirección: {self.user_data['direccion']}"))
        layout.addWidget(QLabel(f"Género: {self.user_data['genero']}"))
        layout.addWidget(QLabel(f"Fecha de nacimiento: {self.user_data['fecha_nacimiento']}"))

        # Añadir botón para cerrar sesión o volver a la pantalla de login
        self.button_logout = QPushButton("Cerrar sesión")
        self.button_logout.clicked.connect(self.logout)

        layout.addWidget(self.button_logout)
        self.setLayout(layout)

    def logout(self):
        # Mostrar mensaje de confirmación de cierre de sesión
        QMessageBox.information(self, "Cerrar sesión", "Sesión cerrada exitosamente.")
        
        # Redirigir a la ventana de login
        self.main_window.stack.setCurrentWidget(self.main_window.login_widget)

# Ventana principal que alterna entre login y registro
class PrincipalLoginWidget(QWidget):
    def __init__(self, gestura_app):
        super().__init__()
        self.gestura_app = gestura_app  # Guardar la referencia a la clase principal
        self.setWindowTitle("Login y Registro")
        self.initUI()

    def initUI(self):
        self.login_container = QWidget()
        # Crear el layout principal para la ventana
        layout = QVBoxLayout()
        
        # Crear un QStackedWidget para alternar entre las pantallas de login y registro
        self.stack = QStackedWidget(self)

        
        # Crear las pantallas de login y registro en dos fases
        self.login_widget = self.create_login_widget()
        self.register_step1_widget_center = self.create_centered_widget(self.create_register_step1_widget())
        self.register_step2_widget_center = self.create_centered_widget(self.create_register_step2_widget())
        
        # Añadir las pantallas al stack
        self.stack.addWidget(self.login_widget)
        self.stack.addWidget(self.register_step1_widget_center)
        self.stack.addWidget(self.register_step2_widget_center)

        
        layout.addWidget(self.stack)

        # Configurar el estilo del widget principal
        self.setStyleSheet("""
            background-color: #111c22;
            border-radius: 8px;
            padding: 5px;
        """)
        
        # Estilo para el tooltip
        ToolTip_box = """
            QToolTip {
                background-color: #444444;
                color: white;
                padding: 5px;
                border-radius: 8px;
                font-family: 'Manrope';
                font-size: 12px;
                opacity: 200;
                
            }
        """

        # Aplicar estilo globalmente usando QApplication
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(ToolTip_box)

        # Configurar el layout principal para el widget
        self.setLayout(layout)

    def create_centered_widget(self, widget):
        """
        Crea un contenedor con un solo widget centrado.
        Args:
            widget: El widget que se quiere centrar.
        """
        container = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)  
        layout.addWidget(widget)
        container.setLayout(layout)
        return container

    
    # Agregar la animación a la ventana de login
    def create_login_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Alinear todo al centro
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Alinear el layout principal al top
        layout.setContentsMargins(0, 0, 0, 0)

        # Establecer tamaño máximo para cuadros de texto y botones
        max_width = 500

        # Configuración de las fuentes
        font_manrope_title = QFont("Manrope", 18, QFont.Weight.Bold)
        font_manrope_label = QFont("Manrope", 10)
        font_manrope_small = QFont("Manrope", 14)
        font_bold_small = QFont("Manrope", 14, QFont.Weight.Bold)

        # Iconos
        google_icon_path = os.path.join(carpeta_recursos, 'google_icon.png')
        facebook_icon_path = os.path.join(carpeta_recursos, 'facebook_icon.png')
        instagram_icon_path = os.path.join(carpeta_recursos, 'instagram_icon.png')

        # Estilos comunes para inputs y botones
        input_style = """
            QLineEdit {
                padding: 12px;
                font-family: 'Manrope';
                font-size: 14px;
                color: #eeeeee;
                background-color: #444444;
                border: 2px solid #666666;
                border-radius: 10px;
                outline: none;
            }
            QLineEdit:focus {
                border: 2px solid #007BFF;
            }
        """

        button_style = """
            QPushButton {
                padding: 10px;
                font-family: 'Manrope';
                font-size: 14px;
                font-weight: bold;
                border-radius: 15px;
                color: white;
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #007BFF, stop:1 #0056b3);
                outline: none;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:focus {
                background-color: #0056b3;
            }
        """

        secondary_button_style = """
            QPushButton {
                padding: 12px;
                font-family: 'Manrope';
                font-size: 14px;
                font-weight: bold;
                border-radius: 20px;
                color: white;
                background-color: #2D2D2D;
                outline: none;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QPushButton:focus {
                background-color: #444444;
            }
        """
        
        button_rec_and_reg_style = """
            QPushButton {
                color: #999999; 
                font-size: 14px; 
                font-family: 'Manrope'; 
                outline: none;
                background-color: transparent;
            }
            QPushButton:focus {
                color: #ffffff;
                background-color: transparent;
            }
        """
        
        logo_icon_path = os.path.join(carpeta_recursos, 'logo_icon.png')
        font_manrope_title = QFont("Manrope", 28, QFont.Weight.Bold)
        
        
        # Crear la escena
        scene = QGraphicsScene()
        view = QGraphicsView(scene)

        # Agregar la imagen
        pixmap = QPixmap(logo_icon_path)
        pixmap = pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        scene.addItem(pixmap_item)

        # Agregar el texto
        text_item = QGraphicsTextItem("estura")
        text_item.setDefaultTextColor(QColor("white"))
        text_item.setFont(font_manrope_title)

        # Calcular las dimensiones totales necesarias
        total_width = pixmap.width() + text_item.boundingRect().width()
        total_height = max(pixmap.height(), text_item.boundingRect().height())

        # Establecer el tamaño y los límites de la escena
        scene.setSceneRect(0, 0, total_width + 20, total_height + 20)  # Añadir padding

        # Posicionar el texto justo después de la imagen
        text_item.setPos(
            pixmap.width(),  # Ajusta este valor para acercar o alejar el texto
            ((total_height - text_item.boundingRect().height()) / 2) - 6  # Centrar verticalmente
        )
        scene.addItem(text_item)

        # Configurar la vista
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setFixedSize(
            int(total_width + 20),  # Añadir padding
            int(total_height + 20)  # Añadir padding
        )
        view.setStyleSheet("""
            QGraphicsView {
                background: transparent;
                border: none;
                margin: 0px;
                padding: 0px;
            }
        """)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        # Centrar la vista en la escena
        view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Añadir la vista al layout principal
        header_widgetL = QVBoxLayout()
        header_widgetL.addWidget(view, alignment=Qt.AlignmentFlag.AlignLeft)
        header_widgetL.setContentsMargins(20, 13, 0, 0)  # Eliminar márgenes del layout

        
        # Título
        title_label = QLabel("Iniciar sesión")
        title_label.setFont(font_manrope_title)
        title_label.setStyleSheet("""
            color: white; 
            background-color: transparent; 
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Campos de entrada
        self.label_user = QLabel("Correo electrónico")
        self.label_user.setFont(font_manrope_label)
        self.label_user.setStyleSheet("""
            color: white; 
            background-color: transparent; 
        """)
        self.textbox_user = QLineEdit()
        self.textbox_user.setPlaceholderText("Correo electrónico")
        self.textbox_user.setStyleSheet(input_style)
        self.textbox_user.setFixedWidth(max_width)

        # Agrupar label y textbox en un layout vertical centrado
        user_layout = QVBoxLayout()
        user_layout.setContentsMargins(0, 0, 0, 5)
        user_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        user_layout.addWidget(self.label_user)
        user_layout.addWidget(self.textbox_user, alignment=Qt.AlignmentFlag.AlignCenter)

        # Contraseña
        self.label_password = QLabel("Contraseña")
        self.label_password.setFont(font_manrope_label)
        self.label_password.setStyleSheet("""
            color: white; 
            background-color: transparent;
        """)
        self.textbox_password = QLineEdit()
        self.textbox_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.textbox_password.setPlaceholderText("Contraseña")
        self.textbox_password.setStyleSheet(input_style)
        self.textbox_password.setFixedWidth(max_width)

        password_layout = QVBoxLayout()  # Agrupar label y textbox de contraseña
        password_layout.setContentsMargins(0, 0, 0, 10)
        password_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        password_layout.addWidget(self.label_password)
        password_layout.addWidget(self.textbox_password, alignment=Qt.AlignmentFlag.AlignCenter)

        # Botones de inicio de sesión
        self.button_login = QPushButton("Iniciar sesión")
        self.button_login.setStyleSheet(button_style)
        self.button_login.setFixedWidth(max_width + 20)  # Fijar ancho del botón
        self.button_login.clicked.connect(self.check_login)
        
        # Añadir sombra al botón de login
        shadow_effect = QGraphicsDropShadowEffect(self.button_login)
        shadow_effect.setBlurRadius(15)
        shadow_effect.setOffset(3, 3)
        shadow_effect.setColor(QColor("black"))
        self.button_login.setGraphicsEffect(shadow_effect)

        login_button_layout = QHBoxLayout()  # Layout centrado para el botón
        login_button_layout.setContentsMargins(0, 0, 0, 5)
        login_button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        login_button_layout.addWidget(self.button_login)

        # Botón de Google con ícono
        google_icon = QIcon(QPixmap(google_icon_path).scaled(15, 15, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.button_google = QPushButton(" Iniciar sesión con Google")
        self.button_google.setStyleSheet(secondary_button_style)
        self.button_google.setIcon(QIcon(google_icon_path))
        self.button_google.setIcon(google_icon)
        self.button_google.setIconSize(QSize(15, 15))  # Tamaño del ícono
        self.button_google.setFixedWidth(max_width + 20)
        self.button_google.clicked.connect(self.google_login)
        
        # Añadir sombra al botón de Google
        shadow_google = QGraphicsDropShadowEffect(self.button_google)
        shadow_google.setBlurRadius(15)
        shadow_google.setOffset(3, 3)
        shadow_google.setColor(QColor("black"))
        self.button_google.setGraphicsEffect(shadow_google)

        google_button_layout = QHBoxLayout()
        google_button_layout.setContentsMargins(0, 0, 0, 5)
        google_button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        google_button_layout.addWidget(self.button_google)

        # Botón de Facebook con ícono
        facebook_icon = QIcon(QPixmap(facebook_icon_path).scaled(15, 15, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.button_facebook = QPushButton(" Iniciar sesión con Facebook")
        self.button_facebook.setStyleSheet(secondary_button_style)
        self.button_facebook.setIcon(facebook_icon)
        self.button_facebook.setIconSize(QSize(15, 15))  # Tamaño del ícono
        self.button_facebook.setIconSize(QSize(15, 15))
        self.button_facebook.setFixedWidth(max_width + 20)
        self.button_facebook.clicked.connect(self.facebook_login)
        
        # Añadir sombra al botón de Facebook
        shadow_facebook = QGraphicsDropShadowEffect(self.button_facebook)
        shadow_facebook.setBlurRadius(15)
        shadow_facebook.setOffset(3, 3)
        shadow_facebook.setColor(QColor("black"))
        self.button_facebook.setGraphicsEffect(shadow_facebook)

        facebook_button_layout = QHBoxLayout()
        facebook_button_layout.setContentsMargins(0, 0, 0, 10)
        facebook_button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        facebook_button_layout.addWidget(self.button_facebook)

        # Botones adicionales
        self.button_register = QPushButton("¿No tienes cuenta? Regístrate")
        self.button_register.setStyleSheet(button_rec_and_reg_style)
        self.button_register.clicked.connect(self.switch_to_register_step1)

        self.button_forgot_password = QPushButton("¿Olvidaste tu contraseña?")
        self.button_forgot_password.setStyleSheet(button_rec_and_reg_style)
        # Modificación del botón de recuperación de contraseña
        self.button_forgot_password.clicked.connect(self.show_password_recovery_popup)

        # Crear varias líneas
        line1 = QFrame()
        line1.setFrameShape(QFrame.Shape.HLine)
        line1.setFrameShadow(QFrame.Shadow.Sunken)
        line1.setStyleSheet("background-color: #18232b; height: 2px;") 

        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setFrameShadow(QFrame.Shadow.Sunken)
        line2.setStyleSheet("background-color: #18232b; height: 2px;")

        # Logos de redes sociales
        social_layout = QHBoxLayout()
        social_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        google_logo = QLabel()
        google_pixmap = QPixmap(google_icon_path)
        google_logo.setPixmap(google_pixmap.scaled(34, 34, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        facebook_logo = QLabel()
        facebook_pixmap = QPixmap(facebook_icon_path)
        facebook_logo.setPixmap(facebook_pixmap.scaled(34, 34, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        instagram_logo = QLabel()
        instagram_pixmap = QPixmap(instagram_icon_path)
        instagram_logo.setPixmap(instagram_pixmap.scaled(34, 34, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))


        social_layout.addWidget(google_logo)
        social_layout.addWidget(facebook_logo)
        social_layout.addWidget(instagram_logo)
        
        # Sección de enlaces adicionales
        footer_layout = QHBoxLayout()
        footer_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        link_about = QLabel("Acerca de nosotros")
        link_about.setFont(font_manrope_small)
        link_about.setStyleSheet("color: #999999;")
        link_contact = QLabel("Contáctanos")
        link_contact.setFont(font_manrope_small)
        link_contact.setStyleSheet("color: #999999;")
        link_terms = QLabel("Términos de servicio")
        link_terms.setFont(font_manrope_small)
        link_terms.setStyleSheet("color: #999999;")
        link_privacy = QLabel("Política de privacidad")
        link_privacy.setFont(font_manrope_small)
        link_privacy.setStyleSheet("color: #999999;")

        footer_layout.addWidget(link_about)
        footer_layout.addWidget(link_contact)
        footer_layout.addWidget(link_terms)
        footer_layout.addWidget(link_privacy)

        # Crear el contenedor para la sección de login
        loginfull_layout = QVBoxLayout(self.login_container)
        loginfull_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Añadir los elementos al layout del contenedor
        loginfull_layout.addWidget(title_label)
        loginfull_layout.addLayout(user_layout)
        loginfull_layout.addLayout(password_layout)
        loginfull_layout.addLayout(login_button_layout)
        loginfull_layout.addLayout(google_button_layout)
        loginfull_layout.addLayout(facebook_button_layout)
        loginfull_layout.addWidget(self.button_register, alignment=Qt.AlignmentFlag.AlignCenter)
        loginfull_layout.addWidget(self.button_forgot_password, alignment=Qt.AlignmentFlag.AlignCenter)

        # Configurar el estilo del login_container
        self.login_container.setStyleSheet("""
            QWidget#login_container {
                background-color: rgba(26, 39, 48, 0.8);  /* Fondo con transparencia */
                border-radius: 15px;  /* Bordes redondeados */
                padding: 20px;  /* Espaciado interno */
                border: 2px solid rgba(255, 255, 255, 0.1);  /* Borde externo sutil */
            }
        """)
        self.login_container.setObjectName("login_container")
        self.login_container.setFixedWidth(max_width + 60)

        # Crear un efecto de sombra para el login_container
        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setBlurRadius(15)
        shadow_effect.setXOffset(0)
        shadow_effect.setYOffset(5)
        shadow_effect.setColor(QColor(0, 0, 0, 80))
        self.login_container.setGraphicsEffect(shadow_effect)

        # Crear un QWidget central para contener el login_container
        self.central_widget = QWidget()
        self.central_layout = QVBoxLayout()
        self.central_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.central_widget.setLayout(self.central_layout)

        # Añadir el login_container al central_layout
        self.central_layout.addWidget(self.login_container)

        # En el layout principal, añadir el central_widget entre los stretches
        layout.addLayout(header_widgetL)
        layout.addWidget(line1)
        layout.addStretch()
        layout.addWidget(self.central_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        layout.addWidget(line2)
        layout.addLayout(social_layout)
        layout.addLayout(footer_layout)

        widget.setLayout(layout)
        self.setStyleSheet("""
            background-color: #111c22;
            border-radius: 8px;
            padding: 5px;
        """)
        return widget


    # Crear la primera fase del registro
    def create_register_step1_widget(self):
        widget = QWidget()

        # Establecer tamaño máximo para cuadros de texto y botones
        max_width_input = 450
        max_width_button = 150

        # Configurar tamaño del widget principal
        widget.setFixedSize(1000, 600)
        widget.setMinimumSize(1000, 600)

        # Logo en posición absoluta
        logo_icon_path = os.path.join(carpeta_recursos, 'logo_icon.png')
        pixmap = QPixmap(logo_icon_path)
        logo_label = QLabel(widget)
        logo_label.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        logo_label.move(50, 150)

        # Línea divisoria
        line = QFrame(widget)
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #ffffff;")
        line.setFixedWidth(1)
        line.setFixedHeight(400)
        line.move(400, 100)

        # Estilo para inputs y botones
        input_style = """
            QLineEdit {
                padding: 12px;
                font-family: 'Manrope';
                font-size: 14px;
                color: #eeeeee;
                background-color: #444444;
                border: 2px solid #666666;
                border-radius: 10px;
                outline: none;
            }
            QLineEdit:focus {
                border: 2px solid #007BFF;
            }
        """

        button_style = """
            QPushButton {
                padding: 10px;
                font-family: 'Manrope';
                font-size: 14px;
                font-weight: bold;
                border-radius: 15px;
                color: white;
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #007BFF, stop:1 #0056b3);
                outline: none;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:focus {
                background-color: #0056b3;
            }
        """

        # Campos de texto y etiquetas en posiciones absolutas
        self.label_username = QLabel("Nombre de usuario:", widget)
        self.label_username.setStyleSheet("color: white;")
        self.label_username.setFont(QFont("Manrope", 10))
        self.label_username.move(450, 100)

        self.textbox_username = QLineEdit(widget)
        self.textbox_username.setPlaceholderText("Ej: Ramon123")
        self.textbox_username.setStyleSheet(input_style)
        self.textbox_username.setFixedWidth(max_width_input)
        self.textbox_username.move(450, 130)
        # Ícono de error para 'Nombre de usuario'
        self.icon_error_username = QLabel(widget)
        self.icon_error_username.setPixmap(QPixmap(os.path.join(carpeta_recursos, 'error_icon.png')).scaled(25, 25, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.icon_error_username.setVisible(False)  # Oculto por defecto
        self.icon_error_username.move(905, 135)  # Posición a la derecha del textbox

        # Tooltip de error
        self.icon_error_username.setToolTip("")

        self.label_email = QLabel("Correo electrónico:", widget)
        self.label_email.setStyleSheet("color: white;")
        self.label_email.setFont(QFont("Manrope", 10))
        self.label_email.move(450, 180)

        self.textbox_email = QLineEdit(widget)
        self.textbox_email.setPlaceholderText("Ej: Ramon123@gmail.com")
        self.textbox_email.setStyleSheet(input_style)
        self.textbox_email.setFixedWidth(max_width_input)
        self.textbox_email.move(450, 210)

        # Ícono de error para 'Correo electrónico'
        self.icon_error_email = QLabel(widget)
        self.icon_error_email.setPixmap(QPixmap(os.path.join(carpeta_recursos, 'error_icon.png')).scaled(25, 25, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.icon_error_email.setVisible(False)  # Oculto por defecto
        self.icon_error_email.move(905, 215)

        # Tooltip de error
        self.icon_error_email.setToolTip("")

        self.label_password_reg = QLabel("Contraseña:", widget)
        self.label_password_reg.setStyleSheet("color: white;")
        self.label_password_reg.setFont(QFont("Manrope", 10))
        self.label_password_reg.move(450, 260)

        self.textbox_password_reg = QLineEdit(widget)
        self.textbox_password_reg.setPlaceholderText("Ej: ·················")
        self.textbox_password_reg.setEchoMode(QLineEdit.EchoMode.Password)
        self.textbox_password_reg.setStyleSheet(input_style)
        self.textbox_password_reg.setFixedWidth(max_width_input)
        self.textbox_password_reg.move(450, 290)

        # Ícono de error para 'Contraseña'
        self.icon_error_password = QLabel(widget)
        self.icon_error_password.setPixmap(QPixmap(os.path.join(carpeta_recursos, 'error_icon.png')).scaled(25, 25, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.icon_error_password.setVisible(False)  # Oculto por defecto
        self.icon_error_password.move(905, 295)

        # Tooltip de error
        self.icon_error_password.setToolTip("")

        self.label_confirm_password = QLabel("Confirmar contraseña:", widget)
        self.label_confirm_password.setStyleSheet("color: white;")
        self.label_confirm_password.setFont(QFont("Manrope", 10))
        self.label_confirm_password.move(450, 340)

        self.textbox_confirm_password = QLineEdit(widget)
        self.textbox_confirm_password.setPlaceholderText("Ej: ·················")
        self.textbox_confirm_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.textbox_confirm_password.setStyleSheet(input_style)
        self.textbox_confirm_password.setFixedWidth(max_width_input)
        self.textbox_confirm_password.move(450, 370)

        # Ícono de error para 'Confirmar contraseña'
        self.icon_error_confirm_password = QLabel(widget)
        self.icon_error_confirm_password.setPixmap(QPixmap(os.path.join(carpeta_recursos, 'error_icon.png')).scaled(25, 25, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.icon_error_confirm_password.setVisible(False)  # Oculto por defecto
        self.icon_error_confirm_password.move(905, 375)

        # Tooltip de error
        self.icon_error_confirm_password.setToolTip("")
        

        # Botones en posiciones absolutas
        self.button_continue = QPushButton("Continuar", widget)
        self.button_continue.setStyleSheet(button_style)
        self.button_continue.clicked.connect(self.validate_step1_fields)
        self.button_continue.setFixedWidth(max_width_button)
        self.button_continue.move(450, 450)

        self.button_back_to_login = QPushButton("Cancelar", widget)
        self.button_back_to_login.setStyleSheet(button_style)
        self.button_back_to_login.clicked.connect(self.switch_to_login)
        self.button_back_to_login.setFixedWidth(max_width_button)
        self.button_back_to_login.move(620, 450)

        widget.setStyleSheet("""
            background-color: #111c22;
            border-radius: 8px;
            padding: 5px;
        """)

        return widget

    # Validación de campos de la ventana 1 y verificación en la base de datos
    def validate_step1_fields(self):
        username = self.textbox_username.text().strip()
        email = self.textbox_email.text().strip()
        password = self.textbox_password_reg.text().strip()
        confirm_password = self.textbox_confirm_password.text().strip()

        email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'

        # Limpiar errores previos
        self.icon_error_username.setVisible(False)
        self.icon_error_email.setVisible(False)
        self.icon_error_password.setVisible(False)
        self.icon_error_confirm_password.setVisible(False)

        valid = True  # Bandera para seguir o no al siguiente paso

        # Validar campos vacíos y formato
        if not username:
            self.icon_error_username.setVisible(True)
            self.icon_error_username.setToolTip("El campo 'Nombre de usuario' está vacío.")
            valid = False
        elif self.check_if_username_exists(username):
            # Verifica si el nombre de usuario ya existe
            self.icon_error_username.setVisible(True)
            self.icon_error_username.setToolTip("El nombre de usuario ya está en uso.")
            valid = False

        if not email or not re.match(email_regex, email):
            self.icon_error_email.setVisible(True)
            self.icon_error_email.setToolTip("El campo 'Correo electrónico' no es válido.")
            valid = False
        elif self.check_if_email_exists(email):
            # Verifica si el correo ya existe
            self.icon_error_email.setVisible(True)
            self.icon_error_email.setToolTip("El correo ya está en uso.")
            valid = False

        if not password:
            self.icon_error_password.setVisible(True)
            self.icon_error_password.setToolTip("El campo 'Contraseña' está vacío.")
            valid = False
        elif len(password) < 8:
            self.icon_error_password.setVisible(True)
            self.icon_error_password.setToolTip("La contraseña debe tener al menos 8 caracteres.")
            valid = False

        if not confirm_password:
            self.icon_error_confirm_password.setVisible(True)
            self.icon_error_confirm_password.setToolTip("El campo 'Confirmar contraseña' está vacío.")
            valid = False
        elif password != confirm_password:
            self.icon_error_confirm_password.setVisible(True)
            self.icon_error_confirm_password.setToolTip("Las contraseñas no coinciden.")
            valid = False

        # Si todo es válido, proceder al siguiente paso
        if valid:
            self.switch_to_register_step2()

    # Método para verificar si el nombre de usuario ya existe en la base de datos
    def check_if_username_exists(self, username):
        conn = connect_db()
        cursor = conn.cursor()

        # Verificar si el nombre de usuario ya existe
        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        username_exists = cursor.fetchone() is not None

        conn.close()
        return username_exists

    # Método para verificar si el correo electrónico ya existe en la base de datos
    def check_if_email_exists(self, email):
        conn = connect_db()
        cursor = conn.cursor()

        # Verificar si el correo ya existe
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        email_exists = cursor.fetchone() is not None

        conn.close()
        return email_exists
    
    # Método para mostrar el popup de recuperación de contraseña
    def show_password_recovery_popup(self):
        recovery_popup = PasswordRecoveryPopup(self)
        recovery_popup.exec()  # Mostrar el popup de manera modal


    def reset_to_login(self):
        """Resetea el stack interno al formulario de login."""
        self.stack.setCurrentWidget(self.login_widget)


    # Crear la segunda fase del registro
    def create_register_step2_widget(self):
        self.widget_register_2 = QWidget()

        # Establecer tamaño máximo para cuadros de texto y botones
        max_width_input = 300
        max_width_button = 150

        # Configurar tamaño del widget principal
        self.widget_register_2.setFixedSize(1000, 600)  # Ajusta el tamaño según lo necesario
        self.widget_register_2.setMinimumSize(1000, 600)  # Evitar que se haga más pequeño que el ancho de los elementos

        # Logo en posición absoluta (columna izquierda)
        logo_icon_path = os.path.join(carpeta_recursos, 'logo_icon.png')
        pixmap = QPixmap(logo_icon_path)
        logo_label = QLabel(self.widget_register_2)
        logo_label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        logo_label.move(50, 150)  # Ajusta la posición (x, y)

        # Línea divisoria entre logo y formulario (primera división)
        line = QFrame(self.widget_register_2)
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #ffffff;")
        line.setFixedWidth(1)
        line.setFixedHeight(400)
        line.move(300, 100)  # Ajusta la posición (x, y)

        # Estilo para inputs y botones
        input_style = """
            QLineEdit {
                padding: 12px;
                font-family: 'Manrope';
                font-size: 14px;
                color: #eeeeee;
                background-color: #444444;
                border: 2px solid #666666;
                border-radius: 10px;
                outline: none;
            }
            QLineEdit:focus {
                border: 2px solid #007BFF;
            }
        """

        button_style = """
            QPushButton {
                padding: 10px;
                font-family: 'Manrope';
                font-size: 14px;
                font-weight: bold;
                border-radius: 15px;
                color: white;
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #007BFF, stop:1 #0056b3);
                outline: none;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:focus {
                background-color: #0056b3;
            }
        """

        # Primera columna de campos (posición manual)
        self.label_nombres = QLabel("Nombres:", self.widget_register_2)
        self.label_nombres.setStyleSheet("color: white;")
        self.label_nombres.setFont(QFont("Manrope", 10))
        self.label_nombres.move(350, 100)

        self.textbox_nombres = QLineEdit(self.widget_register_2)
        self.textbox_nombres.setPlaceholderText("Ej: Javier")
        self.textbox_nombres.setStyleSheet(input_style)
        self.textbox_nombres.setFixedWidth(max_width_input)
        self.textbox_nombres.move(350, 130)
        # Ícono de error para 'Nombres'
        self.icon_error_nombres = QLabel(self.widget_register_2)
        self.icon_error_nombres.setPixmap(QPixmap(os.path.join(carpeta_recursos, 'error_icon.png')).scaled(25, 25, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.icon_error_nombres.setVisible(False)  # Oculto por defecto
        self.icon_error_nombres.move(655, 135)

        self.label_apellidos = QLabel("Apellidos:", self.widget_register_2)
        self.label_apellidos.setStyleSheet("color: white;")
        self.label_apellidos.setFont(QFont("Manrope", 10))
        self.label_apellidos.move(350, 180)

        self.textbox_apellidos = QLineEdit(self.widget_register_2)
        self.textbox_apellidos.setPlaceholderText("Ej: Verdugo")
        self.textbox_apellidos.setStyleSheet(input_style)
        self.textbox_apellidos.setFixedWidth(max_width_input)
        self.textbox_apellidos.move(350, 210)
        # Ícono de error para 'Apellidos'
        self.icon_error_apellidos = QLabel(self.widget_register_2)
        self.icon_error_apellidos.setPixmap(QPixmap(os.path.join(carpeta_recursos, 'error_icon.png')).scaled(25, 25, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.icon_error_apellidos.setVisible(False)
        self.icon_error_apellidos.move(655, 215)

        self.label_telefono = QLabel("Teléfono:", self.widget_register_2)
        self.label_telefono.setStyleSheet("color: white;")
        self.label_telefono.setFont(QFont("Manrope", 10))
        self.label_telefono.move(350, 260)

        self.textbox_telefono = QLineEdit(self.widget_register_2)
        self.textbox_telefono.setPlaceholderText("Ej: 672-122-9825")
        self.textbox_telefono.setStyleSheet(input_style)
        self.textbox_telefono.setFixedWidth(max_width_input)
        self.textbox_telefono.move(350, 290)
        # Ícono de error para 'Teléfono'
        self.icon_error_telefono = QLabel(self.widget_register_2)
        self.icon_error_telefono.setPixmap(QPixmap(os.path.join(carpeta_recursos, 'error_icon.png')).scaled(25, 25, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.icon_error_telefono.setVisible(False)
        self.icon_error_telefono.move(655, 295)

        self.label_direccion = QLabel("Dirección:", self.widget_register_2)
        self.label_direccion.setStyleSheet("color: white;")
        self.label_direccion.setFont(QFont("Manrope", 10))
        self.label_direccion.move(350, 340)

        self.textbox_direccion = QLineEdit(self.widget_register_2)
        self.textbox_direccion.setPlaceholderText("Ej: Culiacan, Sinaloa, Mexico")
        self.textbox_direccion.setStyleSheet(input_style)
        self.textbox_direccion.setFixedWidth(max_width_input)
        self.textbox_direccion.move(350, 370)
        # Ícono de error para 'Dirección'
        self.icon_error_direccion = QLabel(self.widget_register_2)
        self.icon_error_direccion.setPixmap(QPixmap(os.path.join(carpeta_recursos, 'error_icon.png')).scaled(25, 25, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.icon_error_direccion.setVisible(False)
        self.icon_error_direccion.move(655, 375)

        # Género (QLabel y QComboBox)
        self.label_genero = QLabel("Género:", self.widget_register_2)
        self.label_genero.setStyleSheet("""
            color: #eeeeee;
            font-size: 14px;
            font-family: 'Manrope';
            font-weight: normal;
        """)
        self.label_genero.setFont(QFont("Manrope", 10))
        self.label_genero.move(700, 100)

        self.combobox_genero = QComboBox(self.widget_register_2)
        self.combobox_genero.setFixedWidth(max_width_input)
        self.combobox_genero.addItems(["Male", "Female", "Otro"])

        self.combobox_genero.setStyleSheet("""
            QComboBox {
                background-color: #444444;
                color: #eeeeee;
                border: 2px solid #666666;
                border-radius: 10px;
                padding: 8px;
                font-family: 'Manrope';
                font-size: 14px;
            }
            QComboBox:focus {
                border: 2px solid #007BFF;
            }
            QComboBox::drop-down {
                border-left: 0px;
            }
            QComboBox QAbstractItemView {
                background-color: #333333;
                color: #eeeeee;
            }
        """)
        self.combobox_genero.move(700, 130)
        
        # Fecha de nacimiento (QLabel y QDateEdit)
        self.label_fecha_nacimiento = QLabel("Fecha de nacimiento:", self.widget_register_2)
        self.label_fecha_nacimiento.setStyleSheet("""
            color: #eeeeee;
            font-size: 14px;
            font-family: 'Manrope';
            font-weight: normal;
        """)
        self.label_fecha_nacimiento.setFont(QFont("Manrope", 10))
        self.label_fecha_nacimiento.move(700, 180)

        self.date_fecha_nacimiento = QDateEdit(self.widget_register_2)
        self.date_fecha_nacimiento.setDate(QDate.currentDate())
        self.date_fecha_nacimiento.setCalendarPopup(True)
        
        # Asignar el calendario personalizado
        custom_calendar = CustomCalendarWidget(self)
        custom_calendar.setVerticalHeaderFormat(QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader)
        custom_calendar.setHorizontalHeaderFormat(QCalendarWidget.HorizontalHeaderFormat.NoHorizontalHeader)
        self.date_fecha_nacimiento.setCalendarWidget(custom_calendar)
        self.date_fecha_nacimiento.setFixedWidth(max_width_input)

        # Rutas de las imágenes de flechas personalizadas
        calendar_arrow_path = os.path.join(carpeta_recursos, 'calendario_icon.png').replace("\\", "/")
        up_arrow_path = os.path.join(carpeta_recursos, 'up_arrow.png').replace("\\", "/")
        down_arrow_path = os.path.join(carpeta_recursos, 'down_arrow.png').replace("\\", "/")
    
        # Estilos personalizados para QDateEdit
        self.date_fecha_nacimiento.setStyleSheet(f"""
            QDateEdit {{
                background-color: #444444;
                color: #eeeeee;
                border: 2px solid #666666;
                border-radius: 10px;
                padding: 8px;
                font-family: 'Manrope';
                font-size: 14px;
            }}
            QDateEdit:focus {{
                border: 2px solid #007BFF;
            }}
            QDateEdit::drop-down {{
                width: 30px;
                background-color: #444444;
                border-left: 1px solid #666666;
                border-radius: 0px 10px 10px 0px;
            }}
            QDateEdit::down-arrow {{
                image: url({calendar_arrow_path});
                width: 15px;
                height: 15px;
            }}
            QDateEdit QAbstractItemView {{
                background-color: #333333;
                color: #eeeeee;
                selection-background-color: #555555;
                selection-color: white;
                border: 2px solid #666666;
            }}

            QCalendarWidget QWidget#qt_calendar_navigationbar {{
                background-color: #333333;
            }}
            QCalendarWidget QToolButton {{
                color: #eeeeee;
                background-color: #333333;
                font-size: 12px;
                height: 20px;
                border: none;
            }}
            QCalendarWidget QToolButton:hover {{
                background-color: #555555;
            }}
            QCalendarWidget QAbstractItemView {{
                font-size: 12px;
                color: #eeeeee;
                background-color: #333333;
            }}
            QCalendarWidget QAbstractItemView:enabled {{
                selection-background-color: #007BFF;
                selection-color: #ffffff;
            }}
            QCalendarWidget QAbstractItemView:selected {{
                background-color: #007BFF;
                color: #ffffff;
            }}

            /* Personalización del QSpinBox (selector de año) */
            QCalendarWidget QSpinBox {{
                background-color: #222222;
                color: #ffffff;
                border-radius: 8px;
                padding: 1px;
                font-family: 'Manrope';
                font-size: 14px;
                width: 40px;
                height: 30px;
            }}
            QCalendarWidget QSpinBox::up-button, 
            QCalendarWidget QSpinBox::down-button {{
                border-radius: 4px;
                width: 20px;
                height: 20px;
            }}
            QCalendarWidget QSpinBox::up-button:hover, 
            QCalendarWidget QSpinBox::down-button:hover {{
                background-color: #0056b3;
            }}
            QCalendarWidget QSpinBox::up-arrow {{
                image: url({up_arrow_path});
                width: 8px;
                height: 8px;
            }}
            QCalendarWidget QSpinBox::down-arrow {{
                image: url({down_arrow_path});
                width: 8px;
                height: 8px;
            }}

            /* Personalización del menú desplegable de meses */
            QCalendarWidget QToolButton#qt_calendar_monthbutton QMenu {{
                background-color: #444444;
                color: #eeeeee;
                border: 1px solid #666666;
            }}
            QCalendarWidget QToolButton#qt_calendar_monthbutton QMenu::item {{
                background-color: #444444;
                color: #eeeeee;
                padding: 4px;
            }}
            QCalendarWidget QToolButton#qt_calendar_monthbutton QMenu::item:selected {{
                background-color: #007BFF;
                color: white;
            }}
        """)
        
        self.date_fecha_nacimiento.calendarWidget().setFixedSize(300, 250)
        self.date_fecha_nacimiento.move(700, 210)

        self.label_avatar = QLabel("Avatar:", self.widget_register_2)
        self.label_avatar.setStyleSheet("color: white;")
        self.label_avatar.setFont(QFont("Manrope", 10))
        self.label_avatar.move(700, 260)

        self.button_avatar = QPushButton("Subir Avatar", self.widget_register_2)
        self.button_avatar.setStyleSheet(button_style)
        self.button_avatar.clicked.connect(self.open_avatar_selection)
        self.button_avatar.setFixedWidth(max_width_button)
        self.button_avatar.move(700, 290)

        self.label_avatar_info = QLabel("No se ha subido ningún avatar", self.widget_register_2)
        self.label_avatar_info.setStyleSheet("color: white;")
        self.label_avatar_info.setFixedSize(300, 30) 
        self.label_avatar_info.move(700, 330)
        
        # Label para mostrar el avatar después de la carga
        self.label_avatar_image = QLabel(self.widget_register_2)
        self.label_avatar_image.setStyleSheet("border: 2px solid #666666;")
        self.label_avatar_image.setFixedSize(100, 100)
        self.label_avatar_image.move(700, 360)

        # Mostrar el widget de recorte directamente
        self.cropper_widget = ImageCropperWidget(calendar_arrow_path, self.widget_register_2)  # Inicializa el widget de recorte con 'widget' como padre
        self.cropper_widget.move(300, 100)
        self.cropper_widget.hide()

        # Botones en posiciones absolutas
        self.button_register = QPushButton("Registrar", self.widget_register_2)
        self.button_register.setStyleSheet(button_style)
        self.button_register.setFixedWidth(max_width_button)
        self.button_register.clicked.connect(self.validate_step2_fields)
        self.button_register.move(350, 450)
        
        self.button_volver = QPushButton("Regresar", self.widget_register_2)
        self.button_volver.setStyleSheet(button_style)
        self.button_volver.setFixedWidth(max_width_button)
        self.button_volver.clicked.connect(self.switch_to_register_step1)
        self.button_volver.move(510, 450)

        self.widget_register_2.setStyleSheet("""
            background-color: #111c22;
            border-radius: 8px;
            padding: 5px;
        """)

        return self.widget_register_2


    # Validación de campos de la ventana 2
    def validate_step2_fields(self):
        nombres = self.textbox_nombres.text().strip()
        apellidos = self.textbox_apellidos.text().strip()
        telefono = self.textbox_telefono.text().strip()
        direccion = self.textbox_direccion.text().strip()

        # Limpiar errores previos
        self.icon_error_nombres.setVisible(False)
        self.icon_error_apellidos.setVisible(False)
        self.icon_error_telefono.setVisible(False)
        self.icon_error_direccion.setVisible(False)

        valid = True  # Bandera para seguir o no al registro

        # Validar campos vacíos
        if not nombres:
            self.icon_error_nombres.setVisible(True)
            self.icon_error_nombres.setToolTip("El campo 'Nombres' está vacío.")
            valid = False

        if not apellidos:
            self.icon_error_apellidos.setVisible(True)
            self.icon_error_apellidos.setToolTip("El campo 'Apellidos' está vacío.")
            valid = False

        if not telefono or not telefono.isdigit() or len(telefono) < 10:
            self.icon_error_telefono.setVisible(True)
            self.icon_error_telefono.setToolTip("El campo 'Teléfono' no es válido.")
            valid = False

        if not direccion:
            self.icon_error_direccion.setVisible(True)
            self.icon_error_direccion.setToolTip("El campo 'Dirección' está vacío.")
            valid = False

        # Si todo es válido, proceder al registro
        if valid:
            self.register_user()


    # Alternar a la primera fase del registro
    def switch_to_register_step1(self):
        self.stack.setCurrentWidget(self.register_step1_widget_center)

    # Alternar a la segunda fase del registro
    def switch_to_register_step2(self):
        password = self.textbox_password_reg.text()
        confirm_password = self.textbox_confirm_password.text()

        if password != confirm_password:
            QMessageBox.warning(self, "Error", "Las contraseñas no coinciden")
            return
        self.stack.setCurrentWidget(self.register_step2_widget_center)

    # Alternar al formulario de login
    def switch_to_login(self):
        self.stack.setCurrentWidget(self.login_widget)

    def open_avatar_selection(self):
        button_position = self.combobox_genero.mapToGlobal(QPoint(-13, -72))
        avatar_window = AvatarSelectionWindow(self, button_position=button_position)
        avatar_window.show()

    def process_image(self, file):
        if file:
            # Obtener las últimas partes de la ruta
            base_name = os.path.basename(file)  # Nombre del archivo
            dir_name = os.path.dirname(file)  # Ruta del directorio
            # Dividir la ruta del directorio y solo mostrar las últimas dos carpetas
            shortened_dir = os.path.join(
                os.path.split(os.path.split(dir_name)[0])[1],
                os.path.split(dir_name)[1]
            )

            # Mostrar la ruta reducida en el QLabel
            self.label_avatar_info.setText(f"{shortened_dir}\\{base_name}")

            # Verificar si el cropper_widget ya existe y está visible
            if self.cropper_widget is None or not self.cropper_widget.isVisible():
                self.cropper_widget = ImageCropperWidget(file, self)  # Inicializa el cropper_widget
                # Añadir el cropper_widget a la ventana correspondiente
                self.cropper_widget.setParent(self.widget_register_2)  # Asegúrate de que el padre es correcto
                # Conectar el slot para obtener el pixmap recortado y actualizar el QLabel
                self.cropper_widget.image_cropped.connect(self.update_avatar_label)
                self.cropper_widget.move(700, 100)
                self.cropper_widget.show()  # Mostrar el widget de recorte
            else:
                # Si el cropper_widget ya existe y está visible, actualizar la imagen
                self.cropper_widget.original_pixmap = QPixmap(file)
                max_size = QSize(280, 280)
                self.cropper_widget.scaled_pixmap = self.cropper_widget.original_pixmap.scaled(
                    max_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
                self.cropper_widget.crop_label.setPixmap(self.cropper_widget.scaled_pixmap)
                self.cropper_widget.crop_label.setFixedSize(self.cropper_widget.scaled_pixmap.size())
                self.cropper_widget.show()  # Asegurarse de que se muestre el widget de recorte



    def update_avatar_label(self, cropped_pixmap):
        """Actualiza el avatar con la imagen recortada y lo convierte a binario."""
        if cropped_pixmap:
            # Actualizar la vista previa del avatar en la interfaz
            self.label_avatar_image.setPixmap(cropped_pixmap.scaled(90, 90, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.label_avatar_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Convertir el QPixmap recortado a formato binario (JPEG)
            byte_array = QByteArray()
            buffer = QBuffer(byte_array)
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            cropped_pixmap.save(buffer, "JPEG")  # Puedes usar PNG o cualquier otro formato
            self.avatar_binary = byte_array.data()  # Almacenar la imagen en binario




    # Función para registrar al usuario en MySQL
    def register_user(self):
        username = self.textbox_username.text()
        email = self.textbox_email.text()
        password = self.textbox_password_reg.text()
        nombres = self.textbox_nombres.text()
        apellidos = self.textbox_apellidos.text()
        avatar_binary = getattr(self, 'avatar_binary', None)  # Obtener el avatar en binario
        telefono = self.textbox_telefono.text()
        direccion = self.textbox_direccion.text()
        genero = self.combobox_genero.currentText()
        fecha_nacimiento = self.date_fecha_nacimiento.date().toString("yyyy-MM-dd")
        tipo_registro = 'email'  # Establecer el tipo de registro como 'email'

        # Si no se ha cargado ningún avatar, seleccionar la imagen predeterminada según el género
        if avatar_binary is None:
            if genero == 'Male':
                avatar_predef_path = os.path.join(carpeta_recursos, 'avatar_predef_male.png')
            elif genero == 'Female':
                avatar_predef_path = os.path.join(carpeta_recursos, 'avatar_predef_female.png')
            else:
                # Si no es ni 'Male' ni 'Female', usar una imagen predeterminada genérica
                avatar_predef_path = os.path.join(carpeta_recursos, 'avatar_predef.png')
            
            with open(avatar_predef_path, 'rb') as f:
                avatar_binary = f.read()

        conn = connect_db()
        cursor = conn.cursor(dictionary=True)  # Cursor como diccionario

        # Insertar nuevo usuario con los datos completados, incluyendo el avatar en formato binario
        cursor.execute("""
            INSERT INTO users (username, email, password, nombres, apellidos, avatar, telefono, direccion, genero, fecha_nacimiento, es_premium, tipo_registro)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (username, email, password, nombres, apellidos, avatar_binary, telefono, direccion, genero, fecha_nacimiento, False, tipo_registro))
        conn.commit()

        # Recuperar el usuario recién registrado
        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        user_data = cursor.fetchone()  # Esto ahora será un diccionario
        self.open_user_profile(user_data)
        conn.close()


    # Abrir ventana de perfil del usuario
    def check_login(self):
        username = self.textbox_user.text()
        password = self.textbox_password.text()
        try:
            conn = connect_db()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
            result = cursor.fetchone()

            if result:
                # Convertir el resultado en un diccionario para facilitar el acceso
                column_names = [desc[0] for desc in cursor.description]
                user_data = dict(zip(column_names, result))

                # Procesar el avatar si existe y es del tipo bytes
                if 'avatar' in user_data and user_data['avatar']:
                    avatar_data = user_data['avatar']

                    if isinstance(avatar_data, bytes):
                        try:
                            user_data['avatar'] = avatar_data.decode('utf-8')
                        except UnicodeDecodeError:
                            user_data['avatar'] = base64.b64encode(avatar_data).decode('utf-8')

                conn.close()

                # Llamar al método handle_login_success
                self.handle_login_success(user_data)
            else:
                login_fail_box = CustomMessageBox("Error de login", "Usuario o contraseña incorrectos.", message_type="error")
                login_fail_box.exec()
                
        except Exception as e:
            error_message = f"No se pudo conectar con la base de datos. Detalles: {str(e)}"
            connection_fail_box = CustomMessageBox("Error de conexión", error_message, message_type="error")
            connection_fail_box.exec()
                
        

            
    def handle_login_success(self, user_data):
        # Guardar los datos del usuario
        self.user_data = user_data

        # Remover el widget actual del stack si no es el login_widget
        current_widget = self.stack.currentWidget()
        if current_widget != self.login_widget:
            self.stack.removeWidget(current_widget)
            current_widget.deleteLater()

        # Crear el LoadingWidget si no existe o ha sido eliminado
        if not hasattr(self, 'loading_widget') or self.loading_widget is None:
            self.loading_widget = LoadingWidget(int(self.parent().width() * 0.7))

        # Añadir el loading_widget al stack si no está ya añadido
        if self.loading_widget not in [self.stack.widget(i) for i in range(self.stack.count())]:
            self.stack.addWidget(self.loading_widget)

        # Establecer el loading_widget como el widget actual
        self.stack.setCurrentWidget(self.loading_widget)

        # Iniciar el hilo de carga
        self.load_thread = QThread()
        self.worker = LoadApplicationWorker(self.user_data, self.gestura_app)
        self.worker.moveToThread(self.load_thread)

        # Conectar señales y slots
        self.load_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.load_thread.quit)
        self.worker.finished.connect(self.on_loading_finished)
        self.load_thread.start()


    def open_user_profile(self, user_data):
        # Ajustar el código aquí para abrir el perfil del usuario con los datos que recibimos de Google o email
        user_profile = {
            "username": user_data['username'],
            "email": user_data['email'],
            "nombres": user_data['nombres'],
            "apellidos": user_data['apellidos'],
            "avatar": user_data['avatar'],  # Cambié la clave de 'avatar' a 'avatar'
            "telefono": user_data['telefono'],
            "direccion": user_data['direccion'],
            "genero": user_data['genero'],
            "fecha_nacimiento": user_data['fecha_nacimiento']
        }
        
        # Instancia del UserProfileWindow y añadirlo al stack
        self.gestura_app.cargar_gestura_aplicacion(user_profile)

        
    def open_reset_password_link(self, url):
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl(url))  # Abre el enlace de restablecimiento

        layout = QVBoxLayout()
        layout.addWidget(self.browser)

        self.reset_password_widget = QWidget()
        self.reset_password_widget.setLayout(layout)

        # Añadir el widget al stack y mostrarlo
        self.stack.addWidget(self.reset_password_widget)
        self.stack.setCurrentWidget(self.reset_password_widget)

    # Método para mostrar el formulario de recuperación
    def show_password_recovery(self):
        self.recovery_widget = self.create_password_recovery_widget()
        self.stack.addWidget(self.recovery_widget)
        self.stack.setCurrentWidget(self.recovery_widget)


    # Autenticación con Google
    def google_login(self):
        client_id = os.getenv('GOOGLE_CLIENT_ID')
        redirect_uri = 'http://localhost:8000/callback/google'
        authorization_base_url = 'https://accounts.google.com/o/oauth2/auth'

        # Crear sesión OAuth2 con los scopes adecuados
        scope = [
            "profile", "email", "openid",
            "https://www.googleapis.com/auth/user.birthday.read",
            "https://www.googleapis.com/auth/user.gender.read",
            "https://www.googleapis.com/auth/user.phonenumbers.read"
        ]
        google = OAuth2Session(client_id, redirect_uri=redirect_uri, scope=scope)

        # Obtener la URL de autorización
        authorization_url, state = google.authorization_url(authorization_base_url, access_type='offline', prompt='select_account')

        # Guardar el estado de OAuth en una variable de instancia
        self.oauth_state = state

        # Crear la vista de autenticación de Google como un widget y añadirlo al stack
        self.google_login_widget = AlterLoginWidget(authorization_url, self)
        self.stack.addWidget(self.google_login_widget)

        # Cambiar a la vista de autenticación de Google
        self.stack.setCurrentWidget(self.google_login_widget)

    # Autenticación con Facebook
    def facebook_login(self):
        client_id = os.getenv('FACEBOOK_APP_ID')
        redirect_uri = 'http://localhost:8000/callback/facebook'
        authorization_base_url = 'https://www.facebook.com/v20.0/dialog/oauth'

        # Crear sesión OAuth2
        scope = [
            'email', 
            'public_profile', 
            'user_birthday', 
            'user_gender', 
            'user_photos', 
            'user_hometown', 
            'user_location'
        ]
        facebook = OAuth2Session(client_id, redirect_uri=redirect_uri, scope=scope)

        # Obtener la URL de autorización
        authorization_url, state = facebook.authorization_url(authorization_base_url)

        # Guardar el estado de OAuth en una variable de instancia
        self.oauth_state = state
        # Crear la vista de autenticación de Facebook como un widget y añadirla al stack
        self.facebook_login_widget = AlterLoginWidget(authorization_url, self)
        self.stack.addWidget(self.facebook_login_widget)

        # Cambiar a la vista de autenticación de Facebook
        self.stack.setCurrentWidget(self.facebook_login_widget)
    
    def on_loading_finished(self):
        # Remover el loading_widget del central_layout
        self.central_layout.removeWidget(self.loading_widget)
        self.loading_widget.deleteLater()
        self.loading_widget = None  # Añade esta línea

        # Cambiar a la ventana principal
        self.gestura_app.cargar_gestura_aplicacion(self.user_data)
        
class AvatarSelectionWindow(QWidget):
    def __init__(self, parent=None, button_position=None):
        super().__init__(parent)
        self.setFixedSize(300, 400)

        # Posición dinámica
        if button_position:
            self.move(button_position)

        # Crear layout principal
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Crear barra de título personalizada
        title_bar = QLabel("Seleccionar Avatar", self)
        title_bar.setStyleSheet("""
            QLabel {
                color: #333333;
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
            }
        """)
        title_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Rutas de los recursos
        ruta_icono_archivos = os.path.join(carpeta_recursos, "icono_carpeta.png")  # Cambia según tu archivo
        ruta_icono_camara = os.path.join(carpeta_recursos, "icono_camara.png")  # Cambia según tu archivo

        # Crear botón para seleccionar desde archivos
        self.button_files = QPushButton(" Archivos", self)
        self.button_files.setIcon(QIcon(ruta_icono_archivos))
        self.button_files.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                color: #333333;
                font-size: 14px;
                font-weight: 500;
                padding: 10px;
                border: 1px solid #cccccc;
                border-radius: 5px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.button_files.setIconSize(QSize(24, 24))
        self.button_files.setFixedHeight(50)
        self.button_files.clicked.connect(self.open_file_dialog)

        # Crear botón para tomar foto con la cámara
        self.button_camera = QPushButton(" Cámara", self)
        self.button_camera.setIcon(QIcon(ruta_icono_camara))
        self.button_camera.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                color: #333333;
                font-size: 14px;
                font-weight: 500;
                padding: 10px;
                border: 1px solid #cccccc;
                border-radius: 5px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.button_camera.setIconSize(QSize(24, 24))
        self.button_camera.setFixedHeight(50)
        self.button_camera.clicked.connect(self.start_camera)

        # Contenedor horizontal para los botones
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.button_files)
        buttons_layout.addWidget(self.button_camera)
        buttons_layout.setSpacing(10)

        # Elementos de cámara (inicialmente ocultos)
        self.viewfinder = QVideoWidget()
        self.viewfinder.setVisible(False)
        self.viewfinder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Asegura que el tamaño se ajuste
        self.viewfinder.setFixedSize(260, 260)  # Ajustar el tamaño para que encaje perfectamente
        

        self.button_take_photo = QPushButton("Tomar Foto", self)
        self.button_take_photo.setStyleSheet("""
            QPushButton {
                background-color: #007BFF;
                color: white;
                font-size: 14px;
                font-weight: 500;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        self.button_take_photo.setFixedHeight(40)
        self.button_take_photo.setVisible(False)
        self.button_take_photo.clicked.connect(self.take_photo)

        # Añadir elementos al layout principal
        layout.addWidget(title_bar)
        layout.addLayout(buttons_layout)
        layout.addWidget(self.viewfinder, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.button_take_photo, alignment=Qt.AlignmentFlag.AlignCenter)

        # Configurar el layout para el widget
        self.setLayout(layout)


    def open_file_dialog(self):
        # Configurar solo la opción ReadOnly
        options = QFileDialog.Option.ReadOnly
        
        # Usar el explorador de archivos nativo de Windows
        file, _ = QFileDialog.getOpenFileName(self, "TOMATE UNA FOTO", "", "Imágenes (*.png *.jpg *.jpeg)", options=options)
        
        # Verificar si se seleccionó un archivo
        if file:
            self.parent().process_image(file)
            self.close()

    def start_camera(self):
        self.viewfinder.setVisible(True)
        self.button_take_photo.setVisible(True)
        self.button_files.setVisible(False)
        self.button_camera.setVisible(False)

        # Configurar la cámara
        self.camera = QCamera()
        self.camera.setViewfinder(self.viewfinder)
        self.camera.start()

        self.capture = QMediaCaptureSession(self.camera)

    def take_photo(self):
        temp_file = os.path.join(tempfile.gettempdir(), "captured_image.jpg")

        def on_image_captured(id, image):
            image.save(temp_file)
            self.camera.stop()
            self.viewfinder.setVisible(False)
            self.button_take_photo.setVisible(False)
            self.parent().process_image(temp_file)
            self.close()

        self.capture.imageCaptured.connect(on_image_captured)
        self.capture.capture()

    def paintEvent(self, event):
        """Sobrescribimos paintEvent para establecer un fondo oscuro con sombra suave."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Fondo principal
        rect = self.rect()
        painter.setBrush(QColor("#2B2B2B"))  # Fondo oscuro (puedes ajustar el color si deseas)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(rect)

        # Sombra suave alrededor del borde
        shadow_color = QColor(0, 0, 0, 80)  # Color negro con transparencia
        for i in range(1, 8):  # Gradiente de sombra
            shadow_rect = rect.adjusted(i, i, -i, -i)  # Reduce el área de la sombra
            painter.setPen(QPen(shadow_color, 1))
            painter.drawRect(shadow_rect)

        # Llamar al método original para asegurar que otros elementos se dibujen
        super().paintEvent(event)




class LoadingWidget(QWidget):
    def __init__(self, window_width=600):
        super().__init__()

        # Permitir que el fondo sea transparente
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Configurar un fondo oscuro semi-transparente
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(26, 39, 48, 0.8);  /* Fondo con transparencia */
                border-radius: 15px;  /* Bordes redondeados */
                padding: 0px;  /* Espaciado interno */
                border: 2px solid rgba(255, 255, 255, 0.1);  /* Borde externo sutil */
            }
        """)

        # Crear el layout principal
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Eliminar márgenes del layout principal
        layout.setSpacing(0)  # Eliminar el espaciado entre widgets

        # Header con logo
        header_widgetL = self.create_header_widget()
        layout.addLayout(header_widgetL)
        
        # Línea divisora superior
        line1 = self.create_horizontal_line()
        layout.addWidget(line1)

        # Espaciado antes del contenido principal
        layout.addStretch()

        # Contenido principal (GIF y barra de progreso)
        gif_layout = QVBoxLayout()
        gif_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Crear una etiqueta para la animación
        self.loading_label = QLabel()
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Cargar la animación
        loading_movie = QMovie(os.path.join(carpeta_recursos, 'loading.gif'))
        loading_movie.setScaledSize(QSize(800, 400))  # Fijar tamaño del GIF
        self.loading_label.setMovie(loading_movie)
        loading_movie.start()

        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #05B8CC;
                width: 20px;
            }
        """)
        self.progress_bar.setRange(0, 0)  # Barra indeterminada
        self.progress_bar.setFixedHeight(30)
        self.progress_bar.setFixedWidth(window_width)

        # Añadir widgets al layout del GIF
        gif_layout.addWidget(self.loading_label)
        gif_layout.addWidget(self.progress_bar)
        layout.addLayout(gif_layout)

        # Espaciado después del contenido principal
        layout.addStretch()

        # Línea divisora inferior
        line2 = self.create_horizontal_line()
        layout.addWidget(line2)

        # Logos de redes sociales
        social_layout = self.create_social_layout()
        social_layout.setSpacing(7)
        layout.addLayout(social_layout)

        # Footer con enlaces
        footer_layout = self.create_footer_layout()
        footer_layout.setSpacing(7)
        layout.addLayout(footer_layout)

        self.setStyleSheet("""
            background-color: #111c22;
            border-radius: 8px;
            padding: 5px;
        """)
        
        # Establecer el layout en el widget
        self.setLayout(layout)

    def create_horizontal_line(self):
        """Crear una línea divisora."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #18232b; height: 2px;") 

        return line

    def create_header_widget(self):
        """Crear el header con el logo y texto."""
        header_widgetL = QVBoxLayout()
        header_widgetL.setContentsMargins(10, 0, 0, 20)  # Márgenes del layout

        # Logo
        logo_icon_path = os.path.join(carpeta_recursos, 'logo_icon.png')
        font_manrope_title = QFont("Manrope", 28, QFont.Weight.Bold)
        scene = QGraphicsScene()
        view = QGraphicsView(scene)

        # Agregar la imagen
        pixmap = QPixmap(logo_icon_path).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        scene.addItem(pixmap_item)

        # Agregar el texto
        text_item = QGraphicsTextItem("estura")
        text_item.setDefaultTextColor(QColor("white"))
        text_item.setFont(font_manrope_title)

        total_width = pixmap.width() + text_item.boundingRect().width()
        total_height = max(pixmap.height(), text_item.boundingRect().height())

        # Posicionar el texto justo después de la imagen
        text_item.setPos(
            pixmap.width(),
            ((total_height - text_item.boundingRect().height()) / 2) - 6
        )
        scene.addItem(text_item)

        # Configurar la vista
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setFixedSize(int(total_width + 20), int(total_height + 20))
        view.setStyleSheet("""
            QGraphicsView {
                background: transparent;
                border: none;
            }
        """)
        view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        header_widgetL.addWidget(view, alignment=Qt.AlignmentFlag.AlignLeft)
        return header_widgetL

    def create_social_layout(self):
        """Crear los logos de redes sociales."""
        social_layout = QHBoxLayout()
        social_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        for icon_path in ['google_icon.png', 'facebook_icon.png', 'instagram_icon.png']:
            icon_label = QLabel()
            pixmap = QPixmap(os.path.join(carpeta_recursos, icon_path))
            icon_label.setPixmap(pixmap.scaled(34, 34, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            social_layout.addWidget(icon_label)

        return social_layout

    def create_footer_layout(self):
        """Crear el footer con enlaces."""
        footer_layout = QHBoxLayout()
        footer_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font_manrope_small = QFont("Manrope", 14)

        for text in ["Acerca de nosotros", "Contáctanos", "Términos de servicio", "Política de privacidad"]:
            link_label = QLabel(text)
            link_label.setFont(font_manrope_small)
            link_label.setStyleSheet("color: #999999;")
            footer_layout.addWidget(link_label)

        return footer_layout


class LoadApplicationWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, user_data, gestura_app):
        super().__init__()
        self.user_data = user_data
        self.gestura_app = gestura_app

    def run(self):
        # Perform the heavy operations here
        self.gestura_app.load_heavy_components()

        # Emit the finished signal when done
        self.finished.emit()
       
class PasswordRecoveryPopup(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Establecer título y propiedades del QDialog
        self.setWindowTitle("Recuperación de contraseña")
        self.setModal(True)  # Hace que el popup sea modal (bloquea la ventana principal)
        self.setFixedSize(350, 200)  # Tamaño fijo del popup

        # Eliminar el botón de minimizar y el icono dañado
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint)

        # Establecer el icono de la ventana con un tamaño adecuado
        icon_path = os.path.join(carpeta_recursos, 'logo_icon.png')
        icon = QIcon(QPixmap(icon_path).scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.setWindowIcon(icon)

        # Configuración de las fuentes y estilo de inputs
        font_manrope_label = QFont("Manrope", 10)
        
        pywinstyles.change_header_color(self, color="#00524d")

        # Estilo para el campo de entrada
        input_style = """
            QLineEdit {
                padding: 12px;
                font-family: 'Manrope';
                font-size: 14px;
                color: #eeeeee;
                background-color: #444444;
                border: 2px solid #666666;
                border-radius: 10px;
                outline: none;
            }
            QLineEdit:focus {
                border: 2px solid #007BFF;
            }
        """

        # Estilo para los botones
        button_style = """
            QPushButton {
                padding: 10px;
                font-family: 'Manrope';
                font-size: 14px;
                font-weight: bold;
                border-radius: 15px;
                color: white;
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #007BFF, stop:1 #0056b3);
                outline: none;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:focus {
                background-color: #0056b3;
            }
        """

        # Layout y widgets
        layout = QVBoxLayout()

        self.label_email = QLabel("Introduce tu correo electrónico:")
        self.label_email.setFont(font_manrope_label)
        self.label_email.setStyleSheet("color: white;")

        self.textbox_email = QLineEdit()
        self.textbox_email.setStyleSheet(input_style)

        self.button_send_email = QPushButton("Enviar correo de recuperación")
        self.button_send_email.setStyleSheet(button_style)
        self.button_send_email.clicked.connect(self.send_recovery_email)

        # Añadir sombra al botón
        shadow_effect = QGraphicsDropShadowEffect(self.button_send_email)
        shadow_effect.setBlurRadius(15)
        shadow_effect.setOffset(3, 3)
        shadow_effect.setColor(QColor("black"))
        self.button_send_email.setGraphicsEffect(shadow_effect)

        # Añadir widgets al layout
        layout.addWidget(self.label_email)
        layout.addWidget(self.textbox_email)
        layout.addWidget(self.button_send_email)

        # Configuración del layout del QDialog
        self.setLayout(layout)

        # Aplicar un estilo general al QDialog
        self.setStyleSheet("""
            QDialog {
                background-color: #111c22;
                border-radius: 8px;
            }
            QLabel {
                color: white;
                font-family: 'Manrope';
            }
        """)



    # Método para enviar el correo de recuperación
    def send_recovery_email(self):
        email = self.textbox_email.text()
        conn = connect_db()
        cursor = conn.cursor()

        # Obtener el usuario de la base de datos
        cursor.execute("SELECT nombres, apellidos FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        
        if user:
            # Obtener nombre completo del usuario
            nombre_completo = f"{user[0]} {user[1]}"  # Asumiendo que 'nombres' es el primer campo y 'apellidos' el segundo

            # Hacer la solicitud POST a la API enviando también el nombre
            response = requests.post("http://localhost:8000/send_reset_email", data={'email': email, 'nombre': nombre_completo})

            if response.status_code == 200:
                # Mostrar mensaje de información personalizado
                info_box = CustomMessageBox("Correo enviado", "Revisa tu correo para restablecer tu contraseña.", message_type="info", parent=self)
                info_box.exec()
                self.accept()
            else:
                # Mostrar mensaje de advertencia personalizado en caso de error
                error_box = CustomMessageBox("Problemas", "Hubo un problema al enviar el correo.", message_type="error", parent=self)
                error_box.exec()
        else:
            # Mostrar mensaje de error personalizado si el correo no está registrado
            warning_box = CustomMessageBox("Problemas", "El correo no está registrado.", message_type="warning", parent=self)
            warning_box.exec()

        conn.close()

class CustomMessageBox(QDialog):
    def __init__(self, title, message, message_type="info", parent=None):
        super().__init__(parent)

        # Configurar el diálogo
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(350, 200)

        # Eliminar el botón de minimizar y el icono dañado
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint)

        # Establecer el icono de la ventana con un tamaño adecuado
        icon_path = os.path.join(carpeta_recursos, 'logo_icon.png')
        icon = QIcon(QPixmap(icon_path).scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.setWindowIcon(icon)

        # Diccionario de colores de barra de título por tipo de mensaje
        header_colors = {
            "info": "#00524d",
            "warning": "#FFA500",
            "error": "#FF0000"
        }

        # Diccionario de íconos según el tipo de mensaje
        icon_paths = {
            "info": os.path.join(carpeta_recursos, "info_icon.png"),
            "warning": os.path.join(carpeta_recursos, "warning_icon.png"),
            "error": os.path.join(carpeta_recursos, "error_icon.png")
        }

        # Aplicar un color personalizado a la barra de título según el tipo de mensaje
        pywinstyles.change_header_color(self, color=header_colors.get(message_type, "#00524d"))

        # Configuración del fondo y otros estilos del diálogo (aplicamos estilo a todos los elementos)
        self.setStyleSheet("""
            QDialog {
                background-color: #111c22;
                border-radius: 8px;
            }
            QLabel {
                background-color: transparent;  /* Fondo transparente para el ícono y el texto */
                color: white;
                font-family: 'Manrope';
                font-size: 14px;
            }
            QPushButton {
                background-color: #007BFF;
                color: white;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)

        # Crear los elementos con posicionamiento absoluto
        # 1. Ícono según el tipo de mensaje
        self.icon_label = QLabel(self)
        icon_pixmap = QPixmap(icon_paths.get(message_type, icon_paths["info"])).scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.icon_label.setPixmap(icon_pixmap)
        self.icon_label.move(20, 50)  # Mueve el ícono a la posición deseada

        # 2. Etiqueta de mensaje (activar word wrap para ajustar el texto dentro de la ventana)
        self.label = QLabel(message, self)
        self.label.setWordWrap(True)  # Habilitar word wrap para que el texto se ajuste al ancho
        self.label.setFixedWidth(250)  # Fijar el ancho para que el texto no salga de la ventana
        self.label.move(70, 55)  # Mueve el texto del mensaje al lado del ícono

        # 3. Botón de OK (que abarque todo el ancho)
        self.button_ok = QPushButton("OK", self)
        self.button_ok.clicked.connect(self.accept)
        self.button_ok.setFixedWidth(330)  # Fijar el ancho del botón al tamaño total de la ventana menos márgenes
        self.button_ok.move(10, 150)  # Posicionarlo al fondo, ocupando todo el ancho

        # Posicionar el CustomMessageBox en la misma posición que el cuadro padre (PasswordRecoveryPopup)
        if parent:
            self.move(parent.pos())  # Alinear posición con el cuadro padre

class CustomCalendarWidget(QCalendarWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Rutas de las imágenes de flechas
        left_arrow_path = os.path.join(carpeta_recursos, 'left_arrow.png')
        right_arrow_path = os.path.join(carpeta_recursos, 'right_arrow.png')
        
        # Reemplazar el ícono de la flecha izquierda
        left_button = self.findChild(QToolButton, "qt_calendar_prevmonth")
        left_button.setIcon(QIcon(left_arrow_path))
        left_button.setIconSize(QSize(15, 15))
        
        # Reemplazar el ícono de la flecha derecha
        right_button = self.findChild(QToolButton, "qt_calendar_nextmonth")
        right_button.setIcon(QIcon(right_arrow_path))
        right_button.setIconSize(QSize(15, 15))
        
        # Configura la localización
        locale = QLocale(QLocale.Language.Spanish, QLocale.Country.Spain)
        self.setLocale(locale)

        # Sobrescribir el botón del mes
        self._capitalize_month_names()

        # Conectar la señal para actualizar el texto del botón cuando cambie el mes o el año
        self.currentPageChanged.connect(self._update_month_button_text)

        # Actualizar el texto del mes al iniciar para reflejar el mes actual en mayúscula
        self._update_month_button_text()

    def _capitalize_month_names(self):
        # Localiza el botón que despliega los meses
        self.month_button = self.findChild(QToolButton, "qt_calendar_monthbutton")
        
        if self.month_button:
            # Sobrescribir el menú del botón de los meses
            month_menu = self.month_button.menu()
            
            if month_menu:
                # Modificar los textos de cada entrada del menú
                for i in range(1, 13):
                    action = month_menu.actions()[i - 1]
                    # Capitaliza el texto del mes
                    action.setText(action.text().capitalize())

    def _update_month_button_text(self):
        """Actualiza el texto del botón para el mes visible con mayúscula."""
        if self.month_button:
            # Obtener el mes visible actual en el calendario (no la fecha del sistema)
            current_month = self.monthShown()
            # Obtener el nombre capitalizado del mes visible
            month_name = self.locale().monthName(current_month).capitalize()
            # Actualizar el texto del botón
            self.month_button.setText(month_name)

class ImageCropperLabel(QLabel):
    """Un QLabel personalizado que maneja el área de recorte."""
    MIN_CROP_SIZE = 50  # Tamaño mínimo permitido para el área de recorte

    def __init__(self, pixmap, parent=None):
        super().__init__(parent)

        # Ya que la imagen se escala en el diálogo, solo necesitamos establecer el pixmap directamente
        self.setPixmap(pixmap)
        self.setFixedSize(pixmap.size())

        # Posicionar el área de recorte centrada en la imagen
        initial_size = min(self.width(), self.height()) // 2  # Tamaño inicial, 50% del tamaño mínimo de la imagen
        self.crop_rect = QRect((self.width() - initial_size) // 2,
                               (self.height() - initial_size) // 2,
                               initial_size, initial_size)

        self.resizing = False  # Indica si se está redimensionando el área de recorte
        self.dragging = False  # Indica si se está arrastrando el área de recorte
        self.start_point = QPoint()  # Almacena el punto donde se inicia el resizing o el arrastre
        self.image_margin = 5  # Margen entre la imagen y el borde visible

    def paintEvent(self, event):
        """Dibuja la imagen, la sombra y el área de recorte."""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setOpacity(0.5)  # Sombra semitransparente

        # Dibuja las áreas sombreadas fuera del área de recorte
        painter.fillRect(0, 0, self.width(), self.crop_rect.top(), QBrush(QColor("black")))  # Sombra superior
        painter.fillRect(0, self.crop_rect.bottom() + 1, self.width(), self.height() - self.crop_rect.bottom(), QBrush(QColor("black")))  # Sombra inferior
        painter.fillRect(0, self.crop_rect.top(), self.crop_rect.left(), self.crop_rect.height(), QBrush(QColor("black")))  # Sombra izquierda
        painter.fillRect(self.crop_rect.right() + 1, self.crop_rect.top(), self.width() - self.crop_rect.right(), self.crop_rect.height(), QBrush(QColor("black")))  # Sombra derecha

        # Dibuja el área de recorte
        painter.setOpacity(1.0)
        painter.setPen(QPen(QColor("red"), 2, Qt.PenStyle.SolidLine))
        painter.drawRect(self.crop_rect)

        # Dibuja la manija inferior derecha
        self._draw_handle(painter)

    def _draw_handle(self, painter):
        """Dibuja la manija inferior derecha."""
        handle_size = 10
        handle_rect = QRect(self.crop_rect.bottomRight() - QPoint(handle_size // 2, handle_size // 2), QSize(handle_size, handle_size))
        painter.setBrush(QBrush(QColor("white")))
        painter.drawRect(handle_rect)

    def mousePressEvent(self, event):
        """Captura el inicio del redimensionamiento si se presiona la manija, o el arrastre si se presiona el área."""
        handle_size = 10
        handle_rect = QRect(self.crop_rect.bottomRight() - QPoint(handle_size // 2, handle_size // 2), QSize(handle_size, handle_size))
        
        if handle_rect.contains(event.pos()):
            self.resizing = True
            self.start_point = event.pos()
        elif self.crop_rect.contains(event.pos()):  # Si se hace clic dentro del área de recorte
            self.dragging = True
            self.start_point = event.pos()

    def mouseMoveEvent(self, event):
        """Permite redimensionar o mover el área de recorte."""
        if self.resizing:
            # Calcular el nuevo ancho y alto en base al movimiento del ratón
            new_width = max(event.pos().x() - self.crop_rect.left(), self.MIN_CROP_SIZE)
            new_height = max(event.pos().y() - self.crop_rect.top(), self.MIN_CROP_SIZE)

            # Asegurarse de que el área de recorte mantenga una proporción 1:1
            new_size = min(new_width, new_height)

            # Limitar el área de recorte para que no crezca más allá de los bordes inferiores o derechos
            if self.crop_rect.left() + new_size > self.width():  # Si excede el borde derecho
                new_size = self.width() - self.crop_rect.left()

            if self.crop_rect.top() + new_size > self.height():  # Si excede el borde inferior
                new_size = self.height() - self.crop_rect.top()

            # Actualizar el tamaño del área de recorte
            self.crop_rect.setSize(QSize(new_size, new_size))

            # Limitar el área de recorte para que no se salga del borde de la imagen
            self.update()

        elif self.dragging:
            # Calcular cuánto se mueve el recuadro en X e Y
            delta = event.pos() - self.start_point

            # Actualizar la posición del área de recorte
            self.crop_rect.translate(delta)

            # Limitar el área de recorte para que no se salga del borde de la imagen
            self._limit_to_image_bounds()

            # Actualizar el punto inicial para el próximo movimiento
            self.start_point = event.pos()

            self.update()

    def mouseReleaseEvent(self, event):
        """Detiene el redimensionamiento o el arrastre."""
        self.resizing = False
        self.dragging = False

    def _limit_to_image_bounds(self):
        """Asegura que el área de recorte no se salga del borde de la imagen."""
        margin = self.image_margin  # Limitar al margen de la imagen visible

        # Limitar al borde superior
        if self.crop_rect.top() < margin:
            self.crop_rect.moveTop(margin)

        # Limitar al borde inferior
        if self.crop_rect.bottom() > self.height() - margin:
            self.crop_rect.moveBottom(self.height() - margin)

        # Limitar al borde izquierdo
        if self.crop_rect.left() < margin:
            self.crop_rect.moveLeft(margin)

        # Limitar al borde derecho
        if self.crop_rect.right() > self.width() - margin:
            self.crop_rect.moveRight(self.width() - margin)

    def get_crop_rect(self):
        """Devuelve el área de recorte actual."""
        return self.crop_rect

class ImageCropperWidget(QWidget):
    image_cropped = pyqtSignal(QPixmap)  # Señal para indicar que la imagen ha sido recortada

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 400)

        # Crear layout vertical
        layout = QVBoxLayout(self)

        # Crear barra de título personalizada
        title_bar = QLabel("Recortar Imagen", self)
        title_bar.setStyleSheet("background-color: #333333; color: white; font-size: 16px; padding: 10px;")
        title_bar.setFixedHeight(40)
        title_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Cargar la imagen original
        self.original_pixmap = QPixmap(image_path)

        # Escalar la imagen de manera que encaje dentro de un área máxima (p. ej., 300x300)
        max_size = QSize(280, 280)  # Tamaño máximo permitido
        self.scaled_pixmap = self.original_pixmap.scaled(max_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        # Crear un QLabel personalizado para manejar el área de recorte, sin margen ni borde
        self.crop_label = ImageCropperLabel(self.scaled_pixmap, self)
        self.crop_label.setStyleSheet("border: none;")  # Sin borde ni margen

        # Botón para recortar la imagen, estilizado
        self.crop_button = QPushButton("Recortar", self)
        self.crop_button.setStyleSheet("""
            QPushButton {
                background-color: #007BFF;
                color: white;
                font-size: 14px;
                padding: 8px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        self.crop_button.setFixedHeight(40)
        self.crop_button.clicked.connect(self.crop_image)

        # Añadir elementos al layout
        layout.addWidget(title_bar)
        layout.addWidget(self.crop_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.crop_button)

        # Configurar el layout para el widget
        self.setLayout(layout)

    def paintEvent(self, event):
        """Sobrescribimos paintEvent para asegurarnos de que el fondo se dibuje correctamente con un diseño atractivo."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Definir el rectángulo para el área de dibujo (con un margen ligero para mejorar la estética)
        rect = self.rect().adjusted(5, 5, -5, -5)

        # Pintar el fondo con un gradiente radial para darle un efecto suave
        gradient = QLinearGradient(QPointF(rect.topLeft()), QPointF(rect.bottomRight()))
        gradient.setColorAt(0, QColor("#1e2a38"))  # Color superior más claro
        gradient.setColorAt(1, QColor("#111c22"))  # Color inferior más oscuro
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)  # Sin borde
        painter.drawRoundedRect(rect, 15, 15)  # Fondo redondeado con esquinas suaves

        # Añadir un borde fino y suave alrededor del widget
        painter.setPen(QPen(QColor("#007BFF"), 2))  # Borde de color azul suave
        painter.drawRoundedRect(rect, 15, 15)  # Mismo rectángulo, pero con borde

        # Llamar al método original para que otros elementos se dibujen
        super().paintEvent(event)

    def crop_image(self):
        """Recorta la imagen según el área seleccionada."""
        crop_rect = self.crop_label.get_crop_rect()

        # Calcular el factor de escalado entre la imagen visible y la original
        scale_factor_width = self.original_pixmap.width() / self.scaled_pixmap.width()
        scale_factor_height = self.original_pixmap.height() / self.scaled_pixmap.height()
        
        # Aplicar los factores de escalado al área de recorte
        adjusted_crop_x = int(crop_rect.x() * scale_factor_width)
        adjusted_crop_y = int(crop_rect.y() * scale_factor_height)
        adjusted_crop_size = min(int(crop_rect.width() * scale_factor_width), int(crop_rect.height() * scale_factor_height))

        # Asegurarse de que el área recortada mantenga el tamaño cuadrado
        adjusted_crop_rect = QRect(
            adjusted_crop_x,
            adjusted_crop_y,
            adjusted_crop_size,  # Mantener el ancho y alto iguales
            adjusted_crop_size
        )

        # Limitar las coordenadas de recorte para que no excedan los bordes de la imagen original
        if adjusted_crop_rect.right() > self.original_pixmap.width():
            adjusted_crop_rect.setRight(self.original_pixmap.width())

        if adjusted_crop_rect.bottom() > self.original_pixmap.height():
            adjusted_crop_rect.setBottom(self.original_pixmap.height())

        # Realizar el recorte en la imagen original
        self.cropped_pixmap = self.original_pixmap.copy(adjusted_crop_rect)
        
        # Emitir la señal con la imagen recortada
        self.image_cropped.emit(self.cropped_pixmap)
        self.close()  # Cerrar el widget de recorte
