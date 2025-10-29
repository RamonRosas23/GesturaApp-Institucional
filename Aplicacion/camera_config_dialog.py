# camera_config_dialog.py
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QRadioButton, QHBoxLayout, QWidget, 
    QComboBox, QLineEdit, QCheckBox, QPushButton, QSpacerItem, QSizePolicy, QDialogButtonBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextCharFormat, QColor
from PyQt6.QtMultimedia import QMediaDevices  # Para obtener la lista de cámaras disponibles

class CameraConfigDialog(QDialog):
    def __init__(self, default_config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de la Cámara")
        self.setModal(True)
        self.resize(400, 300)
        
        # Indica si se desea cerrar la conexión (se activará al pulsar el botón "Cerrar Conexión")
        self.close_connection = False

        # Si no se recibe un default_config, definimos uno por defecto.
        # Se agrega la clave "connection_open" para saber si hay conexión activa.
        if default_config is None:
            default_config = {
                "video_source": 0,  # Para dispositivo local, se usará un entero (índice)
                "rotation": 0,
                "flip": False,
                "connection_open": False
            }
        self.default_config = default_config

        main_layout = QVBoxLayout(self)
        
        # --- Origen de video ---
        source_layout = QVBoxLayout()
        source_label = QLabel("Origen de video:")
        source_layout.addWidget(source_label)
        
        # Radio buttons para elegir entre local o por IP
        self.radio_local = QRadioButton("Video Local")
        self.radio_ip = QRadioButton("Por IP")
        # Si el valor es un entero o es una cadena que no empieza con "http://", se asume local.
        if (isinstance(default_config["video_source"], int) or 
           (isinstance(default_config["video_source"], str) and not default_config["video_source"].startswith("http://"))):
            self.radio_local.setChecked(True)
        else:
            self.radio_ip.setChecked(True)
        
        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.radio_local)
        radio_layout.addWidget(self.radio_ip)
        source_layout.addLayout(radio_layout)
        
        # Widget contenedor para las opciones específicas
        self.source_option_widget = QWidget()
        self.source_option_layout = QVBoxLayout(self.source_option_widget)
        
        # Para "Local": llenar el combobox con los dispositivos reales usando QMediaDevices
        self.local_combo = QComboBox()
        try:
            devices = QMediaDevices.videoInputs()
            self.local_combo.clear()
            # Se añade cada dispositivo con el índice como dato
            for i, device in enumerate(devices):
                self.local_combo.addItem(f"{i}: {device.description()}", i)
        except Exception as e:
            # Fallback en caso de error
            self.local_combo.addItem("0: Dispositivo 0", 0)
            self.local_combo.addItem("1: Dispositivo 1", 1)
            self.local_combo.addItem("2: Dispositivo 2", 2)
        
        # Para "Por IP": se crea un widget con 4 QLineEdits para la IP y 1 para el puerto
        self.ip_widget = QWidget()
        ip_layout = QHBoxLayout(self.ip_widget)
        ip_layout.setContentsMargins(0, 0, 0, 0)
        ip_layout.setSpacing(5)
        self.ip_edit1 = QLineEdit()
        self.ip_edit2 = QLineEdit()
        self.ip_edit3 = QLineEdit()
        self.ip_edit4 = QLineEdit()
        for ip_edit in (self.ip_edit1, self.ip_edit2, self.ip_edit3, self.ip_edit4):
            ip_edit.setFixedWidth(40)
            ip_edit.setMaxLength(3)
            ip_edit.setPlaceholderText("xxx")
            ip_edit.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.port_edit = QLineEdit()
        self.port_edit.setFixedWidth(50)
        self.port_edit.setMaxLength(5)
        self.port_edit.setPlaceholderText("puerto")
        self.port_edit.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        ip_layout.addWidget(self.ip_edit1)
        ip_layout.addWidget(QLabel("."))
        ip_layout.addWidget(self.ip_edit2)
        ip_layout.addWidget(QLabel("."))
        ip_layout.addWidget(self.ip_edit3)
        ip_layout.addWidget(QLabel("."))
        ip_layout.addWidget(self.ip_edit4)
        ip_layout.addWidget(QLabel(":"))
        ip_layout.addWidget(self.port_edit)
        
        # Según el default, se muestran u ocultan las opciones
        if self.radio_local.isChecked():
            self.local_combo.show()
            self.ip_widget.hide()
            # Si default es entero, se busca en el combo por su dato
            if isinstance(self.default_config["video_source"], int):
                index = self.local_combo.findData(self.default_config["video_source"])
                if index >= 0:
                    self.local_combo.setCurrentIndex(index)
                else:
                    self.local_combo.setCurrentIndex(0)
            else:
                # Si default es una cadena (no IP), intentar extraer el índice del texto (ej. "0: DroidCam Video")
                try:
                    index = int(self.default_config["video_source"].split(":")[0])
                except Exception:
                    index = 0
                found = self.local_combo.findData(index)
                if found >= 0:
                    self.local_combo.setCurrentIndex(found)
                else:
                    self.local_combo.setCurrentIndex(0)
        else:
            self.local_combo.hide()
            self.ip_widget.show()
            current_source = self.default_config["video_source"]
            if current_source.startswith("http://") and current_source.endswith("/video"):
                ip_port = current_source[len("http://"):-len("/video")]
                if ":" in ip_port:
                    ip_str, port_str = ip_port.split(":", 1)
                    ip_parts = ip_str.split(".")
                    if len(ip_parts) == 4:
                        self.ip_edit1.setText(ip_parts[0])
                        self.ip_edit2.setText(ip_parts[1])
                        self.ip_edit3.setText(ip_parts[2])
                        self.ip_edit4.setText(ip_parts[3])
                    self.port_edit.setText(port_str)
                else:
                    self.ip_edit1.setText("")
                    self.ip_edit2.setText("")
                    self.ip_edit3.setText("")
                    self.ip_edit4.setText("")
                    self.port_edit.setText("")
            else:
                self.ip_edit1.setText("")
                self.ip_edit2.setText("")
                self.ip_edit3.setText("")
                self.ip_edit4.setText("")
                self.port_edit.setText("")
        
        self.source_option_layout.addWidget(self.local_combo)
        self.source_option_layout.addWidget(self.ip_widget)
        
        main_layout.addLayout(source_layout)
        main_layout.addWidget(self.source_option_widget)
        
        # Conectar cambio de selección:
        self.radio_local.toggled.connect(self.toggle_source_option)
        
        # --- Rotación (4 valores) ---
        rotation_layout = QHBoxLayout()
        rotation_label = QLabel("Rotación:")
        self.rotation_combo = QComboBox()
        self.rotation_combo.addItems(["0°", "90°", "180°", "270°"])
        self.rotation_combo.setCurrentText(f"{self.default_config.get('rotation', 0)}°")
        rotation_layout.addWidget(rotation_label)
        rotation_layout.addWidget(self.rotation_combo)
        main_layout.addLayout(rotation_layout)
        
        # --- Opción de flip horizontal ---
        self.flip_checkbox = QCheckBox("Invertir horizontalmente")
        self.flip_checkbox.setChecked(self.default_config.get("flip", False))
        main_layout.addWidget(self.flip_checkbox)
        
        # --- Botones ---
        # Layout horizontal para botones:
        # A la izquierda: botón "Cerrar Conexión" (solo si connection_open es True)
        # A la derecha: botones OK y Cancelar
        button_layout = QHBoxLayout()
        if self.default_config.get("connection_open", False):
            self.close_conn_button = QPushButton("Cerrar Conexión")
            self.close_conn_button.setStyleSheet(self.ok_button_style())
            self.close_conn_button.clicked.connect(self.on_close_connection)
            button_layout.addWidget(self.close_conn_button, alignment=Qt.AlignmentFlag.AlignLeft)
        else:
            self.close_conn_button = None
        button_layout.addStretch()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_button, alignment=Qt.AlignmentFlag.AlignRight)
        button_layout.addWidget(self.cancel_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        main_layout.addLayout(button_layout)
    
    def toggle_source_option(self):
        if self.radio_local.isChecked():
            self.local_combo.show()
            self.ip_widget.hide()
        else:
            self.local_combo.hide()
            self.ip_widget.show()
    
    def on_close_connection(self):
        self.close_connection = True
        self.accept()
    
    def get_configuration(self):
        config = {}
        if self.radio_local.isChecked():
            # Usar el dato almacenado (el índice)
            config["video_source"] = self.local_combo.itemData(self.local_combo.currentIndex())
        else:
            ip_parts = [
                self.ip_edit1.text().strip(),
                self.ip_edit2.text().strip(),
                self.ip_edit3.text().strip(),
                self.ip_edit4.text().strip()
            ]
            port = self.port_edit.text().strip()
            if all(ip_parts) and port:
                ip_str = ".".join(ip_parts)
                config["video_source"] = f"http://{ip_str}:{port}/video"
            else:
                config["video_source"] = self.default_config.get("video_source", "")
        rotation_str = self.rotation_combo.currentText().replace("°", "")
        config["rotation"] = int(rotation_str)
        config["flip"] = self.flip_checkbox.isChecked()
        config["close_connection"] = self.close_connection
        return config
    
    def ok_button_style(self):
        return """
            QPushButton {
                background-color: #1E3A5F;
                color: white;
                font-size: 12px;
                font-weight: bold;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #2A496E;
            }
            QPushButton:pressed {
                background-color: #122640;
            }
        """
