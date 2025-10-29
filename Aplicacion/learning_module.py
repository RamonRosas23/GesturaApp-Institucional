from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QApplication, QListWidget,
    QStackedWidget, QHBoxLayout, QDialog, QMessageBox, QFrame, QProgressBar,
    QTextEdit, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, QRect
from PyQt6.QtGui import QPixmap, QImage, QPainter, QIcon, QTextCursor, QTextCharFormat, QColor
import sys, os, cv2, numpy as np, tensorflow as tf
import joblib  # Para cargar el escalador
import re
import json
from threading import Thread, Lock
from queue import Queue
import logging

# Configurar logging para suprimir warnings innecesarios
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('mediapipe').setLevel(logging.ERROR)
import mediapipe as mp

MEDIAPIPE_HANDS_CONFIG = {
    'static_image_mode': False,
    'max_num_hands': 1,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'model_complexity': 1
}

class PracticeWidget(QWidget):
    def __init__(self, sign_name: str, model_path: str, labels: list[str], parent=None):
        super().__init__(parent)
        self.frames_por_muestra = 20  # Igual que en el entrenamiento
        self.practice_paused = False  # Flag para controlar si la práctica está pausada
        
        # Cargar el modelo y el escalador
        self.model = tf.keras.models.load_model(model_path)
        scaler_path = os.path.join(os.path.dirname(model_path), "scaler.skl")
        self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        
        
        # Cargar el mapeo de etiquetas si existe
        label_map_path = os.path.join(os.path.dirname(model_path), "label_mapping.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
                
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #223046, stop:1 #1e293b);
                border-radius: 18px;
            }
            QLabel {
                color: #eaf6ff;
                font-size: 16px;
            }
            QProgressBar {
                background: #18232b;
                border-radius: 6px;
                height: 12px;
                text-align: center;
                color: #9dd6ff;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2a496e, stop:1 #9dd6ff);
                border-radius: 6px;
            }
            QPushButton {
                background-color: #1E3A5F;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                padding: 8px 16px;
                margin-top: 0px;
                min-width: 120px;
                max-width: 200px;
            }
            QPushButton:hover {
                background-color: #2A496E;
            }
            QPushButton:pressed {
                background-color: #122640;
            }
            #finishButton {
                background-color: #1E3A5F;
            }
            #resumeButton {
                background-color: #1a722a;  /* Verde para el botón de reanudar */
            }
        """)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._frame)
        self.timer.start(5)  # Máxima fluidez posible
        
        self.seq_buffer = []                 # buffer de hasta 20 arrays de 90 features
        self.sign_name, self.labels = sign_name, labels
        
        # Layout principal
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Layout para la cámara y panel lateral
        main_content = QHBoxLayout()
        main_content.setSpacing(8)

        # Container para el video
        video_container = QWidget()
        video_container.setFixedSize(640, 480)
        video_container.setStyleSheet("background-color: #18232b; border-radius: 12px;")
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_label = QLabel()
        video_layout.addWidget(self.video_label)
        main_content.addWidget(video_container)

        # Panel lateral
        side_panel = QWidget()
        side_panel.setFixedWidth(200)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setSpacing(12)

        # Etiqueta de feedback
        self.feedback_label = QLabel("Intentalo...")
        self.feedback_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                padding: 8px;
                border-radius: 6px;
                background-color: #18232b;
                color: #eaf6ff;
            }
        """)
        self.feedback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feedback_label.setWordWrap(True)
        side_layout.addWidget(self.feedback_label)

        # Etiqueta y barra de progreso
        progress_label = QLabel("Progreso:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("%p%")  # Mostrar porcentaje
        side_layout.addWidget(progress_label)
        side_layout.addWidget(self.progress_bar)

        # Botón de finalizar práctica (visible todo el tiempo)
        self.finish_button = QPushButton("Finalizar práctica")
        self.finish_button.setObjectName("finishButton")
        self.finish_button.clicked.connect(self.finish)
        
        # Botón de reanudar práctica (inicialmente oculto)
        self.resume_button = QPushButton("Reanudar práctica")
        self.resume_button.setObjectName("resumeButton")
        self.resume_button.clicked.connect(self.resume)
        self.resume_button.hide()  # Inicialmente oculto
        
        side_layout.addWidget(self.finish_button)
        side_layout.addWidget(self.resume_button)
        
        # Añadir stretch al final del panel lateral
        side_layout.addStretch()
        
        main_content.addWidget(side_panel)
        layout.addLayout(main_content)
        self.setLayout(layout)

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Permitir hasta 2 manos
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)

    def _frame(self):
       
        # 1) Leer y procesar cámara
        ret, frame = self.cap.read()
        if not ret:
            return

        # Voltear horizontalmente la imagen (efecto espejo)
        frame = cv2.flip(frame, 1)  # El 1 indica volteo horizontal
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Actualizar configuración de MediaPipe para detectar hasta 2 manos
        if not hasattr(self, 'updated_hands_config'):
            # Cerrar la instancia anterior si existe
            if hasattr(self, 'hands'):
                self.hands.close()
            
            # Crear una nueva instancia de Hands con max_num_hands=2
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,  # Permitir detección de 2 manos
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
            self.updated_hands_config = True
        
        # Si la práctica está en pausa o finalizada, solo mostrar la última imagen con feedback
        if hasattr(self, 'practice_paused') and self.practice_paused:
            return
        
        # Procesar el frame con MediaPipe
        results = self.hands.process(frame_rgb)
        
       # Al mostrar los landmarks en pantalla:
        annotated = frame.copy()
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                self.draw.draw_landmarks(
                    annotated, 
                    hand_landmarks, 
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
        
        # Convertir de BGR a RGB para mostrar correctamente (sin necesidad de voltear de nuevo)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, _ = annotated_rgb.shape
        qimg = QImage(annotated_rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pix)
        # 3) Si no hay manos, reset buffer y UI
        if not results.multi_hand_landmarks:
            self.seq_buffer.clear()
            self.feedback_label.setText("Muestra tus manos...")
            self.feedback_label.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    font-weight: bold;
                    padding: 8px;
                    border-radius: 6px;
                    background-color: #18232b;
                    color: #eaf6ff;
                }
            """)
            self.progress_bar.setValue(0)
            return
        
        # 4) Extraer características para ambas manos en el formato correcto
        # Inicializar vector de características con ceros
        feature_vector = np.zeros(90, dtype=np.float32)
        
        # Clasificar landmarks por mano (izquierda/derecha)
        left_hand_data = None
        right_hand_data = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                # Determinar si es mano izquierda o derecha
                # Nota: MediaPipe etiqueta las manos desde la perspectiva de la cámara,
                # así que 'Left' en MediaPipe es la mano derecha del usuario y viceversa
                # Ajustamos esto para coincidir con la perspectiva del usuario
                is_left = handedness.classification[0].label == "Left"
                
                if is_left and left_hand_data is None:
                    left_hand_data = hand_landmarks
                elif not is_left and right_hand_data is None:
                    right_hand_data = hand_landmarks
        
        # Bandera para indicar qué manos están presentes
        has_left = left_hand_data is not None
        has_right = right_hand_data is not None
        
        # Extraer coordenadas en el mismo orden que en el CSV
        # Estructura: [Left_Elbow_xy, Left_Landmarks_xy, Right_Elbow_xy, Right_Landmarks_xy]
        
        # 1. Coordenadas del codo izquierdo (aproximamos usando la muñeca)
        if has_left:
            wrist = left_hand_data.landmark[0]
            feature_vector[0] = wrist.x * w
            feature_vector[1] = wrist.y * h
        
        # 2. Coordenadas de los 21 puntos de la mano izquierda
        if has_left:
            for i in range(21):
                lm = left_hand_data.landmark[i]
                feature_vector[2 + i*2] = lm.x * w
                feature_vector[2 + i*2 + 1] = lm.y * h
        
        # 3. Coordenadas del codo derecho (aproximamos usando la muñeca)
        if has_right:
            wrist = right_hand_data.landmark[0]
            feature_vector[44] = wrist.x * w
            feature_vector[45] = wrist.y * h
        
        # 4. Coordenadas de los 21 puntos de la mano derecha
        if has_right:
            for i in range(21):
                lm = right_hand_data.landmark[i]
                feature_vector[46 + i*2] = lm.x * w
                feature_vector[46 + i*2 + 1] = lm.y * h
        
        # Información de depuración (opcional, comentar en producción)
        # print(f"[DEBUG] Mano izquierda presente: {has_left}")
        # print(f"[DEBUG] Mano derecha presente: {has_right}")
        
        # 5) Buffer de secuencia
        self.seq_buffer.append(feature_vector)
        if len(self.seq_buffer) > 20:  # Mantenemos 20 frames como en el entrenamiento
            self.seq_buffer.pop(0)

        # 6) Normalización y diferencias
        seq = np.vstack(self.seq_buffer)
        
        # Padding si no tenemos suficientes frames
        if seq.shape[0] < 20:
            pad = np.zeros((20 - seq.shape[0], seq.shape[1]), dtype=np.float32)
            seq = np.vstack([seq, pad])

        # Normalización utilizando el escalador guardado
        if self.scaler is not None:
            # Verificar dimensiones antes de normalizar
            mean = self.scaler['mean']
            std = self.scaler['std']
            
            # Añadir log para depuración (opcional, comentar en producción)
            # print(f"[DEBUG] Forma de seq: {seq.shape}")
            # print(f"[DEBUG] Forma de mean: {mean.shape}")
            # print(f"[DEBUG] Forma de std: {std.shape}")
            
            # Aplicar normalización si las dimensiones coinciden
            if seq.shape[1] == mean.shape[0]:
                seq = (seq - mean) / std
            else:
                # Si las dimensiones no coinciden, usar normalización manual
                print(f"[DEBUG] Dimensiones no coinciden. Usando normalización manual.")
                seq_mean = np.mean(seq, axis=0)
                seq_std = np.std(seq, axis=0)
                seq_std = np.where(seq_std == 0, 1e-8, seq_std)  # Evitar división por cero
                seq = (seq - seq_mean) / seq_std
            
        # Calcular diferencias entre frames consecutivos
        data = np.diff(seq, axis=0, prepend=seq[0:1])
        
        # Preparar batch para el modelo (1, 20, features)
        batch = np.expand_dims(data, 0)
        
        # 7) Predecir y actualizar UI
        try:
            preds = self.model.predict(batch, verbose=0)
            
            idx = int(np.argmax(preds, axis=1)[0])
            conf = float(np.max(preds))
            predicted_label = self.labels[idx]
            
            # Actualizar barra de progreso
            progress = int(conf * 100)
            self.progress_bar.setValue(progress)
            self.progress_bar.setFormat(f"{progress}%")
            
            # Actualizar feedback basado en la predicción
            if predicted_label == self.sign_name and conf > 0.7:
                # ¡La seña es correcta! - Pausar la práctica y mostrar mensaje de éxito
                self.practice_paused = True
                self.timer.stop()  # Detener el timer
                
                # Estilo similar al de la imagen (verde con texto blanco)
                self.feedback_label.setText("¡Excelente!\n¡Ya dominas la\nseña!")
                self.feedback_label.setStyleSheet("""
                    QLabel {
                        font-size: 18px;
                        font-weight: bold;
                        padding: 12px;
                        border-radius: 8px;
                        background-color: #1a722a;  /* Verde oscuro como en la imagen */
                        color: white;  /* Texto blanco */
                        text-align: center;
                    }
                """)
                
                # Mostrar botones de control (finalizar/reanudar) como en la imagen
                self.finish_button.setVisible(True)
                self.finish_button.setEnabled(True)
                self.resume_button.setVisible(True)
                self.resume_button.setEnabled(True)
            else:            
                self.feedback_label.setText("Inténtalo nuevamente...")
                self.feedback_label.setStyleSheet("""
                    QLabel {
                        font-size: 18px;
                        font-weight: bold;
                        padding: 8px;
                        border-radius: 6px;
                        background-color: #18232b;
                        color: #eaf6ff;
                    }
                """)
        
        except Exception as e:
            print(f"[ERROR] Error durante la predicción: {str(e)}")
            self.feedback_label.setText(f"Error: {str(e)}")
            self.feedback_label.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 8px;
                    border-radius: 6px;
                    background-color: #6b1a1a;
                    color: #ff6b6b;
                }
            """)
            
    def finish(self):
        """Finaliza la práctica y detiene la captura."""
        self.practice_paused = True
        self.timer.stop()
        
        # Detener la cámara pero no cerrarla aún (para poder reanudar)
        if hasattr(self, 'cap') and self.cap.isOpened():
            # No cerramos la cámara, solo detenemos el procesamiento
            pass
            
        # Actualizar UI
        self.finish_button.setEnabled(False)  # Deshabilitar el botón finalizar
        self.resume_button.setVisible(True)   # Mostrar el botón reanudar
        self.resume_button.setEnabled(True)
        
        # Enviar señal de finalización al padre si fuera necesario
        if hasattr(self.parent(), 'practice_finished'):
            self.parent().practice_finished(self.sign_name)

    def resume(self):
        """Reanuda la práctica."""
        self.practice_paused = False
        
        # Reiniciar buffer y estados
        self.seq_buffer.clear()
        
        # Reiniciar UI
        self.feedback_label.setText("Intentalo nuevamente...")
        self.feedback_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                padding: 8px;
                border-radius: 6px;
                background-color: #18232b;
                color: #eaf6ff;
            }
        """)
        self.progress_bar.setValue(0)
        
        # Reanudar captura
        if not self.timer.isActive():
            self.timer.start(5)
        
        # Actualizar UI de botones
        self.finish_button.setEnabled(True)
        self.resume_button.setVisible(False)

    def closeEvent(self, event):
        """Maneja el cierre del widget."""
        # Detener el timer y liberar recursos
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        
        # Liberar la cámara
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            
        # Liberar recursos de MediaPipe
        if hasattr(self, 'hands'):
            self.hands.close()
            
        # Continuar con el cierre normal
        super().closeEvent(event)

class TutorialWidget(QWidget):
    def __init__(self, sign_name: str, media_root: str, model_path: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #223046, stop:1 #1e293b);
                border-radius: 18px;
            }
            QLabel {
                color: #eaf6ff;
                font-size: 18px;
            }
            QPushButton {
                background-color: #1E3A5F;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 24px;
                margin-top: 18px;
            }
            QPushButton:hover {
                background-color: #2A496E;
            }
            QPushButton:pressed {
                background-color: #122640;
            }
            #videoContainer {
                background-color: #000;
                border-radius: 14px;
                border: 2px solid #2a496e;
                min-height: 480px;
            }
        """)
        
        # Layout principal con margen reducido para más espacio
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Título
        title = QLabel(f"<h2 style='color:#9dd6ff; font-size:28px; font-weight:bold; margin-bottom:10px;'>Tutorial: {sign_name}</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Contenedor del video con tamaño más pequeño
        video_container = QWidget()
        video_container.setObjectName("videoContainer")
        video_container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        video_container.setFixedSize(400, 300)  # Tamaño fijo más pequeño
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)
        
        self.video_label = QLabel()
        self.video_label.setFixedSize(400, 300)  # Mismo tamaño que el contenedor
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(self.video_label)
        
        # Centrar el contenedor del video
        container_wrapper = QHBoxLayout()
        container_wrapper.addStretch()
        container_wrapper.addWidget(video_container)
        container_wrapper.addStretch()
        layout.addLayout(container_wrapper)

        video_path = os.path.join(media_root, f"{sign_name}.mp4")
        images_dir = os.path.join(media_root, sign_name, "images")
        if os.path.isfile(video_path):
            self.cap = cv2.VideoCapture(video_path)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.next_frame)
            self.timer.start(int(1000/120))  # 120 FPS para máxima fluidez
        elif os.path.isdir(images_dir):
            self.image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(('png','jpg'))])
            self.idx = 0
            self.show_image(0)
            btns = QHBoxLayout()
            prev = QPushButton("Anterior")
            prev.clicked.connect(self.prev_image)
            btns.addWidget(prev)
            nxt = QPushButton("Siguiente")
            nxt.clicked.connect(self.next_image)
            btns.addWidget(nxt)
            layout.addLayout(btns)
        else:
            self.video_label.setText("<span style='color:#ff6b6b'>(No tutorial disponible)</span>")
        
        self.practice_btn = QPushButton("Practicar seña")
        self.practice_btn.clicked.connect(self.go_practice)
        layout.addWidget(self.practice_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            # Cuando llegue al final del video, volver al principio
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                self.timer.stop()
                self.cap.release()
                return
            
        # Convertir el frame a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Redimensionar el frame al tamaño exacto que queremos (400x300)
        frame_resized = cv2.resize(frame_rgb, (400, 300), interpolation=cv2.INTER_AREA)
        
        # Crear QImage directamente del tamaño correcto
        qimg = QImage(frame_resized.data, 400, 300, 3*400, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Establecer el pixmap
        self.video_label.setPixmap(pixmap)
        
    def show_image(self, i):
        # Para imágenes, también aseguramos el tamaño correcto
        pixmap = QPixmap(self.image_paths[i])
        scaled_pixmap = pixmap.scaled(400, 300, 
                                    Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        self.idx = i

    def next_image(self): 
        self.show_image(min(self.idx + 1, len(self.image_paths) - 1))

    def prev_image(self): 
        self.show_image(max(self.idx - 1, 0))

    def go_practice(self): 
        self.parent().setCurrentIndex(1)

    def closeEvent(self, event):
        if hasattr(self, 'timer'):
            self.timer.stop()
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)

class LearningModule(QWidget):
    def __init__(self, media_root: str, model_path: str, signs: list[str], parent=None):
        super().__init__(parent)
        self.media_root, self.model_path, self.signs = media_root, model_path, signs
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1e293b, stop:1 #223046);
                border-radius: 22px;
            }
            QListWidget {
                background-color: #16202a;
                border: none;
                border-radius: 12px;
                padding: 10px;
                color: #eaf6ff;
                font-size: 18px;
            }
            QListWidget::item:hover {
                background-color: #2A496E;
            }
            QListWidget::item:selected {
                background-color: #1E3A5F;
                color: #FFFFFF;
            }
            QStackedWidget {
                background: transparent;
                border-radius: 18px;
            }
        """)
        layout=QHBoxLayout(self)
        layout.setSpacing(24)
        layout.setContentsMargins(24, 24, 24, 24)
        # Contenedor vertical para la lista y su título
        list_container = QVBoxLayout()
        list_container.setSpacing(10)
        list_title = QLabel("<b style='color:#9dd6ff; font-size:20px;'>Practicar las señas</b>")
        list_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        list_container.addWidget(list_title)
        self.list_widget=QListWidget(); self.list_widget.addItems(self.signs); self.list_widget.currentTextChanged.connect(self.load)
        self.signs = signs
        self.list_widget.setFixedWidth(200)
        list_container.addWidget(self.list_widget, 1)
        layout.addLayout(list_container, 1)
        self.stack=QStackedWidget(); self.stack.addWidget(QWidget()); layout.addWidget(self.stack,3)
        self.setLayout(layout)

    def load(self,sign:str):
        for i in reversed(range(self.stack.count())):
            page = self.stack.widget(i)
            self.stack.removeWidget(page)
            page.deleteLater()

        tut=TutorialWidget(sign,self.media_root,self.model_path,self.stack)
        prac=PracticeWidget(sign,self.model_path,self.signs,self.stack)
        self.stack.addWidget(tut);self.stack.addWidget(prac)
        self.stack.setCurrentWidget(tut)
