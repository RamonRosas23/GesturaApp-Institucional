# Librerías estándar de Python
import base64
from base64 import b64decode
import datetime
import os  # Para manejar rutas y nombres de archivos
import re
import sys
import time
from queue import Queue, Full
from threading import Thread, Lock, Event
from typing import Counter
from urllib.parse import parse_qs, unquote, unquote_plus, urlparse

# Librerías externas
import cv2
import leap
import cv2
import mediapipe as mp
import numpy as np
import time
from leap.enums import HandType
import mysql.connector
import numpy as np
import pandas as pd
import pywinstyles
import qtawesome as qta
import requests
import webbrowser
import stripe
from dotenv import load_dotenv
import google.generativeai as genai
import pyttsx3
from PyQt6.QtCore import QThread
from flask import redirect, session
from itsdangerous import URLSafeTimedSerializer
from requests_oauthlib import OAuth2Session
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from sklearn.preprocessing import StandardScaler
import joblib  # o pickle

# PyQt6 - Widgets
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QSlider, QTextEdit, QSpacerItem, QSizePolicy, QFrame, QProgressBar, QGridLayout,
    QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QGraphicsTextItem,
    QGraphicsProxyWidget, QCalendarWidget, QMessageBox, QCheckBox, QFileDialog,
    QStackedWidget, QComboBox, QFormLayout, QGraphicsDropShadowEffect, QDialog
)

# PyQt6 - Core
from PyQt6.QtCore import (
    Qt, QSize, pyqtSignal, QPointF, QSizeF, QTimer, QPropertyAnimation,
    QEasingCurve, QRect, QDate, QUrl, pyqtSlot, QAbstractAnimation
)

# PyQt6 - Gui
from PyQt6.QtGui import (
    QPixmap, QIcon, QTextCursor, QPen, QDoubleValidator, QImage,
    QPainter, QPainterPath, QTextCharFormat, QColor, QFont, QGuiApplication, QTransform
)

# Configuración de Stripe desde variables de entorno
stripe.api_key = os.getenv('STRIPE_SECRET_KEY', 'sk_test_x3BDHyxnpla9rr1Uy3BDoyvq')
PUBLIC_KEY = os.getenv('STRIPE_PUBLISHABLE_KEY', 'pk_test_X5cnLqXxmZKOkaJ0YqfzTFFv')
PRODUCT_ID = os.getenv('STRIPE_PRODUCT_ID', 'prod_RLX46sS2CHX6ek')
    

os.environ['QT_MEDIA_BACKEND'] = 'windows'
os.environ['QTWEBENGINE_DISABLE_SANDBOX'] = '1'

if '__file__' in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    # Si __file__ no está definido, usamos sys.argv[0]
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Agregar script_dir a sys.path para importar módulos en el mismo directorio
sys.path.insert(0, script_dir)

# Agregar el directorio padre de script_dir a sys.path para importar paquetes en la raíz del proyecto
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)


# Carpeta de recursos
carpeta_recursos = os.path.join(os.path.dirname(script_dir), 'assets')

redes_neuronales = os.path.join(os.path.dirname(script_dir), 'RedesNeuronales')

# Importaciones de módulos
from TranscriptionWorker import TranscriptionWorker
from Login.Login import PrincipalLoginWidget, CustomCalendarWidget, connect_db
from learning_module import LearningModule


# Cargar variables de entorno
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

class GeminiParaphraser:
    def __init__(self):
        # Configurar la API de Google Gemini AI Studio desde variable de entorno
        api_key = os.getenv('GOOGLE_GEMINI_API_KEY', "AIzaSyC0PtdE6W6sxVXdPFSEy9E2MaZDPUAaKLI")
        if not api_key or api_key == "your_gemini_api_key":
            raise ValueError("GOOGLE_GEMINI_API_KEY not configured. Please set it in your .env file")
        
        genai.configure(api_key=api_key)
        
        # Usar Gemini 2.5 Flash (rápido y estable)
        # IMPORTANTE: Usar el nombre completo con prefijo 'models/'
        self.model = genai.GenerativeModel('models/gemini-2.5-flash')
        print("[DEBUG] Modelo Gemini 2.5 Flash configurado correctamente")
        
    def paraphrase_signs_to_spanish(self, sign_text):
        """
        Convierte texto de señas a español natural usando Google Gemini
        """
        try:
            system_prompt = """
Eres un experto en lenguaje de señas mexicano. Tu tarea es convertir secuencias de señas detectadas en español natural y fluido.

Reglas importantes:
- Las letras separadas por espacios (como "r a m o n") forman nombres o palabras deletreadas, júntalas: "ramón"
- "MI-NOMBRE-ES" significa "Mi nombre es"
- "yo ser" significa "yo soy"  
- "Hola Maestro" puede quedarse como "Hola maestro"
- Convierte todo a español natural y gramaticalmente correcto
- Mantén el contexto y significado original
- Usa puntuación apropiada
- Responde SOLO con el texto parafraseado, sin explicaciones adicionales

Ejemplos:
Entrada: "Hola Maestro yo ser r a m o n"
Salida: "Hola maestro, yo soy Ramón."

Entrada: "MI-NOMBRE-ES j u a n"
Salida: "Mi nombre es Juan."

Ahora convierte este texto de señas a español natural:
"""
            
            # Generar respuesta con Gemini
            prompt = system_prompt + sign_text
            response = self.model.generate_content(prompt)
            
            return response.text.strip()
            
        except Exception as e:
            print(f"[DEBUG] Error en parafraseo con Gemini: {e}")
            import traceback
            traceback.print_exc()
            return sign_text  # Devolver texto original si hay error


class ParaphraseWorker(QThread):
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, text_to_paraphrase):
        super().__init__()
        self.text = text_to_paraphrase
        self.paraphraser = None
        self.should_stop = False
        
    def run(self):
        try:
            if self.should_stop:
                return
                
            if not self.text.strip():
                self.error_occurred.emit("No hay texto para parafrasear")
                return
            
            # Inicializar el parafraseador aquí para evitar problemas de threading
            self.paraphraser = GeminiParaphraser()
            
            if self.should_stop:
                return
                
            paraphrased_text = self.paraphraser.paraphrase_signs_to_spanish(self.text)
            
            if not self.should_stop:
                self.result_ready.emit(paraphrased_text)
                
        except Exception as e:
            if not self.should_stop:
                self.error_occurred.emit(f"Error: {str(e)}")
                
    def stop_processing(self):
        self.should_stop = True


class NarrationWorker(QThread):
    narration_finished = pyqtSignal()
    narration_error = pyqtSignal(str)
    
    def __init__(self, text_to_narrate):
        super().__init__()
        self.text = text_to_narrate
        self.should_stop = False
        self.engine = None
        
    def run(self):
        try:
            if self.should_stop:
                return
                
            # Inicializar el motor de TTS
            self.engine = pyttsx3.init()
            
            if self.should_stop:
                return
            
            # Configurar propiedades de voz
            voices = self.engine.getProperty('voices')
            if voices:
                # Buscar voz en español si está disponible
                spanish_voice = None
                for voice in voices:
                    if 'spanish' in voice.name.lower() or 'es' in voice.id.lower():
                        spanish_voice = voice
                        break
                if spanish_voice:
                    self.engine.setProperty('voice', spanish_voice.id)
            
            # Configurar velocidad y volumen
            self.engine.setProperty('rate', 150)  # Velocidad de habla
            self.engine.setProperty('volume', 0.9)  # Volumen (0.0 a 1.0)
            
            # Narrar el texto
            if not self.should_stop:
                self.engine.say(self.text)
                self.engine.runAndWait()
                
            if not self.should_stop:
                self.narration_finished.emit()
                
        except Exception as e:
            if not self.should_stop:
                self.narration_error.emit(f"Error en narración: {str(e)}")
        finally:
            # Limpiar el engine
            if self.engine:
                try:
                    self.engine.stop()
                except:
                    pass
            
    def stop_narration(self):
        self.should_stop = True
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass

def scale_coordinate(value, min_range, max_range, min_scale, max_scale):
    # Escalar 'value' desde el rango [min_range, max_range] al rango [min_scale, max_scale]
    return int((value - min_range) / (max_range - min_range) * (max_scale - min_scale) + min_scale)


class Canvas:
    def __init__(self, GesturaApp):
        self.gesturaApp = GesturaApp
        self.name = "Gestura - Interprete de lenguaje de señas"
        self.screen_size = [500, 700]
        self.hands_colour = (255, 255, 255)
        self.font_colour = (0, 255, 44)
        self.dots_colour = (255, 0, 0)
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)
        # Añadir rangos de coordenadas
        self.x_range = (-400, 400)  # Rango en mm para el eje x
        self.z_range = (-400, 400)  # Rango en mm para el eje z
        self.tracking_mode = None
        self.joint_data = []
        self.model = None
        self.is_signing = False
        self.movement_detected = False
        self.required_still_time = 1.3
        self.still_threshold = 5.0  # Define el umbral de quietud
        self.movement_threshold = 3.0  # Umbral para detectar movimiento brusco
        self.last_movement_time = None
        self.data_buffer = []  # Almacena los datos
        self.max_buffer_size = 50  # LÍMITE: máximo 50 frames en buffer
        self.previous_wrist_position = None  # Para una mano, ya no se usará
        self.is_not_active = True
        self.still_duration = 0
        self.start_still_time = None
        self.current_frame_id = 0
        self.recording_start_time = None
        
        # Historial y estados separados para cada mano:
        self.wrist_history_left = []
        self.wrist_history_right = []
        self.max_history_size = 10  # LÍMITE: máximo 10 posiciones en historial
        self.previous_wrist_movement_left = None
        self.previous_wrist_movement_right = None
        self.previous_wrist_still_left = None
        self.previous_wrist_still_right = None
        self.start_still_time_left = None
        self.start_still_time_right = None
        self.still_duration_left = 0
        self.still_duration_right = 0
        self.ready_for_new_sign = False
        self.ready_for_recording = False
        
        # Cargar el scaler
        try:
            scaler_path = os.path.join(redes_neuronales, 'gestura_scaler_v1.0.0_20250519_1241.skl')
            self.scaler = joblib.load(scaler_path)
            print(f"[DEBUG] Scaler cargado correctamente desde: {scaler_path}")
        except Exception as e:
            print(f"[ERROR] No se pudo cargar el scaler: {e}")
            self.scaler = None
        
        self.modelos = []  # En lugar de solo self.model
        self.load_models()  # <-- cargar aquí todas las redes


    def load_models(self):
        """Cargar los modelos para votación por mayoría."""
        from keras.models import load_model
        
        # OPTIMIZACIÓN: Solo cargar 3 modelos en lugar de 4 para mejor rendimiento
        model_paths = [
            os.path.join(redes_neuronales, "gestura_bilstm_v1.0.0_20250519_0130.keras"),
            os.path.join(redes_neuronales, "gestura_bilstm_v1.0.0_20250519_1241.keras"),
            os.path.join(redes_neuronales, "gestura_bilstm_v1.0.0_20250519_0450.keras"),
        ]

        self.modelos = []
        print("[DEBUG] Cargando modelos de predicción...")
        for path in model_paths:
            try:
                modelo = load_model(path)
                self.modelos.append(modelo)
                print(f"[DEBUG] Modelo cargado: {os.path.basename(path)}")
            except Exception as e:
                print(f"[DEBUG] Error al cargar {path}: {e}")

    def get_smoothed_position(self, current_position, side, window=1):
        """
        Calcula la posición suavizada usando un promedio móvil para la mano 'side'
        (side debe ser 'left' o 'right').
        """
        if side == 'left':
            self.wrist_history_left.append(current_position)
            # OPTIMIZACIÓN: Limitar tamaño del historial
            if len(self.wrist_history_left) > max(window, self.max_history_size):
                self.wrist_history_left.pop(0)
            smoothed = tuple(np.mean(self.wrist_history_left, axis=0))
        else:
            self.wrist_history_right.append(current_position)
            # OPTIMIZACIÓN: Limitar tamaño del historial
            if len(self.wrist_history_right) > max(window, self.max_history_size):
                self.wrist_history_right.pop(0)
            smoothed = tuple(np.mean(self.wrist_history_right, axis=0))
        return smoothed

    def detect_movement(self, current_wrist_position, side):
        smoothed = self.get_smoothed_position(current_wrist_position, side, window=1)
        if side == 'left':
            if self.previous_wrist_movement_left is None:
                self.previous_wrist_movement_left = smoothed
                print("Left: No previous movement, returning False")
                return False
            movement_distance = np.linalg.norm(np.array(smoothed) - np.array(self.previous_wrist_movement_left))
            print("Left movement_distance:", movement_distance)
            self.previous_wrist_movement_left = smoothed
            return movement_distance > self.movement_threshold
        else:
            if self.previous_wrist_movement_right is None:
                self.previous_wrist_movement_right = smoothed
                print("Right: No previous movement, returning False")
                return False
            movement_distance = np.linalg.norm(np.array(smoothed) - np.array(self.previous_wrist_movement_right))
            print("Right movement_distance:", movement_distance)
            self.previous_wrist_movement_right = smoothed
            return movement_distance > self.movement_threshold

    def get_joint_position(self, bone):
        if bone:
            x = scale_coordinate(bone.x, self.x_range[0], self.x_range[1], 0, self.screen_size[1])
            z = scale_coordinate(-bone.z, self.z_range[0], self.z_range[1], self.screen_size[0], 0)
            return x, z
        else:
            return None


  
    def is_hand_still(self, current_wrist_position, side):
        """
        Determina si la mano 'side' (left/right) está quieta usando la posición suavizada.
        """
        smoothed = self.get_smoothed_position(current_wrist_position, side, window=5)
        if side == 'left':
            if self.previous_wrist_still_left is None:
                self.previous_wrist_still_left = smoothed
                return False
            movement_distance = np.linalg.norm(np.array(smoothed) - np.array(self.previous_wrist_still_left))
            if movement_distance < self.still_threshold:
                if self.start_still_time_left is None:
                    self.start_still_time_left = time.time()
                else:
                    self.still_duration_left = time.time() - self.start_still_time_left
                    self.gesturaApp.update_progress_bar()  # Aquí puedes actualizar específicamente para la mano izquierda
                    if self.still_duration_left >= self.required_still_time:
                        return True
            else:
                if movement_distance > 3:
                    self.start_still_time_left = None
            self.previous_wrist_still_left = smoothed
            return False
        else:
            if self.previous_wrist_still_right is None:
                self.previous_wrist_still_right = smoothed
                return False
            movement_distance = np.linalg.norm(np.array(smoothed) - np.array(self.previous_wrist_still_right))
            if movement_distance < self.still_threshold:
                if self.start_still_time_right is None:
                    self.start_still_time_right = time.time()
                else:
                    self.still_duration_right = time.time() - self.start_still_time_right
                    self.gesturaApp.update_progress_bar()  # Actualiza para la mano derecha si lo deseas
                    if self.still_duration_right >= self.required_still_time:
                        return True
            else:
                if movement_distance > 3:
                    self.start_still_time_right = None
            self.previous_wrist_still_right = smoothed
            return False

    def pad_sequence(self, seq, max_length=20):
            n_frames, n_features = seq.shape
            if n_frames < max_length:
                pad = np.zeros((max_length - n_frames, n_features))
                seq_padded = np.vstack([seq, pad])
            else:
                seq_padded = seq[:max_length]
            return seq_padded

    def send_data_to_model(self, num_evaluaciones=1):
        """Enviar los datos acumulados a 5 modelos LSTM y usar votación por mayoría."""
        if len(self.data_buffer) == 0 or len(self.modelos) == 0:
            print("[DEBUG] No hay datos o modelos cargados para hacer predicción.")
            return

        try:
            self.is_not_active = True
            self.gesturaApp.toggle_prediccion_icon()

            # 1. Convertir el buffer en DataFrame
            df = self.save_data_to_dataframe(self.data_buffer)
            data_array = df.iloc[:, 2:].to_numpy().astype(np.float32)

            # 2. Normalizar por muestra (como en la versión secundaria)
            mean = np.mean(data_array, axis=0)
            std = np.std(data_array, axis=0)
            std = np.where(std == 0, 1e-8, std)  # Evitar división por cero
            data_array = (data_array - mean) / std

            # 3. Diferencias entre frames
            data_array = np.diff(data_array, axis=0, prepend=data_array[0:1])

            # 4. Padding a 20 frames
            data_array = self.pad_sequence(data_array, max_length=20)

            # 5. Expandir dimensión para batch
            data_array = np.expand_dims(data_array, axis=0)  # (1, 20, features)

            # 6. Predecir con todos los modelos
            from collections import Counter
            predicciones = []

            for idx, modelo in enumerate(self.modelos):
                salida = modelo.predict(data_array, verbose=0)  # (1, clases)
                clase = np.argmax(salida)
                predicciones.append(clase)
                print(f"[DEBUG] Modelo {idx+1} predijo: {clase}")

            # 7. Votación por mayoría
            voto_final = Counter(predicciones).most_common(1)[0][0]

            # 8. Mapeo de clases (26 gestos completos)
            gesture_mapping = {
                0: "A", 1: "Bien", 2: "Buenos_dias", 3: "Cuando", 4: "Cuanto", 
                5: "Dinero", 6: "E", 7: "Gracias", 8: "HOLA", 9: "I", 10: "J", 
                11: "Lo_Repites_por_favor", 12: "MiNombreEs", 13: "No", 
                14: "No_entiendo", 15: "Paciencia", 16: "Pagar", 17: "Para_que", 
                18: "Por_Favor", 19: "Que", 20: "Revisar", 21: "Servicio", 
                22: "Si", 23: "Tengo_una_duda", 24: "V", 25: "Yo"
            }
            seña_predicha = gesture_mapping.get(voto_final, "?")

            # 9. Mostrar predicción
            print(f"[DEBUG] Votación por mayoría: {seña_predicha}")
            print(f"[DEBUG] Votos individuales: {Counter(predicciones)}")
            self.gesturaApp.update_predictions(seña_predicha)

        except Exception as e:
            print(f"[DEBUG] Error en la predicción ensemble: {e}")
            import traceback
            traceback.print_exc()

        # 10. Reset de estado
        self.data_buffer.clear()
        self.movement_detected = False
        self.previous_wrist_movement_left = None
        self.previous_wrist_movement_right = None
        self.previous_wrist_still_left = None
        self.previous_wrist_still_right = None
        self.start_still_time_left = None
        self.start_still_time_right = None
        self.ready_for_new_sign = False




    def save_data_to_dataframe(self, frames):
        """
        Convierte la lista de frames (cada uno con datos de ambas manos) en un DataFrame,
        con la siguiente estructura:
        - Frame_ID, Timestamp, Left_Hand_Present, Right_Hand_Present
        - Para la mano izquierda: Left_Elbow_x, Left_Elbow_y, Left_Elbow_z, y los 21 keypoints
            (Left_0_x, Left_0_y, Left_0_z, …, Left_20_x, Left_20_y, Left_20_z)
        - Para la mano derecha: Similar.
        Si una mano no está presente, sus columnas se llenan con NaN.
        """
        import numpy as np  # Si no está ya importado
        rows = []
        for frame in frames:
            row = {
                'Frame_ID': frame['Frame_ID'],
                'Timestamp': frame['Timestamp'],
                'Left_Hand_Present': frame['Left_Hand_Present'],
                'Right_Hand_Present': frame['Right_Hand_Present']
            }
            # Datos de la mano izquierda
            if frame['Left_Hand_Present'] and frame['left'] is not None:
                left = frame['left']
                if left['elbow'] is not None:
                    row['Left_Elbow_x'] = left['elbow'][0]
                    row['Left_Elbow_y'] = left['elbow'][1]
                else:
                    row['Left_Elbow_x'] = row['Left_Elbow_y'] = 0
                for i, kp in enumerate(left['keypoints']):
                    row[f'Left_{i}_x'] = kp[0]
                    row[f'Left_{i}_y'] = kp[1]
            else:
                row['Left_Elbow_x'] = row['Left_Elbow_y'] = 0
                for i in range(21):
                    row[f'Left_{i}_x'] = 0
                    row[f'Left_{i}_y'] = 0

            # Datos de la mano derecha
            if frame['Right_Hand_Present'] and frame['right'] is not None:
                right = frame['right']
                if right['elbow'] is not None:
                    row['Right_Elbow_x'] = right['elbow'][0]
                    row['Right_Elbow_y'] = right['elbow'][1]
                else:
                    row['Right_Elbow_x'] = row['Right_Elbow_y'] = 0
                for i, kp in enumerate(right['keypoints']):
                    row[f'Right_{i}_x'] = kp[0]
                    row[f'Right_{i}_y'] = kp[1]
            else:
                row['Right_Elbow_x'] = row['Right_Elbow_y'] = 0
                for i in range(21):
                    row[f'Right_{i}_x'] = 0
                    row[f'Right_{i}_y'] = 0

            rows.append(row)
        import pandas as pd
        df = pd.DataFrame(rows)

        # Guardar el DataFrame en un archivo CSV
        tipo_gesto = 'Maestro'  # Cambiar por el tipo de gesto actual
        output_folder = 'Gestos'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        subfolder = os.path.join(output_folder, tipo_gesto)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        existing_files = len([name for name in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, name))])
        csv_filename = f"Gesto_{tipo_gesto}_{existing_files + 1}.csv"
        csv_path = os.path.join(subfolder, csv_filename)
        df.to_csv(csv_path, index=False)
        return df
    
    # Nuevos métodos en la clase Canvas para capturar datos de ambas manos:
    def capture_hand_data(self, hand):
        hand_data = {}
        # Obtener la posición del codo usando la información de PseudoArm
        elbow = self.get_joint_position(hand.arm.next_joint)
        hand_data['elbow'] = elbow
        # Capturar los 21 keypoints a partir de los landmarks
        keypoints = []
        for lm in hand.landmarks:
            joint = PseudoJoint.from_hand_landmark(lm)
            pos = self.get_joint_position(joint)
            keypoints.append(pos)
        hand_data['keypoints'] = keypoints
        # Agregar el lado (left/right)
        hand_data['side'] = hand.type.lower()
        return hand_data

    def process_frame_movement(self, frame_data):
        """
        Procesa el movimiento en el frame completo.
        frame_data es un diccionario con las siguientes claves:
        'Frame_ID', 'Timestamp', 'Left_Hand_Present', 'Right_Hand_Present',
        'left' (datos de la mano izquierda) y 'right' (datos de la mano derecha).
        
        Se evalúa el movimiento (usando el keypoint 0, la muñeca) de cada mano.
        Si alguna mano se mueve bruscamente (movimiento_distance > threshold),
        se activa ready_for_recording y se acumula el frame en data_buffer.
        Cuando la mano (o manos) se estabilicen por el tiempo requerido, se finaliza la grabación.
        """
        # Inicializamos flags locales para cada mano
        left_trigger = False
        right_trigger = False

        # Verificar la mano izquierda
        if frame_data['Left_Hand_Present'] and frame_data['left'] is not None:
            left_wrist = frame_data['left']['keypoints'][0]  # Asumimos que el índice 0 es la muñeca
            # Detectamos movimiento para la mano izquierda:
            if self.detect_movement(left_wrist, 'left'):
                left_trigger = True
                print(f"[DEBUG] Movimiento brusco detectado para left en Frame_ID {frame_data['Frame_ID']}")
                if not self.ready_for_recording:
                    frame_data['Frame_ID'] = 1
            
        # Verificar la mano derecha
        if frame_data['Right_Hand_Present'] and frame_data['right'] is not None:
            right_wrist = frame_data['right']['keypoints'][0]
            if self.detect_movement(right_wrist, 'right'):
                right_trigger = True
                print(f"[DEBUG] Movimiento brusco detectado para right en Frame_ID {frame_data['Frame_ID']}")
                if not self.ready_for_recording:
                    frame_data['Frame_ID'] = 1
                
        # Si se detecta movimiento brusco en alguna mano, activar la grabación
        if left_trigger or right_trigger:
            
            if not self.ready_for_recording:
                self.current_frame_id = 1
                self.recording_start_time = time.time()
            self.ready_for_recording = True
            print(f"[DEBUG] ready_for_recording ACTIVADO en Frame_ID {frame_data['Frame_ID']}")
        
        # Si estamos en modo grabación, acumulamos el frame
        if getattr(self, 'ready_for_recording', False):
            self.data_buffer.append(frame_data)
            # OPTIMIZACIÓN: Limitar el tamaño del buffer para evitar consumo excesivo de memoria
            if len(self.data_buffer) > self.max_buffer_size:
                self.data_buffer.pop(0)  # Eliminar el frame más antiguo

    
    def capture_frame_data(self, hands):
        """
        Captura los datos de todas las manos detectadas en el frame.
        Retorna un diccionario con:
        - 'Frame_ID': ID del frame
        - 'Timestamp': Marca temporal en formato ISO
        - 'Left_Hand_Present': Booleano que indica si se detectó la mano izquierda
        - 'Right_Hand_Present': Booleano que indica si se detectó la mano derecha
        - 'left': Diccionario con los datos de la mano izquierda (o None)
        - 'right': Diccionario con los datos de la mano derecha (o None)
        """
        timestamp = 0.0
        
        if self.ready_for_recording and self.recording_start_time is not None:
            now = time.time()
            timestamp = now - self.recording_start_time  # Delta en segundos
        
        frame_dict = {
            'Frame_ID': self.current_frame_id if hasattr(self, 'current_frame_id') else 0,
            'Timestamp': timestamp,
            'Left_Hand_Present': False,
            'Right_Hand_Present': False,
            'left': None,
            'right': None
        }
        
        for hand in hands:
            hd = self.capture_hand_data(hand)
            if hand.type.lower() == "left":
                frame_dict['Left_Hand_Present'] = True
                frame_dict['left'] = hd
            else:
                frame_dict['Right_Hand_Present'] = True
                frame_dict['right'] = hd
        return frame_dict
    
    def check_still_and_finalize(self, frame_data):
        still_left = False
        still_right = False

        if frame_data['Left_Hand_Present'] and frame_data['left'] is not None:
            left_wrist = frame_data['left']['keypoints'][0]
            still_left = self.is_hand_still(left_wrist, 'left')
            print(f"[DEBUG] still_duration_left: {self.still_duration_left}")
        if frame_data['Right_Hand_Present'] and frame_data['right'] is not None:
            right_wrist = frame_data['right']['keypoints'][0]
            still_right = self.is_hand_still(right_wrist, 'right')
            print(f"[DEBUG] still_duration_right: {self.still_duration_right}")

        if (still_left or not frame_data['Left_Hand_Present']) and (still_right or not frame_data['Right_Hand_Present']):
            if getattr(self, 'ready_for_recording', False) and len(self.data_buffer) > 0:
                print(f"[DEBUG] Finalizando grabación en Frame_ID {frame_data['Frame_ID']}")
                self.send_data_to_model()
                self.ready_for_recording = False
                self.recording_start_time = None

    def get_current_still_duration(self):
        durations = []
        if self.previous_wrist_still_left is not None:
            durations.append(self.still_duration_left)
        if self.previous_wrist_still_right is not None:
            durations.append(self.still_duration_right)
        if durations:
            return min(durations)
        return 0


class PseudoJoint:
    def __init__(self, x, z):
        self.x = x  # Valor en "mm" dentro del rango [-400, 400]
        self.z = z

    @classmethod
    def from_hand_landmark(cls, landmark):
        # Convertir la coordenada normalizada a un rango de [-400, 400]
        x = (landmark.x - 0.5) * 800  
        # Escalar el valor de z (puedes ajustar el factor según tus pruebas)
        z = landmark.z * 800  
        return cls(x, z)

    @classmethod
    def from_pose(cls, coord, frame_width=1080):
        # coord es una tupla (cx, cy) obtenida de MediaPipe Pose
        # Convertimos la coordenada x de píxeles a un rango [-400, 400]
        x = (coord[0] / frame_width) * 800 - 400
        # Como no disponemos de profundidad, se asigna 0
        z = 0
        return cls(x, z)

class PseudoBone:
    def __init__(self, prev_joint, next_joint):
        self.prev_joint = prev_joint  # Instancia de PseudoJoint
        self.next_joint = next_joint

class PseudoDigit:
    def __init__(self, finger_id, bones):
        self.finger_id = finger_id  # Nombre: "Thumb", "Index", etc.
        self.bones = bones  # Lista de PseudoBone

class PseudoArm:
    def __init__(self, hand_landmarks, pose_landmarks, hand_type, frame_width):
        # Usar los landmarks de la pose para asignar los puntos de la muñeca y el codo.
        if hand_type.lower() == "left":
            if "left_wrist" in pose_landmarks:
                self.prev_joint = PseudoJoint.from_pose(pose_landmarks["left_wrist"], frame_width)
            else:
                self.prev_joint = PseudoJoint.from_hand_landmark(hand_landmarks[0])
            if "left_elbow" in pose_landmarks:
                self.next_joint = PseudoJoint.from_pose(pose_landmarks["left_elbow"], frame_width)
            else:
                self.next_joint = PseudoJoint.from_hand_landmark(hand_landmarks[0])
        else:
            if "right_wrist" in pose_landmarks:
                self.prev_joint = PseudoJoint.from_pose(pose_landmarks["right_wrist"], frame_width)
            else:
                self.prev_joint = PseudoJoint.from_hand_landmark(hand_landmarks[0])
            if "right_elbow" in pose_landmarks:
                self.next_joint = PseudoJoint.from_pose(pose_landmarks["right_elbow"], frame_width)
            else:
                self.next_joint = PseudoJoint.from_hand_landmark(hand_landmarks[0])

class PseudoHand:
    def __init__(self, hand_landmarks, handedness, pose_landmarks, hand_id, frame_width):
        self.id = hand_id
        # handedness es un string ("Left" o "Right")
        self.type = handedness  
        # Guardamos los landmarks completos para acceder a los 21 keypoints
        self.landmarks = hand_landmarks  
        self.arm = PseudoArm(hand_landmarks, pose_landmarks, self.type, frame_width)
        self.digits = []
        self._build_digits(hand_landmarks)


    def _build_digits(self, landmarks):
        # Mapeo de índices para cada dedo (incluye el punto de la muñeca en posición 0)
        mapping = {
            "Thumb": [0, 1, 2, 3, 4],
            "Index": [0, 5, 6, 7, 8],
            "Middle": [0, 9, 10, 11, 12],
            "Ring": [0, 13, 14, 15, 16],
            "Pinky": [0, 17, 18, 19, 20]
        }
        for finger_name, indices in mapping.items():
            bones = []
            # Cada dedo tendrá 4 segmentos (huesos)
            for i in range(len(indices) - 1):
                joint_start = PseudoJoint.from_hand_landmark(landmarks[indices[i]])
                joint_end = PseudoJoint.from_hand_landmark(landmarks[indices[i+1]])
                bones.append(PseudoBone(joint_start, joint_end))
            self.digits.append(PseudoDigit(finger_name, bones))


class MediaPipeListener:
    def __init__(self, canvas, image_queue, lock, data_full, app, video_source=0):
        self.canvas = canvas
        self.image_queue = image_queue
        self.lock = lock
        self.data_full = data_full
        self.app = app
        self.stop_flag = False
        self.fps_limit = 30  # Reducido de 60 a 30 FPS para mejor rendimiento
        self.previous_time = time.time()

        # Inicializar MediaPipe Hands y Pose con configuración optimizada
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=0.5,  # Aumentado para reducir falsos positivos
            min_tracking_confidence=0.5,   # Aumentado para mejor estabilidad
            max_num_hands=2,
            model_complexity=0  # Reducido a 0 para mejor rendimiento
        )
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,  # Reducido de 0.9 para mejor balance
            min_tracking_confidence=0.7,   # Reducido de 0.9 para mejor balance
            model_complexity=1  # Reducido de 2 a 1 para mejor rendimiento
        )

        # Abrir la fuente de video (puede ser un índice o una URL)
        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reducido de 1080 para mejor rendimiento
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Reducido de 920 para mejor rendimiento
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Limitar FPS de la cámara
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.hand_id_counter = 0  # Para asignar IDs únicos a cada mano

        # Tiempo máximo sin recibir un frame (en segundos)
        self.timeout_seconds = 3
        self.no_frame_start = time.time()
        
        print(f"[DEBUG] MediaPipeListener iniciado - Resolución: {self.frame_width}x{self.frame_height}, FPS: 30")

    def run(self):
        while not self.stop_flag:
            current_time = time.time()
            # Limitar FPS
            if current_time - self.previous_time < 1.0 / self.fps_limit:
                time.sleep(0.01)  # Pequeña pausa para no consumir CPU innecesariamente
                continue
            self.previous_time = current_time

            ret, frame = self.cap.read()
            if not ret:
                # Si no se pudo leer el frame, se verifica el timeout
                if time.time() - self.no_frame_start > self.timeout_seconds:
                    # Marcar cámara como desconectada
                    self.app.set_camera_status(False)
                    break
                else:
                    continue  # Intenta leer frame nuevamente
            else:
                # Reiniciamos el timer de no_frame
                self.no_frame_start = time.time()
                # Si se recibe un frame y la cámara no está marcada como online, la marcamos
                if not self.app.camera_on:
                    self.app.set_camera_status(True)

            # Procesamiento de imagen: flip y conversión a RGB
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesar el frame con MediaPipe Hands y Pose
            hand_results = self.hands.process(frame_rgb)
            pose_results = self.pose.process(frame_rgb)

            # Crear una imagen en negro para dibujar (opcional)
            skeleton_frame = np.zeros_like(frame)

            # Procesar landmarks de la pose (por ejemplo, muñecas y codos)
            pose_landmarks = {}
            if pose_results.pose_landmarks:
                for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    if idx in [13, 14, 15, 16]:
                        h, w, _ = skeleton_frame.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        if idx in [15, 16]:
                            key = "left_wrist" if idx == 15 else "right_wrist"
                            pose_landmarks[key] = (cx, cy)
                        elif idx in [13, 14]:
                            key = "left_elbow" if idx == 13 else "right_elbow"
                            pose_landmarks[key] = (cx, cy)

            # Construir pseudo manos (usando tu clase PseudoHand, que ya tienes definida)
            pseudo_hands = []
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    pseudo_hand = PseudoHand(
                        hand_landmarks.landmark,
                        handedness.classification[0].label,
                        pose_landmarks,
                        self.hand_id_counter,
                        self.frame_width
                    )
                    self.hand_id_counter += 1
                    pseudo_hands.append(pseudo_hand)
                    # Dibujar conexiones (opcional)
                    for connection in self.mp_hands.HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        h, w, _ = skeleton_frame.shape
                        start_landmark = hand_landmarks.landmark[start_idx]
                        end_landmark = hand_landmarks.landmark[end_idx]
                        start_pos = (int(start_landmark.x * w), int(start_landmark.y * h))
                        end_pos = (int(end_landmark.x * w), int(end_landmark.y * h))
                        cv2.line(skeleton_frame, start_pos, end_pos, (255, 255, 255), 8)
                        cv2.circle(skeleton_frame, start_pos, 4, (255, 0, 0), 10)
                        cv2.circle(skeleton_frame, end_pos, 4, (255, 0, 0), 10)

            combined_frame = cv2.addWeighted(frame, 1, skeleton_frame, 1, 0)
            #combined_frame = skeleton_frame

            # Creamos un objeto "evento" para simular la estructura de Ultraleap
            pseudo_event = type("PseudoEvent", (), {})()
            pseudo_event.hands = pseudo_hands

            # Limpiar datos anteriores y procesar movimiento
            self.canvas.joint_data.clear()
            if len(pseudo_event.hands) > 0:
                self.lost_hands_start_time = None
                frame_data = self.canvas.capture_frame_data(pseudo_event.hands)
                #print("Frame Data:", frame_data)
                self.canvas.process_frame_movement(frame_data)
                self.canvas.check_still_and_finalize(frame_data)
            else:
                # Si no hay manos detectadas y se estaba grabando, iniciamos el contador.
                if self.canvas.ready_for_recording:
                    if not hasattr(self, 'lost_hands_start_time') or self.lost_hands_start_time is None:
                        self.lost_hands_start_time = time.time()
                    elif time.time() - self.lost_hands_start_time > 5.0:  # umbral de 1 segundo
                        print("[DEBUG] Manos perdidas por más de 5 segundos. Reiniciando grabación.")
                        # Reiniciamos el estado de grabación:
                        self.canvas.ready_for_recording = False
                        self.canvas.recording_start_time = None
                        
                        self.canvas.movement_detected = False
                        self.canvas.previous_wrist_movement_left = None
                        self.canvas.previous_wrist_movement_right = None
                        self.canvas.previous_wrist_still_left = None
                        self.canvas.previous_wrist_still_right = None
                        self.canvas.start_still_time_left = None
                        self.canvas.start_still_time_right = None
                        self.canvas.ready_for_new_sign = False
        
                        self.canvas.current_frame_id = 0
                        self.canvas.data_buffer.clear()
                self.canvas.output_image[:, :] = 0
            
            # Al final de cada iteración, incrementamos el frame_id
            if self.canvas.ready_for_recording and len(pseudo_event.hands) > 0:
                self.canvas.current_frame_id += 1


            with self.lock:
                try:
                    self.image_queue.put(combined_frame.copy(), block=False)
                except Exception:
                    pass

        # Al salir del bucle, liberar recursos
        print("[DEBUG] MediaPipeListener finalizando...")
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print("[DEBUG] VideoCapture liberado")
        if self.hands:
            self.hands.close()
            print("[DEBUG] MediaPipe Hands cerrado")
        if self.pose:
            self.pose.close()
            print("[DEBUG] MediaPipe Pose cerrado")

    def stop(self):
        print("[DEBUG] Deteniendo MediaPipeListener...")
        self.stop_flag = True
        # Forzamos la liberación del VideoCapture
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print("[DEBUG] VideoCapture liberado en stop()")


class GesturaApp(QWidget):
    update_progress_signal = pyqtSignal(int)  # Señal para actualizar la barra de progreso
    update_text_signal = pyqtSignal(str)  # Señal para actualizar el texto
    
    def __init__(self):
        super().__init__()
        
        self.user_data = None
        self.stop_threads = False  # Inicializar la bandera de hilos detenidos
        self.listener = None  # Añadir atributo para el listener
        self.connection = None  # Añadir atributo para la conexión
        self.partial_text = ""
        
        self.update_progress_signal.connect(self.update_progress_bar_from_signal)
        self.update_text_signal.connect(self._update_predictions)
        
        # Variables para narración
        self.is_narrating = False
        self.paraphrase_worker = None
        self.narration_worker = None
        
        # Variables para control de hilos y recursos
        self.capture_thread = None
        self.timer = None
        self.learning_module = None
        self.current_page_index = -1
        
        # Conectar señal de cierre de aplicación para limpiar hilos
        QApplication.instance().aboutToQuit.connect(self.cleanup_threads)
        
        self.setWindowTitle(" ")
        self.setStyleSheet("background-color: #111c22; font-family: 'Manrope', 'Noto Sans', sans-serif;")
        self.transcription_worker = None  # Variable para el trabajador de transcripción
        self.is_transcribing = False  # Bandera para saber si está en proceso de transcripción
        
        self.initUI()
    
    def initUI(self):
        self.video_container = None  # Inicialización del atributo
        layout = QVBoxLayout()
        
        # Crear un QStackedWidget para alternar entre las pantallas de login y la aplicación
        self.stack = QStackedWidget(self)

        # Crear las pantallas de login y registro en dos fases
        self.login_widget = PrincipalLoginWidget(self)
        self.login_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Añadir las pantallas al stack
        self.stack.addWidget(self.login_widget)

        # Añadir el header y el stack al layout principal
        layout.addWidget(self.stack)
        self.setLayout(layout)
        
    class AnimatedStackedWidget(QGraphicsView):
        def __init__(self):
            super().__init__()
            self.scene = QGraphicsScene()
            self.setScene(self.scene)
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.setFrameStyle(0)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.current_widget = None
            self.animation = None
            
        def wheelEvent(self, event):
            """Bloquear el desplazamiento con la rueda del ratón"""
            event.ignore()  # Ignorar el evento para que no provoque desplazamiento

        def resizeEvent(self, event):
            """Redimensionar todos los widgets dentro del QGraphicsView al cambiar tamaño"""
            super().resizeEvent(event)
            
            # Reajustar el tamaño de la escena para que coincida con el nuevo tamaño de la vista
            self.scene.setSceneRect(0, 0, self.viewport().width(), self.viewport().height())

            # Reajustar la posición y tamaño de todos los widgets en la escena
            for item in self.scene.items():
                if isinstance(item, QGraphicsProxyWidget):
                    widget = item.widget()
                    # Ajustar el tamaño del widget al nuevo tamaño de la vista
                    widget.setFixedSize(self.viewport().size())
                    item.setMinimumSize(QSizeF(self.viewport().size()))
                    item.setMaximumSize(QSizeF(self.viewport().size()))
                    
                    # Reposicionar el widget para que quede centrado en la escena
                    if item == self.current_widget:
                        item.setPos(0, 0)  # Mantén el widget actual en la posición (0, 0)
                    else:
                        item.setPos(0, self.viewport().height())  # Otros widgets fuera de la vista



        def add_widget(self, widget):
            """Añade un QWidget a la escena como un QGraphicsProxyWidget"""
            proxy = QGraphicsProxyWidget()
            proxy.setWidget(widget)

            # Configurar el tamaño del widget para que ocupe todo el espacio
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            widget.setFixedSize(self.viewport().size())
            widget.setStyleSheet("background-color: #111c22; border: none;")

            proxy.setMinimumSize(QSizeF(self.viewport().size()))
            proxy.setMaximumSize(QSizeF(self.viewport().size()))

            self.scene.addItem(proxy)

            if not self.current_widget:
                self.current_widget = proxy
                proxy.setPos(0, 0)  # Deja el primer widget en su lugar
            else:
                proxy.setPos(0, self.height())  # Mueve el siguiente widget fuera de la vista
                proxy.hide()

            return proxy

        def slide_to_widget(self, target_widget):
            """Anima la transición hacia el widget objetivo siempre hacia abajo"""
            if self.current_widget == target_widget:
                return  # No animar si ya estamos en el widget objetivo

            if self.animation and self.animation.state() == QAbstractAnimation.State.Running:
                return  # Evitar colisiones de animaciones

            # Configurar la posición inicial del nuevo widget
            target_widget.setPos(0, -self.height())  # Siempre inicia arriba
            target_widget.setZValue(1)  # Asegurar que esté por encima
            if self.current_widget:
                self.current_widget.setZValue(0)  # Actual widget por debajo

            target_widget.show()

            # Animar la entrada del nuevo widget
            entry_animation = QPropertyAnimation(target_widget, b"pos")
            entry_animation.setDuration(500)
            entry_animation.setStartValue(QPointF(0, -self.height()))
            entry_animation.setEndValue(QPointF(0, 0))
            entry_animation.setEasingCurve(QEasingCurve.Type.OutQuad)

            # Ocultar el widget anterior al finalizar la animación del nuevo
            def finalize():
                if self.current_widget:
                    self.current_widget.hide()
                self.current_widget = target_widget

            # Conectar la animación al finalizar
            entry_animation.finished.connect(finalize)
            self.animation = entry_animation
            entry_animation.start()
            
    class ClickableLabel(QLabel):
        clicked = pyqtSignal()  # Señal personalizada para clics

        def mousePressEvent(self, event):
            self.clicked.emit()  # Emitir señal cuando se detecta clic
            super().mousePressEvent(event)
            
    # Método para cambiar la página del QStackedWidget
    def change_page(self, index):
        """Método para cambiar la página con animaciones"""
        # Si estamos cambiando de página, limpiar recursos de la página anterior
        if self.current_page_index != index:
            self._cleanup_current_page()
            self.current_page_index = index
            
            # OPTIMIZACIÓN: Reiniciar recursos de la nueva página
            self._start_page_resources(index)
        
        target_widget = [self.proxy1, self.proxy2, self.proxy3, self.proxy4, self.proxy5][index]
        self.stack_widget.slide_to_widget(target_widget)
        
        for i, link_text in enumerate(self.nav_items):
            button = self.nav_buttons[link_text]
            line = self.nav_lines[link_text]
            animation = self.nav_animations[link_text]
            if i == index:
                # Animar el ancho de la línea
                animation.stop()
                animation.setStartValue(line.maximumWidth())
                animation.setEndValue(button.width() + 20)
                animation.start()

                # Cambiar color del texto del botón
                button.setStyleSheet("""
                    QPushButton {
                        color: #9dd6ff; /* Color resaltado */
                        font-size: 14px; 
                        font-weight: bold;
                        background-color: transparent;
                    }
                    QPushButton:hover {
                        text-decoration: underline;
                    }
                """)
            else:
                # Ocultar línea y restaurar estilo del botón
                animation.stop()
                animation.setStartValue(line.maximumWidth())
                animation.setEndValue(0)
                animation.start()
                button.setStyleSheet("""
                    QPushButton {
                        color: white; 
                        font-size: 14px; 
                        background-color: transparent; 
                        font-weight: 500;
                    }
                    QPushButton:hover {
                        text-decoration: underline;
                    }
                """)
            
    def header_gestura(self):
        # Header: logo, título y elementos de navegación
        self.header_frame = QFrame()
        self.header_layout = QVBoxLayout()  # Cambiar a QVBoxLayout para apilar header, línea y contenido
        self.header_layout.setContentsMargins(16, 19, 10, 0)
        self.header_layout.setSpacing(5)  # Ajustar el espaciado para mantener proximidad

        # Crear layout horizontal para el contenido del header
        header_content_layout = QHBoxLayout()
        header_content_layout.setContentsMargins(0, 0, 0, 0)
        header_content_layout.setSpacing(25)

        # Logo y título en un layout horizontal
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
                margin: 0px;
                padding: 0px;
            }
        """)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        # Centrar la vista en la escena
        view.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        # Añadir la vista al layout principal
        header_widgetL = QVBoxLayout()
        header_widgetL.addWidget(view, alignment=Qt.AlignmentFlag.AlignLeft)
        header_widgetL.setContentsMargins(20, 10, 0, 0)
        header_content_layout.addLayout(header_widgetL)

        # Spacer para empujar los widgets hacia la derecha
        header_content_layout.addStretch()

        # Enlaces de navegación
        self.nav_buttons = {}
        self.nav_lines = {}
        self.nav_animations = {}
        self.nav_items = ["Traducción", "Aprendizaje", "Configuración"]

        for index, link_text in enumerate(self.nav_items):
            # Crear un contenedor para el botón y la línea
            button_container = QVBoxLayout()
            button_container.setSpacing(0)
            button_container.setContentsMargins(0, 0, 0, 0)

            # Crear el botón
            nav_link = QPushButton(link_text)
            nav_link.setStyleSheet("""
                QPushButton {
                    color: white; 
                    font-size: 14px; 
                    background-color: transparent; 
                    font-weight: 500;
                }
                QPushButton:hover {
                    text-decoration: underline;
                }
            """)
            nav_link.setCursor(Qt.CursorShape.PointingHandCursor)
            nav_link.setFlat(True)
            self.nav_buttons[link_text] = nav_link

            # Crear la línea inferior
            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setFrameShadow(QFrame.Shadow.Sunken)
            line.setStyleSheet("background-color: #9dd6ff;")
            line.setFixedHeight(2)
            line.setFixedWidth(0)  # Ancho inicial 0 para animación
            self.nav_lines[link_text] = line

            # Crear animación para la línea
            animation = QPropertyAnimation(line, b"maximumWidth")
            animation.setDuration(300)
            animation.setEasingCurve(QEasingCurve.Type.OutQuad)
            self.nav_animations[link_text] = animation

            # Añadir el botón y la línea al contenedor
            button_container.addWidget(nav_link, alignment=Qt.AlignmentFlag.AlignCenter)
            button_container.addWidget(line)

            # Añadir el contenedor al layout principal
            header_content_layout.addLayout(button_container)

            # Conectar el botón a la función para cambiar la página del QStackedWidget
            nav_link.clicked.connect(lambda checked, idx=index: self.change_page(idx))

        # Botón "Cerrar sesión"
        logout_button = QPushButton("Cerrar sesión")
        logout_button.setStyleSheet("""
            QPushButton {
                color: white; 
                font-size: 14px; 
                background-color: transparent; 
                font-weight: 500;
            }
            QPushButton:hover {
                text-decoration: underline;
            }
        """)
        logout_button.setCursor(Qt.CursorShape.PointingHandCursor)
        logout_button.setFlat(True)
        header_content_layout.addWidget(logout_button)
        logout_button.clicked.connect(self.logout)  # Conectar a la función logout

        # Botón de activar
        self.activate_button = QPushButton("ACTIVAR")
        self.activate_button.setStyleSheet("""
            QPushButton {
                height: 40px;
                padding-left: 20px;
                padding-right: 20px;
                background-color: #1E3A5F;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2A496E;
            }
            QPushButton:pressed {
                background-color: #122640;
            }
        """)
        # Conectar el botón a la función para cambiar la página del QStackedWidget
        self.activate_button.clicked.connect(lambda: self.change_page(4))
        header_content_layout.addWidget(self.activate_button)

        # Imagen de perfil
        self.profile_pic_label = self.ClickableLabel()
        self.profile_pixmap = self.get_rounded_pixmap(
            self.user_data["avatar"], 60, 60, 10, frame_color=QColor("#2a496e"), frame_thickness=5
        )
        self.profile_pic_label.setPixmap(self.profile_pixmap)
        header_content_layout.addWidget(self.profile_pic_label)

        # Conectar el clic de la imagen a la función para cambiar la página
        self.profile_pic_label.clicked.connect(lambda: self.change_page(3))

        # Añadir el contenido del header al layout principal
        self.header_layout.addLayout(header_content_layout)

        # Línea horizontal debajo del header
        line1 = QFrame()
        line1.setFrameShape(QFrame.Shape.HLine)
        line1.setFrameShadow(QFrame.Shadow.Sunken)
        line1.setStyleSheet("background-color: #18232b; height: 2px;")
        self.header_layout.addWidget(line1)

        # QStackedWidget para cambiar las vistas
        self.stack_widget = self.AnimatedStackedWidget()
        self.header_layout.addWidget(self.stack_widget)

        self.proxy1 = self.stack_widget.add_widget(self.translation_widget())
        self.proxy2 = self.stack_widget.add_widget(self.learning_widget())
        self.proxy3 = self.stack_widget.add_widget(self.settings_widget())
        self.proxy4 = self.stack_widget.add_widget(self.perfile_widget())
        self.proxy5 = self.stack_widget.add_widget(self.suscripcion_widget())
        
        # Resaltar el botón inicial ("Traducción") 
        # NOTA: No llamar _start_page_resources aquí porque se hará en cargar_gestura_aplicacion
        self.current_page_index = 0  # Establecer índice inicial sin activar recursos
        target_widget = self.proxy1
        self.stack_widget.slide_to_widget(target_widget)
        
        # Actualizar UI de navegación
        for i, link_text in enumerate(self.nav_items):
            if i == 0:
                self.nav_buttons[link_text].setStyleSheet("""
                    QPushButton {
                        color: #9dd6ff;
                        font-size: 16px;
                        background-color: transparent;
                        font-weight: bold;
                        text-decoration: underline;
                    }
                """)
                self.nav_animations[link_text].setEndValue(120)
                self.nav_animations[link_text].start()
            else:
                self.nav_buttons[link_text].setStyleSheet("""
                    QPushButton {
                        color: white;
                        font-size: 16px;
                        background-color: transparent;
                        font-weight: 500;
                    }
                """)
                self.nav_animations[link_text].setEndValue(0)
                self.nav_animations[link_text].start()

        # Asignar el layout principal al gestura_widget
        self.gestura_widget.setLayout(self.header_layout)

    def learning_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(20, 40, 20, 20)
        layout.setSpacing(20)

        # Contenedor principal con estilo moderno
        main_container = QFrame()
        main_container.setStyleSheet("""
        QFrame {
            background-color: #1E293B;
            border-radius: 20px;
            padding: 0px;
            border: 1px solid #2A496E;
            margin-top: -10px;
        }
    """)
        main_layout = QVBoxLayout(main_container)
        main_layout.setSpacing(20)

        # Título del módulo con estilo mejorado
        title_label = QLabel("Módulo de Aprendizaje")
        title_label.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 28px;
                font-weight: bold;
                padding: 10px;
                border-bottom: 2px solid #2A496E;
            }
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # Rutas
        media_root = os.path.join(os.path.dirname(script_dir), "tutoriales")
        model_path = os.path.join(os.path.dirname(script_dir), "RedesNeuronales",
                                "gestura_bilstm_v1.0.0_20250519_1241.keras")

        def check_resources():
            errors = []
            if not os.path.exists(media_root):
                errors.append("No se encontró la carpeta de tutoriales.")
            if not os.path.exists(model_path):
                errors.append("No se encontró el modelo de red neuronal.")
            if os.path.exists(media_root) and not os.listdir(media_root):
                errors.append("La carpeta de tutoriales está vacía.")
            return errors

        def show_error(error_messages):
            # Limpia errores anteriores
            for i in reversed(range(main_layout.count())):
                w = main_layout.itemAt(i).widget()
                if w and w.objectName() == "error_container":
                    w.deleteLater()

            error_container = QFrame()
            error_container.setObjectName("error_container")
            error_container.setStyleSheet("""
                QFrame {
                    background-color: #2A1C1C;
                    border-radius: 15px;
                    padding: 20px;
                    border: 1px solid #FF6B6B;
                }
            """)
            error_layout = QVBoxLayout(error_container)

            icon = QLabel()
            error_icon_path = os.path.join(carpeta_recursos, "error_icon.png")
            if os.path.exists(error_icon_path):
                pix = QPixmap(error_icon_path)
                icon.setPixmap(pix.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio))
                icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
                error_layout.addWidget(icon)

            for msg in error_messages:
                lbl = QLabel(msg)
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setWordWrap(True)
                lbl.setStyleSheet("color: #FF6B6B; font-size: 16px;")
                error_layout.addWidget(lbl)

            retry_btn = QPushButton("Reintentar")
            retry_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1E3A5F;
                    color: white;
                    padding: 10px 20px;
                    font-size: 16px;
                    border-radius: 5px;
                }
                QPushButton:hover { background-color: #2A496E; }
                QPushButton:pressed { background-color: #122640; }
            """)
            retry_btn.clicked.connect(retry_loading)
            error_layout.addWidget(retry_btn)

            main_layout.addWidget(error_container)

        def retry_loading():
            # Lista de señas válida (26 gestos)
            signs = [
                "A", "Bien", "Buenos_dias", "Cuando", "Cuanto", 
                "Dinero", "E", "Gracias", "HOLA", "I", "J", 
                "Lo_Repites_por_favor", "MiNombreEs", "No", 
                "No_entiendo", "Paciencia", "Pagar", "Para_que", 
                "Por_Favor", "Que", "Revisar", "Servicio", 
                "Si", "Tengo_una_duda", "V", "Yo"
            ]

            errors = check_resources()
            if errors:
                show_error(errors)
                return

            # Limpiar módulo anterior si existe
            if hasattr(self, 'learning_module') and self.learning_module:
                print("[DEBUG] Limpiando módulo de aprendizaje anterior...")
                self.learning_module.cleanup_resources()
                self.learning_module.deleteLater()
            
            # Cargar el módulo de aprendizaje
            self.learning_module = LearningModule(media_root, model_path, signs, parent=self)
            self.learning_module.setStyleSheet("""
                QWidget {
                    background-color: #243B55;
                    border-radius: 15px;
                }
                QListWidget {
                    background-color: #1E293B;
                    border: none;
                    border-radius: 10px;
                    padding: 10px;
                    color: white;
                    font-size: 16px;
                }
                QListWidget::item:hover {
                    background-color: #2A496E;
                }
                QListWidget::item:selected {
                    background-color: #1E3A5F;
                    color: #FFFFFF;
                }
            """)
            main_layout.addWidget(self.learning_module)
            print("[DEBUG] Módulo de aprendizaje cargado correctamente")

        # Al iniciar, comprobamos recursos
        errs = check_resources()
        if errs:
            show_error(errs)
        else:
            retry_loading()

        layout.addWidget(main_container)
        widget.setLayout(layout)
        return widget

    def settings_widget(self):
        # Crear el widget principal para las configuraciones
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Alinear todo hacia la parte superior

        # Título del menú de configuración
        title_label = QLabel("Configuraciones de Traducción")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #555555;
            margin-bottom: 20px;
        """)
        layout.addWidget(title_label)

        # Configuración 1: Modelos de Traducción
        model_label = QLabel("Modelo de Traducción:")
        model_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #777777;")
        layout.addWidget(model_label)

        self.model_selector = QComboBox()
        self.model_selector.addItems(["Modelo Transformer v1", "Modelo Transformer v2"])
        self.model_selector.setStyleSheet("""
            QComboBox {
                background-color: #2A3A53;
                color: white;
                border: 1px solid #4B6E85;
                border-radius: 5px;
                padding: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #2A3A53;
                selection-background-color: #4B6E85;
                color: white;
            }
        """)
        layout.addWidget(self.model_selector)

        # Configuración 2: Umbrales de Quietud/Movimiento
        thresholds_label = QLabel("Umbrales de Quietud/Movimiento:")
        thresholds_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #777777; margin-top: 20px;")
        layout.addWidget(thresholds_label)

        self.movement_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.movement_threshold_slider.setRange(1, 10)  # Rango de sensibilidad
        self.movement_threshold_slider.setValue(int(self.canvas.movement_threshold))  # Valor inicial
        self.movement_threshold_slider.setStyleSheet("background-color: #2A3A53;")
        layout.addWidget(QLabel("Sensibilidad al movimiento (1-10):"))
        layout.addWidget(self.movement_threshold_slider)

        self.still_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.still_threshold_slider.setRange(1, 10)
        self.still_threshold_slider.setValue(int(self.canvas.still_threshold))
        self.still_threshold_slider.setStyleSheet("background-color: #2A3A53;")
        layout.addWidget(QLabel("Umbral de quietud (1-10):"))
        layout.addWidget(self.still_threshold_slider)

        # Configuración 3: Tiempo Requerido para Quietud
        still_time_label = QLabel("Tiempo Requerido para Quietud (segundos):")
        still_time_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #777777; margin-top: 20px;")
        layout.addWidget(still_time_label)

        self.still_time_input = QLineEdit()
        self.still_time_input.setText(str(self.canvas.required_still_time))
        self.still_time_input.setValidator(QDoubleValidator(0.1, 10.0, 1))  # Validación: Número decimal entre 0.1 y 10.0
        self.still_time_input.setStyleSheet("""
            background-color: #2A3A53;
            color: white;
            border: 1px solid #4B6E85;
            border-radius: 5px;
            padding: 8px;
        """)
        layout.addWidget(self.still_time_input)

        # Botón para guardar los cambios
        save_button = QPushButton("Guardar Cambios")
        save_button.setStyleSheet("""
            QPushButton {
                background-color: #1E3A5F;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2A496E;
            }
            QPushButton:pressed {
                background-color: #122640;
            }
        """)
        save_button.clicked.connect(self.save_settings)  # Conectar al método para guardar
        layout.addWidget(save_button)

        # Establecer el layout al widget principal
        widget.setLayout(layout)
        return widget
    
    def open_camera_config_dialog(self):
        from camera_config_dialog import CameraConfigDialog
        
        # Define la configuración actual (puedes definir valores por defecto)
        current_config = {
            "video_source": getattr(self, "current_video_source", "Dispositivo 0"),
            "rotation": getattr(self, "camera_rotation", 0),
            "flip": getattr(self, "camera_flip", False),
            "connection_open": self.listener is not None  # Si hay listener activo, hay conexión
        }


        
        dialog = CameraConfigDialog(default_config=current_config, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_config = dialog.get_configuration()
            print("[DEBUG] Nueva configuración de cámara:", new_config)
            
            # Si se ha marcado la opción para cerrar la conexión, actúa en consecuencia:
            if new_config.get("close_connection", False):
                # Aquí podrías llamar a tu método para cerrar la conexión
                self.close_camera_connection()
                return
            
            # Procesa video_source: si es un string tipo "Dispositivo X", conviértelo a índice
            video_source = new_config["video_source"]
            if isinstance(video_source, str) and video_source.startswith("Dispositivo"):
                try:
                    index = int(video_source.split()[-1])
                    video_source = index
                except Exception as e:
                    print("[DEBUG] Error al convertir video_source:", e)
            
            # Si la configuración ha cambiado, reinicia la conexión (como en el código previo)
            config_changed = (
                video_source != getattr(self, "current_video_source", None) or
                new_config["rotation"] != getattr(self, "camera_rotation", 0) or
                new_config["flip"] != getattr(self, "camera_flip", False)
            )
            
            if config_changed:
                print("[DEBUG] Cerrando la conexión previa...")
                if hasattr(self, "listener") and self.listener is not None:
                    self.listener.stop()
                    if hasattr(self, "capture_thread") and self.capture_thread.is_alive():
                        self.capture_thread.join(timeout=2)
                    time.sleep(1.0)
                
                self.current_video_source = video_source
                self.camera_rotation = new_config["rotation"]
                self.camera_flip = new_config["flip"]
                
                print("[DEBUG] Reiniciando la conexión de video con la nueva configuración.")
                self.listener = MediaPipeListener(self.canvas, self.image_queue, self.lock, self.data_full, self, video_source=video_source)
                self.capture_thread = Thread(target=self.listener.run)
                self.capture_thread.start()
            else:
                print("[DEBUG] La configuración no ha cambiado. No se reinicia la conexión.")
        else:
            print("[DEBUG] El usuario canceló la configuración.")


    def close_camera_connection(self):
        print("[DEBUG] Cerrando la conexión actual de la cámara...")
        if hasattr(self, "listener") and self.listener is not None:
            self.listener.stop()
            # Espera hasta que el hilo termine
            if hasattr(self, "capture_thread") and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2)
            self.listener = None
            self.capture_thread = None
        # Opcional: actualiza el estado de la cámara (por ejemplo, actualizando el icono)
        self.set_camera_status(False)  # Si tienes este método para actualizar el label



    def save_settings(self):
        selected_model = self.model_selector.currentText()

        try:
            # Nota: Los modelos ya se cargan automáticamente en __init__ del Canvas
            # Esta sección se mantiene para compatibilidad pero los modelos BILSTM se usan por defecto
            print(f"[DEBUG] Configuración guardada. Modelos BILSTM activos.")
            # Si en el futuro deseas cambiar modelos dinámicamente, puedes descomentar:
            # if selected_model == "Modelo Transformer v1":
            #     self.canvas.load_model(os.path.join(redes_neuronales, "Modelo_Transformer_v1.keras"))
            # elif selected_model == "Modelo Transformer v2":
            #     self.canvas.load_model(os.path.join(redes_neuronales, "Modelo_Transformer_v2.keras"))

        except Exception as e:
            print(f"[DEBUG] Error al guardar configuración: {e}")

        # Guardar otras configuraciones
        try:
            self.canvas.movement_threshold = self.movement_threshold_slider.value()
            self.canvas.still_threshold = self.still_threshold_slider.value()
            self.canvas.required_still_time = float(self.still_time_input.text())
        except Exception as e:
            print(f"[DEBUG] Error al guardar configuraciones adicionales: {e}")

        QMessageBox.information(self, "Configuración Guardada", "Los cambios han sido aplicados correctamente.")

    def suscripcion_widget(self):
        """ Crea un widget con la página de pago de Stripe """
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Vista web para mostrar la página de pago de Stripe
        self.web_view = QWebEngineView()
         
        self.web_view.setSizePolicy(
            QSizePolicy.Policy.Expanding,  # Ancho
            QSizePolicy.Policy.Expanding   # Alto
        )
        self.web_view.setMinimumSize(1, 1)
        
        layout.addWidget(self.web_view)
        # Mensaje de compra completada (se oculta inicialmente)
        self.completed_label = QLabel("¡Compra completada con éxito! Ya tienes acceso PREMIUM.")
        self.completed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.completed_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #28a745;
        """)
        self.completed_label.setVisible(False)  # Oculto inicialmente
        layout.addWidget(self.completed_label)
        widget.setLayout(layout)

        # Cargar la página de pago al abrir la ventana
        self.iniciar_pago()

        # Detectar si la URL cambió a éxito o cancelación
        self.web_view.urlChanged.connect(lambda url: self.check_payment_status(url))
        
        return widget

    def iniciar_pago(self):
        """ Carga la página de Stripe para realizar el pago de suscripción mensual """
        # Primero, verificar si la suscripción ya está activa
        if self.verificar_suscripcion_activa():
            self.activar_boton_premium()
            return

        # Si no existe suscripción activa, entonces creas la sesión de pago.
        try:
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price': 'price_1QSq0GAGqFTWbMDaoOZpx9pC',
                    'quantity': 1,
                }],
                mode='subscription',
                success_url='http://localhost:5000/success?session_id={CHECKOUT_SESSION_ID}',
                cancel_url='http://localhost:5000/cancel?session_id={CHECKOUT_SESSION_ID}',
            )
            payment_url = session.url
            self.web_view.setUrl(QUrl(payment_url))
        except stripe.error.StripeError as e:
            QMessageBox.critical(self, "Error", f"Stripe Error: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error inesperado: {str(e)}")
            
    def verificar_suscripcion_activa(self):
        user_id = self.user_data["user_id"]
        customer_id = self.obtener_customer_id_desde_db(user_id)  # Debes implementar este método
        
        if not customer_id:
            return False  # Si no existe customer_id, es que no ha pagado antes o no está vinculado a Stripe.

        # Consultar las suscripciones activas del cliente
        subscriptions = stripe.Subscription.list(customer=customer_id, status='active')
        if len(subscriptions.data) > 0:
            # Hay al menos una suscripción activa
            return True
        return False
    
    def obtener_customer_id_desde_db(self, user_id):
        conn = connect_db()
        cursor = conn.cursor()

        # Ejecutar la consulta para obtener el customer_id
        cursor.execute("SELECT customer_id FROM users WHERE user_id=%s", (user_id,))
        result = cursor.fetchone()

        conn.close()

        # Verificar si hay un resultado y devolver el customer_id
        if result:
            customer_id = result[0]  # El resultado de fetchone() es una tupla, por eso se accede a result[0]
        else:
            customer_id = None  # En caso de que no se encuentre, retorna None
        
        return customer_id

    def check_payment_status(self, url):
        """ Detecta si la URL contiene /success o /cancel para manejar el estado del pago """
        url_str = url.toString()
        if 'success' in url_str:
            # Extraer el session_id de la URL
            parsed_url = urlparse(url_str)
            query_params = parse_qs(parsed_url.query)
            session_id = query_params.get('session_id', [None])[0]
            if session_id:
                # Recuperar la sesión desde Stripe
                session = stripe.checkout.Session.retrieve(session_id)
                customer_id = session.customer
                # Guardar el customer_id en la base de datos
                self.guardar_customer_id_en_db(self.user_data["user_id"], customer_id)

            # Cuando la compra es exitosa
            self.web_view.setVisible(False)  # Ocultar la vista web
            self.completed_label.setVisible(True)  # Mostrar mensaje de compra completada
            self.activar_boton_premium()  # Cambiar el botón a "PREMIUM"
            QMessageBox.information(self, "Pago Exitoso", "¡Felicidades! Ahora tienes acceso premium.")
        elif 'cancel' in url_str:
            # Cuando el usuario cancela el pago
            self.web_view.setVisible(False)
            self.completed_label.setVisible(False)
            self.iniciar_pago()
            self.web_view.setVisible(True)
            QMessageBox.warning(self, "Pago Cancelado", "El pago fue cancelado. Inténtalo de nuevo.")

    def guardar_customer_id_en_db(self, user_id, customer_id):
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET customer_id = %s WHERE user_id = %s", (customer_id, user_id))
        conn.commit()
        conn.close()


    def activar_boton_premium(self):
        """ Cambia el botón ACTIVAR a PREMIUM """
        if hasattr(self, 'activate_button'):  # Verifica si la referencia al botón existe
            self.activate_button.setText('PREMIUM')
            self.activate_button.setStyleSheet("""
                QPushButton {
                    height: 40px;
                    padding-left: 20px;
                    padding-right: 20px;
                    background-color: #9dd6ff;  /* Color verde de éxito */
                    color: #092710;
                    font-size: 14px;
                    font-weight: bold;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
                QPushButton:pressed {
                    background-color: #1c7430;
                }
            """)
            self.activate_button.setEnabled(False)  # Deshabilitar el botón para que no se pueda presionar de nuevo


    def perfile_widget(self):
        # Crear el widget principal
        widget = QWidget()
        self.profile_layout = QHBoxLayout()  # Usamos HBoxLayout para colocar las tarjetas una al lado de la otra
        self.profile_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Crear contenedor para la tarjeta original
        self.card_container = QWidget()  # Contenedor para la tarjeta original
        self.card_container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)  # Ajuste fijo
        self.card = QFrame()
        self.card_layout = QVBoxLayout(self.card)  # Layout específico para la tarjeta
        self.card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.card_layout.setContentsMargins(40, 40, 40, 40)  # Márgenes aumentados

        # Estilo de la tarjeta
        self.card_container.setStyleSheet("""
            QWidget {
                background-color: #1E293B;
                border-radius: 20px; 
            }
        """)

        # Avatar
        avatar_label = QLabel()
        avatar_pixmap = self.user_data["avatar"].scaled(
            180, 180, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation  # Tamaño aumentado del avatar
        )
        rounded_avatar = self.get_rounded_pixmap(avatar_pixmap, 180, 180, 90)
        avatar_label.setPixmap(rounded_avatar)
        avatar_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Nombre completo
        full_name_label = QLabel(f"{self.user_data['nombres']} {self.user_data['apellidos']}")
        full_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        full_name_label.setStyleSheet("""
            font-size: 28px;  /* Tamaño de fuente aumentado */
            font-weight: bold;
            color: #FFFFFF;
            margin-top: 20px;  /* Margen superior aumentado */
        """)

        # Información adicional en un grid layout
        self.info_layout = QGridLayout()
        self.info_layout.setContentsMargins(0, 30, 0, 0)  # Márgenes personalizados

        for i, (key, label) in enumerate([("username", "Usuario"),
                                        ("email", "Correo Electrónico"),
                                        ("telefono", "Teléfono"),
                                        ("direccion", "Dirección"),
                                        ("genero", "Género"),
                                        ("fecha_nacimiento", "Fecha de Nacimiento")]):
            key_label = QLabel(f"<b>{label}:</b>")
            value_label = QLabel(f"{self.user_data[key]}")

            # Configurar estilo y tamaño mínimo
            key_label.setStyleSheet("color: #93B6C8; font-size: 20px;")  # Fuente personalizada
            key_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            key_label.setMinimumHeight(30)

            value_label.setStyleSheet("color: #FFFFFF; font-size: 20px;")  # Fuente personalizada
            value_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            value_label.setMinimumHeight(30)

            # Añadir etiquetas a la cuadrícula
            self.info_layout.addWidget(key_label, i * 2, 0, alignment=Qt.AlignmentFlag.AlignRight)
            self.info_layout.addWidget(value_label, i * 2, 1, alignment=Qt.AlignmentFlag.AlignLeft)

            # Añadir un espaciador después de cada fila
            spacer = QSpacerItem(0, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            self.info_layout.addItem(spacer, i * 2 + 1, 0, 1, 2)


        # Botón para acciones (Editar Perfil)
        edit_button = QPushButton("EDITAR PERFIL")
        edit_button.setStyleSheet("""
            QPushButton {
                background-color: #1E3A5F;
                color: white;
                font-size: 20px;  /* Fuente aumentada */
                font-weight: bold;
                border-radius: 10px;  /* Bordes redondeados aumentados */
            }
            QPushButton:hover {
                background-color: #2A496E;
            }
            QPushButton:pressed {
                background-color: #122640;
            }
        """)
        edit_button.setCursor(Qt.CursorShape.PointingHandCursor)
        edit_button.setFixedSize(200, 60)  # Tamaño del botón aumentado
        edit_button.clicked.connect(self.on_edit_clicked)

        # Añadir los elementos al layout de la tarjeta
        self.card_layout.addWidget(avatar_label)
        self.card_layout.addWidget(full_name_label)
        self.card_layout.addLayout(self.info_layout)

        # Añadir un espacio (espacio de 40px de alto)
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)  # Espaciado aumentado
        self.card_layout.addItem(spacer)

        # Botón centrado dinámico
        self.card_layout.addWidget(edit_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # Añadir el contenedor de la tarjeta original al layout principal
        self.card_container.setLayout(self.card_layout)
        self.profile_layout.addWidget(self.card_container)

        # Captura Card Animacion
        self.image_label = QLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.profile_layout.addWidget(self.image_label)
        
        # Flecha en el centro
        self.flecha_label = QLabel()
        Flecha_pixmap = QPixmap(os.path.join(carpeta_recursos, "flecha_editar.png")).scaled(
            200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.flecha_label.setPixmap(Flecha_pixmap)
        self.flecha_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.flecha_label.setVisible(False)
        # Añadir Flecha al layout
        self.profile_layout.addWidget(self.flecha_label, alignment=Qt.AlignmentFlag.AlignLeft)

        # Crear el contenedor de la tarjeta de edición, pero no lo mostramos aún
        self.edit_card_container = QWidget()  # Contenedor para la tarjeta de edición
        self.edit_card_container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.edit_card = QFrame()
        self.edit_card_layout = QVBoxLayout(self.edit_card)
        self.edit_card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.edit_card_layout.setContentsMargins(20, 20, 20, 20)

        # Título de la tarjeta de edición
        edit_title = QLabel("EDITAR PERFIL")
        edit_title.setStyleSheet("""
            font-size: 24px;
            color: #FFFFFF;
            font-weight: bold;
            margin-bottom: 20px;
        """)
        self.edit_card_layout.addWidget(edit_title, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        
        # Línea horizontal debajo del header
        line1 = QFrame()
        line1.setFrameShape(QFrame.Shape.HLine)
        line1.setFrameShadow(QFrame.Shadow.Sunken)
        line1.setStyleSheet("background-color: #18232b; height: 2px;")
        self.edit_card_layout.addWidget(line1)

        # Crear un layout horizontal para dividir los campos en dos columnas
        fields_layout = QHBoxLayout()  # Layout horizontal para la división izquierda y derecha

        # Crear el layout para la parte izquierda
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Crear los campos de edición
        self.edit_fields = {}
        fields = [("username", "Usuario"),
                ("email", "Correo Electrónico"),
                ("telefono", "Teléfono"),
                ("direccion", "Dirección"),
                ("genero", "Género")]

        # Dividir los campos entre las dos columnas (izquierda y derecha)
        for idx, (key, label) in enumerate(fields):
            field_layout = QVBoxLayout()  # Cambiar a QVBoxLayout para colocar verticalmente
            field_label = QLabel(f"{label}:")
            field_label.setStyleSheet("font-size: 16px; color: #E1E1E1; font-weight: 500;")
            
            # Obtener el valor de la clave
            value = self.user_data[key]
            
            # Verificar si el valor es una fecha (datetime.date) y convertir a string
            if isinstance(value, datetime.date):
                value = value.strftime('%Y-%m-%d')  # Formato de fecha

            # Crear el campo de entrada y asignar el texto
            field_input = QLineEdit()
            field_input.setText(str(value))  # Asegurarse de que sea una cadena
            field_input.setStyleSheet("""
                QLineEdit {
                    background-color: #2A3A53;
                    color: #FFFFFF;
                    font-size: 14px;
                    padding: 8px;
                    border: 1px solid #4B6E85;
                    border-radius: 5px;
                }
                QLineEdit:focus {
                    border-color: #1E3A5F;
                    background-color: #344B66;
                }
            """)
            self.edit_fields[key] = field_input
            
            # Añadir la etiqueta encima del campo de entrada
            field_layout.addWidget(field_label)
            field_layout.addWidget(field_input)
            
            # Añadir el campo a la izquierda o derecha dependiendo del índice
            if idx % 2 == 0:
                left_layout.addLayout(field_layout)
            else:
                right_layout.addLayout(field_layout)

        # Añadir las dos columnas al layout principal
        fields_layout.addLayout(left_layout)
        fields_layout.addLayout(right_layout)

        # Añadir el layout de los campos al layout principal
        self.edit_card_layout.addLayout(fields_layout)
        
        # Fecha de nacimiento (QLabel y QDateEdit)
        label_fecha_nacimiento = QLabel("Fecha de nacimiento:", self.edit_card_container)
        label_fecha_nacimiento.setStyleSheet("""
            font-size: 16px; color: #E1E1E1; font-weight: 500;
        """)

        # Campo de entrada para la fecha
        self.date_fecha_nacimiento = QLineEdit(self.edit_card_container)
        self.date_fecha_nacimiento.setPlaceholderText("Selecciona una fecha...")
        self.date_fecha_nacimiento.setReadOnly(True)
        self.date_fecha_nacimiento.setStyleSheet("""
            background-color: #2A3A53;
            color: #FFFFFF;
            font-size: 14px;
            padding: 8px;
            border: 1px solid #4B6E85;
            border-radius: 5px;
        """)
        selected_date = self.user_data.get('fecha_nacimiento')  # Obtén la fecha de manera segura
        formatted_date = "No especificada"

        if selected_date:  # Verifica que no sea None
            try:
                if isinstance(selected_date, str):
                    try:
                        # Valida si ya está en DD/MM/YYYY
                        datetime.datetime.strptime(selected_date, '%d/%m/%Y')  # Esto valida el formato
                        formatted_date = selected_date  # Deja la fecha como está
                    except ValueError:
                        # Si no está en DD/MM/YYYY, intenta convertir desde otros formatos
                        try:
                            # Caso 1: MM/DD/YYYY
                            parsed_date = datetime.datetime.strptime(selected_date, '%m/%d/%Y')  # De MM/DD/YYYY a objeto datetime
                            formatted_date = parsed_date.strftime('%d/%m/%Y')  # Formatear a DD/MM/YYYY
                        except ValueError:
                            try:
                                # Caso 2: YYYY-MM-DD
                                parsed_date = datetime.datetime.strptime(selected_date, '%Y-%m-%d')  # De YYYY-MM-DD a objeto datetime
                                formatted_date = parsed_date.strftime('%d/%m/%Y')  # Formatear a DD/MM/YYYY
                            except ValueError:
                                formatted_date = "Fecha inválida"
                elif isinstance(selected_date, datetime.date):  # Si es un objeto date
                    formatted_date = selected_date.strftime('%d/%m/%Y')
                else:
                    formatted_date = "Fecha inválida"  # Caso donde el formato sea irreconocible
            except Exception as e:
                print(f"Error inesperado: {e}")
                formatted_date = "Fecha inválida"
        else:
            print("selected_date es None.")

        self.date_fecha_nacimiento.setText(formatted_date)
        
        # Rutas de las imágenes de flechas personalizadas
        calendar_arrow_path = os.path.join(carpeta_recursos, 'calendario_icon.png').replace("\\", "/")
        up_arrow_path = os.path.join(carpeta_recursos, 'up_arrow.png').replace("\\", "/")
        down_arrow_path = os.path.join(carpeta_recursos, 'down_arrow.png').replace("\\", "/")
        

        def show_custom_calendar():
            # Crear el calendario personalizado
            custom_calendar = CustomCalendarWidget()
            custom_calendar.setVerticalHeaderFormat(QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader)
            custom_calendar.setWindowFlags(Qt.WindowType.Popup)
            custom_calendar.setFixedSize(360, 250)
            custom_calendar.setMinimumDate(QDate.currentDate().addYears(-100))
            custom_calendar.setMaximumDate(QDate.currentDate().addYears(100))
            # Crear un objeto QTextCharFormat para el formato del encabezado
            header_format = QTextCharFormat()
            header_format.setFont(QFont('Arial', 10, QFont.Weight.Bold))  # Fuente Arial, tamaño 10, negrita
            header_format.setForeground(QColor("black"))  # Color del texto blanco

            # Aplicar el formato al encabezado horizontal
            custom_calendar.setHeaderTextFormat(header_format)
            # Estilo personalizado del calendario
            custom_calendar.setStyleSheet(f"""
                QCalendarWidget QWidget#qt_calendar_navigationbar {{
                    background-color: #1E3A5F; /* Fondo de la barra de navegación */
                }}
                QCalendarWidget QToolButton {{
                    color: #FFFFFF;
                    background-color: #1E3A5F; /* Fondo del botón */
                    font-size: 14px;
                    height: 20px;
                    border: none;
                }}
                QCalendarWidget QToolButton:hover {{
                    background-color: #2A496E; /* Fondo más claro al pasar el cursor */
                }}
                QCalendarWidget QAbstractItemView {{
                    font-size: 14px;
                    color: #FFFFFF;
                    background-color: #1E293B;
                }}
                QCalendarWidget QAbstractItemView:enabled {{
                    selection-background-color: #93B6C8;
                    selection-color: #1E293B; /* Texto oscuro para contraste */
                }}
                QCalendarWidget QAbstractItemView:selected {{
                    background-color: #93B6C8; /* Fondo de selección */
                    color: #1E293B;            /* Texto oscuro */
                }}
                /* Personalización del QSpinBox (selector de año) */
                QCalendarWidget QSpinBox {{
                    background-color: #1E293B;
                    color: #FFFFFF;
                    border-radius: 8px;
                    padding: 1px;
                    font-family: 'Manrope';
                    font-size: 16px;
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
                    background-color: #2A496E;
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
                    background-color: #1E293B; /* Fondo del menú */
                    color: #E1E1E1;
                    border: 1px solid #93B6C8;
                }}
                QCalendarWidget QToolButton#qt_calendar_monthbutton QMenu::item {{
                    background-color: #1E293B;
                    color: #E1E1E1;
                    padding: 4px;
                }}
                QCalendarWidget QToolButton#qt_calendar_monthbutton QMenu::item:selected {{
                    background-color: #93B6C8; /* Fondo al seleccionar */
                    color: #1E293B;
                }}
            """)

            # Posición del calendario emergente
            pos = self.date_fecha_nacimiento.mapToGlobal(self.date_fecha_nacimiento.rect().bottomLeft())
            custom_calendar.move(pos)
            custom_calendar.show()

            # Conexión para actualizar el QLineEdit con la fecha seleccionada
            def update_date(selected_date):
                self.date_fecha_nacimiento.setText(selected_date.toString("dd-MM-yyyy"))
                custom_calendar.close()

            custom_calendar.clicked.connect(update_date)

        # Conectar el evento de clic en el QLineEdit al calendario emergente
        self.date_fecha_nacimiento.mousePressEvent = lambda event: show_custom_calendar()

        # Añadir al layout
        right_layout.addWidget(label_fecha_nacimiento)
        right_layout.addWidget(self.date_fecha_nacimiento)

        # Botón de Confirmar
        confirm_button = QPushButton("Confirmar")
        confirm_button.setStyleSheet("""
            QPushButton {
                background-color: #1E3A5F;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #2A496E;
            }
            QPushButton:pressed {
                background-color: #122640;
            }
        """)
        confirm_button.setCursor(Qt.CursorShape.PointingHandCursor)
        confirm_button.setFixedSize(150, 40)
        confirm_button.clicked.connect(self.on_confirm_clicked)
        
        # Añadir un espacio (espacio de 20px de alto)
        spacer = QSpacerItem(20, 40)  # 20px de ancho, 40px de alto (ajusta según lo que necesites)
        self.edit_card_layout.addItem(spacer)  # Añadir el espacio al layout
        
        # Línea horizontal debajo del header
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setFrameShadow(QFrame.Shadow.Sunken)
        line2.setStyleSheet("background-color: #18232b; height: 2px;")
        self.edit_card_layout.addWidget(line2)

        self.edit_card_layout.addWidget(confirm_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # Estilo de la tarjeta de edición (similar a la tarjeta original, pero más grande)
        self.edit_card_container.setStyleSheet("""
            QWidget {
                background-color: #1E293B;
                border-radius: 15px;
                padding: 20px;
            }
        """)

        self.edit_card_container.setLayout(self.edit_card_layout)

        # Añadir el contenedor de la tarjeta de edición al layout principal, pero inicialmente no lo mostramos
        self.edit_card_container.setVisible(False)

        # Obtener el tamaño actual de card_container
        container_size = self.card_container.size()
        container_width = container_size.width()
        container_height = container_size.height()

        # Aumentar el tamaño un 20% (multiplicando por 1.2)
        new_width = int(container_width * 1.2)
        new_height = int(container_height * 1.2)

        # Establecer el nuevo tamaño en edit_card_container
        self.edit_card_container.setFixedSize(new_width, new_height)

        # Añadir el contenedor de la tarjeta de edición al layout principal
        self.profile_layout.addWidget(self.edit_card_container, alignment=Qt.AlignmentFlag.AlignLeft)


        # Estilo general del widget
        widget.setStyleSheet("""
            background-color: #0F172A;
        """)
        widget.setLayout(self.profile_layout)  # Usar profile_layout para el layout principal
        return widget

    def on_edit_clicked(self):
        # Primero mostramos flecha_label y edit_card_container (pero ocultos al final)
        self.flecha_label.setVisible(True)
        self.edit_card_container.setVisible(True)

        # Forzamos el layout a recalcular posiciones
        self.profile_layout.activate()
        self.card_container.hide()  # Ya la tienes oculta tras la captura

        # ---- Captura del card_container y creación de image_label ----
        edit_button = self.card_layout.itemAt(self.card_layout.count() - 1).widget()
        edit_button.setVisible(False)  # Ocultar el botón temporalmente
        card_pixmap = QPixmap(self.card_container.size())
        painter = QPainter(card_pixmap)
        self.card_container.render(painter)
        painter.end()
        edit_button.setVisible(True)
        
        scaled_pixmap = card_pixmap.scaled(
            int(card_pixmap.width() * 0.6), 
            int(card_pixmap.height() * 0.6),
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        rounded_pixmap = self.get_rounded_pixmap(scaled_pixmap, scaled_pixmap.width(), scaled_pixmap.height(), 15)
        self.image_label.setPixmap(rounded_pixmap)
        self.image_label.adjustSize()

        # Ahora calculamos el centro del widget principal
        parent_widget = self.image_label.parentWidget()
        parent_widget_size = parent_widget.size()
        img_width = self.image_label.width()
        img_height = self.image_label.height()

        center_x = (parent_widget_size.width() - img_width) // 2
        center_y = (parent_widget_size.height() - img_height) // 2

        # Posicionar image_label inicialmente en el centro
        self.image_label.setGeometry(center_x, center_y, img_width, img_height)
        self.image_label.show()

        # Ejemplo asumiendo que están todos en el mismo QHBoxLayout:
        self.profile_layout.update()
        QApplication.processEvents()  # Forzar actualización de layouts

        # Obtener geometrías
        flecha_geom = self.flecha_label.geometry()
        edit_geom = self.edit_card_container.geometry()
        img_final_x = (parent_widget_size.width() - (img_width + flecha_geom.width() + edit_geom.width())) // 2
        img_final_y = center_y
        final_rect = QRect(img_final_x, img_final_y, img_width, img_height)

        # Ahora ocultamos flecha_label y edit_card_container otra vez para la animación inicial
        self.flecha_label.setVisible(False)
        self.edit_card_container.setVisible(False)

        # Restauramos image_label al centro
        self.image_label.setGeometry(center_x, center_y, img_width, img_height)

        # ---- Crear la animación desde el centro hasta final_rect ----
        self.animation = QPropertyAnimation(self.image_label, b"geometry")
        self.animation.setDuration(300)
        self.animation.setStartValue(QRect(center_x, center_y, img_width, img_height))
        self.animation.setEndValue(final_rect)

        # Cuando la animación termine, mostramos flecha_label y edit_card_container
        def after_animation():
            self.flecha_label.setVisible(True)
            self.edit_card_container.setVisible(True)
            self.profile_layout.update()

        self.animation.finished.connect(after_animation)
        self.animation.start()


    def show_edit_card(self):
        # Mostrar la tarjeta de edición después de la animación
        self.edit_card_container.setVisible(True)
        self.flecha_label.setVisible(True)
        self.profile_layout.update()

    def on_confirm_clicked(self):
        # Actualizar los datos con los nuevos valores de los campos de edición
        for key, field in self.edit_fields.items():
            self.user_data[key] = field.text()

        # Eliminar la tarjeta de edición y mostrar la original con los datos actualizados
        self.image_label.setPixmap(QPixmap())
        self.image_label.hide()
        self.image_label.setGeometry(self.card_container.geometry())
        self.flecha_label.setVisible(False)
        self.edit_card_container.setVisible(False)
        self.card_container.setVisible(True)

        self.profile_layout.update()

    def translation_widget(self):
        # Widget Principal
        self.widget_traduccion = QWidget()
        
        # Layout principal
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(0)
        
        # Main Content Section
        self.content_layout = QHBoxLayout()
        
        # Left Section (Video and Info)
        self.left_layout = QVBoxLayout()
        self.left_layout.setContentsMargins(10, 10, 10, 10)
        self.left_layout.setSpacing(0)
        
        # Crear un QWidget para contener el left_layout
        self.left_widget = QWidget()
        self.left_widget.setLayout(self.left_layout)
        
        # Video Container
        self.video_container = QFrame()
        self.video_container.setStyleSheet("background-color: black; border-radius: 10px;")
        
        # Establecer la política de tamaño para que se expanda o contraiga
        self.video_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Usamos QGridLayout para superponer y posicionar widgets
        self.video_container_layout = QGridLayout()
        self.video_container_layout.setContentsMargins(0, 0, 0, 0)
        self.video_container_layout.setSpacing(0)
        
        # Ajustar el tamaño del video_container después de crearlo
        self.adjust_video_container_size()

        # Image widget (video)
        self.image_widget = QLabel()
        self.image_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_container_layout.addWidget(self.image_widget, 0, 0)  # Añade en la posición (0, 0)

        
        # Icono de estado de la cámara
        self.camera_on = False  # Estado inicial de la cámara
        # En el método header_gestura o donde se cree el camera_status_label:
        self.camera_status_label = self.ClickableLabel()  # Usamos la clase ClickableLabel ya existente
        self.camera_pixmap = QPixmap(os.path.join(carpeta_recursos, "camera_off.png")).scaled(
            32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.camera_status_label.setPixmap(self.camera_pixmap)
        self.camera_status_label.setContentsMargins(0, 10, 10, 0)
        # Conecta el click para abrir el diálogo de configuración
        self.camera_status_label.clicked.connect(self.open_camera_config_dialog)

        # Agrega espacio usando márgenes en el QLabel
        self.camera_status_label.setContentsMargins(0, 10, 10, 0)  # 10 píxeles a la derecha y arriba

        # Barra de progreso personalizada
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 7000)
        self.progress_bar.setFixedHeight(3)  # Ajusta la altura de la barra de progreso a 5 píxeles
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)  # Expande horizontalmente, pero fija en altura

        # Estilo para hacerla cuadrada y sin texto
        self.progress_bar.setTextVisible(False)  # Oculta el texto dentro de la barra
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #2b2b2b;  /* Fondo */
                border: 1px solid #1e2d3a;  /* Borde */
                border-radius: 0px;         /* Bordes redondeados */
            }
            QProgressBar::chunk {
                background-color: #19a2e6;  /* Color del progreso */
            }
        """)
        # Configurar la barra de progreso para que ocupe todo el ancho del video_container_layout
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)  # Expande en ancho, fija en altura
        self.progress_bar.setFixedHeight(5)  # Ajusta la altura de la barra de progreso

        # Inicializar valor de la barra de progreso
        self.progress_value = 0
        
        # Etiqueta de estado de predicción
        self.prediccion_status_label = QLabel()
        self.prediccion_on = False
        self.prediccion_pixmap = QPixmap(os.path.join(carpeta_recursos, "prediccion_off.png")).scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.prediccion_status_label.setPixmap(self.prediccion_pixmap)

        # Crear un layout horizontal para progress_bar y prediccion_status_label
        self.progress_layout = QHBoxLayout()

        # Añadir la barra de progreso al layout
        self.progress_layout.addWidget(self.progress_bar)

        # Asegúrate de que el layout ocupe el espacio adecuadamente
        self.progress_layout.setContentsMargins(0, 0, 0, 0)
        
        # Añadir el progress_layout a la esquina inferior izquierda del video_container_layout
        self.video_container_layout.addLayout(self.progress_layout, 0, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        
        # Añadir camera_status_label en la esquina inferior derecha
        self.video_container_layout.addWidget(self.camera_status_label, 0, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        
        # Establece el layout en el video_container
        self.video_container.setLayout(self.video_container_layout)
        
        # Añadir el video_container al left_layout
        self.left_layout.addWidget(self.video_container)
        
        # Contenedor principal para `prediction_label`
        self.prediction_container = QWidget()
        self.prediction_container.setStyleSheet("""
            background-color: #2E3B4E; /* Fondo oscuro */
            border-radius: 8px;       /* Bordes redondeados */
            border: 2px solid #1C2533; /* Borde ligeramente más oscuro */
        """)

        # Crear un layout vertical para el contenedor
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(5, 5, 5, 5)  # Ajustar márgenes para dar espacio arriba
        container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Alinear todo hacia arriba
        self.prediction_container.setLayout(container_layout)

        # Etiqueta "Traducción" (arriba a la derecha dentro del contenedor)
        self.translation_label = QLabel("Traducción")
        self.translation_label.setStyleSheet("""
            color: white; 
            font-size: 32px; 
            font-weight: bold;
            border: None;
        """)
        
        # Crear varias líneas
        line3 = QFrame()
        line3.setFrameShape(QFrame.Shape.HLine)
        line3.setFrameShadow(QFrame.Shadow.Sunken)
        line3.setStyleSheet("background-color: #18232b; height: 2px;") 
        
        # Etiqueta de texto dinámico
        self.prediction_label = QTextEdit()
        self.prediction_label.setReadOnly(True)  # Hacerlo no editable
        self.prediction_label.setStyleSheet("""
            background-color: #2E3B4E; 
            color: #B0BEC5; 
            font-size: 28px;
            border: None;
        """)

        # Añadir las etiquetas al layout del contenedor
        container_layout.addWidget(self.translation_label, alignment=Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop)  # Texto fijo
        container_layout.addWidget(line3)
        container_layout.addWidget(self.prediction_label)

        # Ajustar la altura del contenedor
        label_height = int(self.gestura_widget.height() * 0.50)
        self.prediction_container.setMinimumHeight(label_height)

        # Añadir el contenedor al layout principal
        self.left_layout.addWidget(self.prediction_container)
        
        # Crear el botón
        self.prediction_button = QPushButton("NARRAR")
        self.prediction_button.setStyleSheet("""
            QPushButton {
                background-color: #1E3A5F;  /* Fondo azulado */
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: none;
                border-bottom-left-radius: 8px; /* Bordes redondeados en la parte inferior izquierda */
                border-bottom-right-radius: 8px; /* Bordes redondeados en la parte inferior derecha */
                border-top-left-radius: 0px; /* Plano en la parte superior izquierda */
                border-top-right-radius: 0px; /* Plano en la parte superior derecha */
            }
            QPushButton:hover {
                background-color: #2A496E; /* Fondo azulado más claro al pasar el mouse */
            }
            QPushButton:pressed {
                background-color: #122640; /* Fondo azulado oscuro al presionar */
            }
        """)
        self.prediction_button.setCursor(Qt.CursorShape.PointingHandCursor)  # Cursor de mano al pasar por encima
        self.prediction_button.setFixedSize(200, 40)  # Ancho: 200px, Alto: 40px
        self.prediction_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        
        # Crear el botón de texto de prueba
        self.test_button = QPushButton("TEXTO DE PRUEBA")
        self.test_button.setStyleSheet("""
            QPushButton {
                background-color: #2A496E;  /* Color diferente para distinguir */
                color: white;
                font-size: 14px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #1E3A5F;
            }
            QPushButton:pressed {
                background-color: #122640;
            }
        """)
        self.test_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.test_button.setFixedSize(150, 35)  # Más pequeño que el botón NARRAR
        
        # Conectar los botones a sus funciones (UNA SOLA VEZ cada uno)
        self.test_button.clicked.connect(self.add_test_text)
        self.prediction_button.clicked.connect(self.start_narration_process)

        # Crear un layout horizontal para centrar los botones y controlar su tamaño
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)  # Sin márgenes para pegarlo al prediction_container
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Centrar los botones horizontalmente
        button_layout.addWidget(self.test_button)
        button_layout.addWidget(self.prediction_button)

        # Crear un QWidget para contener el botón con su layout
        button_container = QWidget()
        button_container.setLayout(button_layout)

        # Añadir el contenedor del botón al layout principal, justo después del prediction_container
        self.left_layout.addWidget(button_container)
        
        # Añadir un estiramiento para empujar los widgets hacia arriba
        self.left_layout.addStretch()
        
        # Añadir el left_widget al content_layout
        self.content_layout.addWidget(self.left_widget)
        
        # Sidebar (Right Section)
        self.sidebar_layout = QVBoxLayout()
        self.sidebar_layout.setContentsMargins(10, 10, 10, 10)
        self.sidebar_layout.setSpacing(15)
        
        # Sidebar Box
        self.sidebar_widget = QFrame()
        self.sidebar_widget.setStyleSheet("background-color: #243b47; border-radius: 20px;")
        self.sidebar_box_layout = QHBoxLayout()
        self.sidebar_box_layout.setContentsMargins(10, 10, 10, 10)
        self.sidebar_box_layout.setSpacing(10)
        
        # Imagen de la caja
        self.box_image_label = QLabel()
        self.box_pixmap = self.get_rounded_pixmap(os.path.join(carpeta_recursos, "box_image.png"), 56, 56, 10)
        self.box_image_label.setPixmap(self.box_pixmap)
        self.box_image_label.setFixedSize(56, 56)
        
        # Título y descripción
        self.text_layout = QVBoxLayout()
        self.box_title_label = QLabel("Habla ahora")
        self.box_title_label.setStyleSheet("color: white; font-weight: bold; font-size: 18px;")
        
        self.box_description_label = QLabel("Presiona para convertir tu voz en texto...")
        self.box_description_label.setStyleSheet("color: #93b6c8; font-size: 18px;")
        
        self.text_layout.addWidget(self.box_title_label)
        self.text_layout.addWidget(self.box_description_label)
        
        # Botón de voz
        self.voice_button = QPushButton()
        self.voice_button.setFixedSize(45, 45)
        self.voice_button.setStyleSheet("""
            QPushButton {
                background-color: #1E3A5F;
                border-radius: 22px;
                border: none;
            }
            QPushButton:hover {
                background-color: #2A496E; /* Fondo azulado más claro al pasar el mouse */
            }
            QPushButton:pressed {
                background-color: #122640; /* Fondo azulado oscuro al presionar */
            }
        """)
        self.voice_button.setIcon(QIcon(os.path.join(carpeta_recursos, "voice_icon.png")))
        self.voice_button.setIconSize(QSize(24, 24))
        
        # Añadir widgets al sidebar_box_layout
        self.sidebar_box_layout.addWidget(self.box_image_label)
        self.sidebar_box_layout.addLayout(self.text_layout)
        self.sidebar_box_layout.addWidget(self.voice_button)
        
        # Establecer el layout en el sidebar_widget
        self.sidebar_widget.setLayout(self.sidebar_box_layout)
        
        # Añadir el sidebar_widget al sidebar_layout
        self.sidebar_layout.addWidget(self.sidebar_widget)
        
        # Inicializar elementos relacionados con la transcripción
        self.init_transcription_elements()
        
        # Añadir un estiramiento para empujar los widgets hacia arriba
        self.sidebar_layout.addStretch()
        
        # Añadir el sidebar_layout al content_layout
        self.content_layout.addLayout(self.sidebar_layout)
        
        # Establecer el stretch para que ambas secciones se ajusten proporcionalmente
        self.content_layout.setStretch(0, 2)  # Sección izquierda (más ancha)
        self.content_layout.setStretch(1, 2)  # Sección derecha (más estrecha)
        
        # Añadir el content_layout al main_layout
        self.main_layout.addLayout(self.content_layout)
        
        # Asignar el layout principal al gestura_widget
        self.widget_traduccion.setLayout(self.main_layout)
        
        self.left_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.sidebar_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        
        return self.widget_traduccion
    
    def add_test_text(self):
        """Agregar texto de prueba al área de predicciones"""
        test_messages = [
        "Hola me llamo j u a n p e r e z yo necesitar abrir cuenta bancaria nueva yo acabar conseguir trabajo y necesitar depositar mi salario ayudar con requisitos por favor",
        
        "Buenos dias mi nombre es m a r i a l o p e z yo venir para solicitar credito hipotecario yo querer comprar casa primera vez necesitar informacion sobre documentos y tasas interes",
        
        "Hola me llamo c a r l o s r o d r i g u e z yo necesitar tramitar reposicion de tarjeta debito porque perder en taxi ayer necesitar bloquear cuenta y conseguir nueva tarjeta rapido",
        
        "Buenos dias yo ser l u i s a g u i l a r estudiante universidad yo necesitar solicitar beca estudiantil porque mis padres no poder pagar colegiatura completa ayudar con formulario",
        
        "Hola mi nombre es a n a t o r r e s yo venir para inscribir a mi hijo en primaria necesitar saber que documentos traer y cuando empezar clases siguiente año",
        
        "Buenos dias me llamo d a v i d s a n c h e z yo necesitar certificado estudios preparatoria porque aplicar para universidad necesitar documento oficial con calificaciones",
        
        "Hola yo ser s o f i a m e n d o z a madre familia yo necesitar inscribir tres hijos en escuela publica pero no tener algunos documentos ayudar con proceso",
        
        "Buenos dias mi nombre es r o b e r t o c a s t r o yo necesitar solicitar cita medico especialista porque tener problema corazon doctor general mandar con cardiologo",
        
        "Hola me llamo p a t r i c i a r u i z yo venir para renovar receta medica diabetes porque pastillas acabar y necesitar continuar tratamiento urgente",
        
        "Buenos dias yo ser m i g u e l v a r g a s yo necesitar tramitar pension jubilacion porque cumplir sesenta y cinco años y trabajar cuarenta años en empresa",
        
        "Hola mi nombre es c a r m e n f l o r e s yo necesitar solicitar apoyo gobierno para adultos mayores porque mi esposo estar enfermo y no poder trabajar mas",
        
        "Buenos dias me llamo f e r n a n d o m o r a l e s yo venir para pagar impuestos predial mi casa pero no entender cuanto deber ayudar calcular cantidad correcta",
        
        "Hola yo ser g a b r i e l a h e r n a n d e z yo necesitar tramitar visa para viajar estados unidos por trabajo mi empresa mandar conferencia internacional",
        
        "Buenos dias mi nombre es r i c a r d o j i m e n e z yo necesitar abrir cuenta ahorro para mi hija pequeña yo querer guardar dinero para su educacion universidad",
        
        "Hola me llamo v e r o n i c a r a m i r e z yo venir para cambiar beneficiario mi seguro vida porque divorciar recientemente y querer poner mis hijos como beneficiarios"
    ]

        # Seleccionar un mensaje aleatorio
        import random
        selected_message = random.choice(test_messages)
        
        # Limpiar el área y agregar el texto de prueba
        self.prediction_label.clear()
        
        # Crear formato para el texto de prueba (color diferente para distinguir)
        cursor = self.prediction_label.textCursor()
        test_format = QTextCharFormat()
        test_format.setForeground(QColor("#FFD700"))  # Color dorado para distinguir que es de prueba
        cursor.setCharFormat(test_format)
        cursor.insertText(f"[PRUEBA] {selected_message}")
        
        print(f"[DEBUG] Texto de prueba agregado: {selected_message}")

    def logout(self):
        print("[DEBUG] Cerrando sesión...")
        # Si hay conexión abierta, detener el listener y esperar el hilo
        if hasattr(self, "listener") and self.listener is not None:
            self.listener.stop()
            if hasattr(self, "capture_thread") and self.capture_thread is not None:
                self.capture_thread.join(timeout=2)
                self.capture_thread = None
            self.listener = None

        # Detener el temporizador, si existe
        if hasattr(self, "timer") and self.timer is not None:
            self.timer.stop()
            self.timer = None
            
        # Detener narración si está activa
        if self.is_narrating:
            self.stop_narration()

        # Limpiar la información del usuario y la interfaz
        self.user_data = None
        if hasattr(self, "gestura_widget") and self.gestura_widget is not None:
            self.stack.removeWidget(self.gestura_widget)
            self.gestura_widget = None

        # Liberar recursos adicionales (por ejemplo, la instancia de Canvas)
        self.canvas = None

        # Reiniciar bandera de hilos
        self.stop_threads = False

        # Volver a la pantalla de login
        self.stack.setCurrentWidget(self.login_widget)
        self.login_widget.reset_to_login()



    def resizeEvent(self, event):
        self.adjust_video_container_size()
        super().resizeEvent(event)

    def adjust_video_container_size(self):
        # Obtener el tamaño actual de la ventana
        current_width = self.width()
        current_height = self.height()
        
        # Obtener la resolución de la pantalla
        screen_geometry = QGuiApplication.primaryScreen().geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Calcular el tamaño dinámico del video_container
        new_width = min(int(current_width * 0.5), int(screen_width * 0.5))  # Máximo 50% del ancho de la pantalla
        new_height = min(int(current_height * 0.5), int(screen_height * 0.5))  # Máximo 50% de la altura de la pantalla

        if self.video_container:
            # En lugar de un tamaño fijo, usamos un tamaño mínimo para que se expanda
            self.video_container.setMinimumSize(new_width, new_height)

        
    def cargar_gestura_aplicacion(self, user_perfile):
        avatar = self.convertir_avatar(user_perfile['avatar'])
        user_perfile['avatar'] = avatar
        self.user_data = user_perfile
        
        # Reinicializar los atributos relacionados con el hilo y el listener
        self.stop_threads = False
        self.listener = None
        self.connection = None

        # Reinicializar canvas y otros recursos
        self.load_heavy_components()

        # Crear un nuevo widget de la aplicación principal
        self.gestura_widget = QWidget()
        self.header_gestura()
        self.stack.addWidget(self.gestura_widget)

        # Iniciar la cámara
        self.start_camera()

        # Cambiar al widget de la aplicación principal
        self.stack.setCurrentWidget(self.gestura_widget)

    def load_heavy_components(self):
        # Inicializar Canvas (los modelos se cargan automáticamente en __init__)
        self.canvas = Canvas(self)
        # Los modelos BILSTM ya se cargan automáticamente en Canvas.__init__()
        print(f"[DEBUG] Canvas inicializado con {len(self.canvas.modelos)} modelos BILSTM")

        self.image_queue = Queue(maxsize=3)
        self.data_full = []
        self.lock = Lock()

    
    # Función en GesturaApp para actualizar la barra de progreso usando el still_duration de canvas
    def update_progress_bar(self):
        # Usamos el tiempo de quietud efectivo (mínimo entre las manos presentes)
        current_still = self.canvas.get_current_still_duration()  # En segundos
        # Solo actualizamos si estamos en modo recording:
        if not self.canvas.ready_for_recording:
            self.update_progress_signal.emit(0)
            return

        # La barra no avanza hasta 0.5 segundos de quietud
        if current_still < 0.5:
            self.update_progress_signal.emit(0)
        else:
            # Mapear la quietud desde 0.5 segundos hasta required_still_time a un rango 0 - 7000
            # Por ejemplo, si required_still_time es 1.3, el rango es 0.8 segundos.
            progress_value = int(((current_still - 0.5) / (self.canvas.required_still_time - 0.5)) * 7000)
            # Limitar el valor máximo:
            if progress_value > 7000:
                progress_value = 7000
            elif progress_value < 0:
                progress_value = 0
            self.update_progress_signal.emit(progress_value)


            
    def update_progress_bar_from_signal(self, value):
        self.progress_bar.setValue(value)

    def convertir_avatar(self, avatar_data):
        try:
            if not avatar_data:
                raise ValueError("No hay avatar disponible.")
            
            if isinstance(avatar_data, bytes):
                # Caso: el avatar es un archivo de imagen binario cargado desde la computadora
                image = QPixmap()
                image.loadFromData(avatar_data)
                return image.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

            if isinstance(avatar_data, str):
                # Verificar si el avatar es una URL
                if avatar_data.startswith("http"):
                    try:
                        response = requests.get(avatar_data)
                        if response.status_code == 200:
                            avatar_bytes = response.content
                            image = QPixmap()
                            if image.loadFromData(avatar_bytes):
                                return image.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                            else:
                                raise ValueError("Error al cargar el avatar desde los datos descargados.")
                        elif response.status_code == 404:
                            raise ValueError("Error 404: El enlace del avatar no es válido o ha expirado")
                        else:
                            raise ValueError(f"No se pudo descargar el avatar desde la URL. Código de estado: {response.status_code}")
                    except Exception as e:
                        raise ValueError(f"Error al descargar el avatar desde la URL: {str(e)}")
                else:
                    # Decodificar la cadena base64 si no es una URL
                    try:
                        avatar_bytes = base64.b64decode(avatar_data)
                        image = QPixmap()
                        if image.loadFromData(avatar_bytes):
                            return image.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                        else:
                            raise ValueError("Error al cargar el avatar desde los datos base64.")
                    except Exception as e:
                        raise ValueError(f"Error al decodificar el avatar en base64: {str(e)}")   
            raise ValueError("Formato de avatar no soportado.")
        except Exception as e:
            print(f"Error al convertir el avatar: {str(e)}")
            return None

        
        
    def get_rounded_pixmap(self, source, width, height, radius, frame_color=QColor("black"), frame_thickness=0):
        # Determinar si la fuente es un QPixmap o una ruta de archivo
        if isinstance(source, QPixmap):
            original_pixmap = source.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        elif isinstance(source, str):  # Asume que es una ruta de archivo
            original_pixmap = QPixmap(source).scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        else:
            raise ValueError("La fuente debe ser un QPixmap o una ruta de archivo (str).")

        # Crear una imagen con transparencia
        canvas_width = width + 2 * frame_thickness
        canvas_height = height + 2 * frame_thickness
        rounded_pixmap = QPixmap(canvas_width, canvas_height)
        rounded_pixmap.fill(QColor("transparent"))

        # Usar un QPainter para dibujar la imagen redondeada
        painter = QPainter(rounded_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Crear una máscara con un rectángulo de esquinas redondeadas
        path = QPainterPath()
        path.addRoundedRect(frame_thickness, frame_thickness, width, height, radius, radius)

        # Dibujar el marco antes de recortar la imagen
        if frame_thickness > 0:
            pen = QPen(frame_color, frame_thickness)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

        # Ajustar la posición de la imagen para que quede centrada
        x_offset = (width - original_pixmap.width()) // 2
        y_offset = (height - original_pixmap.height()) // 2

        # Aplicar la máscara para recortar la imagen con bordes redondeados
        painter.setClipPath(path)
        painter.drawPixmap(frame_thickness + x_offset, frame_thickness + y_offset, original_pixmap)
        painter.end()

        return rounded_pixmap
            
    def start_camera(self):
        """Iniciar la captura de la cámara en un hilo separado."""
        # Si ya hay un hilo de captura activo, no crear otro
        if self.capture_thread and self.capture_thread.is_alive():
            print("[DEBUG] La cámara ya está en ejecución")
            # Pero asegurarse de que el timer esté activo
            if not self.timer or not self.timer.isActive():
                print("[DEBUG] Reiniciando timer de actualización de cámara")
                self.timer = QTimer(self)
                self.timer.timeout.connect(self.update_camera_view)
                self.timer.start(33)
            return
        
        # Si ya hay un timer activo, no crear otro
        if self.timer and self.timer.isActive():
            print("[DEBUG] El timer ya está activo")
            # Pero verificar si hay hilo de captura
            if not self.capture_thread or not self.capture_thread.is_alive():
                print("[DEBUG] Reiniciando hilo de captura")
                self.capture_thread = Thread(target=self.data_capture_main, daemon=True)
                self.capture_thread.start()
            return
        
        print("[DEBUG] Iniciando captura de cámara...")
        self.capture_thread = Thread(target=self.data_capture_main, daemon=True)
        self.capture_thread.start()

        # Usar QTimer para actualizar la vista de la cámara
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_view)
        self.timer.start(33)  # 30 FPS en lugar de 60 para mejor rendimiento
        print("[DEBUG] ✅ Cámara y timer iniciados correctamente")


    def data_capture_main(self):
        self.listener = MediaPipeListener(self.canvas, self.image_queue, self.lock, self.data_full, self)
        self.listener.run()


    def set_camera_status(self, online: bool):
        self.camera_on = online
        if online:
            pixmap = QPixmap(os.path.join(carpeta_recursos, "camera_on.png")).scaled(
                32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        else:
            pixmap = QPixmap(os.path.join(carpeta_recursos, "camera_off.png")).scaled(
                32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.camera_status_label.setPixmap(pixmap)


    def update_camera_view(self):
        if not hasattr(self, 'image_widget'):
            # Si image_widget no existe aún, no hacer nada
            return
            
        with self.lock:
            if not self.image_queue.empty():
                frame = self.image_queue.get()
                if isinstance(frame, np.ndarray):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, _ = frame_rgb.shape
                    qimg = QImage(frame_rgb.data, width, height, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                    
                    # Aplicar rotación y flip si están configurados
                    if hasattr(self, "camera_rotation") and self.camera_rotation:
                        transform = QTransform().rotate(self.camera_rotation)
                        pixmap = pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)
                    if hasattr(self, "camera_flip") and self.camera_flip:
                        pixmap = pixmap.transformed(QTransform().scale(-1, 1))
                    
                    pixmap = pixmap.scaled(self.image_widget.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    
                    # Mostrar en el label del video:
                    self.image_widget.setPixmap(pixmap)

    def update_predictions(self, predictions):
        """Emitir la señal para actualizar el QTextEdit desde cualquier hilo."""
        self.update_text_signal.emit(predictions)

        
    def _update_predictions(self, predictions):
        """Manejar la actualización segura del QTextEdit con colores diferenciados."""
        # Crear un cursor para el QTextEdit
        cursor = self.prediction_label.textCursor()
        
        # Seleccionar todo el texto actual y cambiar su color a gris claro
        cursor.movePosition(QTextCursor.MoveOperation.Start)  # Mover al inicio
        cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor) # Seleccionar todo el texto
        
        # Aplicar formato gris claro al texto anterior
        gray_format = QTextCharFormat()
        gray_format.setForeground(QColor("#B0BEC5"))  # Gris claro
        cursor.setCharFormat(gray_format)
        
        # Mover el cursor al final del QTextEdit para añadir el nuevo texto
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Aplicar formato azul claro al nuevo texto
        blue_format = QTextCharFormat()
        blue_format.setForeground(QColor("#9CDCFE"))  # Azul claro
        cursor.setCharFormat(blue_format)
        
        # Insertar el nuevo texto
        cursor.insertText(f"{predictions} ")
        
        # Asegurarse de que el cursor permanece al final
        self.prediction_label.setTextCursor(cursor)
        
        # Emitir señal para reiniciar la barra de progreso (opcional)
        self.update_progress_signal.emit(0)




    def toggle_prediccion_icon(self):
        if not self.prediccion_on:
            # Cambiar a 'camera_on.png'
            self.prediccion_pixmap = QPixmap(os.path.join(carpeta_recursos, "prediccion_on.png")).scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.prediccion_on = True  # Actualizamos el estado a "encendido"
        else:
            # Cambiar a 'camera_off.png'
            self.prediccion_pixmap = QPixmap(os.path.join(carpeta_recursos, "prediccion_off.png")).scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.prediccion_on = False  # Actualizamos el estado a "apagado"
        
        self.prediccion_status_label.setPixmap(self.prediccion_pixmap)
        
    
    # Método para alternar entre camera_off.png y camera_on.png
    def toggle_camera_icon(self):
        """Alterna entre las imágenes de camera_off y camera_on."""
        if not self.camera_on:
            # Cambiar a 'camera_on.png'
            self.camera_pixmap = QPixmap(os.path.join(carpeta_recursos, "camera_on.png")).scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.camera_on = True  # Actualizamos el estado a "encendido"
        else:
            # Cambiar a 'camera_off.png'
            self.camera_pixmap = QPixmap(os.path.join(carpeta_recursos, "camera_off.png")).scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.camera_on = False  # Actualizamos el estado a "apagado"

        # Actualizar la imagen en el QLabel
        self.camera_status_label.setPixmap(self.camera_pixmap)

    
    def init_transcription_elements(self):
        """Inicializar la parte de transcripción"""
        # Conectar el botón de voz para iniciar la transcripción
        self.voice_button.clicked.connect(self.toggle_transcription)

        # Crear el área de texto para la transcripción
        self.transcription_areaText = QTextEdit(self)
        self.transcription_areaText.setStyleSheet("""
            background-color: #243b47; 
            color: white; 
            border-radius: 10px;
            padding: 10px;
            border: none;
            font-size: 34px;
        """)
        self.transcription_areaText.setPlaceholderText("Esperando transcripción...")
        self.transcription_areaText.setReadOnly(True)
        self.transcription_areaText.setEnabled(True)
        self.sidebar_layout.addWidget(self.transcription_areaText)

    def toggle_transcription(self):
        """Iniciar o detener la transcripción"""
        if self.is_transcribing:
            self.stop_transcription()
        else:
            self.start_transcription()

    def start_transcription(self):
        """Iniciar la transcripción de voz"""
        self.is_transcribing = True
        self.voice_button.setEnabled(False)  # Desactivar el botón mientras se transcribe
        self.transcription_areaText.clear()  # Limpiar el área de texto
        self.transcription_worker = TranscriptionWorker()
        self.transcription_worker.text_update.connect(self.update_transcription_text)
        self.transcription_worker.timer_update.connect(self.update_transcription_timer)
        self.transcription_worker.finished.connect(self.transcription_finished)

        # Iniciar la transcripción en un hilo separado
        self.transcription_thread = Thread(target=self.transcription_worker.run)
        self.transcription_thread.start()

    def stop_transcription(self):
        """Detener la transcripción"""
        self.is_transcribing = False
        self.transcription_worker = None  # Desactivar el trabajador
        self.voice_button.setEnabled(True)  # Reactivar el botón
        
    def update_transcription_text(self, text, partial=False, final=False):
        """Actualizar el cuadro de texto con la transcripción recibida."""
        cursor = self.transcription_areaText.textCursor()

        if partial:
            text = text.strip()
            # Verificar diferencias con el texto parcial actual
            if text != self.partial_text:
                current_words = self.partial_text.split()
                new_words = text.split()

                # Identificar la última palabra nueva
                if len(new_words) > len(current_words):
                    last_word = new_words[-1]

                    # Actualizar texto en el cuadro
                    self.transcription_areaText.clear()

                    # Escribir palabras anteriores en gris claro
                    format_previous = QTextCharFormat()
                    format_previous.setForeground(QColor("#B0BEC5"))  # Gris claro
                    cursor.setCharFormat(format_previous)
                    cursor.insertText(" ".join(new_words[:-1]) + " ")

                    # Escribir la última palabra en azul claro
                    format_new = QTextCharFormat()
                    format_new.setForeground(QColor("#9CDCFE"))  # Azul claro
                    cursor.setCharFormat(format_new)
                    cursor.insertText(last_word)

                    # Actualizar el estado interno
                    self.partial_text = text

                    # Mover el cursor al final
                    cursor.movePosition(QTextCursor.MoveOperation.End)

        elif final:
            clean_text = re.sub(r'\s+', ' ', text.strip())
            # Al confirmar el texto, se almacena todo como gris claro
            self.transcription_areaText.clear()
            format_final = QTextCharFormat()
            format_final.setForeground(QColor("#B0BEC5"))  # Gris claro
            cursor.setCharFormat(format_final)
            cursor.insertText(clean_text.strip())

            # Resetear el estado interno
            self.partial_text = ""

    def update_transcription_timer(self, elapsed_time):
        """Actualizar el temporizador de la transcripción si es necesario (opcional)"""
        pass

    def transcription_finished(self):
        """Método llamado cuando la transcripción finaliza"""
        self.is_transcribing = False
        self.voice_button.setEnabled(True)  # Habilitar nuevamente el botón para iniciar transcripción
    
    def start_narration_process(self):
        """Iniciar el proceso completo de parafraseo y narración"""
        if self.is_narrating:
            self.stop_narration()
            return
            
        # Obtener el texto del área de predicciones
        text_to_paraphrase = self.prediction_label.toPlainText().strip()
        
        if not text_to_paraphrase:
            QMessageBox.information(self, "Sin texto", "No hay texto para narrar.")
            return
            
        # Cambiar estado del botón
        self.is_narrating = True
        self.prediction_button.setText("PROCESANDO...")
        self.prediction_button.setEnabled(False)
        
        # Iniciar parafraseo con ChatGPT
        self.paraphrase_worker = ParaphraseWorker(text_to_paraphrase)
        self.paraphrase_worker.result_ready.connect(self.on_paraphrase_ready)
        self.paraphrase_worker.error_occurred.connect(self.on_paraphrase_error)
        self.paraphrase_worker.start()
        
        # Asegurar que previous workers estén completamente terminados
        if hasattr(self, 'paraphrase_worker') and self.paraphrase_worker and self.paraphrase_worker != self.paraphrase_worker:
            old_worker = getattr(self, '_old_paraphrase_worker', None)
            if old_worker and old_worker.isRunning():
                old_worker.stop_processing()
                old_worker.terminate()
                old_worker.wait(1000)
        
        print(f"[DEBUG] Iniciando parafraseo del texto: {text_to_paraphrase}")

    def on_paraphrase_ready(self, paraphrased_text):
        """Callback cuando el parafraseo está listo"""
        print(f"[DEBUG] Texto parafraseado: {paraphrased_text}")
        
        # NUEVA FUNCIONALIDAD: Actualizar el área de predicciones con el texto parafraseado
        self.prediction_label.clear()  # Limpiar el contenido actual
        
        # Crear formato para el texto parafraseado (color verde para distinguir que fue procesado)
        cursor = self.prediction_label.textCursor()
        paraphrased_format = QTextCharFormat()
        paraphrased_format.setForeground(QColor("#90EE90"))  # Verde claro para indicar que es texto parafraseado
        cursor.setCharFormat(paraphrased_format)
        cursor.insertText(paraphrased_text)
        
        # Cambiar botón a estado de narración
        self.prediction_button.setText("NARRANDO...")
        
        # Iniciar narración con el texto parafraseado
        self.narration_worker = NarrationWorker(paraphrased_text)
        self.narration_worker.narration_finished.connect(self.on_narration_finished)
        self.narration_worker.narration_error.connect(self.on_narration_error)
        self.narration_worker.start()

    def on_paraphrase_error(self, error_message):
        """Callback cuando hay error en el parafraseo"""
        print(f"[DEBUG] Error en parafraseo: {error_message}")
        QMessageBox.warning(self, "Error de Parafraseo", f"No se pudo parafrasear el texto:\n{error_message}")
        self.reset_narration_button()

    def on_narration_finished(self):
        """Callback cuando la narración termina"""
        print("[DEBUG] Narración completada")
        self.reset_narration_button()

    def on_narration_error(self, error_message):
        """Callback cuando hay error en la narración"""
        print(f"[DEBUG] Error en narración: {error_message}")
        QMessageBox.warning(self, "Error de Narración", f"No se pudo narrar el texto:\n{error_message}")
        self.reset_narration_button()

    def stop_narration(self):
        """Detener el proceso de narración"""
        # Detener paraphrase_worker si está ejecutándose
        if self.paraphrase_worker and self.paraphrase_worker.isRunning():
            self.paraphrase_worker.terminate()
            self.paraphrase_worker.wait(1000)  # Esperar máximo 1 segundo
            
        # Detener narration_worker si está ejecutándose
        if self.narration_worker and self.narration_worker.isRunning():
            self.narration_worker.stop_narration()
            self.narration_worker.terminate()
            self.narration_worker.wait(1000)  # Esperar máximo 1 segundo
            
        self.reset_narration_button()

    def reset_narration_button(self):
        """Resetear el botón de narración a su estado original"""
        self.is_narrating = False
        self.prediction_button.setText("NARRAR")
        self.prediction_button.setEnabled(True)
        
        # Limpiar paraphrase_worker
        if self.paraphrase_worker:
            if self.paraphrase_worker.isRunning():
                self.paraphrase_worker.terminate()
                self.paraphrase_worker.wait(1000)
            self.paraphrase_worker.deleteLater()
            self.paraphrase_worker = None
            
        # Limpiar narration_worker
        if self.narration_worker:
            if self.narration_worker.isRunning():
                self.narration_worker.stop_narration()
                self.narration_worker.terminate()
                self.narration_worker.wait(1000)
            self.narration_worker.deleteLater()
            self.narration_worker = None
    
    def _cleanup_current_page(self):
        """Limpiar recursos de la página actual antes de cambiar"""
        print(f"[DEBUG] Limpiando recursos de página {self.current_page_index}...")
        
        # Página 0: Traducción - detener cámara y listener
        if self.current_page_index == 0:
            self._stop_translation_resources()
        
        # Página 1: Aprendizaje - detener módulo de aprendizaje
        elif self.current_page_index == 1:
            self._stop_learning_resources()
    
    def _start_page_resources(self, page_index):
        """Iniciar recursos específicos de la página"""
        print(f"[DEBUG] Iniciando recursos para página {page_index}...")
        
        # Página 0: Traducción - reiniciar cámara
        if page_index == 0:
            self._start_translation_resources()
        
        # Página 1: Aprendizaje - no necesita reiniciar nada (lazy loading)
        elif page_index == 1:
            print("[DEBUG] Módulo de aprendizaje usa lazy loading")
    
    def _start_translation_resources(self):
        """Iniciar/reiniciar recursos de traducción"""
        try:
            print("[DEBUG] Iniciando recursos de traducción...")
            
            # Solo reiniciar si no están activos
            if not self.capture_thread or not self.capture_thread.is_alive():
                if not self.timer or not self.timer.isActive():
                    self.start_camera()
                    print("[DEBUG] ✅ Recursos de traducción iniciados")
                else:
                    print("[DEBUG] ⚡ Recursos de traducción ya activos")
            else:
                print("[DEBUG] ⚡ Cámara ya en ejecución")
        except Exception as e:
            print(f"[ERROR] Error al iniciar recursos de traducción: {e}")
    
    def _stop_translation_resources(self):
        """Detener recursos de traducción"""
        try:
            # Detener timer de actualización de cámara
            if self.timer and self.timer.isActive():
                self.timer.stop()
                self.timer = None
            
            # Detener listener de MediaPipe
            if self.listener:
                print("[DEBUG] Deteniendo MediaPipeListener...")
                self.listener.stop()
                self.listener = None
            
            # Esperar a que termine el hilo de captura
            if self.capture_thread and self.capture_thread.is_alive():
                print("[DEBUG] Esperando hilo de captura...")
                self.capture_thread.join(timeout=2.0)
                self.capture_thread = None
            
            print("[DEBUG] Recursos de traducción liberados")
        except Exception as e:
            print(f"[ERROR] Error al limpiar recursos de traducción: {e}")
    
    def _stop_learning_resources(self):
        """Detener recursos del módulo de aprendizaje"""
        try:
            if self.learning_module:
                print("[DEBUG] Limpiando módulo de aprendizaje...")
                # El módulo de aprendizaje limpiará sus propios recursos
                if hasattr(self.learning_module, 'cleanup_resources'):
                    self.learning_module.cleanup_resources()
                # NO eliminamos self.learning_module para permitir cambio rápido de vuelta
                # self.learning_module = None
            print("[DEBUG] Recursos de aprendizaje pausados (no eliminados)")
        except Exception as e:
            print(f"[ERROR] Error al limpiar recursos de aprendizaje: {e}")
    
    def cleanup_threads(self):
        """Limpiar todos los hilos antes de cerrar la aplicación"""
        print("[DEBUG] Limpiando hilos antes de cerrar...")
        
        # Limpiar página actual
        self._cleanup_current_page()
        
        # Limpiar módulo de aprendizaje completamente
        if self.learning_module:
            print("[DEBUG] Limpieza final del módulo de aprendizaje...")
            if hasattr(self.learning_module, 'cleanup_resources'):
                self.learning_module.cleanup_resources()
            self.learning_module = None
            
            # Limpiar caché global de modelos
            try:
                from Aplicacion.learning_module import _model_cache
                _model_cache.cleanup()
            except Exception as e:
                print(f"[DEBUG] No se pudo limpiar caché de modelos: {e}")
        
        # Detener transcripción si está activa
        if self.is_transcribing and self.transcription_worker:
            self.stop_transcription()
        
        # Detener narración si está activa
        if self.is_narrating:
            self.stop_narration()
            
        # Forzar terminación de cualquier hilo restante
        for worker in [self.paraphrase_worker, self.narration_worker, self.transcription_worker]:
            if worker and worker.isRunning():
                worker.terminate()
                worker.wait(2000)  # Esperar máximo 2 segundos
        
        print("[DEBUG] Todos los hilos limpiados")

if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = GesturaApp()

    # Cambiar el color de la barra de título
    pywinstyles.change_header_color(window, color="#243b48")  # Color verde oscuro

    # Cambiar el color del borde de la ventana
    pywinstyles.change_border_color(window, color="#252a40")  # Borde color cian

    # Agregar el icono desde la carpeta de recursos
    icon_path = os.path.join(carpeta_recursos, 'empty_icon.png')  # Ruta completa al icono
    window.setWindowIcon(QIcon(icon_path))

    # Mostrar la ventana maximizada
    window.showMaximized()

    # Ejecutar la aplicación
    sys.exit(app.exec())