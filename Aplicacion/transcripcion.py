import os
import pyaudio
import time
import threading
from google.cloud import speech_v1p1beta1 as speech
from six.moves import queue  # type: ignore

# Configuración de audio
RATE = 16000  # Frecuencia de muestreo
CHUNK = int(RATE / 10)  # Dividimos la entrada en fragmentos de 100ms

# Duración máxima sin actividad (en segundos)
MAX_SILENCE_TIME = 5

class MicrophoneStream:
    """Captura el audio del micrófono y lo convierte en flujos de audio para la API"""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self, stop_event):
        """Genera fragmentos de audio y detiene el flujo si se activa el stop_event."""
        while not self.closed:
            if stop_event.is_set():  # Si el evento de cierre se activa, dejamos de enviar audio
                return
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None or stop_event.is_set():  # Detenemos el generador si se activa el stop_event
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)


def monitor_silence(last_activity_time_ref, stop_event, text_update_callback, partial_transcript):
    """Monitorea el tiempo de inactividad y termina si se excede el tiempo máximo de silencio."""
    while not stop_event.is_set():
        current_time = time.time()
        last_activity_time = last_activity_time_ref[0]
        time_since_last_activity = current_time - last_activity_time

        if time_since_last_activity > MAX_SILENCE_TIME:
            text_update_callback(partial_transcript[0], final=True)
            stop_event.set()  # Detenemos el stream después del tiempo de silencio
            break

        time.sleep(0.1)


def transcribe_streaming(text_update_callback, timer_callback):
    """Transcripción de audio en tiempo real"""
    # Usar la variable de entorno o valor por defecto
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 
                                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'credentials', 'google-cloud-speech.json'))
    
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"Google Cloud credentials file not found at: {credentials_path}")
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="es-ES",
        enable_automatic_punctuation=True
    )

    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

    last_activity_time_ref = [time.time()]
    stop_event = threading.Event()
    partial_transcript = [""]  # Guarda el estado continuo del texto parcial
    final_transcript = [""]  # Acumulador de todo el texto final

    # Iniciar el hilo que monitorea el silencio
    monitor_thread = threading.Thread(target=monitor_silence, args=(last_activity_time_ref, stop_event, text_update_callback, final_transcript))
    monitor_thread.start()

    start_time = time.time()  # Temporizador para la duración de la grabación
    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator(stop_event)  # Pasamos el stop_event al generador
        requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)

        try:
            responses = client.streaming_recognize(streaming_config, requests)

            for response in responses:
                if stop_event.is_set():  # Si se activa el evento de silencio, se detiene el stream
                    break

                if not response.results:
                    continue

                result = response.results[0]
                last_activity_time_ref[0] = time.time()  # Actualizar el tiempo de la última actividad

                # Actualizar temporizador
                elapsed_time = time.time() - start_time
                timer_callback(elapsed_time)

                # Actualizar el flujo de texto parcial continuo
                if result.is_final:
                    # Consolidar el texto final al acumulador
                    final_transcript[0] += " " + result.alternatives[0].transcript
                    text_update_callback(final_transcript[0], final=True)
                elif result.alternatives:
                    # Actualizar el parcial
                    partial_transcript[0] = result.alternatives[0].transcript
                    text_update_callback(final_transcript[0] + " " + partial_transcript[0], partial=True)
        
        except Exception as e:
            print(f"Error during speech recognition: {e}")  # Capturamos el error para evitar que el hilo muera
        
        finally:
            stop_event.set()  # Aseguramos que el evento de parada se active al finalizar
            stream.closed = True  # Cerramos explícitamente el stream
            monitor_thread.join()  # Esperar a que termine el monitoreo de silencio

