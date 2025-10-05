# moodscope.py
import os

os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
import sys
import cv2
import numpy as np
import sqlite3
import tensorflow as tf
from mtcnn import MTCNN
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QMutex, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QProgressBar, QFileDialog, QMessageBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time
from collections import deque, Counter
import urllib3
import pandas as pd
from PyQt5.QtWidgets import QStackedWidget
import requests
import json

# --- Disable Warnings ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- Metal Configuration (for macOS GPU acceleration) ---
try:
    tf.keras.backend.clear_session()
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.set_soft_device_placement(True)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("--- GPU detected:", gpus[0].name, "---")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print("--- Warning: No GPU detected. Using CPU. ---")
except Exception as e:
    print(f"Error during TensorFlow GPU configuration: {e}")
    print("--- Proceeding with CPU. ---")

# --- Application Configuration ---
MODEL_PATH = 'emotion_detection_model.h5'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
MODEL_INPUT_SIZE = (48, 48)
DATABASE_FILE = 'emotion_data.db'
FRAME_SKIP = 2
GRAPH_UPDATE_INTERVAL = 500
MAX_GRAPH_POINTS = 500
COLOR_PALETTE = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0),
    (0, 255, 255), (255, 0, 255), (255, 255, 0),
]


class ThreadSafeDB:
    """Thread-safe SQLite database handler using a Singleton pattern."""
    _instance = None
    _mutex = QMutex()

    def __new__(cls):
        if cls._instance is None:
            cls._mutex.lock()
            try:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.conn = None
                    cls._instance.c = None
                    cls._instance._init_connection()
            finally:
                cls._mutex.unlock()
        return cls._instance

    def _init_connection(self):
        """Initializes the database connection and cursor."""
        try:
            self._instance.conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
            self._instance.c = self._instance.conn.cursor()
            self._instance._init_db_schema()
            print(f"Database connection established: {DATABASE_FILE}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            self._instance.conn = None
            self._instance.c = None

    def _init_db_schema(self):
        """Creates the emotions table if it doesn't exist."""
        if self.c:
            try:
                self.c.execute('''CREATE TABLE IF NOT EXISTS emotions (
                                    timestamp REAL PRIMARY KEY,
                                    emotion TEXT,
                                    confidence REAL
                                 )''')
                self.conn.commit()
            except sqlite3.Error as e:
                print(f"Error creating database table: {e}")

    def log_emotion(self, emotion, confidence):
        """Logs the detected emotion and confidence with a timestamp."""
        if not self.conn or not self.c:
            print("Database connection not available. Cannot log emotion.")
            return

        self._mutex.lock()
        try:
            timestamp = time.time()
            self.c.execute("INSERT INTO emotions (timestamp, emotion, confidence) VALUES (?, ?, ?)",
                           (timestamp, emotion, confidence))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error logging emotion to database: {e}")
        finally:
            self._mutex.unlock()

    def get_recent_emotions(self, limit=MAX_GRAPH_POINTS):
        """Retrieves the most recent emotion records."""
        if not self.conn or not self.c:
            print("Database connection not available. Cannot retrieve emotions.")
            return []

        self._mutex.lock()
        try:
            self.c.execute("SELECT timestamp, emotion, confidence FROM emotions ORDER BY timestamp DESC LIMIT ?",
                           (limit,))
            results = self.c.fetchall()[::-1]
            return results
        except sqlite3.Error as e:
            print(f"Error retrieving emotions from database: {e}")
            return []
        finally:
            self._mutex.unlock()

    def export_to_csv(self, output_path):
        """Exports the emotions table to a CSV file at output_path."""
        if not self.conn or not self.c:
            print("Database connection not available. Cannot export.")
            return False
        self._mutex.lock()
        try:
            df = pd.read_sql_query("SELECT timestamp, emotion, confidence FROM emotions ORDER BY timestamp", self.conn)
            df.to_csv(output_path, index=False)
            print(f"Exported emotions to {output_path}")
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
        finally:
            self._mutex.unlock()

    def close(self):
        """Closes the database connection."""
        self._mutex.lock()
        try:
            if self.conn:
                self.conn.close()
                self.conn = None
                self.c = None
                print("Database connection closed.")
        except sqlite3.Error as e:
            print(f"Error closing database connection: {e}")
        finally:
            self._mutex.unlock()


def load_emotion_model(model_path=MODEL_PATH):
    """Loads the Keras emotion detection model with error handling."""
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return None

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✅ Emotion model loaded successfully from {model_path}")
        print(f"📐 Model input shape: {model.input_shape}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


class VideoProcessor(QThread):
    """QThread that captures video frames, performs face detection and emotion recognition."""
    frame_processed = pyqtSignal(QImage, list)
    emotion_logged = pyqtSignal()
    group_emotion = pyqtSignal(str)
    engagement_score = pyqtSignal(float)
    alert = pyqtSignal(str)

    def __init__(self, emotion_model, parent=None):
        super().__init__(parent)
        self.emotion_model = emotion_model
        self.detector = None
        self.running = False
        self.frame_count = 0
        self.db = ThreadSafeDB()
        self._recent_emotions = deque(maxlen=30)
        self.last_confidences = deque(maxlen=30)
        self.emotion_timestamps = deque(maxlen=300)  # Track emotion history with timestamps
        self.input_video_path = 0  # Default to webcam

    def _initialize_detector(self):
        """Initializes the MTCNN detector."""
        try:
            self.detector = MTCNN()
            print("MTCNN face detector initialized successfully.")
            return True
        except Exception as e:
            print(f"Error initializing MTCNN detector: {e}")
            return False

    def run(self):
        """Main processing loop."""
        self.running = True
        self.frame_count = 0

        if not self._initialize_detector():
            print("Failed to initialize face detector. Thread stopping.")
            self.running = False
            return

        if self.emotion_model is None:
            print("Emotion model not loaded. Thread stopping.")
            self.running = False
            return

        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            print("Error: Could not open video capture device.")
            self.running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Video capture started.")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("Warning: Failed to grab frame.")
                    time.sleep(0.1)
                    continue

                if self.frame_count % FRAME_SKIP == 0:
                    self.process_frame(frame)

                self.frame_count += 1

        except Exception as e:
            print(f"Error during video processing loop: {e}")
        finally:
            cap.release()
            print("Video capture released.")

    def process_frame(self, frame):
        """Detects faces, predicts emotions, draws results, and logs data."""
        detected_emotions_in_frame = []

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detector.detect_faces(rgb_frame)

            if not faces:
                h, w, ch = frame.shape
                qt_image = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
                self.frame_processed.emit(qt_image.copy(), [])
                return

            for face in faces:
                try:
                    x, y, w, h = face['box']
                    x, y = max(0, x), max(0, y)
                    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                    if w <= 0 or h <= 0:
                        continue

                    roi_rgb = rgb_frame[y:y2, x:x2]
                    if roi_rgb.size == 0:
                        continue

                    # Preprocess face ROI - Convert to grayscale for the model
                    roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
                    roi_resized = cv2.resize(roi_gray, MODEL_INPUT_SIZE)
                    roi_normalized = roi_resized.astype(np.float32) / 255.0
                    # Add channel dimension for grayscale: (48, 48) -> (48, 48, 1)
                    roi_input = np.expand_dims(roi_normalized, axis=-1)
                    # Add batch dimension: (48, 48, 1) -> (1, 48, 48, 1)
                    roi_input = np.expand_dims(roi_input, axis=0)

                    # Predict emotion
                    predictions = self.emotion_model.predict(roi_input, verbose=0)[0]
                    emotion_index = np.argmax(predictions)
                    emotion_label = EMOTION_LABELS[emotion_index]
                    confidence = float(np.max(predictions))

                    detected_emotions_in_frame.append((emotion_label, confidence))
                    self.db.log_emotion(emotion_label, confidence)
                    self.emotion_logged.emit()

                    # Track emotions with timestamps for time-based analysis
                    current_time = time.time()
                    self.emotion_timestamps.append((current_time, emotion_label, confidence))

                    # Track for engagement analysis
                    self._recent_emotions.append(emotion_label)
                    self.last_confidences.append(confidence)

                    # Draw bounding box and emotion label
                    color = (0, 255, 0)
                    text = f"{emotion_label} ({confidence:.2f})"
                    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x, max(20, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                except Exception as face_err:
                    print(f"Error processing face: {face_err}")
                    continue

            # Compute group dominant emotion
            if detected_emotions_in_frame:
                group_counts = Counter([e for e, _ in detected_emotions_in_frame])
                dominant_emotion = group_counts.most_common(1)[0][0]
                self.group_emotion.emit(dominant_emotion)

                # Alert for prolonged negative emotions (10 minutes)
                current_time = time.time()
                time_threshold = 600  # 10 minutes in seconds

                # Check each negative emotion type
                for negative_emotion in ['Sad', 'Angry', 'Fear']:
                    # Filter emotions from the last 10 minutes
                    recent_negative = [
                        (t, e, c) for t, e, c in self.emotion_timestamps
                        if e == negative_emotion and (current_time - t) <= time_threshold
                    ]

                    # If we have continuous negative emotion detection
                    if len(recent_negative) > 0:
                        # Calculate what percentage of the last 10 minutes showed this emotion
                        # Check if the emotion has been consistently present
                        oldest_time = recent_negative[0][0]
                        time_span = current_time - oldest_time

                        # Alert if emotion persisted for more than 8 minutes out of last 10
                        # (allows for brief interruptions)
                        if time_span >= 480 and len(recent_negative) >= 40:  # ~40 detections over 8 min
                            self.alert.emit(f"⚠️ Prolonged {negative_emotion} detected for over 8 minutes!")
                            # Clear old entries to avoid repeated alerts
                            self.emotion_timestamps = deque(
                                [(t, e, c) for t, e, c in self.emotion_timestamps if (current_time - t) <= 60],
                                maxlen=300
                            )
                            break

                # Compute engagement score
                avg_confidence = np.mean([c for _, c in detected_emotions_in_frame])
                engagement = min(100, avg_confidence * 100)
                self.engagement_score.emit(engagement)

                # Detect emotional volatility
                if len(self.last_confidences) >= 5:
                    std_conf = np.std(self.last_confidences)
                    mean_conf = np.mean(self.last_confidences)

                    if std_conf > 0.25:
                        self.alert.emit("⚠️ High emotional volatility detected!")
                    elif mean_conf < 0.4:
                        self.alert.emit("😟 Low emotional intensity — possible disengagement.")

            # Emit processed frame to UI
            h, w, ch = frame.shape
            qt_image = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
            self.frame_processed.emit(qt_image.copy(), detected_emotions_in_frame)

        except Exception as e:
            print(f"Error in process_frame: {e}")
            try:
                h, w, ch = frame.shape
                qt_image = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
                self.frame_processed.emit(qt_image.copy(), [])
            except Exception:
                pass

    def stop(self):
        """Sets the running flag to False to stop the thread."""
        print("Stopping video processor thread...")
        self.running = False


class EmotionGraphCanvas(FigureCanvas):
    """Matplotlib canvas widget to display emotion confidence over time."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.timestamps = deque(maxlen=MAX_GRAPH_POINTS)
        self.confidences = {label: deque(maxlen=MAX_GRAPH_POINTS) for label in EMOTION_LABELS}
        self.lines = {}

        self.setup_plot()

    def setup_plot(self):
        """Initial setup of the plot axes and lines."""
        self.axes.set_title("Emotion Confidence Over Time")
        self.axes.set_xlabel("Time (seconds ago)")
        self.axes.set_ylabel("Confidence")
        self.axes.set_ylim(0, 1.05)
        self.axes.grid(True)

        for label in EMOTION_LABELS:
            line, = self.axes.plot([], [], label=label, marker='.', linestyle='-')
            self.lines[label] = line

        self.axes.legend(loc='upper left', fontsize='small')
        self.fig.tight_layout()

    def update_plot(self, data, max_seconds=60):
        """
        Updates the graph with continuous data over time.
        :param data: List of (timestamp, emotion, confidence)
        :param max_seconds: How many seconds back to display (default 60)
        """
        if not data:
            return

        current_time = time.time()

        # Filter data to last max_seconds
        recent_data = [(t, e, c) for t, e, c in data if current_time - t <= max_seconds]

        if not recent_data:
            return

        # Organize data by emotion
        emotion_data = {label: {'times': [], 'confidences': []} for label in EMOTION_LABELS}

        for timestamp, emotion, confidence in recent_data:
            seconds_ago = current_time - timestamp
            emotion_data[emotion]['times'].append(seconds_ago)
            emotion_data[emotion]['confidences'].append(confidence)

        # Update lines
        for label, line in self.lines.items():
            times = emotion_data[label]['times']
            confidences = emotion_data[label]['confidences']

            if times:
                # Sort by time for proper line plotting
                sorted_pairs = sorted(zip(times, confidences))
                sorted_times, sorted_confidences = zip(*sorted_pairs)
                line.set_data(sorted_times, sorted_confidences)
            else:
                line.set_data([], [])

        # Set axes limits dynamically
        if recent_data:
            self.axes.set_xlim(0, max_seconds)
            self.axes.set_ylim(0, 1.05)
            self.axes.invert_xaxis()  # Most recent (0) on right

        self.draw()


class StartupScreen(QWidget):
    def __init__(self, on_start_detection, on_import_csv):
        super().__init__()
        from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
        from PyQt5.QtCore import Qt
        import pandas as pd

        self.on_start_detection = on_start_detection
        self.on_import_csv = on_import_csv

        self.setWindowTitle("MoodScope - Welcome")
        self.setStyleSheet("background-color: #121212; color: white; font-family: 'Arial';")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("MoodScope")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 36px; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel("Understand emotions in real time or through reflection.")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 16px; color: #ccc;")
        layout.addWidget(subtitle)

        layout.addSpacing(40)

        start_btn = QPushButton("🎥 Start Real-Time Detection")
        import_btn = QPushButton("📂 Import Emotion Log (CSV)")
        exit_btn = QPushButton(" Exit")

        layout.addWidget(start_btn)
        layout.addWidget(import_btn)
        layout.addWidget(exit_btn)

        start_btn.clicked.connect(self.on_start_detection)
        import_btn.clicked.connect(self.import_csv)
        exit_btn.clicked.connect(self.close)

    def import_csv(self):
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        import pandas as pd
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
            if not {'timestamp', 'emotion', 'confidence'}.issubset(df.columns):
                QMessageBox.warning(self, "Invalid CSV", "CSV must have columns: timestamp, emotion, confidence.")
                return
            self.on_import_csv(df, file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {e}")


class OllamaInsights:
    """Handler for Ollama AI-powered insights generation"""

    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama3.2"  # or llama3, mistral, etc.
        self.available = self.check_availability()

    def check_availability(self):
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def generate_insights(self, analysis, df):
        """Generate AI-powered insights using Ollama"""
        if not self.available:
            print("⚠️ Ollama not available. Using rule-based insights.")
            return None

        # Prepare comprehensive data summary
        emotion_breakdown = "\n".join([f"  • {k}: {v:.1f}%" for k, v in analysis['emotion_percentages'].items()])

        prompt = f"""You are a compassionate mental health and wellness advisor. Analyze this emotion tracking data and provide personalized insights.

EMOTION DATA SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Records: {analysis['total_records']}
Dominant Emotion: {analysis['dominant_emotion']}
Average Confidence: {analysis['avg_confidence']:.1%}

Emotion Distribution:
{emotion_breakdown}

Positive Emotions: {analysis['positive_percentage']:.1f}%
Negative Emotions: {analysis['negative_percentage']:.1f}%
Neutral: {analysis['neutral_percentage']:.1f}%

Emotional Volatility: {analysis['emotion_volatility']:.2f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Please provide:
1. **Key Insights** (2-3 observations about their emotional patterns)
2. **Wellness Assessment** (overall mental health state)
3. **Actionable Recommendations** (3-5 specific, practical tips)

Be empathetic, supportive, and focus on actionable advice. Keep response under 400 words."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                print(f"Ollama API error: {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            print("⚠️ Ollama request timed out")
            return None
        except Exception as e:
            print(f"⚠️ Ollama error: {e}")
            return None

    def generate_wellness_tips(self, analysis):
        """Generate personalized wellness tips using Ollama"""
        if not self.available:
            return None

        prompt = f"""Based on this emotional profile, provide 5 specific, actionable wellness tips:

Dominant Emotion: {analysis['dominant_emotion']}
Negative Emotions: {analysis['negative_percentage']:.1f}%
Positive Emotions: {analysis['positive_percentage']:.1f}%

Give practical, specific advice that someone can implement today. Format as a numbered list. Keep it under 250 words."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7}
                },
                timeout=20
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            return None
        except Exception as e:
            print(f"⚠️ Ollama wellness tips error: {e}")
            return None


class CSVAnalysisWindow(QWidget):
    def __init__(self, df, file_path):
        super().__init__()
        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QTextEdit
        from PyQt5.QtCore import Qt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        import pandas as pd

        self.setWindowTitle(f"Analysis - {file_path}")
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")

        # Initialize Ollama
        self.ollama = OllamaInsights()
        if self.ollama.available:
            print("✅ Ollama AI available - using AI-powered insights")
        else:
            print("ℹ️ Ollama not available - using rule-based insights")

        main_layout = QHBoxLayout(self)

        # LEFT SIDE: Graph and Stats
        left_layout = QVBoxLayout()

        title = QLabel("📊 Emotion Analysis Report")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #4CAF50; padding: 10px;")
        left_layout.addWidget(title)

        # Analyze the data
        analysis = self.analyze_emotions(df)

        # Stats summary
        stats_label = QLabel(self.format_stats(analysis))
        stats_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #2d2d2d; border-radius: 5px;")
        stats_label.setWordWrap(True)
        left_layout.addWidget(stats_label)

        # Plot
        fig = Figure(figsize=(7, 5))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor('#2d2d2d')
        fig.patch.set_facecolor('#1e1e1e')

        df_copy = df.copy()
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], unit='s', errors='coerce')

        colors = {
            'Happy': '#4CAF50', 'Sad': '#2196F3', 'Angry': '#F44336',
            'Fear': '#9C27B0', 'Surprise': '#FF9800', 'Disgust': '#795548',
            'Neutral': '#9E9E9E'
        }

        for e in df_copy['emotion'].unique():
            sub = df_copy[df_copy['emotion'] == e]
            ax.plot(sub['timestamp'], sub['confidence'],
                    label=e, color=colors.get(e, '#ffffff'), linewidth=2, marker='o', markersize=3)

        ax.legend(facecolor='#2d2d2d', edgecolor='#4CAF50', loc='upper left')
        ax.set_title("Emotion Confidence Over Time", color='white', fontsize=14, pad=10)
        ax.set_xlabel("Time", color='white')
        ax.set_ylabel("Confidence", color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        left_layout.addWidget(canvas)

        # RIGHT SIDE: Insights and Recommendations
        right_layout = QVBoxLayout()

        # AI Status indicator
        ai_status = QLabel("🤖 AI-Powered Analysis" if self.ollama.available else "📊 Rule-Based Analysis")
        ai_status.setStyleSheet(f"""
            font-size: 12px; 
            color: {'#4CAF50' if self.ollama.available else '#FF9800'}; 
            padding: 5px;
            background-color: #2d2d2d;
            border-radius: 3px;
        """)
        ai_status.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(ai_status)

        insights_title = QLabel("💡 Insights & Recommendations")
        insights_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #FF9800; padding: 10px;")
        right_layout.addWidget(insights_title)

        # Scrollable insights area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background-color: #2d2d2d;")

        insights_widget = QWidget()
        insights_layout = QVBoxLayout(insights_widget)

        # Generate insights with AI or fallback
        if self.ollama.available:
            ai_insights = self.ollama.generate_insights(analysis, df)
            if ai_insights:
                insights_text = self.format_ai_insights(ai_insights)
            else:
                insights_text = self.generate_insights(analysis)  # Fallback
        else:
            insights_text = self.generate_insights(analysis)

        insights_label = QLabel(insights_text)
        insights_label.setWordWrap(True)
        insights_label.setStyleSheet("""
            font-size: 13px; 
            padding: 15px; 
            line-height: 1.6;
            background-color: #2d2d2d;
        """)
        insights_layout.addWidget(insights_label)

        # Wellness tips
        tips_title = QLabel("🌟 Personalized Wellness Tips")
        tips_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50; padding: 10px; margin-top: 20px;")
        insights_layout.addWidget(tips_title)

        # Generate wellness tips with AI or fallback
        if self.ollama.available:
            ai_tips = self.ollama.generate_wellness_tips(analysis)
            if ai_tips:
                tips_text = self.format_ai_tips(ai_tips)
            else:
                tips_text = self.generate_wellness_tips(analysis)  # Fallback
        else:
            tips_text = self.generate_wellness_tips(analysis)

        tips_label = QLabel(tips_text)
        tips_label.setWordWrap(True)
        tips_label.setStyleSheet("""
            font-size: 13px; 
            padding: 15px;
            background-color: #1e1e1e;
            border-left: 4px solid #4CAF50;
            line-height: 1.6;
        """)
        insights_layout.addWidget(tips_label)

        insights_layout.addStretch()
        scroll.setWidget(insights_widget)
        right_layout.addWidget(scroll)

        # Back button
        back_btn = QPushButton("⬅️ Back to Home")
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        back_btn.clicked.connect(self.close)
        right_layout.addWidget(back_btn)

        # Add layouts to main
        main_layout.addLayout(left_layout, 6)
        main_layout.addLayout(right_layout, 4)

    def analyze_emotions(self, df):
        """Analyze emotion patterns and return insights data"""
        import pandas as pd
        import numpy as np

        analysis = {}

        # Basic stats
        analysis['total_records'] = len(df)
        analysis['dominant_emotion'] = df['emotion'].value_counts().idxmax()
        analysis['avg_confidence'] = df['confidence'].mean()

        # Emotion distribution
        analysis['emotion_counts'] = df['emotion'].value_counts().to_dict()
        analysis['emotion_percentages'] = (df['emotion'].value_counts() / len(df) * 100).to_dict()

        # Negative emotions
        negative = ['Sad', 'Angry', 'Fear', 'Disgust']
        analysis['negative_percentage'] = (df[df['emotion'].isin(negative)].shape[0] / len(df)) * 100

        # Positive emotions
        positive = ['Happy', 'Surprise']
        analysis['positive_percentage'] = (df[df['emotion'].isin(positive)].shape[0] / len(df)) * 100

        # Neutral
        analysis['neutral_percentage'] = (df[df['emotion'] == 'Neutral'].shape[0] / len(df)) * 100

        # Time-based patterns
        df_copy = df.copy()
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], unit='s', errors='coerce')
        df_copy['hour'] = df_copy['timestamp'].dt.hour

        if not df_copy['hour'].isna().all():
            hour_emotions = df_copy.groupby('hour')['emotion'].agg(lambda x: x.value_counts().idxmax())
            analysis['peak_hours'] = hour_emotions.to_dict()
        else:
            analysis['peak_hours'] = {}

        # Volatility
        analysis['emotion_volatility'] = len(df['emotion'].unique()) / len(df) * 100

        # Longest streaks
        analysis['longest_streak'] = self.find_longest_streak(df)

        return analysis

    def find_longest_streak(self, df):
        """Find the longest consecutive streak of each emotion"""
        streaks = {}
        current_emotion = None
        current_count = 0

        for emotion in df['emotion']:
            if emotion == current_emotion:
                current_count += 1
            else:
                if current_emotion:
                    streaks[current_emotion] = max(streaks.get(current_emotion, 0), current_count)
                current_emotion = emotion
                current_count = 1

        if current_emotion:
            streaks[current_emotion] = max(streaks.get(current_emotion, 0), current_count)

        return streaks

    def format_ai_insights(self, ai_text):
        """Format AI-generated insights for HTML display"""
        # Convert markdown-style formatting to HTML
        formatted = ai_text.replace('\n\n', '<br><br>')
        formatted = formatted.replace('\n', '<br>')

        # Bold headers (e.g., **Key Insights**)
        import re
        formatted = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', formatted)

        # Add some styling
        formatted = f"<div style='line-height: 1.8;'>{formatted}</div>"
        return formatted

    def format_ai_tips(self, ai_text):
        """Format AI-generated wellness tips for HTML display"""
        formatted = ai_text.replace('\n\n', '<br><br>')
        formatted = formatted.replace('\n', '<br>')

        import re
        formatted = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', formatted)
        formatted = re.sub(r'\*(.+?)\*', r'<i>\1</i>', formatted)

        return formatted

    def format_stats(self, analysis):
        """Format basic statistics"""
        text = f"""
<b>📈 Overview</b><br>
• Total Records: {analysis['total_records']}<br>
• Dominant Emotion: <span style='color: #4CAF50;'>{analysis['dominant_emotion']}</span><br>
• Average Confidence: {analysis['avg_confidence']:.2%}<br><br>

<b>🎭 Emotion Distribution</b><br>
• Positive Emotions: <span style='color: #4CAF50;'>{analysis['positive_percentage']:.1f}%</span><br>
• Negative Emotions: <span style='color: #F44336;'>{analysis['negative_percentage']:.1f}%</span><br>
• Neutral: <span style='color: #9E9E9E;'>{analysis['neutral_percentage']:.1f}%</span>
        """
        return text.strip()

    def generate_insights(self, analysis):
        """Generate personalized insights based on emotion data"""
        insights = []

        # Overall emotional state
        if analysis['positive_percentage'] > 60:
            insights.append(
                "✨ <b>Great news!</b> Your emotional state appears predominantly positive. You're experiencing more happiness and pleasant emotions than negative ones.")
        elif analysis['negative_percentage'] > 60:
            insights.append(
                "⚠️ <b>Attention needed:</b> A significant portion of your emotions are negative. This might indicate stress, anxiety, or challenging circumstances that deserve attention.")
        else:
            insights.append(
                "⚖️ <b>Balanced state:</b> You're experiencing a mix of positive and negative emotions, which is normal. However, there's room for improvement.")

        # Dominant emotion insights
        dom = analysis['dominant_emotion']
        dom_pct = analysis['emotion_percentages'][dom]

        insights.append(f"<br>🎯 <b>Dominant Pattern:</b> '{dom}' appears {dom_pct:.1f}% of the time.")

        if dom == 'Happy':
            insights.append(
                "This suggests you're in a generally positive mental state. Keep nurturing the activities and relationships that bring you joy!")
        elif dom == 'Sad':
            insights.append(
                "Persistent sadness may indicate depression or grief. Consider reaching out to friends, family, or a mental health professional.")
        elif dom == 'Angry':
            insights.append(
                "Frequent anger can affect your relationships and health. Identifying triggers and practicing anger management techniques could be beneficial.")
        elif dom == 'Fear':
            insights.append(
                "High levels of fear might indicate anxiety. Consider relaxation techniques, mindfulness, or speaking with a counselor.")
        elif dom == 'Neutral':
            insights.append(
                "A neutral state is stable but might lack emotional engagement. Consider activities that bring more joy and excitement to your life.")

        # Volatility
        if analysis['emotion_volatility'] > 0.3:
            insights.append(
                "<br>🌊 <b>Emotional Volatility:</b> Your emotions shift frequently, which might indicate sensitivity to external factors or internal processing. Grounding techniques may help.")

        # Longest streak
        if analysis['longest_streak']:
            longest = max(analysis['longest_streak'].items(), key=lambda x: x[1])
            if longest[1] > 20:
                insights.append(
                    f"<br>⏱️ <b>Notable Pattern:</b> You experienced '{longest[0]}' for an extended period ({longest[1]} consecutive detections). Consider what triggered or sustained this emotional state.")

        return "<br>".join(insights)

    def generate_wellness_tips(self, analysis):
        """Generate actionable wellness recommendations"""
        tips = []

        # Based on dominant emotion
        dom = analysis['dominant_emotion']

        if dom in ['Sad', 'Fear', 'Angry']:
            tips.append(
                "<b>🧘 Mindfulness & Meditation</b><br>Practice 10-15 minutes of daily meditation using apps like Headspace or Calm. Focus on breath awareness and body scans.")
            tips.append(
                "<br><b>🏃 Physical Activity</b><br>Exercise releases endorphins that naturally improve mood. Aim for 30 minutes of moderate activity daily—even a walk helps!")
            tips.append(
                "<br><b>💬 Social Connection</b><br>Reach out to a trusted friend or family member. Social support is crucial for emotional wellbeing.")

        if analysis['negative_percentage'] > 50:
            tips.append(
                "<br><b>📝 Journaling</b><br>Write down your thoughts and feelings daily. This helps process emotions and identify patterns or triggers.")
            tips.append(
                "<br><b>🎯 Professional Support</b><br>Consider speaking with a therapist or counselor. There's no shame in seeking professional help—it's a sign of strength.")

        if dom == 'Angry':
            tips.append(
                "<br><b>🧊 Cooling Techniques</b><br>When angry, take a 5-minute break. Practice deep breathing: inhale for 4, hold for 4, exhale for 4.")
            tips.append(
                "<br><b>🥊 Physical Outlet</b><br>Channel anger through exercise, boxing, or other physical activities. Punching a bag is better than punching walls (or people)!")

        if dom == 'Sad':
            tips.append(
                "<br><b>☀️ Light Therapy</b><br>Get 15-30 minutes of sunlight daily. Sunlight boosts serotonin and can improve mood significantly.")
            tips.append(
                "<br><b>🎵 Music & Arts</b><br>Engage with uplifting music, art, or creative hobbies. Creative expression can be therapeutic.")

        if dom == 'Fear':
            tips.append(
                "<br><b>🌬️ Breathing Exercises</b><br>Practice 4-7-8 breathing: inhale for 4 seconds, hold for 7, exhale for 8. This activates your parasympathetic nervous system.")
            tips.append(
                "<br><b>🛡️ Grounding Techniques</b><br>Use the 5-4-3-2-1 method: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.")

        if analysis['positive_percentage'] > 60:
            tips.append(
                "<br><b>🌱 Maintain Your Positivity</b><br>You're doing great! Continue the habits that are working. Consider keeping a gratitude journal to sustain this positive momentum.")

        # General tips
        tips.append(
            "<br><b>😴 Sleep Hygiene</b><br>Aim for 7-9 hours of quality sleep. Maintain a consistent schedule and avoid screens 1 hour before bed.")
        tips.append(
            "<br><b>🥗 Nutrition</b><br>Eat balanced meals with omega-3s (fish, nuts), complex carbs, and plenty of water. Nutrition directly impacts mood.")
        tips.append(
            "<br><b>📱 Digital Detox</b><br>Limit social media and news consumption. Constant connectivity can increase anxiety and decrease wellbeing.")

        return "".join(tips)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroScope - Real-Time Emotion Detection")
        self.setGeometry(50, 50, 1600, 900)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left Side: Video Feed
        self.video_layout = QVBoxLayout()
        self.video_label = QLabel("Initializing Video...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_layout.addWidget(self.video_label)

        # Right Side: Controls and Graph
        self.controls_graph_layout = QVBoxLayout()

        self.graph_canvas = EmotionGraphCanvas(self.central_widget)
        self.controls_graph_layout.addWidget(self.graph_canvas)

        self.dominant_label = QLabel("Dominant: N/A")
        self.dominant_label.setAlignment(Qt.AlignCenter)
        self.controls_graph_layout.addWidget(self.dominant_label)

        self.engagement_bar = QProgressBar()
        self.engagement_bar.setRange(0, 100)
        self.engagement_bar.setValue(100)
        self.controls_graph_layout.addWidget(self.engagement_bar)

        controls_row = QHBoxLayout()
        self.export_btn = QPushButton("Export CSV")
        self.upload_btn = QPushButton("Upload Video")
        controls_row.addWidget(self.export_btn)
        controls_row.addWidget(self.upload_btn)
        self.controls_graph_layout.addLayout(controls_row)

        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.controls_graph_layout.addWidget(self.status_label)

        self.main_layout.addLayout(self.video_layout, 5)
        self.main_layout.addLayout(self.controls_graph_layout, 5)

        # Load Model
        self.emotion_model = load_emotion_model()

        if self.emotion_model is None:
            self.status_label.setText("Status: Error loading emotion model!")
            self.video_label.setText("Error: Model not loaded. Cannot start video processing.")
            return

        # Video Processing Thread
        self.video_thread = VideoProcessor(self.emotion_model, self)
        self.video_thread.frame_processed.connect(self.update_video_feed)
        self.video_thread.emotion_logged.connect(self.schedule_graph_update)
        self.video_thread.group_emotion.connect(lambda e: self.dominant_label.setText(f"Dominant: {e}"))
        self.video_thread.engagement_score.connect(lambda s: self.engagement_bar.setValue(int(s)))
        self.video_thread.alert.connect(self.show_alert)

        self.export_btn.clicked.connect(self.export_csv)
        self.upload_btn.clicked.connect(self.upload_video)

        # Graph Update Timer
        self.graph_update_timer = QTimer(self)
        self.graph_update_timer.timeout.connect(self.update_emotion_graph)
        self.needs_graph_update = False

        # Start Processing
        self.video_thread.start()
        self.graph_update_timer.start(GRAPH_UPDATE_INTERVAL)
        self.status_label.setText("Status: Running")

    @pyqtSlot(QImage, list)
    def update_video_feed(self, qt_image, emotions):
        """Updates the video feed QLabel with the new frame."""
        try:
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(),
                                                     Qt.KeepAspectRatio,
                                                     Qt.SmoothTransformation))
        except Exception as e:
            print(f"Error updating video feed: {e}")

    @pyqtSlot()
    def schedule_graph_update(self):
        """Sets a flag indicating the graph needs updating."""
        self.needs_graph_update = True

    @pyqtSlot()
    def update_emotion_graph(self):
        """Fetches recent data from DB and updates the graph if needed."""
        if self.needs_graph_update:
            try:
                db = ThreadSafeDB()
                # Fetch more data points to show longer history
                recent_data = db.get_recent_emotions(limit=500)
                if recent_data:
                    # Display last 60 seconds by default (you can adjust this)
                    self.graph_canvas.update_plot(recent_data, max_seconds=60)
                    self.needs_graph_update = False
            except Exception as e:
                print(f"Error updating emotion graph: {e}")

    def export_csv(self):
        """Export the emotion DB to CSV via file dialog."""
        try:
            db = ThreadSafeDB()
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getSaveFileName(self, "Export Emotions to CSV",
                                                      "emotions.csv",
                                                      "CSV Files (*.csv);;All Files (*)",
                                                      options=options)
            if filename:
                ok = db.export_to_csv(filename)
                if ok:
                    QMessageBox.information(self, "Export", f"Exported emotions to {filename}")
                else:
                    QMessageBox.warning(self, "Export", "Failed to export emotions. See console for details.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Unexpected error: {e}")

    def upload_video(self):
        """Allow user to pick a video file and process it instead of the webcam."""
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Select Video File", "",
                                                  "Video Files (*.mp4 *.avi *.mov);;All Files (*)",
                                                  options=options)
        if not filename:
            return

        if self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()

        self.video_thread = VideoProcessor(self.emotion_model, self)
        self.video_thread.input_video_path = filename
        self.video_thread.frame_processed.connect(self.update_video_feed)
        self.video_thread.emotion_logged.connect(self.schedule_graph_update)
        self.video_thread.group_emotion.connect(lambda e: self.dominant_label.setText(f"Dominant: {e}"))
        self.video_thread.engagement_score.connect(lambda s: self.engagement_bar.setValue(int(s)))
        self.video_thread.alert.connect(self.show_alert)
        self.video_thread.start()
        self.status_label.setText(f"Processing video: {filename}")

    def show_alert(self, message):
        QMessageBox.warning(self, "Alert", message)

    def closeEvent(self, event):
        """Handles the window closing event."""
        print("Closing application...")
        self.graph_update_timer.stop()

        if self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()

        db = ThreadSafeDB()
        db.close()

        print("Application closed.")
        event.accept()


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("MoodScope")

    # Create stacked widget for navigation
    from PyQt5.QtWidgets import QStackedWidget

    stacked = QStackedWidget()


    def start_detection():
        main_window = MainWindow()
        stacked.addWidget(main_window)
        stacked.setCurrentWidget(main_window)


    def open_csv_analysis(df, path):
        csv_window = CSVAnalysisWindow(df, path)
        stacked.addWidget(csv_window)
        stacked.setCurrentWidget(csv_window)


    startup = StartupScreen(start_detection, open_csv_analysis)
    stacked.addWidget(startup)
    stacked.setCurrentWidget(startup)
    stacked.show()
    sys.exit(app.exec_())
