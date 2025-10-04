#neuroscope.py
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
                           QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time
from collections import deque
import urllib3

# --- Disable Warnings ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# Suppress TensorFlow logging (1 = INFO, 2 = WARNING, 3 = ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- Metal Configuration (for macOS GPU acceleration) ---
try:
    tf.keras.backend.clear_session()
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.set_soft_device_placement(True)

    # Verify GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("--- GPU detected:", gpus[0].name, "---")
        # Allow memory growth to avoid allocating all GPU memory at once
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print("--- Warning: No GPU detected. Using CPU. ---")
        print("--- Ensure TensorFlow-Metal is installed correctly for GPU acceleration on macOS. ---")
except Exception as e:
    print(f"Error during TensorFlow GPU configuration: {e}")
    print("--- Proceeding with CPU. ---")


# --- Application Configuration ---
MODEL_PATH = 'emotion_detection_model.h5' # Path to your trained Keras model
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
MODEL_INPUT_SIZE = (48, 48) # Input size expected by the emotion model
DATABASE_FILE = 'emotion_data.db' # SQLite database file
FRAME_SKIP = 2 # Process every Nth frame to save resources (1 = process all)
GRAPH_UPDATE_INTERVAL = 500 # Milliseconds between graph updates
MAX_GRAPH_POINTS = 50 # Maximum number of data points to show on the graph

class ThreadSafeDB:
    """
    Thread-safe SQLite database handler using a Singleton pattern.
    Ensures only one connection is made and operations are serialized.
    """
    _instance = None
    _mutex = QMutex() # Mutex to protect database access from multiple threads

    def __new__(cls):
        # Use double-checked locking for efficiency
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
            # check_same_thread=False is needed because the DB will be accessed
            # from the VideoProcessor thread, different from the main thread.
            # The mutex handles the thread safety.
            self._instance.conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
            self._instance.c = self._instance.conn.cursor()
            self._instance._init_db_schema()
            print(f"Database connection established: {DATABASE_FILE}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            self._instance.conn = None # Ensure connection is None if failed
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
            # print(f"Logged: {timestamp}, {emotion}, {confidence:.2f}") # Optional: for debugging
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
            # Fetch the latest 'limit' records ordered by timestamp
            self.c.execute("SELECT timestamp, emotion, confidence FROM emotions ORDER BY timestamp DESC LIMIT ?", (limit,))
            # Fetchall and reverse to get chronological order for plotting
            results = self.c.fetchall()[::-1]
            return results
        except sqlite3.Error as e:
            print(f"Error retrieving emotions from database: {e}")
            return []
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


def load_emotion_model(model_path):
    """
    Loads the Keras emotion detection model.
    Includes error handling and basic input shape validation.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path, compile=False) # compile=False speeds up loading if not retraining
        print(f"Emotion model loaded successfully from {model_path}")

        # Optional: Check input shape if needed (some models might not have it defined explicitly)
        # This adds an Input layer if the loaded model doesn't have one defined.
        if not hasattr(model, '_input_shape') and not model.inputs:
             print("Model lacks explicit input shape. Adding Input layer.")
             input_shape = (*MODEL_INPUT_SIZE, 1) # Grayscale image
             new_model = tf.keras.Sequential([
                 tf.keras.layers.InputLayer(input_shape=input_shape, name="input_layer"),
                 model
             ])
             model = new_model
             # Re-check input shape
             # print(f"Model input shape after adding InputLayer: {model.input_shape}")

        # You might want to print the actual expected input shape
        # print(f"Model expected input shape: {model.input_shape}")

        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

class VideoProcessor(QThread):
    """
    QThread that captures video frames, performs face detection and emotion recognition.
    Emits signals to update the UI.
    """
    # Signals:
    # frame_processed: Emits the processed QImage and a list of detected (emotion, confidence) tuples
    frame_processed = pyqtSignal(QImage, list)
    # emotion_logged: Signal that an emotion was logged (can be used for graph updates)
    emotion_logged = pyqtSignal()

    def __init__(self, emotion_model, parent=None):
        super().__init__(parent)
        self.emotion_model = emotion_model
        self.detector = None # Initialize MTCNN later to handle potential errors
        self.running = False
        self.frame_count = 0
        self.db = ThreadSafeDB() # Get the singleton DB instance

    def _initialize_detector(self):
        """Initializes the MTCNN detector."""
        try:
            self.detector = MTCNN()
            print("MTCNN face detector initialized successfully.")
            return True
        except Exception as e:
            print(f"Error initializing MTCNN detector: {e}")
            # Potentially try alternative initialization or handle the error
            # For example, check TensorFlow/CUDA compatibility if using GPU MTCNN
            return False

    def run(self):
        """Main processing loop."""
        self.running = True
        self.frame_count = 0

        if not self._initialize_detector():
            print("Failed to initialize face detector. Thread stopping.")
            self.running = False
            return # Stop the thread if detector fails

        if self.emotion_model is None:
            print("Emotion model not loaded. Thread stopping.")
            self.running = False
            return # Stop if model isn't loaded

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video capture device.")
            self.running = False
            return

        # Optional: Set camera properties (might not work on all cameras/OS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Video capture started.")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("Warning: Failed to grab frame.")
                    time.sleep(0.1) # Avoid busy-waiting if frames fail
                    continue

                # Process frame only if frame skip condition is met
                if self.frame_count % FRAME_SKIP == 0:
                    self.process_frame(frame)

                self.frame_count += 1
                # Small sleep to yield control and prevent 100% CPU usage if frame rate is very high
                # Adjust as needed based on performance
                # time.sleep(0.01)

        except Exception as e:
            print(f"Error during video processing loop: {e}")
        finally:
            cap.release()
            print("Video capture released.")
            # self.db.close() # Close DB when app closes, not when thread stops

    def process_frame(self, frame):
        """Detects faces, predicts emotions, draws results, and logs data."""
        detected_emotions_in_frame = []
        try:
            # Convert frame to RGB (MTCNN expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces using MTCNN
            faces = self.detector.detect_faces(rgb_frame)

            for face in faces:
                if face['confidence'] < 0.90: # Confidence threshold for face detection
                     continue

                x, y, w, h = face['box']
                # Ensure coordinates are valid and within frame boundaries
                x, y = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                w, h = x2 - x, y2 - y # Recalculate width/height after clamping

                if w <= 0 or h <= 0: # Skip invalid boxes
                    continue

                try:
                    # Extract Region of Interest (ROI) - the detected face
                    roi_rgb = rgb_frame[y:y2, x:x2]

                    # Preprocess ROI for the emotion model
                    # 1. Convert to Grayscale
                    roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
                    # 2. Resize to model's expected input size
                    roi_resized = cv2.resize(roi_gray, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)
                    # 3. Normalize pixel values (usually to [0, 1])
                    roi_normalized = roi_resized.astype(np.float32) / 255.0
                    # 4. Expand dimensions to match model input (batch_size, height, width, channels)
                    roi_input = np.expand_dims(roi_normalized, axis=(0, -1)) # Add batch and channel dims

                    # Predict emotion using the loaded model
                    # Use verbose=0 to suppress prediction logs
                    predictions = self.emotion_model.predict(roi_input, verbose=0)[0]

                    # Get the predicted emotion label and confidence score
                    emotion_index = np.argmax(predictions)
                    emotion_label = EMOTION_LABELS[emotion_index]
                    confidence_score = float(np.max(predictions))

                    detected_emotions_in_frame.append((emotion_label, confidence_score))

                    # Log the primary detected emotion to the database
                    self.db.log_emotion(emotion_label, confidence_score)
                    self.emotion_logged.emit() # Signal that data was logged

                    # --- Draw bounding box and emotion label on the original frame ---
                    # Box color
                    color = (0, 255, 0) # Green
                    # Text to display
                    text = f"{emotion_label} ({confidence_score:.2f})"

                    # Draw rectangle around the face
                    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

                    # Draw background rectangle for text for better visibility
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y), color, -1) # Filled rect

                    # Put text above the bounding box
                    cv2.putText(frame, text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA) # Black text

                except cv2.error as cv_err:
                     # Handle errors during ROI processing (e.g., if face box is tiny)
                     # print(f"OpenCV error processing face ROI: {cv_err}")
                     pass # Continue to next face
                except Exception as e:
                    print(f"Error processing individual face: {e}")
                    # Optionally log detailed error or stack trace here
                    # import traceback
                    # traceback.print_exc()

            # --- Emit the processed frame (with drawings) ---
            # Convert the OpenCV frame (BGR) to QImage (RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            # Emit the QImage and the list of detected emotions in this frame
            self.frame_processed.emit(qt_image.copy(), detected_emotions_in_frame) # Emit a copy

        except tf.errors.OpError as tf_err:
             print(f"TensorFlow runtime error during prediction: {tf_err}")
             # This might indicate issues with the model, input data, or TF setup
        except Exception as e:
            print(f"Error in process_frame: {e}")
            # Emit the original frame even if processing failed?
            # h, w, ch = frame.shape
            # bytes_per_line = ch * w
            # qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            # self.frame_processed.emit(qt_image.copy(), []) # Emit with empty emotion list

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

        # Data storage using deque for efficient fixed-size plotting
        self.timestamps = deque(maxlen=MAX_GRAPH_POINTS)
        self.confidences = {label: deque(maxlen=MAX_GRAPH_POINTS) for label in EMOTION_LABELS}
        self.lines = {} # Store line objects for efficient updates

        self.setup_plot()

    def setup_plot(self):
        """Initial setup of the plot axes and lines."""
        self.axes.set_title("Emotion Confidence Over Time")
        self.axes.set_xlabel("Time (seconds ago)")
        self.axes.set_ylabel("Confidence")
        self.axes.set_ylim(0, 1.05) # Confidence range [0, 1]
        self.axes.grid(True)

        # Create plot lines for each emotion, initially empty
        for label in EMOTION_LABELS:
            line, = self.axes.plot([], [], label=label, marker='.', linestyle='-') # Add markers
            self.lines[label] = line

        self.axes.legend(loc='upper left', fontsize='small')
        self.fig.tight_layout() # Adjust layout

    def update_plot(self, data):
        """Updates the plot with new data points."""
        if not data: # No data to plot
            return

        current_time = time.time()
        # Clear old data points before adding new ones
        self.timestamps.clear()
        for label in EMOTION_LABELS:
            self.confidences[label].clear()

        # Populate deques with new data (timestamp, emotion, confidence)
        for timestamp, emotion, confidence in data:
             if emotion in self.confidences:
                 time_ago = current_time - timestamp # Calculate time relative to now
                 self.timestamps.append(time_ago) # Store relative time
                 # Store confidence for the specific emotion, others get NaN or None for gaps
                 for label in EMOTION_LABELS:
                     if label == emotion:
                         self.confidences[label].append(confidence)
                     else:
                         # Append NaN to create gaps for emotions not detected at this timestamp
                         self.confidences[label].append(np.nan)


        if not self.timestamps: # No valid data points added
             return

        # Update the plot lines
        min_time_ago = 0
        max_time_ago = max(self.timestamps) if self.timestamps else 0

        for label, line in self.lines.items():
            # Get corresponding confidence values, handling potential length mismatches if needed
            conf_data = list(self.confidences[label])
            time_data = list(self.timestamps) # Use the common timestamp deque

            # Ensure data lengths match (should match due to deque logic)
            min_len = min(len(time_data), len(conf_data))
            line.set_data(time_data[:min_len], conf_data[:min_len])


        # Adjust x-axis limits dynamically based on the time range in the deque
        # Add a small buffer for better visualization
        self.axes.set_xlim(max_time_ago + 1, min_time_ago -1 ) # Reversed for 'seconds ago'

        # Redraw the canvas
        try:
            self.draw()
        except Exception as e:
            print(f"Error drawing graph: {e}")


class MainWindow(QMainWindow):
    """Main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroScope - Real-Time Emotion Detection")
        self.setGeometry(100, 100, 1000, 600) # x, y, width, height

        # --- Central Widget and Layout ---
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget) # Main horizontal layout

        # --- Left Side: Video Feed ---
        self.video_layout = QVBoxLayout()
        self.video_label = QLabel("Initializing Video...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        # Make video label expand to fill available space
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_layout.addWidget(self.video_label)

        # --- Right Side: Controls and Graph ---
        self.controls_graph_layout = QVBoxLayout()

        # Emotion Graph
        self.graph_canvas = EmotionGraphCanvas(self.central_widget)
        self.controls_graph_layout.addWidget(self.graph_canvas)

        # Optional: Add a status label or other controls if needed
        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.controls_graph_layout.addWidget(self.status_label)

        # --- Assemble Main Layout ---
        self.main_layout.addLayout(self.video_layout, 7) # Video takes 70% width
        self.main_layout.addLayout(self.controls_graph_layout, 3) # Controls/Graph takes 30%

        # --- Load Model ---
        self.emotion_model = load_emotion_model(MODEL_PATH)
        if self.emotion_model is None:
             self.status_label.setText("Status: Error loading emotion model!")
             # Optionally disable features or show an error message box
             # QMessageBox.critical(self, "Model Error", f"Could not load model: {MODEL_PATH}")

        # --- Video Processing Thread ---
        self.video_thread = VideoProcessor(self.emotion_model, self)
        self.video_thread.frame_processed.connect(self.update_video_feed)
        # Connect the emotion_logged signal to trigger graph updates
        self.video_thread.emotion_logged.connect(self.schedule_graph_update)

        # --- Graph Update Timer ---
        # Use a timer to update the graph periodically from the DB,
        # rather than every single frame/detection, for performance.
        self.graph_update_timer = QTimer(self)
        self.graph_update_timer.timeout.connect(self.update_emotion_graph)
        self.needs_graph_update = False # Flag to indicate new data is available

        # --- Start Processing ---
        if self.emotion_model: # Only start if model loaded
            self.video_thread.start()
            self.graph_update_timer.start(GRAPH_UPDATE_INTERVAL) # Start the timer
            self.status_label.setText("Status: Running")
        else:
            self.video_label.setText("Error: Model not loaded. Cannot start video processing.")


    @pyqtSlot(QImage, list)
    def update_video_feed(self, qt_image, emotions):
        """Updates the video feed QLabel with the new frame."""
        try:
            pixmap = QPixmap.fromImage(qt_image)
            # Scale pixmap to fit the label while maintaining aspect ratio
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
                # Fetch recent data from the database
                db = ThreadSafeDB()
                recent_data = db.get_recent_emotions(limit=MAX_GRAPH_POINTS)
                if recent_data:
                    self.graph_canvas.update_plot(recent_data)
                    self.needs_graph_update = False # Reset flag after update
                # else:
                #     print("No recent data found for graph update.") # Optional debug log
            except Exception as e:
                print(f"Error updating emotion graph: {e}")


    def closeEvent(self, event):
        """Handles the window closing event."""
        print("Closing application...")
        # Stop the graph update timer
        self.graph_update_timer.stop()

        # Stop the video processing thread
        if self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait() # Wait for the thread to finish cleanly

        # Close the database connection
        db = ThreadSafeDB()
        db.close()

        print("Application closed.")
        event.accept() # Accept the close event


if __name__ == "__main__":
    # Set application attributes for better high-DPI scaling and naming
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("NeuroScope")

    # --- Check if model file exists before starting ---
    if not os.path.exists(MODEL_PATH):
         from PyQt5.QtWidgets import QMessageBox
         QMessageBox.critical(None, "Error",
                              f"Emotion model file not found:\n{MODEL_PATH}\n\nPlease ensure the model file is in the correct location.")
         sys.exit(1) # Exit if model is missing

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    #using a model trained on vector location through 100s of points acorss the fasce across multiple poijts and then detect abnormalities . this shousd be accursate since even a little deviation of vectors
    # from normal space would depict there is a diversion of emotion and we only need to look at remaining data to determine what the emotion is
