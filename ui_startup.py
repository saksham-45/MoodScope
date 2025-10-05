# ui_startup.py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
import pandas as pd


class StartupScreen(QWidget):
    def __init__(self, on_start_detection, on_import_csv):
        super().__init__()
        self.on_start_detection = on_start_detection
        self.on_import_csv = on_import_csv

        self.setWindowTitle("MoodScope - Welcome")
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: white;
                font-family: 'Arial';
            }
            QPushButton {
                background-color: #2E86DE;
                color: white;
                font-size: 18px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1B4F72;
            }
        """)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("MoodScope")
        title.setStyleSheet("font-size: 36px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Understand emotions in real time or through reflection.")
        subtitle.setStyleSheet("font-size: 16px; color: #CCCCCC;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(30)

        start_btn = QPushButton("🎥 Start Real-Time Detection")
        start_btn.clicked.connect(self.on_start_detection)
        layout.addWidget(start_btn)

        import_btn = QPushButton("📂 Import Emotion Log (CSV)")
        import_btn.clicked.connect(self.import_csv)
        layout.addWidget(import_btn)

        exit_btn = QPushButton("❌ Exit")
        exit_btn.clicked.connect(self.close)
        layout.addWidget(exit_btn)

        self.setLayout(layout)

    def import_csv(self):
        """Open a file dialog to select a CSV and trigger analysis."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Emotion Log CSV", "",
                                                  "CSV Files (*.csv);;All Files (*)")
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
            if not {'timestamp', 'emotion', 'confidence'}.issubset(df.columns):
                QMessageBox.warning(self, "Invalid File", "CSV must contain columns: timestamp, emotion, confidence.")
                return
            self.on_import_csv(df, file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read CSV: {e}")