# ui_analysis.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np


class CSVAnalysisWindow(QWidget):
    def __init__(self, df, file_path):
        super().__init__()
        self.df = df
        self.setWindowTitle(f"MoodScope - Analysis of {file_path.split('/')[-1]}")
        self.setMinimumSize(900, 1600)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Emotion Log Analysis")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        # Summary stats
        emotion_counts = df['emotion'].value_counts()
        dominant = emotion_counts.idxmax()
        avg_conf = df['confidence'].mean()
        summary = QLabel(f"Dominant Emotion: {dominant} | Avg Confidence: {avg_conf:.2f}")
        summary.setAlignment(Qt.AlignCenter)
        layout.addWidget(summary)

        # Graph
        fig = Figure(figsize=(6, 4), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        for emotion in df['emotion'].unique():
            subset = df[df['emotion'] == emotion]
            ax.plot(subset['timestamp'], subset['confidence'], label=emotion)

        ax.set_title("Emotion Confidence Over Time")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Confidence")
        ax.legend()
        fig.tight_layout()
        layout.addWidget(canvas)

        # Back button
        back_btn = QPushButton("⬅️ Back to Home")
        back_btn.clicked.connect(self.close)
        layout.addWidget(back_btn)

        self.setLayout(layout)