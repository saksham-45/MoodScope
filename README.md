# 🎭 MoodScope
**AI-Powered Real-Time Emotion Detection and Analysis**

MoodScope is an intelligent emotion recognition system that integrates **face detection**, **deep learning-based emotion classification**, and **AI-generated wellness insights**.  
It can analyze emotions **in real time via webcam** or from **pre-recorded videos / CSV logs**, visualize trends, and generate personalized insights using **Ollama AI (Llama 3)**.
<img width="1800" height="1169" alt="Screenshot 2025-10-04 at 11 36 14 PM" src="https://github.com/user-attachments/assets/8a6965f9-1baa-4ccc-9bb2-1fdfed7d78dd" />
<img width="1432" height="868" alt="Screenshot 2025-10-04 at 11 22 59 PM" src="https://github.com/user-attachments/assets/9bacaa78-fe86-4628-b066-0bc41a8b60ff" />



---

## 🚀 Features

### 🧠 Real-Time Emotion Detection
- Detects faces using **MTCNN** and classifies emotions with a trained **CNN model (`emotion_detection_model.h5`)**.
- Displays bounding boxes, emotion labels, and confidence scores directly on live video.
- Supports webcam input or uploaded video files.

### 📊 Emotion Analytics Dashboard
- Real-time emotion logging to an **SQLite database**.
- Interactive emotion confidence graph using **Matplotlib**.
- Exports emotion data to **CSV** for offline analysis.

### 🤖 AI-Powered Insights
- Integrates with **Ollama (Llama3.2)** to generate:
  - Key emotional insights  
  - Personalized wellness recommendations  
  - Actionable tips for improving emotional wellbeing  
- Falls back to rule-based insights if Ollama isn’t running.

### 🧩 CSV Reflection Mode
- Import CSV emotion logs and visualize trends over time.
- Generates detailed emotional reports and recommendations.

---

## 🖥️ Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend / UI** | PyQt5 |
| **Backend / Processing** | OpenCV, TensorFlow/Keras |
| **Face Detection** | MTCNN |
| **Emotion Model** | Custom CNN (`emotion_detection_model.h5`) |
| **Database** | SQLite3 |
| **Visualization** | Matplotlib |
| **AI Insights** | Ollama (Llama3.2 or compatible local model) |

---

## ⚙️ Installation
```bash
1️⃣ Clone the Repository

git clone https://github.com/saksham-45/MoodScope.git
cd MoodScope

2️⃣ Create a Virtual Environment

python3 -m venv venv
source venv/bin/activate     # On macOS/Linux
venv\Scripts\activate        # On Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Download or Train the Emotion Model

Place your trained model in the project root as:

emotion_detection_model.h5


⸻

▶️ Usage

🧍 Real-Time Detection

python main.py

	•	Starts webcam feed
	•	Detects and logs emotions live
	•	Displays emotion trends on a graph

📂 Import and Analyze Emotion Logs

In the startup screen:
	•	Click “📂 Import Emotion Log (CSV)”
	•	Select a previously exported emotion log

⸻

🧠 Optional: Ollama AI Setup (for Insights)

To enable AI-powered insights:
	1.	Install Ollama
	2.	Run Ollama locally:

ollama run llama3.2


	3.	MoodScope will automatically detect and use the local Ollama server at:

http://localhost:11434



⸻

📈 Output Examples

🖼️ Real-Time Detection Window
	•	Bounding boxes with emotion labels and confidence
	•	Dynamic engagement bar and dominant emotion indicator

📊 Analytics Report
	•	Emotion Confidence vs. Time plot
	•	Emotion distribution summary
	•	AI-generated insights and wellness tips

⸻

🧾 Export Options
	•	Export real-time data to CSV (Export CSV button)
	•	Import exported CSVs for reflection and trend analysis

⸻

💡 Future Enhancements
	•	Cloud synchronization for emotion data
	•	Advanced model fine-tuning with FER2013+
	•	Group emotion analytics
	•	API integration for real-world emotion tracking

⸻

🧑‍💻 Author

Saksham Srivastava
📍 Developer & AI Enthusiast

⸻

🪪 License

This project is licensed under the MIT License — feel free to use and modify with attribution.

⸻

🌈 Summary

MoodScope bridges emotion recognition, data visualization, and AI-driven self-awareness.
Whether for research, wellness tracking, or human-computer interaction studies, it offers an all-in-one tool to understand emotions in real time.
