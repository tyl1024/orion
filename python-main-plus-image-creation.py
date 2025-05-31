import sys
import json
import ollama
import speech_recognition as sr
from PyQt6.QtWidgets import (
    QApplication, QTextEdit, QVBoxLayout, QWidget, QPushButton, 
    QLineEdit, QCalendarWidget, QLabel, QMessageBox, QInputDialog, QHBoxLayout
)
from PyQt6.QtGui import QFont, QPalette, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDate, QTimer
from datetime import datetime
from PIL import Image
from io import BytesIO
import base64
import threading
from diffusers import StableDiffusionPipeline
import torch
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel
import tempfile
import os
from diffusers import StableDiffusionPipeline


# --- Event Persistence ---
event_file = "events.json"

def load_events():
    try:
        with open(event_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_events(events):
    with open(event_file, "w") as f:
        json.dump(events, f, indent=4)

# --- Voice Thread ---
class VoiceRecognitionThread(QThread):
    recognition_complete = pyqtSignal(str)

    def run(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source, timeout=15, phrase_time_limit=30)
                text = recognizer.recognize_google(audio)
                self.recognition_complete.emit(text)
            except sr.UnknownValueError:
                self.recognition_complete.emit("🤔 Could not understand the audio.")
            except sr.RequestError:
                self.recognition_complete.emit("⚠️ Speech service unavailable.")
            except sr.WaitTimeoutError:
                self.recognition_complete.emit("⏳ No speech detected.")

# --- Image Generation Module (Graceful fallback) ---
# Load Stable Diffusion model once
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
except Exception as e:
    print(f"Failed to load Stable Diffusion: {e}")
    sd_pipe = None

# Replace existing generate_image() function with this one
def generate_image(prompt):
    global sd_pipe
    try:
        if not sd_pipe:
            return None
        image = sd_pipe(prompt).images[0]
        return image
    except Exception as e:
        print(f"Image generation error: {e}")
        return None

# In your CalendarAI class, update handle_image_request() like this:
def handle_image_request(self):
    prompt = self.input_field.text().strip()
    if not prompt:
        return
    self.input_field.clear()
    self.event_display.append(f"🧑‍🎨 Image Prompt: {prompt}\n")
    image = generate_image(prompt)
    if image:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            image.save(tmp_file.name)
            pixmap = QPixmap(tmp_file.name)
            image_label = QLabel()
            image_label.setPixmap(pixmap.scaledToWidth(400, Qt.TransformationMode.SmoothTransformation))
            self.layout.addWidget(image_label)
            os.unlink(tmp_file.name)  # Clean up temp image file after showing
    else:
        self.event_display.append("⚠️ Failed to generate image.\n")


# --- Video Generation (Mock) ---
def generate_video(prompt):
    return f"[🎞️ Video generated for: '{prompt}']"

# --- Multilingual Placeholder ---
def translate_text(text, language="en"):
    return text  # Future: plug in multilingual translation model

# --- Main Calendar App ---
class CalendarAI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Calendar Assistant")
        self.setGeometry(200, 200, 950, 620)
        self.setStyleSheet("font-size: 14px; background-color: #121212; color: #e0e0e0;")
        self.events = load_events()
        self.pending_event = None

        # Dark theme palette fix
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor("#121212"))
        dark_palette.setColor(QPalette.ColorRole.Base, QColor("#1e1e1e"))
        dark_palette.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
        dark_palette.setColor(QPalette.ColorRole.Button, QColor("#2a2a2a"))
        self.setPalette(dark_palette)

        self.layout = QVBoxLayout()

        self.event_list_label = QLabel("📆 Events This Month:")
        self.event_list_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.layout.addWidget(self.event_list_label)

        self.monthly_event_display = QTextEdit(self)
        self.monthly_event_display.setReadOnly(True)
        self.monthly_event_display.setFixedHeight(80)
        self.monthly_event_display.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")
        self.layout.addWidget(self.monthly_event_display)

        self.calendar = QCalendarWidget(self)
        self.calendar.setFixedSize(400, 300)
        self.calendar.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")
        self.calendar.clicked.connect(self.confirm_event)
        self.layout.addWidget(self.calendar)

        self.event_display = QTextEdit(self)
        self.event_display.setReadOnly(True)
        self.event_display.setFixedHeight(200)
        self.event_display.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")
        self.layout.addWidget(self.event_display)

        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Type event details, ask AI, or describe a scene...")
        self.input_field.returnPressed.connect(self.send_message)
        self.input_field.setStyleSheet("background-color: #2a2a2a; color: #ffffff;")
        self.layout.addWidget(self.input_field)

        button_layout = QHBoxLayout()
        button_style = "background-color: #2a2a2a; color: #ffffff; padding: 5px; border-radius: 5px;"

        self.add_button = QPushButton("➕ Add Event", self)
        self.add_button.setStyleSheet(button_style)
        self.add_button.clicked.connect(self.prepare_event)
        button_layout.addWidget(self.add_button)

        self.clear_button = QPushButton("🗑 Clear Event", self)
        self.clear_button.setStyleSheet(button_style)
        self.clear_button.clicked.connect(self.clear_event)
        button_layout.addWidget(self.clear_button)

        self.clear_all_button = QPushButton("🚨 Clear All", self)
        self.clear_all_button.setStyleSheet(button_style)
        self.clear_all_button.clicked.connect(self.clear_all_events)
        button_layout.addWidget(self.clear_all_button)

        self.voice_button = QPushButton("🎙️ Voice Input", self)
        self.voice_button.setStyleSheet(button_style)
        self.voice_button.clicked.connect(self.start_voice_input)
        button_layout.addWidget(self.voice_button)

        self.ask_ai_button = QPushButton("🧠 Ask AI", self)
        self.ask_ai_button.setStyleSheet(button_style)
        self.ask_ai_button.clicked.connect(self.send_message)
        button_layout.addWidget(self.ask_ai_button)

        self.image_button = QPushButton("🖼 Generate Image", self)
        self.image_button.setStyleSheet(button_style)
        self.image_button.clicked.connect(self.handle_image_request)
        button_layout.addWidget(self.image_button)

        self.video_button = QPushButton("🎞 Generate Video", self)
        self.video_button.setStyleSheet(button_style)
        self.video_button.clicked.connect(self.handle_video_request)
        button_layout.addWidget(self.video_button)

        self.minimize_button = QPushButton("📉 Toggle Calendar", self)
        self.minimize_button.setStyleSheet(button_style)
        self.minimize_button.clicked.connect(self.toggle_calendar)
        button_layout.addWidget(self.minimize_button)

        self.layout.addLayout(button_layout)
        self.setLayout(self.layout)

        self.reminder_timer = QTimer(self)
        self.reminder_timer.timeout.connect(self.check_reminders)
        self.reminder_timer.start(60000)

        self.update_monthly_events()

    def prepare_event(self):
        self.pending_event = self.input_field.text().strip()
        if not self.pending_event:
            QMessageBox.warning(self, "Input Error", "Please enter an event before adding.")
            return

        choice, ok = QInputDialog.getItem(self, "Date Selection", "Select date input method:", ["Click on Calendar", "Enter Date"], 0, False)
        if ok:
            if choice == "Click on Calendar":
                self.input_field.setPlaceholderText("Click a date on the calendar to add event.")
            else:
                date, ok = QInputDialog.getText(self, "Enter Date", "Enter event date (YYYY-MM-DD):")
                if ok:
                    self.confirm_event(QDate.fromString(date, "yyyy-MM-dd"))

    def confirm_event(self, date=None):
        if self.pending_event:
            if not date:
                date = self.calendar.selectedDate().toString("yyyy-MM-dd")
            else:
                date = date.toString("yyyy-MM-dd")

            if date not in self.events:
                self.events[date] = []
            self.events[date].append(self.pending_event)
            save_events(self.events)
            self.event_display.append(f"📅 {date}: {self.pending_event}\n")
            self.update_monthly_events()
            self.input_field.clear()
            self.pending_event = None

    def clear_event(self):
        date = self.calendar.selectedDate().toString("yyyy-MM-dd")
        events = self.events.get(date, [])
        if not events:
            QMessageBox.information(self, "Clear Event", "No events to remove for this date.")
            return

        event, ok = QInputDialog.getItem(self, "Clear Event", "Select event to remove:", events, 0, False)
        if ok and event:
            self.events[date].remove(event)
            if not self.events[date]:
                del self.events[date]
            save_events(self.events)
            self.update_monthly_events()

    def clear_all_events(self):
        confirm = QMessageBox.question(self, "Clear All", "Are you sure you want to delete all events?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.Yes:
            self.events = {}
            save_events(self.events)
            self.update_monthly_events()
            self.event_display.clear()

    def send_message(self):
        user_input = self.input_field.text().strip()
        if not user_input:
            return

        self.input_field.clear()
        self.event_display.append(f"🧑‍💻 You: {user_input}\n")

        def get_ai_response():
            try:
                response = ollama.chat(model="mistral", messages=[{"role": "user", "content": user_input}])
                ai_response = response['message']['content']
                self.event_display.append(f"🤖 AI: {ai_response}\n")
            except Exception as e:
                self.event_display.append(f"⚠️ Error: {str(e)}\n")

        threading.Thread(target=get_ai_response).start()

    def handle_image_request(self):
        prompt = self.input_field.text().strip()
        if not prompt:
            return
        self.input_field.clear()
        self.event_display.append(f"🧑‍🎨 Image Prompt: {prompt}\n")
        image = generate_image(prompt)
        if image:
            image.show()
        else:
            self.event_display.append("⚠️ Failed to generate image.\n")

    def handle_video_request(self):
        prompt = self.input_field.text().strip()
        if not prompt:
            return
        self.input_field.clear()
        video_msg = generate_video(prompt)
        self.event_display.append(video_msg + "\n")

    def start_voice_input(self):
        self.event_display.append("🎤 Listening...\n")
        self.voice_thread = VoiceRecognitionThread()
        self.voice_thread.recognition_complete.connect(self.process_voice_input)
        self.voice_thread.start()

    def process_voice_input(self, text):
        self.input_field.setText(text)
        self.send_message()

    def toggle_calendar(self):
        self.calendar.setVisible(not self.calendar.isVisible())

    def update_monthly_events(self):
        month_events = [f"{date}: {', '.join(events)}" for date, events in sorted(self.events.items())]
        self.monthly_event_display.setText("\n".join(month_events) if month_events else "No events this month.")

    def check_reminders(self):
        today = datetime.today().strftime("%Y-%m-%d")
        if today in self.events:
            QMessageBox.information(self, "Reminder", f"📌 Today's Events:\n{', '.join(self.events[today])}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CalendarAI()
    window.show()
    sys.exit(app.exec())

