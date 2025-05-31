import sys
import os
import json
import torch
import speech_recognition as sr
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLineEdit,
    QWidget, QCalendarWidget, QMessageBox, QLabel
)
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon
from diffusers import StableDiffusionPipeline
from PIL import Image

class CalendarAI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Calendar Assistant")
        self.setGeometry(100, 100, 800, 600)

        self.init_ui()
        self.events = self.load_events()
        self.pipeline = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_reminders)
        self.timer.start(3600000)  # check every hour

    def init_ui(self):
        main_layout = QVBoxLayout()

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        main_layout.addWidget(self.output_text)

        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.returnPressed.connect(self.handle_input)
        input_layout.addWidget(self.input_field)

        send_button = QPushButton("Send")
        send_button.clicked.connect(self.handle_input)
        input_layout.addWidget(send_button)

        mic_button = QPushButton("🎤")
        mic_button.clicked.connect(self.listen_voice)
        input_layout.addWidget(mic_button)

        main_layout.addLayout(input_layout)

        button_layout = QHBoxLayout()

        self.toggle_calendar_btn = QPushButton("Hide Calendar")
        self.toggle_calendar_btn.clicked.connect(self.toggle_calendar)
        button_layout.addWidget(self.toggle_calendar_btn)

        daily_btn = QPushButton("📅 Daily Events")
        daily_btn.clicked.connect(self.show_daily_events)
        button_layout.addWidget(daily_btn)

        monthly_btn = QPushButton("🗓️ Monthly Events")
        monthly_btn.clicked.connect(self.show_monthly_events)
        button_layout.addWidget(monthly_btn)

        clear_btn = QPushButton("❌ Clear Events")
        clear_btn.clicked.connect(self.confirm_clear_events)
        button_layout.addWidget(clear_btn)

        main_layout.addLayout(button_layout)

        self.calendar = QCalendarWidget()
        self.calendar.clicked.connect(self.display_events_for_date)
        main_layout.addWidget(self.calendar)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def handle_input(self):
        text = self.input_field.text().strip()
        self.input_field.clear()
        self.output_text.append(f"You: {text}")

        if text.lower().startswith("/event"):
            self.add_event(text[6:].strip())
        elif text.lower().startswith("/ask"):
            self.respond_ai(text[4:].strip())
        elif text.lower().startswith("/image"):
            self.generate_image(text[6:].strip())
        else:
            self.output_text.append("🤖: Please use /event, /ask, or /image commands.")

    def listen_voice(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self.output_text.append("🎙️ Listening...")
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            self.input_field.setText(text)
            self.handle_input()
        except sr.UnknownValueError:
            self.output_text.append("🤖: Sorry, could not understand the audio.")
        except sr.RequestError as e:
            self.output_text.append(f"🤖: Could not request results; {e}")

    def add_event(self, event_text):
        selected_date = self.calendar.selectedDate().toString("yyyy-MM-dd")
        if selected_date not in self.events:
            self.events[selected_date] = []
        self.events[selected_date].append(event_text)
        self.save_events()
        self.output_text.append(f"📌 Event added on {selected_date}: {event_text}")

    def display_events_for_date(self):
        selected_date = self.calendar.selectedDate().toString("yyyy-MM-dd")
        events = self.events.get(selected_date, [])
        if events:
            self.output_text.append(f"📆 Events on {selected_date}:\n" + "\n".join(events))
        else:
            self.output_text.append(f"📆 No events for {selected_date}.")

    def show_daily_events(self):
        today = datetime.now().strftime("%Y-%m-%d")
        events = self.events.get(today, [])
        if events:
            self.output_text.append(f"📅 Today's Events:\n" + "\n".join(events))
        else:
            self.output_text.append("📅 No events today.")

    def show_monthly_events(self):
        current_month = datetime.now().strftime("%Y-%m")
        self.output_text.append(f"📅 Events for {current_month}:")
        found = False
        for date, items in self.events.items():
            if date.startswith(current_month):
                self.output_text.append(f"{date}:\n  - " + "\n  - ".join(items))
                found = True
        if not found:
            self.output_text.append("No events this month.")

    def confirm_clear_events(self):
        confirm = QMessageBox.question(self, "Clear Events", "Are you sure you want to delete all events?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.Yes:
            self.events = {}
            self.save_events()
            self.output_text.append("🗑️ All events cleared.")

    def toggle_calendar(self):
        if self.calendar.isVisible():
            self.calendar.hide()
            self.toggle_calendar_btn.setText("Show Calendar")
        else:
            self.calendar.show()
            self.toggle_calendar_btn.setText("Hide Calendar")

    def respond_ai(self, question):
        # Dummy logic – replace with real AI call or local LLM
        self.output_text.append(f"🤖: Let me think about '{question}'... Here’s a possible answer!")

    def generate_image(self, prompt):
        if not self.pipeline:
            self.output_text.append("🔄 Loading Stable Diffusion pipeline...")
            model_id = "CompVis/stable-diffusion-v1-4"
            self.pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
            self.pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
        self.output_text.append(f"🎨 Generating image for prompt: '{prompt}'")
        image = self.pipeline(prompt).images[0]
        filename = f"generated_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        image.save(filename)
        self.output_text.append(f"🖼️ Image saved as {filename}")

    def check_reminders(self):
        today = datetime.now().strftime("%Y-%m-%d")
        todays_events = self.events.get(today, [])
        if todays_events:
            reminders = "\n".join(f"🔔 {event}" for event in todays_events)
            QMessageBox.information(self, "Today's Reminders", reminders)

    def load_events(self):
        if os.path.exists("events.json"):
            with open("events.json", "r") as f:
                return json.load(f)
        return {}

    def save_events(self):
        with open("events.json", "w") as f:
            json.dump(self.events, f, indent=2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CalendarAI()
    window.show()
    sys.exit(app.exec())
