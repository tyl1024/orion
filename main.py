import sys
import json
import ollama
import speech_recognition as sr
import requests
from PyQt6.QtWidgets import (
    QApplication, QTextEdit, QVBoxLayout, QWidget, QPushButton, 
    QLineEdit, QCalendarWidget, QLabel, QMessageBox, QInputDialog, QHBoxLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDate, QTimer
from datetime import datetime

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

def is_ollama_running():
    try:
        response = requests.get("http://localhost:11434")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

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

class CalendarAI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Calendar Assistant")
        self.setGeometry(200, 200, 900, 600)
        self.events = load_events()
        self.pending_event = None

        self.layout = QVBoxLayout()

        self.event_list_label = QLabel("📆 Events This Month:")
        self.layout.addWidget(self.event_list_label)

        self.monthly_event_display = QTextEdit(self)
        self.monthly_event_display.setReadOnly(True)
        self.monthly_event_display.setFixedHeight(80)
        self.layout.addWidget(self.monthly_event_display)

        self.calendar = QCalendarWidget(self)
        self.calendar.setFixedSize(400, 300)
        self.calendar.clicked.connect(self.confirm_event)
        self.layout.addWidget(self.calendar)

        self.event_display = QTextEdit(self)
        self.event_display.setReadOnly(True)
        self.event_display.setFixedHeight(180)
        self.layout.addWidget(self.event_display)

        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Type event details or ask AI...")
        self.input_field.returnPressed.connect(self.send_message)
        self.layout.addWidget(self.input_field)

        button_layout = QHBoxLayout()

        self.add_button = QPushButton("➕ Add Event", self)
        self.add_button.clicked.connect(self.prepare_event)
        button_layout.addWidget(self.add_button)

        self.clear_button = QPushButton("🗑 Clear Event", self)
        self.clear_button.clicked.connect(self.clear_event)
        button_layout.addWidget(self.clear_button)

        self.clear_all_button = QPushButton("🚨 Clear All", self)
        self.clear_all_button.clicked.connect(self.clear_all_events)
        button_layout.addWidget(self.clear_all_button)

        self.voice_button = QPushButton("🎙️ Voice Input", self)
        self.voice_button.clicked.connect(self.start_voice_input)
        button_layout.addWidget(self.voice_button)

        self.ask_ai_button = QPushButton("🧠 Ask AI", self)
        self.ask_ai_button.clicked.connect(self.send_message)
        button_layout.addWidget(self.ask_ai_button)

        self.minimize_button = QPushButton("📉 Minimize Calendar", self)
        self.minimize_button.clicked.connect(self.toggle_calendar)
        button_layout.addWidget(self.minimize_button)

        self.layout.addLayout(button_layout)
        self.setLayout(self.layout)

        self.reminder_timer = QTimer(self)
        self.reminder_timer.timeout.connect(self.check_reminders)
        self.reminder_timer.start(60000)

        self.update_monthly_events()

        # Check Ollama on start
        if not is_ollama_running():
            QMessageBox.critical(
                self,
                "Ollama Not Running",
                "⚠️ Failed to connect to Ollama.\n\nPlease:\n1. Install Ollama: https://ollama.com/download\n2. Open it so it runs in the background.\n\nAfter starting Ollama, re-open this app."
            )

    def prepare_event(self):
        self.pending_event = self.input_field.text().strip()
        if not self.pending_event:
            QMessageBox.warning(self, "Input Error", "Please enter an event before adding.")
            return

        choice, ok = QInputDialog.getItem(self, "Date Selection", "How do you want to select the date?", ["Click on Calendar", "Enter Date"], 0, False)

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

        if not is_ollama_running():
            self.event_display.append("⚠️ Ollama is not running. Please open Ollama from your Start Menu or Applications.\n")
            return

        try:
            response = ollama.chat(model="mistral", messages=[{"role": "user", "content": user_input}])
            ai_response = response['message']['content']
            self.event_display.append(f"🤖 AI: {ai_response}\n")
        except Exception as e:
            self.event_display.append(f"⚠️ Error talking to Ollama: {str(e)}\n")

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

app = QApplication(sys.argv)
window = CalendarAI()
window.show()
sys.exit(app.exec())
