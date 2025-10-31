#############################################
#
# CareBot Command System App
# Author: Dharunish Yugeswardeenoo
#
#############################################

import sys,os
import json
import queue
import threading
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QTextEdit, QLabel, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QIcon
from PyQt6.QtGui import QPixmap, QTransform
import pyqtgraph as pg
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import subprocess


class AudioSignals(QObject):
    transcript_update = pyqtSignal(str)
    waveform_update = pyqtSignal(np.ndarray)
    command_detected = pyqtSignal(str)


class VoiceCommandUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CareBot Command System")
        self.setGeometry(100, 100, 1200, 700)

        # Audio
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recognizer = None
        self.stream = None
        self.signals = AudioSignals()

        # Models
        self.MODEL_PATH = "/Users/dharunish/Downloads/vosk-model-small-en-us-0.15"
        self.command_models = {'feed me': 'feed_model', 'clean': 'clean_model', 'pick up medicine': 'medicine_model'}
        self.current_model_thread = None
        self.stop_model_flag = threading.Event()
        self.mode = "voice_command"

        # UI
        self.init_ui()

        # Signals
        self.signals.transcript_update.connect(self.update_transcript)
        self.signals.waveform_update.connect(self.update_waveform)
        self.signals.command_detected.connect(self.handle_command)

        # Theme
        self.apply_theme()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)


        # --- Create the top horizontal layout ---
        top_layout = QHBoxLayout()
        top_layout.setSpacing(20)

        # Left image
        left_image = QLabel()
        left_pix = QPixmap("icon.png").scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        left_image.setPixmap(left_pix)
        left_image.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Title label (center)
        title = QLabel("CareBot Command System")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setObjectName("title_label")

        # Right image
        right_image = QLabel()
        right_pix = QPixmap("icon.png").scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        right_pix = right_pix.transformed(QTransform().scale(-1, 1))
        right_image.setPixmap(right_pix)
        right_image.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # --- Add them to the horizontal layout ---
        top_layout.addWidget(left_image)
        top_layout.addStretch()
        top_layout.addWidget(title)
        top_layout.addStretch()
        top_layout.addWidget(right_image)

        # --- Add the top layout to the main vertical layout ---
        main_layout.addLayout(top_layout)



        # title = QLabel("CareBot Command System")
        # title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # title.setObjectName("title_label")
        # main_layout.addWidget(title)

        # Mode buttons
        button_layout = QHBoxLayout()
        self.voice_cmd_btn = QPushButton("Voice Command Mode")
        self.voice_cmd_btn.setCheckable(True)
        self.voice_cmd_btn.clicked.connect(self.toggle_voice_command)
        self.training_btn = QPushButton("Training Mode")
        self.training_btn.setCheckable(True)
        self.training_btn.clicked.connect(self.toggle_training_mode)
        button_layout.addWidget(self.voice_cmd_btn)
        button_layout.addWidget(self.training_btn)
        main_layout.addLayout(button_layout)

        # Status label
        self.status_label = QLabel("Voice Status: Press a button to begin")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setObjectName("status_label")
        main_layout.addWidget(self.status_label)

        # Content area
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        main_layout.addWidget(self.content_widget)

        # Default view
        self.init_voice_command_view()

    def init_voice_command_view(self):
        self.clear_content_layout()
        split_layout = QHBoxLayout()
        split_layout.setSpacing(15)

        # Left panel - waveform
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_panel.setObjectName("left_panel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        
        waveform_label = QLabel("Audio Waveform")
        waveform_label.setObjectName("panel_header")
        waveform_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(waveform_label)
        
        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setBackground('#1a1a2e')
        self.waveform_plot.setYRange(-0.4, 0.4)
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.3)
        self.waveform_curve = self.waveform_plot.plot(pen=pg.mkPen(color='#00fff0', width=2))
        left_layout.addWidget(self.waveform_plot)

        # Right panel - transcript (with centered content)
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.Shape.StyledPanel)
        right_panel.setObjectName("right_panel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 15, 15, 15)
        right_layout.setSpacing(10)
        
        transcript_label = QLabel("Transcribed Text")
        transcript_label.setObjectName("panel_header")
        transcript_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(transcript_label)
        
        # Container for centered text
        transcript_container = QWidget()
        transcript_container.setObjectName("transcript_container")
        container_layout = QVBoxLayout(transcript_container)
        container_layout.setContentsMargins(20, 20, 20, 20)
        
        self.transcript_text = QLabel("")
        self.transcript_text.setWordWrap(True)
        self.transcript_text.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.transcript_text.setObjectName("transcript_text")
        container_layout.addWidget(self.transcript_text)
        
        right_layout.addWidget(transcript_container, 1)

        split_layout.addWidget(left_panel, 1)
        split_layout.addWidget(right_panel, 1)
        self.content_layout.addLayout(split_layout, 1)

        # Model status
        self.model_status = QLabel("Model Status: No model running")
        self.model_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_status.setObjectName("model_status")
        self.content_layout.addWidget(self.model_status)

    def init_training_view(self):
        self.clear_content_layout()
        training_label = QLabel("Teleoperation Mode")
        training_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        training_label.setObjectName("training_header")
        self.content_layout.addWidget(training_label)
        
        info_label = QLabel("Use Second arm to control the robot.\nData will be recorded for training.")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setObjectName("info_label")
        self.content_layout.addWidget(info_label)
        
        teleop_widget = QTextEdit()
        teleop_widget.setReadOnly(True)
        teleop_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        teleop_widget.setObjectName("teleop_text")
        teleop_widget.setPlainText(
            "Teleoperation is ongoing.\n\n"
            "Use the leader arm to control the robot and collect data"
        )
        self.content_layout.addWidget(teleop_widget)

    def clear_content_layout(self):
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    # ---------------- Voice Command ----------------
    def toggle_voice_command(self):
        if self.voice_cmd_btn.isChecked():
            self.training_btn.setChecked(False)
            self.mode = "voice_command"
            self.init_voice_command_view()
            self.start_voice_recording()
        else:
            self.stop_voice_recording()

    def toggle_training_mode(self):
        if self.training_btn.isChecked():
            if self.is_recording:
                self.stop_voice_recording()
                self.voice_cmd_btn.setChecked(False)

            self.mode = "training"
            self.status_label.setText("Status: Training Mode Active - Starting Data Collection...")
            self.init_training_view()

            # --- START EXECUTION ---
            command = (
                "python lerobot/scripts/control_robot.py record "
                "--robot-path lerobot/configs/robot/moss.yaml "
                "--fps 30 --root data --repo-id test/cup_stack "
                "--tags moss --warmup-time-s 0 --episode-time-s 60 "
                "--reset-time-s 50 --num-episodes 30 --push-to-hub 0"
            )

            # Open a new macOS Terminal window and execute the command
            new_terminal_command = [
                "osascript", "-e",
                f'tell application "Terminal" to do script "cd {os.getcwd()} && {command}; echo \\"Command finished (exit code $?)\\"; exec bash -i"'
            ]

            try:
                self.training_process = subprocess.Popen(new_terminal_command)
                self.status_label.setText("Status: Training Mode Active - Data Collection Running!")
            except FileNotFoundError:
                self.status_label.setText("Status: Error - 'python' or script not found. Check environment.")
            except Exception as e:
                self.status_label.setText(f"Status: Error - Failed to start process: {e}")
            # --- END EXECUTION ---

            
        else:
            # --- STOP EXECUTION ---
            if hasattr(self, 'training_process') and self.training_process.poll() is None:
                self.training_process.terminate() # or .kill() for a hard stop
                self.training_process.wait(timeout=5)
            # --- END STOP EXECUTION ---
            self.mode = "voice_command"
            self.status_label.setText("Status: Ready")
            self.init_voice_command_view()

    def start_voice_recording(self):
        try:
            model = Model(self.MODEL_PATH)
            self.recognizer = KaldiRecognizer(model, 16000)
            self.transcript_text.setText("")
            self.is_recording = True
            self.status_label.setText("Voice Status: Listening...")
            self.voice_cmd_btn.setText("Stop Recording")

            self.stream = sd.RawInputStream(
                samplerate=16000,
                blocksize=8000,
                dtype='int16',
                channels=1,
                callback=self.audio_callback
            )
            self.stream.start()
            self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
            self.processing_thread.start()
        except Exception as e:
            self.status_label.setText(f"Voice Status: Error - {e}")
            self.voice_cmd_btn.setChecked(False)

    def stop_voice_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.transcript_text.setText("")
        self.status_label.setText("Voice Status: Recording stopped")
        self.voice_cmd_btn.setText("Voice Command Mode")

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, flush=True)
        self.audio_queue.put(bytes(indata))
        audio_data = np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0
        self.signals.waveform_update.emit(audio_data)

    def process_audio(self):
        while self.is_recording:
            try:
                data = self.audio_queue.get(timeout=1)
                if self.recognizer and self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        self.signals.transcript_update.emit(text)
                        self.check_for_commands(text)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")

    def update_transcript(self, text):
        self.transcript_text.setText(text)

    def update_waveform(self, audio_data):
        self.waveform_curve.setData(audio_data)

    def check_for_commands(self, text):
        text_lower = text.lower()
        if any(word in text_lower for word in ['abort', 'stop', 'cancel', 'halt']):
            if self.current_model_thread and self.current_model_thread.is_alive():
                self.stop_model_flag.set()
                self.signals.command_detected.emit("STOP")
            return
        for command, model_name in self.command_models.items():
            if command in text_lower:
                self.signals.command_detected.emit(command)
                break

    def handle_command(self, command):
        if command == "STOP":
            self.model_status.setText("Model Status: Execution stopped")
            self.model_status.setStyleSheet("color:#ff0000;font-weight:bold;")
            return
        if self.current_model_thread and self.current_model_thread.is_alive():
            self.stop_model_flag.set()
            self.current_model_thread.join(timeout=2)
        model_name = self.command_models.get(command)
        if model_name:
            self.stop_model_flag.clear()
            self.current_model_thread = threading.Thread(
                target=self.run_model, args=(model_name, command), daemon=True
            )
            self.current_model_thread.start()

    def run_model(self, model_name, command):
        # Validate model name
        valid_models = ["feed_model", "clean_model", "medicine_model"]
        if model_name not in valid_models:
            self.model_status.setText(f"Model Status: Unknown model '{model_name}'")
            self.model_status.setStyleSheet("color:#ff0000;font-weight:bold;")
            return

        self.model_status.setText(f"Model Status: Running {model_name}")
        self.model_status.setStyleSheet("color:#00ff00;font-weight:bold;")

        # --- Base command (common to all models) ---
        base_command = (
            "python lerobot/scripts/control_robot.py record "
            "--robot-path lerobot/configs/robot/moss.yaml "
            "--fps 30 --root data --repo-id test/eval_paper_task "
            "--tags moss tutorial eval "
            "--warmup-time-s 70 --episode-time-s 100 --reset-time-s 60 "
            "--num-episodes 24 "
            f"-p /Users/dharunish/Desktop/Robotics/trained_models/{model_name}"
        )

        # --- Open a new macOS Terminal and run the command ---
        terminal_command = [
            "osascript", "-e",
            f'tell application "Terminal" to do script "cd {os.getcwd()} && {base_command}; echo \\"Model {model_name} finished (exit code $?)\\"; exec bash -i"'
        ]

        try:
            self.model_process = subprocess.Popen(terminal_command)
        except Exception as e:
            self.model_status.setText(f"Model Status: Error starting process - {e}")
            self.model_status.setStyleSheet("color:#ff0000;font-weight:bold;")
            return

    def apply_theme(self):
        """Enhanced theme with better color panels and centered text"""
        self.setStyleSheet(
            "QMainWindow{background:#0f0f1e;} "
            "QWidget{background:#0f0f1e;color:#ffffff;font-family:'Segoe UI',Arial,sans-serif;} "
            "QLabel#title_label{color:#00fff0;font-size:52px;font-weight:bold;padding:10px;} "
            "QLabel#status_label{color:#00ff88;font-size:16px;font-weight:bold;padding:8px;} "
            "QLabel#model_status{font-size:18px;font-weight:bold;padding:12px;background:#1a1a2e;border-radius:8px;margin-top:10px;} "
            "QPushButton{font-size:18px;font-weight:bold;padding:15px;border-radius:12px;border:2px solid #3a3a4e;background:#2a2a3e;color:#aaaaaa;} "
            "QPushButton:hover{background:#3a3a4e;color:#ffffff;border:2px solid #4a4a5e;} "
            "QPushButton#voice_cmd_btn:checked{background:#ff0066;color:white;border:2px solid #ff0066;} "
            "QPushButton#training_btn:checked{background:#ff0066;color:white;border:2px solid #ff0066;} "
            "QFrame#left_panel,QFrame#right_panel{background:#1a1a2e;border-radius:15px;border:2px solid #2a2a3e;} "
            "QLabel#panel_header{color:#00fff0;font-size:18px;font-weight:bold;padding:10px;background:transparent;} "
            "QWidget#transcript_container{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #1a1a2e,stop:0.5 #252540,stop:1 #1a1a2e);border-radius:12px;border:2px solid #3a3a5e;} "
            "QLabel#transcript_text{color:#ff00ff;font-size:52px;font-weight:bold;background:transparent;padding:30px;qproperty-alignment:AlignCenter;} "
            "QTextEdit#teleop_text{background:#1a1a2e;color:#00fff0;font-size:20px;border-radius:12px;border:2px solid #2a2a3e;padding:20px;} "
            "QLabel#training_header{color:#ff00ff;font-size:28px;font-weight:bold;padding:15px;} "
            "QLabel#info_label{color:#00fff0;font-size:24px;padding:10px;}"
        )
        
        self.voice_cmd_btn.setObjectName("voice_cmd_btn")
        self.training_btn.setObjectName("training_btn")

    def closeEvent(self, event):
        self.stop_voice_recording()
        if self.current_model_thread and self.current_model_thread.is_alive():
            self.stop_model_flag.set()
            self.current_model_thread.join(timeout=2)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CareBot Command System")
    app.setOrganizationName("CareBot")
    # Optional: Set app icon if you have an icon file
    app.setWindowIcon(QIcon('/Users/dharunish/Desktop/Robotics/App/icon.png'))
    window = VoiceCommandUI()
    #window.show()
    window.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()