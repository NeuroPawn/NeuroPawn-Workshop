import sys
import random
import csv
import os
import numpy as np
import time
import json
from enum import Enum

from pydantic import BaseModel, Field
from typing import Annotated

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
    QVBoxLayout, QSpinBox, QLineEdit, QFileDialog
)
from PyQt5.QtCore import QTimer, Qt, QEventLoop, QElapsedTimer
from PyQt5.QtGui import QColor, QPalette


class Config(BaseModel):
    num_classes: Annotated[int, Field(ge=1, le=6)] = 2  # Number of classes must be at least 1
    num_sessions: Annotated[int, Field(ge=1)] = 3  # Number of trials must be at least 1
    trial_cooldown_duration: Annotated[int, Field(ge=1)] = 5 # Cooldown duration must be non-negative
    stimuli_size: Annotated[int, Field(ge=150)] = 300
    trial_start_countdown: Annotated[int, Field(ge=1)] = 5
    target_display_duration: Annotated[int, Field(ge=4)] = 4
    flashing_duration: Annotated[int, Field(ge=5)] = 5
    experiment_path: str = ''

class State(Enum):
    COUNTDOWN = 'countdown'
    TARGET = 'target'
    FLASHING = 'flashing'
    COOLDOWN = 'cooldown'
    STOPPED = 'stopped'

def get_widget_centers(num_classes):
    if num_classes == 1:
        return np.array([[1/2, 1/2]])
    elif num_classes == 2:
        return np.array([[1/3 - 0.1, 1/2],[2/3 + 0.1, 1/2]])
    elif num_classes == 3:
        return np.array([[1/4 - 0.1, 1/2],[1/2, 1/2],[3/4 + 0.1, 1/2]])
    elif num_classes == 4:
        return np.array([[1/3 - 0.1, 1/3 - 0.1],[2/3 + 0.1, 1/3 - 0.1],[1/3 - 0.1, 2/3 + 0.1], [2/3 + 0.1, 2/3 + 0.1]])
    elif num_classes == 5:
        return np.array([[1/3 - 0.1, 1/3 - 0.1],[2/3 + 0.1, 1/3 - 0.1],[1/4 - 0.1, 2/3 + 0.1], [1/2, 2/3 + 0.1], [3/4 + 0.1, 2/3 + 0.1]])
    else:
        return np.array([[1/4 - 0.1, 1/3 - 0.1],[1/2, 1/3 - 0.1], [3/4 + 0.1, 1/3 - 0.1], [1/4 - 0.1, 2/3 + 0.1], [1/2, 2/3 + 0.1], [3/4 + 0.1, 2/3 + 0.1]])
    
def get_flashing_frequencies(num_classes):
    if num_classes == 1:
        output = np.array([11])
    elif num_classes == 2:
        output = np.array([5, 15])
    elif num_classes == 3:
        output = np.array([5, 10, 15])
    elif num_classes == 4:
        output = np.array([5, 8, 12, 15])
    elif num_classes == 5:
        output = np.array([5, 7.5, 10, 12.5, 15])
    else:
        output = np.array([5, 7, 9, 11, 13, 15])
    np.random.shuffle(output)
    return output

class FlashingWidget(QWidget):
    def __init__(self, frequency):
        super().__init__()
        self.elapsed_timer = QElapsedTimer()
        self.elapsed_timer.start()

        self.frequency = frequency
        self.is_white = False
        self.toggle_interval = int(1000 / (self.frequency * 2))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.toggle_flash)

        self.setAutoFillBackground(True)
        self.update_color(QColor(0, 0, 0, 0))

    def start_flashing(self):
        self.timer.start(self.toggle_interval)

    def stop_flashing(self):
        self.timer.stop()
        self.update_color(QColor(0, 0, 0, 0))

    def toggle_flash(self):
        self.is_white = not self.is_white
        self.update_color()

    def update_color(self, color=None):
        color = color or (QColor("white") if self.is_white else QColor("black"))
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, color)
        self.setPalette(palette)

    def update_frequency(self, new_frequency):
        self.frequency = new_frequency
        self.toggle_interval = int(1000 / (self.frequency * 2))

class ConfigWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroPawn SSVEP Configuration")
        self.resize(300, 300)

        # Default Config
        self.config = Config()
        self.folder = None
        # Initialize layout
        layout = QVBoxLayout()

        # Experiment ID input (this will still be used for the experiment ID)
        self.experiment_id_input = QLineEdit()
        self.experiment_id_input.setPlaceholderText("Enter Experiment ID")
        layout.addWidget(QLabel("Experiment ID:"))
        layout.addWidget(self.experiment_id_input)
        self.experiment_id_input.textChanged.connect(self.validate_path)

        # Add configuration controls with limits
        self.num_classes_spinbox = self.create_spinbox("Number of Classes", self.config.num_classes, layout, min_val=1, max_val=6)
        self.num_sessions_spinbox = self.create_spinbox("Number of Sessions", self.config.num_sessions, layout, min_val=1, max_val=20)
        self.trial_cooldown_duration_spinbox = self.create_spinbox("Trial Cooldown Duration (s)", self.config.trial_cooldown_duration, layout, min_val=1, max_val=60)
        self.trial_start_countdown_spinbox = self.create_spinbox("Trial Start Countdown (s)", self.config.trial_start_countdown, layout, min_val=1, max_val=10)
        self.target_display_duration_spinbox = self.create_spinbox("Target Display Duration (s)", self.config.target_display_duration, layout, min_val=3, max_val=5)
        self.flashing_duration_spinbox = self.create_spinbox("Flashing Duration (s)", self.config.flashing_duration, layout, min_val=3, max_val=20)

        # Folder picker for experiment path
        self.folder_picker_button = QPushButton("Choose Experiment Folder")
        self.folder_picker_button.clicked.connect(self.open_folder_picker)
        layout.addWidget(self.folder_picker_button)

        # Path status label
        self.path_status_label = QLabel("Path status: Not checked")
        layout.addWidget(self.path_status_label)

        # Start button
        self.start_button = QPushButton("Start Experiment")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_experiment)
        layout.addWidget(self.start_button)

        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def create_spinbox(self, label_text, default_value, layout, min_val, max_val):
        label = QLabel(label_text)
        spinbox = QSpinBox()
        spinbox.setValue(default_value)
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        layout.addWidget(label)
        layout.addWidget(spinbox)
        return spinbox

    def open_folder_picker(self):
        # Open a folder picker dialog to choose the experiment folder
        self.folder = QFileDialog.getExistingDirectory(self, "Select Experiment Folder", os.getcwd())
        if self.folder:
            self.validate_path()
        else:
            self.path_status_label.setText("Path not selected")
            self.start_button.setEnabled(False)

    def validate_path(self):
        # If no specific path is passed, use the default based on experiment_id
        if self.folder is None: return
        self.experiment_id = self.experiment_id_input.text().strip()
        if not self.experiment_id: 
            self.path_status_label.setText(f"Empty Experiment ID")
            self.start_button.setEnabled(False)
            return

        experiment_path = os.path.join(self.folder, self.experiment_id)
        if os.path.exists(experiment_path):
            self.path_status_label.setText(f"Path already exists: {experiment_path}")
            self.start_button.setEnabled(False)
        else:
            self.path_status_label.setText(f"Path available: {experiment_path}")
            self.start_button.setEnabled(True)

    def start_experiment(self):
        self.config.experiment_path = os.path.join(self.folder, self.experiment_id)

        # Update config with UI values
        self.config.num_classes = self.num_classes_spinbox.value()
        self.config.num_sessions = self.num_sessions_spinbox.value()
        self.config.trial_cooldown_duration = self.trial_cooldown_duration_spinbox.value()
        self.config.trial_start_countdown = self.trial_start_countdown_spinbox.value()
        self.config.target_display_duration = self.target_display_duration_spinbox.value()
        self.config.flashing_duration = self.flashing_duration_spinbox.value()

        # Open Experiment Window
        self.experiment_window = ExperimentWindow(self.config)
        self.experiment_window.show()
        self.close()


class ExperimentWindow(QMainWindow):
    current_state: State
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.flashing_widgets: list[FlashingWidget] = []
        self.current_iteration = 0
        self.countdown_label_text = f'Progress: {self.get_progress(self.current_iteration)}\nStarting In: '
        self.frequencies = np.random.permutation(np.repeat(get_flashing_frequencies(self.config.num_classes), self.config.num_sessions))

        os.makedirs(self.config.experiment_path, exist_ok=True)
        self.file_name = os.path.join(self.config.experiment_path, 'state.csv')
        self.config_file_name = os.path.join(self.config.experiment_path, 'config.json')
        
        with open(self.file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "iteration",  "state", "notes"])

        self.setup_UI()
        self.set_state(State.COUNTDOWN)
        self.show_countdown(self.config.trial_start_countdown, self.start_trial)
        self.dump_config_to_json()

    def dump_config_to_json(self):
        try:
            with open(self.config_file_name, mode='w', newline='') as config_file:
                config_file.write(json.dumps(self.config.dict(), indent=4))
        except Exception as e:
            print(f"Error dumping config to JSON: {e}")

    def log_state(self, notes: str):
        timestamp = str(time.time())
        with open(self.file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, self.current_iteration if self.current_state != State.COOLDOWN else (self.current_iteration - 1), self.current_state.value, notes])

    def setup_UI(self):
        self.setWindowTitle("NeuroPawn SSVEP")
        screen = QApplication.primaryScreen().availableGeometry()
        self.window_width = int(screen.width() * 0.8)
        self.window_height = int(screen.height() * 0.9)
        self.setFixedSize(self.window_width, self.window_height)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)

        self.flashing_area = QWidget()
        self.flashing_area.setFixedHeight(int(self.window_height * 0.85))
        self.flashing_area.setFixedWidth(int(self.window_width * 0.95))
        main_layout.addWidget(self.flashing_area, alignment=Qt.AlignmentFlag.AlignCenter)

        self.countdown_label = QLabel("", self)
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.countdown_label.setStyleSheet("font-size: 20px;")
        
        self.stop_button = QPushButton("Stop Experiment")
        self.stop_button.clicked.connect(self.stop_experiment)

        bottom_layout = QVBoxLayout()
        bottom_layout.addWidget(self.countdown_label)
        bottom_layout.addWidget(self.stop_button)
        main_layout.addLayout(bottom_layout)

        self.render_flashing_widgets()

    def get_progress(self, iteration):
        return f"{round(iteration * 100 / (self.config.num_classes * self.config.num_sessions))} %"
    
    def get_random_frequency(self):
        selected_frequency = random.choice(self.frequencies)
        indices = np.where(self.frequencies == selected_frequency)[0]
        self.frequencies = np.delete(self.frequencies, indices[0])
        return selected_frequency
    
    def render_flashing_widgets(self):
        for widget in self.flashing_widgets:
            widget.stop_flashing()
            widget.deleteLater()
        self.flashing_widgets = []

        widget_coords_normalized = get_widget_centers(self.config.num_classes)
        frequencies = get_flashing_frequencies(self.config.num_classes)

        for i in range(self.config.num_classes):
            frequency = frequencies[i]
            widget = FlashingWidget(frequency)
            widget.setFixedSize(int(self.config.stimuli_size*1.5), self.config.stimuli_size)

            widget.setParent(self.flashing_area)
            widget.move(int(self.flashing_area.width() * widget_coords_normalized[i, 0] - self.config.stimuli_size *1.5 / 2), int(self.flashing_area.height() * widget_coords_normalized[i, 1] - self.config.stimuli_size / 2))
            widget.show()

            self.flashing_widgets.append(widget)
    
    def shuffle_frequencies(self):
        new_frequencies = get_flashing_frequencies(self.config.num_classes)
        for i , new_frequency in enumerate(new_frequencies):
            self.flashing_widgets[i].update_frequency(new_frequency)

    def show_countdown(self, seconds, callback):
        self.countdown_value = seconds
        self.countdown_label.setText(f'{self.countdown_label_text}{self.countdown_value}')
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(lambda: self.update_countdown(callback))
        self.countdown_timer.start(1000)

    def update_countdown(self, callback):
        self.countdown_value -= 1
        if self.countdown_value <= 0:
            self.countdown_timer.stop()
            self.countdown_label.setText("")
            callback()
        else:
            self.countdown_label.setText(f'{self.countdown_label_text}{self.countdown_value}')

    def start_trial(self):
        target_frequency = self.get_random_frequency()
        
        target_widget = None
        for widget in self.flashing_widgets:
            if widget.frequency == target_frequency:
                target_widget = widget
                break
        
        self.set_state(State.TARGET, notes=f"target_frequency={target_frequency}_Hz")
        target_widget.update_color(QColor("red"))
        self.delay(self.config.target_display_duration * 1000)
        target_widget.update_color(QColor(0,0,0,0))

        self.set_state(State.FLASHING)
        self.flash_all_widgets()
        self.delay(self.config.flashing_duration * 1000)

        self.current_iteration += 1
        if self.current_iteration < self.config.num_sessions * self.config.num_classes:
            self.start_cooldown()
        else:
            self.stop_experiment(notes='Finished Experiment')


    def start_cooldown(self):
        self.set_state(State.COOLDOWN)
        self.countdown_label_text = f'Progress: {self.get_progress(self.current_iteration)}\nCooldown: '
        def callback():
            self.countdown_label_text = f'Progress: {self.get_progress(self.current_iteration)}\nStarting In: '
            self.shuffle_frequencies()
            self.set_state(State.COUNTDOWN)
            self.show_countdown(self.config.trial_start_countdown, self.start_trial)

        self.show_countdown(self.config.trial_cooldown_duration, callback)

    def flash_all_widgets(self):
        for widget in self.flashing_widgets:
            widget.start_flashing()
        QTimer.singleShot(self.config.flashing_duration * 1000, lambda: [widget.stop_flashing() for widget in self.flashing_widgets])

    def delay(self, milliseconds):
        loop = QEventLoop()
        QTimer.singleShot(milliseconds, loop.quit)
        loop.exec_()

    def stop_experiment(self, *args, notes: str = ""):
        if notes == "": notes = "Experiment Stopped"
        self.set_state(State.STOPPED, notes=notes)
        self.close()
        sys.exit()

    def set_state(self, new_state: State, notes=""):
        self.current_state = new_state
        self.log_state(notes)


app = QApplication(sys.argv)
config_window = ConfigWindow()
config_window.show()
sys.exit(app.exec_())