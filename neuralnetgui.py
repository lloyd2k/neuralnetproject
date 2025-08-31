# neuralnet_gui.py
# -------------------------------------------------------------
# GUI for Neural Network training and testing 
# Features:
# - Allows user to configure and train/test a neural network
# -------------------------------------------------------------

import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel,
    QSpinBox, QHBoxLayout, QLineEdit, QComboBox, QFrame
)
from PyQt5.QtCore import Qt
from gitneuralnet import NeuralNetwork

class NeuralNetApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.nn = None  # Placeholder for neural network

    def initUI(self):
        self.setWindowTitle("Neural Network Trainer")
        self.setGeometry(100, 100, 600, 450)

        layout = QVBoxLayout()

        # ---------------- Header Banner ----------------
        header = QLabel("Neural Network Trainer")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 22px; font-weight: bold; color: #1E3D59; padding: 12px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #DFF6FF, stop:1 #B6E6FF); border-radius: 10px;")
        layout.addWidget(header)

        # ---------------- Status Display ----------------
        self.status = QLabel("Ready")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("background-color: #E8F5E9; color: #2E7D32; font-size: 14px; padding: 6px; border-radius: 8px;")
        layout.addWidget(self.status)

        # ---------------- Hidden Layers Input ----------------
        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("Hidden Layers (comma-separated):"))
        self.hidden_input = QLineEdit("64,32")  # default hidden layers
        hlayout.addWidget(self.hidden_input)
        layout.addLayout(hlayout)

        # ---------------- Epochs Input ----------------
        hlayout2 = QHBoxLayout()
        hlayout2.addWidget(QLabel("Epochs:"))
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 100)
        self.epochs_input.setValue(10)
        hlayout2.addWidget(self.epochs_input)
        layout.addLayout(hlayout2)

        # ---------------- Batch Size Input ----------------
        hlayout3 = QHBoxLayout()
        hlayout3.addWidget(QLabel("Batch Size:"))
        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 256)
        self.batch_input.setValue(32)
        hlayout3.addWidget(self.batch_input)
        layout.addLayout(hlayout3)

        # ---------------- Activation Function Selection ----------------
        hlayout4 = QHBoxLayout()
        hlayout4.addWidget(QLabel("Activation Function:"))
        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["relu", "leaky_relu", "elu", "tanh", "sigmoid", "silu"])
        hlayout4.addWidget(self.activation_combo)
        layout.addLayout(hlayout4)

        # ---------------- Initialization Method Selection ----------------
        hlayout5 = QHBoxLayout()
        hlayout5.addWidget(QLabel("Initialization Method:"))
        self.init_combo = QComboBox()
        self.init_combo.addItems(["he", "xavier", "default"])
        hlayout5.addWidget(self.init_combo)
        layout.addLayout(hlayout5)

        # ---------------- Buttons ----------------
        self.init_button = QPushButton("Initialize Model")
        self.init_button.clicked.connect(self.init_model)
        layout.addWidget(self.init_button)

        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        self.test_button = QPushButton("Test Model")
        self.test_button.clicked.connect(self.test_model)
        layout.addWidget(self.test_button)

        # Apply styling to buttons
        button_style = """
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border-radius: 10px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """
        self.init_button.setStyleSheet(button_style)
        self.train_button.setStyleSheet(button_style)
        self.test_button.setStyleSheet(button_style)

        self.setLayout(layout)

    # ---------------- Model Functions ----------------
    def init_model(self):
        try:
            hidden_layers = list(map(int, self.hidden_input.text().split(',')))
            batch_size = self.batch_input.value()
            activation = self.activation_combo.currentText()
            init_method = self.init_combo.currentText()

            self.nn = NeuralNetwork(hidden_layers=hidden_layers,
                                    batch_size=batch_size,
                                    activation=activation,
                                    init_method=init_method)
            self.status.setText(f"Model initialized with {activation} activation and {init_method} init")
            self.status.setStyleSheet("background-color: #E3F2FD; color: #1565C0; font-size: 14px; padding: 6px; border-radius: 8px;")
        except Exception as e:
            self.status.setText(f"Error: {e}")
            self.status.setStyleSheet("background-color: #FFEBEE; color: #C62828; font-size: 14px; padding: 6px; border-radius: 8px;")

    def train_model(self):
        if not self.nn:
            self.status.setText("Model not initialized")
            return
        try:
            epochs = self.epochs_input.value()
            self.nn.train(epochs=epochs)
            self.status.setText("Training complete")
            self.status.setStyleSheet("background-color: #E8F5E9; color: #2E7D32; font-size: 14px; padding: 6px; border-radius: 8px;")
        except Exception as e:
            self.status.setText(f"Training error: {e}")
            self.status.setStyleSheet("background-color: #FFEBEE; color: #C62828; font-size: 14px; padding: 6px; border-radius: 8px;")

    def test_model(self):
        if not self.nn:
            self.status.setText("Model not initialized")
            return
        try:
            self.nn.test()
            self.status.setText("Testing complete")
            self.status.setStyleSheet("background-color: #E8F5E9; color: #2E7D32; font-size: 14px; padding: 6px; border-radius: 8px;")
        except Exception as e:
            self.status.setText(f"Testing error: {e}")
            self.status.setStyleSheet("background-color: #FFEBEE; color: #C62828; font-size: 14px; padding: 6px; border-radius: 8px;")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = NeuralNetApp()
    window.show()
    sys.exit(app.exec_())
