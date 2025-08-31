# -------------------------------------------------------------
# A fully customizable neural network implementation from scratch
# Features:
# - Supports multiple activation functions (ReLU, Leaky ReLU, Tanh, Sigmoid)
# - Supports different weight initialization methods (He, Xavier, Default small random)
# - Tracks both loss and accuracy during training
# - Plots training vs validation curves
# - Saves and loads models
# - Confusion matrix for evaluation
# -------------------------------------------------------------

import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ---------------- Activation Functions -----------------
def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01): return np.where(x > 0, x, alpha * x)
def leaky_relu_derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

def elu(x):
    return np.where(x > 0, x, np.exp(x) - 1)
def elu_derivative(x):
    return np.where(x > 0, 1, np.exp(x))

def tanh(x): return np.tanh(x)
def tanh_derivative(x): return 1 - np.tanh(x) ** 2

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return sigmoid(x) * (1 - sigmoid(x))

def silu(x): return x/(1 + np.exp(-x))
def silu_derivative(x):
    s = 1/(1 + np.exp(-x))
    return s + x * s * (1 - s)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9))


# ---------------- Neural Network Class -----------------
class NeuralNetwork:
    def __init__(self, hidden_layers, learning_rate=0.01, batch_size=32,
                 activation="relu", init_method="he"):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.layers = hidden_layers
        self.weights, self.biases = [], []
        self.gradient_norms = []
        self.training_losses, self.validation_losses = [], []
        self.training_accuracies, self.validation_accuracies = [], []

        # Select activation function
        self.activation_name = activation
        self.activation, self.activation_derivative = self._select_activation(activation)

        # Initialization method
        self.init_method = init_method

        # Load dataset (MNIST IDX files must be in same folder)
        self._load_data()
        self._initialize_weights()

    # ---------------- Helper Methods -----------------
    def _select_activation(self, name):
        if name == "relu": return relu, relu_derivative
        if name == "leaky_relu": return leaky_relu, leaky_relu_derivative
        if name == "elu": return elu, elu_derivative
        if name == "tanh": return tanh, tanh_derivative
        if name == "sigmoid": return sigmoid, sigmoid_derivative
        if name == "silu": return silu, silu_derivative
        raise ValueError(f"Unknown activation: {name}")

    def _load_data(self):
        # Training set
        self.training_data = idx2numpy.convert_from_file('train-images.idx3-ubyte').astype('float32') / 255.0
        self.training_data = self.training_data.reshape(self.training_data.shape[0], -1).T
        self.training_labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
        # Test set
        self.testing_data = idx2numpy.convert_from_file('t10k-images.idx3-ubyte').astype('float32') / 255.0
        self.testing_data = self.testing_data.reshape(self.testing_data.shape[0], -1).T
        self.testing_labels = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

    def _initialize_weights(self):
        layer_sizes = [784] + self.layers + [10]
        for i in range(len(layer_sizes) - 1):
            shape = (layer_sizes[i+1], layer_sizes[i])            
            if self.init_method == "he_n":
                sigma = np.sqrt(2. / layer_sizes[i])
                weight = np.random.normal(0, sigma, size=shape)
            elif self.init_method == "he_u":
                weight = np.random.uniform(-(np.sqrt(6/layer_sizes[i])), np.sqrt(6/layer_sizes[i+1]), size=shape)
            elif self.init_method == "xavier":
                sigma = np.sqrt(2/(layer_sizes[i] + layer_sizes[i+1]))
                weight = np.random.normal(0, sigma, size=shape)
            elif self.init.method == "lecun":
                sigma = 1/layer_sizes[i]
                weight = np.random.normal(0, sigma, size=shape)
            else:
                weight = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.01
            bias = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(weight)
            self.biases.append(bias)

    # ---------------- Forward & Backward -----------------
    def forward(self, x):
        activations, z_values = [x], []
        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            z_values.append(z)
            activations.append(self.activation(z))
        z = np.dot(self.weights[-1], activations[-1]) + self.biases[-1]
        z_values.append(z)
        activations.append(softmax(z))
        return activations, z_values

    def backward(self, x, y, activations, z_values):
        grads_w, grads_b = [0]*len(self.weights), [0]*len(self.biases)
        # Output layer delta
        delta = activations[-1] - y
        grads_w[-1] = np.dot(delta, activations[-2].T)
        grads_b[-1] = delta
        # Hidden layers delta
        for l in range(2, len(self.layers)+2):
            z = z_values[-l]
            sp = self.activation_derivative(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            grads_w[-l] = np.dot(delta, activations[-l-1].T)
            grads_b[-l] = delta
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    # ---------------- Training -----------------
    def train(self, epochs):
        self.gradient_norms = [[] for _ in range(len(self.weights))] # Reset for each training
        samples = self.training_data.shape[1]
        for epoch in range(epochs):
            permutation = np.random.permutation(samples)
            data_shuffled = self.training_data[:, permutation]
            labels_shuffled = self.training_labels[permutation]
            epoch_loss, correct = 0, 0

            epoch_grads = [[] for _ in range(len(self.weights))] # Store batch grads for this epoch

            for i in tqdm(range(0, samples, self.batch_size), desc=f"Epoch {epoch+1}"):
                x_batch = data_shuffled[:, i:i+self.batch_size]
                y_batch = labels_shuffled[i:i+self.batch_size]
                for j in range(x_batch.shape[1]):
                    x = x_batch[:, j].reshape(-1, 1)
                    y = np.zeros((10, 1)); y[y_batch[j]] = 1
                    activations, z_values = self.forward(x)
                    loss = cross_entropy_loss(y, activations[-1])
                    epoch_loss += loss

                    # --- Begin: Gradient norm tracking ---
                    # Compute gradients but do not update weights yet
                    grads_w, grads_b = [0]*len(self.weights), [0]*len(self.biases)
                    delta = activations[-1] - y
                    grads_w[-1] = np.dot(delta, activations[-2].T)
                    grads_b[-1] = delta
                    for l in range(2, len(self.layers)+2):
                        z = z_values[-l]
                        sp = self.activation_derivative(z)
                        delta = np.dot(self.weights[-l+1].T, delta) * sp
                        grads_w[-l] = np.dot(delta, activations[-l-1].T)
                        grads_b[-l] = delta
                    # Store L2 norm for each layer
                    for idx, gw in enumerate(grads_w):
                        epoch_grads[idx].append(np.linalg.norm(gw))
                    # --- End: Gradient norm tracking ---

                    # Now update weights as before
                    for k in range(len(self.weights)):
                        self.weights[k] -= self.learning_rate * grads_w[k]
                        self.biases[k] -= self.learning_rate * grads_b[k]

                    if np.argmax(activations[-1]) == y_batch[j]: correct += 1

            for i in range(len(self.weights)):
                avg_norm = np.mean(epoch_grads[i])
                self.gradient_norms[i].append(avg_norm)

            avg_loss, accuracy = epoch_loss/samples, correct/samples
            self.training_losses.append(avg_loss)
            self.training_accuracies.append(accuracy)
            val_loss, val_acc = self.evaluate_loss_and_accuracy()
            self.validation_losses.append(val_loss)
            self.validation_accuracies.append(val_acc)
            print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}, Acc: {accuracy:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        self.plot_metrics()
        self.save_model("model.pkl")

    def evaluate_loss_and_accuracy(self):
        total_loss, correct = 0, 0
        for i in range(self.testing_data.shape[1]):
            x = self.testing_data[:, i].reshape(-1, 1)
            y = np.zeros((10, 1)); y[self.testing_labels[i]] = 1
            activations, _ = self.forward(x)
            total_loss += cross_entropy_loss(y, activations[-1])
            if np.argmax(activations[-1]) == self.testing_labels[i]: correct += 1
        return total_loss/self.testing_data.shape[1], correct/self.testing_data.shape[1]
        

    # ---------------- Visualization -----------------
    def plot_metrics(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        ax1.plot(self.training_losses, label="Train Loss")
        ax1.plot(self.validation_losses, label="Val Loss")
        ax1.set_title("Loss per Epoch"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid()
        ax2.plot(self.training_accuracies, label="Train Acc")
        ax2.plot(self.validation_accuracies, label="Val Acc")
        ax2.set_title("Accuracy per Epoch"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.grid()
        plt.show()

    # ---------------- Save & Load -----------------
    def save_model(self, filename):
        with open(filename, 'wb') as f: pickle.dump((self.weights, self.biases), f)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        with open(filename, 'rb') as f: self.weights, self.biases = pickle.load(f)
        print(f"Model loaded from {filename}")

    # ---------------- Testing -----------------
    def test(self):
        correct, all_preds, all_labels = 0, [], []
        for i in range(self.testing_data.shape[1]):
            x = self.testing_data[:, i].reshape(-1, 1)
            label = self.testing_labels[i]
            activations, _ = self.forward(x)
            prediction = np.argmax(activations[-1])
            all_preds.append(prediction); all_labels.append(label)
            if prediction == label: correct += 1
        acc = correct/self.testing_data.shape[1]
        print(f"Test Accuracy: {acc*100:.2f}%")
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.show()


if __name__ == '__main__':
    nn = NeuralNetwork(hidden_layers=[64, 32], learning_rate=0.01, batch_size=32, activation="relu", init_method="he")
    nn.train(epochs=10)
    nn.test()
