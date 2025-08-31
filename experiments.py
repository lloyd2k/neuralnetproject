# activation_experiments.py
# -------------------------------------------------------------
# Script to compare different activation functions and initialization methods
# Runs multiple experiments and plots loss/accuracy curves side by side
# -------------------------------------------------------------

import matplotlib.pyplot as plt
from gitneuralnet import NeuralNetwork
import numpy as np

# Default experiment configurations
experiments = [
    {"activation": "relu", "init": "he_n"},
    {"activation": "relu", "init": "he_u"},
    {"activation": "relu", "init": "xavier"},
    # {"activation": "leaky_relu", "init": "he_n"},
    # {"activation": "tanh", "init": "xavier"},
    # {"activation": "tanh", "init": "lecun"},
    # {"activation": "sigmoid", "init": "xavier"},
    # {"acctivation": "silu", "init": "lecun"}
    ]

results = {}
weight_distributions = {}

# Run experiments
for exp in experiments:
    print(f"Running {exp['activation']} with {exp['init']} init...")
    nn = NeuralNetwork(hidden_layers=[128, 64], learning_rate=0.01, batch_size=32,
                       activation=exp['activation'], init_method=exp['init'])
    initial_weights = np.concatenate([w.flatten() for w in nn.weights])
    nn.train(epochs=50)
    final_weights = np.concatenate([w.flatten() for w in nn.weights])
    results[(exp['activation'], exp['init'])] = (nn.training_losses, nn.validation_losses,
                                               nn.training_accuracies, nn.validation_accuracies, nn.gradient_norms)
    weight_distributions[exp['activation'], exp['init']] = (initial_weights, final_weights)

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
for (act, init), (train_loss, val_loss, train_acc, val_acc, gradient_norms) in results.items():
    ax1.plot(val_loss, label=f"{act}-{init}")
    ax2.plot(val_acc, label=f"{act}-{init}")
ax1.set_title("Validation Loss Comparison"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()
ax2.set_title("Validation Accuracy Comparison"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend()
plt.show()

final_accuracies = []
labels = []
for (act, init), (_, _, _, val_acc, _) in results.items():
    final_accuracies.append(val_acc[-1] * 100)
    labels.append(f"{act}-{init}")

plt.figure(figsize=(10,6))
plt.bar(labels, final_accuracies, color='skyblue')
plt.ylabel("Final Validation Accuracy (%)")
plt.title("Final Validation Accuracy by Activation+Init Combination")
plt.xticks(rotation=45)
min_acc = min(final_accuracies)
max_acc = max(final_accuracies)
plt.ylim(bottom=(min_acc - 0.5), top=(max_acc + 0.5))
plt.tight_layout()
plt.show()

# --- Convergence Rate Plot ---
threshold = 97  # percent
epochs_to_converge = []
for (act, init), (_, _, _, val_acc, _) in results.items():
    # val_acc is a list of floats (0-1), so multiply by 100 for percent
    found = False
    for epoch_idx, acc in enumerate(val_acc):
        if acc * 100 >= threshold:
            epochs_to_converge.append(epoch_idx + 1)  # epochs are 1-based
            found = True
            break
    if not found:
        epochs_to_converge.append(len(val_acc) + 1)  # did not converge

plt.figure(figsize=(10,6))
plt.bar(labels, epochs_to_converge, color='orange')
plt.ylabel(f"Epochs to reach {threshold}% Validation Accuracy")
plt.title(f"Convergence Rate by Activation+Init Combination ({threshold}%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot gradient norms per layer for each experiment
for (act, init), (train_loss, val_loss, train_acc, val_acc, gradient_norms) in results.items():
    nn = NeuralNetwork(hidden_layers=[64,32], learning_rate=0.01, batch_size=32,
                       activation=act, init_method=init)
    # Instead of retraining, you should store nn.gradient_norms in results during the experiment loop!
    # For now, let's assume you stored it as a 5th element:
    # results[(exp['activation'], exp['init'])] = (nn.training_losses, nn.validation_losses,
    #                                              nn.training_accuracies, nn.validation_accuracies,
    #                                              nn.gradient_norms)
    gradient_norms = results[(act, init)][4]
    plt.figure(figsize=(10,6))
    for layer_idx, norms in enumerate(gradient_norms):
        plt.plot(norms, label=f"Layer {layer_idx+1}")
    plt.title(f"Gradient Norms per Layer: {act}-{init}")
    plt.xlabel("Epoch")
    plt.ylabel("Average Gradient L2 Norm")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot weight distribution histograms (before vs after training)
for (act, init), (initial_weights, final_weights) in weight_distributions.items():
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(initial_weights, bins=50, color='blue', alpha=0.7)
    plt.title(f"Initial Weights: {act}-{init}")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.subplot(1,2,2)
    plt.hist(final_weights, bins=50, color='green', alpha=0.7)
    plt.title(f"Final Weights: {act}-{init}")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()