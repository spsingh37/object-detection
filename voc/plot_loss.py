import numpy as np
import matplotlib.pyplot as plt

# Load loss values
loss_data = np.load("epoch_losses.npy", allow_pickle=True).item()

# Extract loss values
epochs = range(1, len(loss_data["rpn_classification"]) + 1)
rpn_classification = loss_data["rpn_classification"]
rpn_localization = loss_data["rpn_localization"]
frcnn_classification = loss_data["frcnn_classification"]
frcnn_localization = loss_data["frcnn_localization"]

# Plot RPN Classification Loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, rpn_classification, label="RPN Classification Loss", marker='o', color='blue')
plt.title("RPN Classification Loss Per Epoch", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("rpn_classification_loss.png", dpi=300)
plt.show()

# Plot RPN Localization Loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, rpn_localization, label="RPN Localization Loss", marker='o', color='green')
plt.title("RPN Localization Loss Per Epoch", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("rpn_localization_loss.png", dpi=300)
plt.show()

# Plot FRCNN Classification Loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, frcnn_classification, label="FRCNN Classification Loss", marker='o', color='red')
plt.title("FRCNN Classification Loss Per Epoch", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("frcnn_classification_loss.png", dpi=300)
plt.show()

# Plot FRCNN Localization Loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, frcnn_localization, label="FRCNN Localization Loss", marker='o', color='purple')
plt.title("FRCNN Localization Loss Per Epoch", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("frcnn_localization_loss.png", dpi=300)
plt.show()

