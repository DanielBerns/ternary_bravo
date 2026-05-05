import pickle
import numpy as np
import matplotlib.pyplot as plt
from model import DeepTernaryNetworkMHot

def main(epochs=50, lr=0.01):
    with open("dataset.pkl", "rb") as f:
        data = pickle.load(f)
        
    vocab_size = len(data["word_to_id"])
    
    # Reconstruct model and load weights
    model = DeepTernaryNetworkMHot(vocab_size, [64, 64], vocab_size)
    model.load_weights("weights.pkl")

    train_losses, test_accuracies = [], []

    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0
        
        # Training Pass
        for x_ctx, y_target in zip(data["X_train"], data["Y_train"]):
            y_one_hot = np.zeros(vocab_size)
            y_one_hot[y_target] = 1.0
            
            loss, _ = model.train_step(x_ctx, y_one_hot, lr=lr)
            epoch_loss += loss
            
        train_losses.append(epoch_loss / len(data["X_train"]))
        
        # Testing Pass
        correct = 0
        for x_ctx, y_target in zip(data["X_test"], data["Y_test"]):
            y_pred = model.forward(x_ctx)
            if np.argmax(y_pred) == y_target:
                correct += 1
                
        accuracy = correct / max(1, len(data["X_test"]))
        test_accuracies.append(accuracy)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_losses[-1]:.4f} - Test Acc: {accuracy:.4f}")

    # Save updated weights
    model.save_weights("weights.pkl")
    print("Training complete. Weights saved.")

    # Generate Performance Graphics
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='red')
    plt.title('Categorical Cross-Entropy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy', color='blue')
    plt.title('Prediction Accuracy')
    plt.legend()
    
    plt.savefig("training_metrics.png")
    print("Saved training_metrics.png")

if __name__ == "__main__":
    main()