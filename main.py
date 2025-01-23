# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import wandb
from datetime import datetime
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd

# CONFIGS
EPOCHS = 500
LEARNING_RATE = 0.001
NUM_SAMPLES = 1000
NUM_CLASSES = 2
HIDDEN_LAYERS = (100,)

CONFIG = dict(epochs=EPOCHS, learning_rate=LEARNING_RATE, hidden_layers=HIDDEN_LAYERS, num_samples=NUM_SAMPLES, num_classes=NUM_CLASSES)


# Get the timestamp for the current run
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS

# Step 5: Data Visualization
def visualize_data(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k") # !! Only plotting the first two features
    plt.title("Classification Data")
    # plt.show()
    wandb.log({"generated_samples_plot": plt}) # Save the plot to WandB


# Step 6: User Input and Output Functionality
def predict_user_input(model, user_data):
    try:
        prediction = model.predict([user_data])
        return prediction
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Skeleton Workflow
if __name__ == "__main__":
    
    # Step 4: Setup WandB
    timestamp = get_timestamp()
    wandb.init(project="foai-classification-project", config=CONFIG, name=f"run_{timestamp}")
    
    # Step 1: Generate sample data
    X, y = make_classification(n_samples=NUM_SAMPLES, n_features=10, n_informative=8, 
                               n_redundant=2, n_classes=NUM_CLASSES, random_state=40)
    # random_state for reproducibility.

    # # Visualize the generated data
    visualize_data(X, y)
    # Create a DataFrame with X and y
    df = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(X.shape[1])])
    df["Class"] = y
    # Pair plot
    sns.pairplot(df, hue="Class", diag_kind="kde")
    plt.show()

    # Step 2: Split data (and shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40, shuffle=True)

    # Step 3: Build and train the model
    model = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS, max_iter=EPOCHS, random_state=40, verbose=True)
    model.fit(X_train, y_train)
    # Log the entire loss curve to WandB
    for epoch, loss in enumerate(model.loss_curve_):
        wandb.log({"loss": loss, "epoch": epoch + 1})

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Log the accuracy to WandB
    wandb.summary['test_accuracy'] = accuracy
    wandb.finish() # Finish the run

    # # Step 6: User prediction (example)
    # user_data = X_test[0]  # Example input
    # prediction = predict_user_input(model, user_data)
    # print("Prediction for user input:", prediction)