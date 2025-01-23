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
from math import ceil

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

# Step 6: User Input Functionality
def predict_user_input(model, inp_min, inp_max):
    try:
        # Prompt the user for input
        print("\nEnter feature values separated by spaces (10 features expected):")
        user_input = input().strip()

        # Convert input to a numpy array
        user_data = np.array([float(x) for x in user_input.split()]).reshape(1, -1)
        
        # Validate the range of user inputs
        if np.any(user_data < inp_min) or np.any(user_data > inp_max):
            print(f"Warning: Some values are outside the expected range ({inp_min} to {inp_max}).")

        # Ensure the correct number of features
        if user_data.shape[1] != 10:
            print("Error: You must enter exactly 10 features.")
            return

        # Make a prediction
        prediction = model.predict(user_data)
        return prediction

    except Exception as e:
        print(f"Error during prediction: {e}")


# Skeleton Workflow
if __name__ == "__main__":
    
    # Step 4: Setup WandB
    timestamp = get_timestamp()
    wandb.init(project="foai-classification-project", config=CONFIG, name=f"run_{timestamp}")
    
    # Step 1: Generate sample data
    X, y = make_classification(n_samples=NUM_SAMPLES, n_features=10, n_informative=8, 
                               n_redundant=2, n_classes=NUM_CLASSES, random_state=40)
    # random_state for reproducibility. Same sample data will be generated each time.

    # Visualize the generated data
    # Create a DataFrame with X and y
    df = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(X.shape[1])])
    df["Class"] = y
    # Pair plot - pairwise scatter plots for all combinations of features
    sns.pairplot(df, hue="Class", diag_kind="kde")
    plt.savefig(f"pairplot-{timestamp}.png")

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

    # Log to WandB
    # Log evaluation metrics to WandB
    wandb.sklearn.plot_roc(y_test, model.predict_proba(X_test), labels=[0, 1])                  # Receiver Operating Characteristic (ROC) curve
    wandb.sklearn.plot_precision_recall(y_test, model.predict_proba(X_test), labels=[0, 1])     # Precision-Recall curve
    wandb.sklearn.plot_confusion_matrix(y_test, model.predict(X_test), labels=[0, 1])           # Confusion matrix
    wandb.summary['test_accuracy'] = accuracy
    wandb.finish() # Finish the run

    # Step 6: User prediction (example)
    print(f"Feature ranges:\nMin: {X.min(axis=0)}\nMax: {X.max(axis=0)}") # To see the range of values that each of the features can take
    inp_min = np.min(X)
    inp_max = np.max(X)
    prediction = predict_user_input(model, ceil(inp_min), ceil(inp_max))
    print("Predicted Class for user input:", prediction)