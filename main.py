# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import wandb
from datetime import datetime

# CONFIGS
EPOCHS = 500
LEARNING_RATE = 0.001

CONFIG = dict(epochs=EPOCHS, learning_rate=LEARNING_RATE, hidden_layers=(100,))


# Get the timestamp for the current run
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS

# Step 1: Sample Data Generation
def generate_data():
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, 
                               n_redundant=2, random_state=42)
    return X, y

# Step 2: Dataset Preparation
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    return X_train, X_test, y_train, y_test

# Step 3: Neural Network Model Construction
def build_and_train_model(X_train, y_train, hidden_layer_sizes=(100,)):
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=EPOCHS, random_state=42)
    wandb.watch(model)  # Watch the model for logging gradients and parameters
    model.fit(X_train, y_train)
    return model

# Step 4: WandB Integration
def setup_wandb():
    timestamp = get_timestamp()
    wandb.init(project="foai-classification-project", config=CONFIG, name=f"run_{timestamp}")

# Step 5: Data Visualization
def visualize_data(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k")
    plt.title("Classification Data")
    plt.show()
    # plt.close() # Close the plot to avoid overlapping plots

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
    
    # Step 4: Setup WandB (currently not logging details, placeholder for later)
    setup_wandb()
    
    # Step 1: Generate data
    X, y = generate_data()

    # Visualize the generated data
    visualize_data(X, y)

    # Step 2: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 3: Build and train the model
    model = build_and_train_model(X_train, y_train, hidden_layer_sizes=(100,))


    # Step 6: User prediction (example)
    user_data = X_test[0]  # Example input
    prediction = predict_user_input(model, user_data)
    print("Prediction for user input:", prediction)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Log the accuracy to WandB
    wandb.summary['test_accuracy'] = accuracy
    wandb.finish() # Finish the run
