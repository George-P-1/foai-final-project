import numpy as np
from joblib import load

# Load the trained model
model_path = "model-.joblib"
model = load(model_path)
print(f"Model loaded from {model_path}")

# Step 6: User Input Functionality
def predict_user_input(model):
    try:
        # Prompt the user for input
        print("\nEnter feature values separated by spaces (10 features expected):")
        user_input = input().strip()

        # Convert input to a numpy array
        user_data = np.array([float(x) for x in user_input.split()]).reshape(1, -1)

        # Ensure the correct number of features
        if user_data.shape[1] != 10:
            print("Error: You must enter exactly 10 features.")
            return

        # Make a prediction
        prediction = model.predict(user_data)
        return prediction

    except Exception as e:
        print(f"Error during prediction: {e}")

# Predict for user input
predict_user_input(model)
