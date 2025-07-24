import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os

# Define constants for file paths
MODEL_DIR = 'artifacts'
MODEL_NAME = 'model.joblib'
METRICS_FILE = 'metrics.txt'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
TRAINING_DATA_PATH = 'data/iris_prepared.csv'

def train_and_evaluate():
    """
    This function encapsulates the entire model training and evaluation process.
    It reads data from a local CSV, trains a model, evaluates it,
    saves the artifacts, and returns the accuracy.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading training data from {TRAINING_DATA_PATH}...")
    try:
        # Load the data directly from the local prepared CSV file
        training_df = pd.read_csv(TRAINING_DATA_PATH)
        print(f"Found {len(training_df)} records.")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error: {TRAINING_DATA_PATH} not found. "
            "Please run prepare_data.py first."
        )

    print("\n--- Training Data Sample ---")
    print(training_df.head())
    print("----------------------------\n")

    print("Splitting data...")
    # Define features (X) and target (y)
    # The species column needs to be mapped from string to integer for the model
    species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    training_df['species_encoded'] = training_df['species'].map(species_map)

    X = training_df.drop(columns=['iris_id', 'event_timestamp', 'species', 'species_encoded'])
    y = training_df['species_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    print("Data split complete.")

    print("Training Decision Tree model...")
    mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
    mod_dt.fit(X_train, y_train)
    print("Model training complete.")

    print("Evaluating model...")
    prediction = mod_dt.predict(X_test)
    accuracy = metrics.accuracy_score(prediction, y_test)
    print(f'The accuracy of the Decision Tree is: {accuracy:.3f}')

    print(f"Saving metrics to {METRICS_FILE}...")
    with open(METRICS_FILE, "w") as f:
        f.write(f"Accuracy: {accuracy:.3f}\n")
    print("Metrics saved.")

    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(mod_dt, MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")

    return accuracy


if __name__ == "__main__":
    print("Running simplified train.py as a script...")
    train_and_evaluate()
    print("\ntrain.py script finished.")

##
