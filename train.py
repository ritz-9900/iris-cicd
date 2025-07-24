# train.py (Refactored for Hyperparameter Tuning with MLflow)

import pandas as pd

from sklearn.model_selection import train_test_split


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os


# Define constants for file paths


MODEL_DIR = 'artifacts'
MODEL_NAME = 'best_model.joblib' # We now save the 'best' model
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
    
    # --- 2. Set up MLflow Experiment ---
    print(f"Setting up MLflow experiment: '{MLFLOW_EXPERIMENT_NAME}'")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # --- 3. Hyperparameter Tuning with GridSearchCV ---
    print("Setting up hyperparameter grid search...")
    # Define the parameter grid to search over
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }

    # Initialize the Decision Tree model and the Grid Search
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    print("Running grid search...")
    grid_search.fit(X_train, y_train)

    # Get the best model found by the grid search
    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")

    # --- 4. MLflow Tracking ---
    # Start an MLflow run to log the results of our experiment
    with mlflow.start_run():
        print("Logging experiment results to MLflow...")

        # Log the best parameters found
        mlflow.log_params(grid_search.best_params_)

        # Evaluate the best model
        prediction = best_model.predict(X_test)
        accuracy = metrics.accuracy_score(prediction, y_test)
        
        # Log the final evaluation metric
        mlflow.log_metric("accuracy", accuracy)
        print(f"Best model accuracy: {accuracy:.3f}")

        # Log the trained model as an artifact. This is a key feature of MLflow.
        # It allows us to version and retrieve the exact model object later.
        mlflow.sklearn.log_model(best_model, "model")

        # Log a tag to easily identify this run
        mlflow.set_tag("model_type", "DecisionTreeClassifier")

    # --- 5. Save Artifacts (Same as before, but for the best model) ---
    print(f"Saving metrics to {METRICS_FILE}...")
    with open(METRICS_FILE, "w") as f:
        f.write(f"Accuracy: {accuracy:.3f}\n")
        f.write(f"Best Params: {grid_search.best_params_}\n")

    print(f"Saving best model to {MODEL_PATH}...")
    joblib.dump(best_model, MODEL_PATH)

    print("Training, tuning, and tracking complete.")
    return accuracy

if __name__ == "__main__":

    print("Running simplified train.py as a script...")
    train_and_evaluate()
    print("\ntrain.py script finished.")


##

