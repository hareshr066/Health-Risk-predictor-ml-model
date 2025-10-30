import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV # NEW: GridSearch
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import os
from sklearn.metrics import classification_report, accuracy_score, f1_score

# --- CONFIGURATION ---
DATA_FOLDER_PATH = r'C:\Users\hrato\OneDrive\Desktop\datasets\Training datasets'
TARGET_COLUMN = 'Outcome'

# Define the columns used in your final model.
FINAL_FEATURE_COLS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

COLS_TO_IMPUTE = ['Glucose', 'BloodPressure', 'BMI', 'SkinThickness', 'Insulin']

# --- MAPPING CRITICAL: Generalized mapping for all potential column names ---
COLUMN_MAPPING = {
    'Pregnancies': 'Pregnancies', 'Glucose': 'Glucose', 'PlasmaGlucose': 'Glucose',
    'Avg_Glucose_Level': 'Glucose', 'BloodPressure': 'BloodPressure', 'DiastolicBP': 'BloodPressure',
    'RestingBP': 'BloodPressure', 'SkinThickness': 'SkinThickness', 'Triceps_Skin_Fold': 'SkinThickness',
    'Insulin': 'Insulin', '2_Hour_Serum_Insulin': 'Insulin', 'BMI': 'BMI',
    'Body_Mass_Index': 'BMI', 'DiabetesPedigreeFunction': 'DiabetesPedigreeFunction',
    'Age': 'Age', 'Age_in_Years': 'Age',
    'Outcome': 'Outcome', 'Diabetes_binary': 'Outcome', 'HeartDisease': 'Outcome',
    'target': 'Outcome', 'HeartDiseaseorAttack': 'Outcome'
}


def load_and_standardize_data(folder_path, mapping, final_cols, target_col):
    """Loads, renames, and combines all CSV files in a given folder."""
    all_dfs = []
    required_cols_with_target = final_cols + [target_col]

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)

            print(f"Loading and processing {filename}...")

            try:
                df = pd.read_csv(file_path, index_col=0)
            except (ValueError, KeyError, TypeError):
                df = pd.read_csv(file_path)

            df.rename(columns={k: v for k, v in mapping.items() if k in df.columns}, inplace=True)

            for col in final_cols:
                if col not in df.columns:
                    df[col] = 0 if col == 'Pregnancies' else np.nan

            if target_col not in df.columns:
                print(f"WARNING: Target column 'Outcome' not found after mapping in {filename}. Skipping.")
                continue

            df = df[[col for col in required_cols_with_target if col in df.columns]]
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No valid CSV datasets were loaded from the folder. Check path and file formats.")

    final_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n--- Merging Complete. Total samples: {final_df.shape[0]} ---")
    return final_df


def tune_and_train_model():
    """Performs cleaning, tuning, and trains the final Ensemble model."""

    df = load_and_standardize_data(DATA_FOLDER_PATH, COLUMN_MAPPING, FINAL_FEATURE_COLS, TARGET_COLUMN)

    # --- STEP 1: CLEANING AND IMPUTATION ---
    for col in COLS_TO_IMPUTE:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # --- STEP 2: SPLIT AND SCALE ---
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, 'scaler.pkl')
    print("\nScaler saved as 'scaler.pkl'")

    # --- STEP 3: HYPERPARAMETER TUNING FOR XGBOOST (Focus on maximizing Precision/F1) ---
    print("\n--- Starting Grid Search for XGBoost Parameters (Optimizing F1-Score) ---")

    ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

    # Define a smaller parameter grid for hackathon speed (3x2=6 combinations)
    param_grid = {
        'n_estimators': [100, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }

    xgb_base = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False, scale_pos_weight=ratio)

    # Use GridSearch, optimizing for the F1-score of the positive (risk=1) class
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='f1',
        cv=2, # Use low CV count (2) for speed
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)

    best_xgb_params = grid_search.best_params_
    print(f"\nBest XGBoost Parameters Found: {best_xgb_params}")

    # --- STEP 4: TRAIN FINAL ENSEMBLE MODEL ---
    print("\n--- Training Final Voting Classifier with Optimized XGBoost ---")

    # 1. Instantiate Optimized Base Models
    log_clf = LogisticRegression(solver='liblinear', C=1.0, random_state=42)
    svm_clf = SVC(probability=True, kernel='linear', C=0.1, random_state=42)

    # Instantiate XGBoost with the best parameters found
    optimized_xgb_clf = XGBClassifier(
        random_state=42, eval_metric='logloss', use_label_encoder=False,
        scale_pos_weight=ratio, **best_xgb_params # Unpack the best parameters here
    )

    # 2. Create and Train Ensemble
    ensemble_model = VotingClassifier(
        estimators=[
            ('lr', log_clf),
            ('xgb', optimized_xgb_clf),
            ('svc', svm_clf)
        ],
        voting='soft',
        weights=[1, 3, 1]
    )

    ensemble_model.fit(X_train_scaled, y_train)

    # --- STEP 5: EVALUATION AND SAVING ---
    ensemble_pred = ensemble_model.predict(X_test_scaled)

    print("\n--- FINAL ENSEMBLE MODEL PERFORMANCE ---")
    print(f"Overall Accuracy: {accuracy_score(y_test, ensemble_pred):.4f}")
    print(f"F1 Score (Risk=1): {f1_score(y_test, ensemble_pred, pos_label=1):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, ensemble_pred))

    # Save the new ensemble model
    joblib.dump(ensemble_model, 'risk_prediction_model.pkl')
    print("\nFINAL OPTIMIZED ENSEMBLE Model saved as 'risk_prediction_model.pkl'")
    print("\n--- All steps complete. Run 'python app.py' to deploy the optimized model! ---")


if __name__ == "__main__":
    try:
        if not os.path.exists(DATA_FOLDER_PATH):
             raise ValueError(f"Data folder not found: {DATA_FOLDER_PATH}. Please verify the path.")

        tune_and_train_model()
    except ValueError as e:
        print(f"FATAL ERROR: {e}")