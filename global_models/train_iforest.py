import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def main():
    # Ensure working directory is global_models so it finds normal.csv and attack.csv
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Loading datasets...")
    normal_df = pd.read_csv('../normal.csv')
    attack_df = pd.read_csv('../attack.csv')

    print("Preprocessing...")
    # Strip column names just in case
    normal_df.columns = normal_df.columns.str.strip()
    attack_df.columns = attack_df.columns.str.strip()

    # Merge datasets for training
    df = pd.concat([normal_df, attack_df], ignore_index=True)

    # Drop timestamp as it's not numeric
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    if 'attack' not in df.columns:
        print("Error: 'attack' column not found in datasets.")
        return

    X_train = df.drop(columns=['attack'])
    y_train = df['attack']

    # Handle potential missing values or non-numeric columns
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    means = X_train.mean()
    X_train = X_train.fillna(means)
    
    # Apply normalization
    print("Normalizing datasets...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Map labels for Isolation Forest
    # IsolationForest outputs 1 for inliers (normal) and -1 for outliers (anomalies)
    y_train_true = np.where(y_train == 0, 1, -1)

    print("Preparing test dataset from attack.csv...")
    test_df = attack_df.copy()
    if 'Timestamp' in test_df.columns:
        test_df = test_df.drop(columns=['Timestamp'])
    
    X_test = test_df.drop(columns=['attack'])
    y_test = test_df['attack']
    
    # Use the same means from training set to fill NAs in test set
    X_test = X_test.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.fillna(means)
    X_test_scaled = scaler.transform(X_test)
    y_test_true = np.where(y_test == 0, 1, -1)

    print("Training Isolation Forest on the whole merged dataset...")
    # Contamination is set to an estimated proportion of outliers in the data.
    outlier_fraction = (y_train == 1).mean()
    # Isolation forest requires contamination to be in (0, 0.5]
    if outlier_fraction == 0:
        contamination = 'auto'
    elif outlier_fraction > 0.5:
        contamination = 0.5
    else:
        contamination = outlier_fraction

    print(f"Estimated contamination: {contamination}")
    clf = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled)

    print("Evaluating on the attack dataset...")
    y_pred_test = clf.predict(X_test_scaled)

    print("--- Test Set Performance (attack.csv) ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_true, y_pred_test))
    print("\nClassification Report:")
    # 1: Normal, -1: Attack
    print(classification_report(y_test_true, y_pred_test, target_names=["Attack (-1)", "Normal (1)"]))

    print("Saving model...")
    joblib.dump(clf, 'isolation_forest_model.pkl')
    print("Model saved to isolation_forest_model.pkl")

if __name__ == "__main__":
    main()
