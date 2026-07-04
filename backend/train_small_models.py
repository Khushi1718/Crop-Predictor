import os
# pyrefly: ignore [missing-import]
import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# This script trains lightweight version of the machine learning models
# that consume very little memory (< 100MB in RAM combined) and are small 
# enough to be saved in Git and deployed on free hosting tiers (e.g., Render/Railway 512MB RAM).

def main():
    print("🌾 Training Lightweight Models for Production Deployment...")
    
    # Check for dataset
    csv_path = "crop_yield.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found in backend directory. Trying root...")
        csv_path = "../crop_yield.csv"
        if not os.path.exists(csv_path):
            print("Error: Could not locate crop_yield.csv. Please ensure it is present.")
            return

    print(f"Reading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Downsample to 50,000 rows (Saves memory, speed up training, reduces model sizes)
    print("Downsampling dataset to 50,000 rows...")
    df_sample = df.sample(n=50000, random_state=42)

    # Convert datatypes
    df_sample['Fertilizer_Used'] = df_sample['Fertilizer_Used'].astype(int)
    df_sample['Irrigation_Used'] = df_sample['Irrigation_Used'].astype(int)

    encoders = {}
    for col in ['Crop', 'Region']:
        le = LabelEncoder()
        df_sample[col + "_enc"] = le.fit_transform(df_sample[col])
        encoders[col] = le

    q1 = df_sample['Yield_tons_per_hectare'].quantile(0.33)
    q2 = df_sample['Yield_tons_per_hectare'].quantile(0.66)

    def categorize(y):
        if y <= q1:
            return 0  # low
        elif y <= q2:
            return 1  # medium
        else:
            return 2  # high

    df_sample['Yield_Class'] = df_sample['Yield_tons_per_hectare'].apply(categorize)

    features = [
        'Rainfall_mm',
        'Temperature_Celsius',
        'Irrigation_Used',
        'Fertilizer_Used',
        'Crop_enc',
        'Region_enc'
    ]

    X = df_sample[features]
    y_class = df_sample['Yield_Class']
    y_reg = df_sample['Yield_tons_per_hectare']

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.25, random_state=42
    )

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.25, random_state=42
    )

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # 2. Train classifiers with restricted complexity to keep model sizes minimal
    print("\n[1/5] Training KNN Classifier (n_neighbors=5)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_c, y_train_c)
    joblib.dump(knn, "models/KNN_model.pkl")
    print(f"KNN Saved! Size: {os.path.getsize('models/KNN_model.pkl')/1024/1024:.2f} MB")

    print("\n[2/5] Training SVM Classifier (subset of 5,000 for training speed)...")
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train_c[:5000], y_train_c[:5000])
    joblib.dump(svm, "models/SVM_model.pkl")
    print(f"SVM Saved! Size: {os.path.getsize('models/SVM_model.pkl')/1024/1024:.2f} MB")

    print("\n[3/5] Training Decision Tree Classifier...")
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train_c, y_train_c)
    joblib.dump(dt, "models/DecisionTree_model.pkl")
    print(f"Decision Tree Saved! Size: {os.path.getsize('models/DecisionTree_model.pkl')/1024/1024:.2f} MB")

    print("\n[4/5] Training Random Forest Classifier (Compact: 15 estimators, max_depth=10)...")
    rf = RandomForestClassifier(n_estimators=15, max_depth=10, random_state=42)
    rf.fit(X_train_c, y_train_c)
    joblib.dump(rf, "models/RandomForest_model.pkl")
    print(f"Random Forest Saved! Size: {os.path.getsize('models/RandomForest_model.pkl')/1024/1024:.2f} MB")

    # 3. Train Linear Regressor (already very small)
    print("\n[5/5] Training Linear Regressor...")
    regressor = LinearRegression()
    regressor.fit(X_train_r, y_train_r)
    joblib.dump(regressor, "models/regressor_model.pkl")
    print(f"Linear Regressor Saved! Size: {os.path.getsize('models/regressor_model.pkl')/1024:.2f} KB")

    print("\n🎉 Success! All production-friendly models saved successfully inside 'backend/models/'.")
    print("These models are ready to be pushed to git / deployed to cloud environments.")

if __name__ == "__main__":
    main()
