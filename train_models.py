# train_models.py
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from backend.utils import load_data

# Load data
(class_data, reg_data) = load_data()
X_train, X_test, y_train_class, y_test_class = class_data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = reg_data

# -------- Classifiers --------
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train_class)
    joblib.dump(model, f"models/{name}_model.pkl")
    print(f"{name} saved!")

# -------- Regressor --------
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
print("Training regressor...")
regressor.fit(X_train_reg, y_train_reg)
joblib.dump(regressor, "models/regressor_model.pkl")
print("Regressor saved!")
