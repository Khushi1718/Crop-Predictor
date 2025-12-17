# # import numpy as np
# # import pandas as pd
# # import joblib
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # # 1oad all models
# # knn = joblib.load("models/KNN_model.pkl")
# # svm = joblib.load("models/SVM_model.pkl")
# # dt = joblib.load("models/DecisionTree_model.pkl")
# # rf = joblib.load("models/RandomForest_model.pkl")           # Random Forest
# # regressor_model = joblib.load("models/regressor_model.pkl")

# # # Format: [Rainfall_mm, Temperature_C, Irrigation_Used, Fertilizer_Used]
# # new_input = np.array([[120, 30, 1, 50]])

# # # Classification Predictions (High/Low Yield)
# # # print("===== Classification Predictions (High/Low Yield) =====")
# # # print("KNN:", knn.predict(new_input)[0])
# # # print("SVM:", svm.predict(new_input)[0])
# # # print("Decision Tree:", dt.predict(new_input)[0])
# # # print("Random Forest:", rf.predict(new_input)[0])

# # #  Exact Yield Prediction
# # exact_yield = regressor_model.predict(new_input)[0]
# # print("\n===== Exact Yield Prediction =====")
# # print(f"Predicted Yield (tons/hectare): {exact_yield:.2f}")

# # inputs = pd.DataFrame([
# #     [120, 30, 1, 50],
# #     [80, 25, 0, 40],
# #     [150, 28, 1, 60]
# # ], columns=['Rainfall_mm', 'Temperature_Celsius', 'Irrigation_Used', 'Fertilizer_Used'])

# # for i, row in inputs.iterrows():
# #     exact = regressor_model.predict([row])
# #     print("Predicted Exact Yield (tons/hectare):", exact)
# # try:
# #     models = {"KNN": knn, "SVM": svm, "DTree": dt, "RForest": rf}
# #     plt.figure(figsize=(8,5))
# #     for i, (name, model) in enumerate(models.items()):
# #         if hasattr(model, "predict_proba"):
# #             probs = model.predict_proba(new_input)[0]
# #             plt.bar([f"{name}-Low", f"{name}-High"], probs, color='lightblue', alpha=0.7)
# #     plt.title("Predicted Probabilities for High/Low Yield")
# #     plt.ylabel("Probability")
# #     plt.show()
# # except:
# #     print("\nSome models do not support probability prediction (e.g., SVM without probability=True).")

# import numpy as np
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import LabelEncoder

# df = pd.read_csv("backend/crop_yield.csv")
# le = LabelEncoder()
# df['Crop_encoded'] = le.fit_transform(df['Crop'])
# crop_mapping = dict(zip(df['Crop'], df['Crop_encoded']))

# print("\nCROP ENCODING USED IN MODELS")
# print(crop_mapping)

# regressor = joblib.load("backend/models/regressor_model.pkl")

# test_data = pd.DataFrame([
#     [897.077, 27.67, 1, 0, crop_mapping["Cotton"]],
#     [992.67, 18.02, 1, 1, crop_mapping["Rice"]],
#     [147.99, 29.79, 0, 0, crop_mapping["Barley"]]
# ], columns=['Rainfall_mm', 'Temperature_Celsius', 'Irrigation_Used', 'Fertilizer_Used', 'Crop_encoded'])

# actual_yields = [6.55, 8.52, 1.12]

# predicted_yields = regressor.predict(test_data)

# comparison = pd.DataFrame({
#     "Crop": ["Cotton", "Rice", "Barley"],
#     "Actual_Yield": actual_yields,
#     "Predicted_Yield": predicted_yields,
#     "Absolute_Error": np.abs(np.array(actual_yields) - predicted_yields)
# })

# print("\n Actual vs Predicted Yield (tons/ha)")
# print(comparison)

# plt.figure(figsize=(7,4))
# sns.barplot(x="Crop", y="value", hue="variable",
#             data=pd.melt(comparison[["Crop", "Actual_Yield", "Predicted_Yield"]], ["Crop"]),
#             palette="Set2")
# plt.title("Actual vs Predicted Yield (tons/ha)")
# plt.ylabel("Yield (tons/ha)")
# plt.xlabel("Crop")
# plt.show()

# mean_error = comparison["Absolute_Error"].mean()
# print(f"\nAverage Absolute Error: {mean_error:.3f} tons/ha")
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# --------------------
# Load CSV & encoders
# --------------------
df = pd.read_csv("backend/crop_yield.csv")

# Encode Crop
crop_le = LabelEncoder()
df["Crop_enc"] = crop_le.fit_transform(df["Crop"])
crop_mapping = dict(zip(df["Crop"], df["Crop_enc"]))

# Encode Region
region_le = LabelEncoder()
df["Region_enc"] = region_le.fit_transform(df["Region"])
region_mapping = dict(zip(df["Region"], df["Region_enc"]))

print("\nCROP ENCODING:")
print(crop_mapping)

print("\nREGION ENCODING:")
print(region_mapping)

# --------------------
# Load Regression Model
# --------------------
regressor = joblib.load("backend/models/regressor_model.pkl")

# --------------------
# Create Test Data
# --------------------
test_data = pd.DataFrame([
    [
        897.077,         # Rainfall
        27.67,           # Temperature
        1,               # Irrigation_Used
        0,               # Fertilizer_Used
        crop_mapping["Cotton"],
        region_mapping["West"]
    ],
    [
        992.67,
        18.02,
        1,
        1,
        crop_mapping["Rice"],
        region_mapping["South"]
    ],
    [
        147.99,
        29.79,
        0,
        0,
        crop_mapping["Barley"],
        region_mapping["North"]
    ]
], columns=[
    'Rainfall_mm',
    'Temperature_Celsius',
    'Irrigation_Used',
    'Fertilizer_Used',
    'Crop_enc',
    'Region_enc'
])

# --------------------
# Actual yields for comparison
# --------------------
actual_yields = [6.55, 8.52, 1.12]

# --------------------
# Predict using regression model
# --------------------
predicted_yields = regressor.predict(test_data)

comparison = pd.DataFrame({
    "Crop": ["Cotton", "Rice", "Barley"],
    "Region": ["West", "South", "North"],
    "Actual_Yield": actual_yields,
    "Predicted_Yield": predicted_yields,
    "Absolute_Error": np.abs(np.array(actual_yields) - predicted_yields)
})

print("\n Actual vs Predicted Yield (tons/ha)")
print(comparison)

# --------------------
# Plot
# --------------------
plt.figure(figsize=(8,5))
sns.barplot(
    x="Crop",
    y="value",
    hue="variable",
    data=pd.melt(comparison[["Crop", "Actual_Yield", "Predicted_Yield"]], ["Crop"]),
    palette="Set2"
)
plt.title("Actual vs Predicted Yield (tons/ha)")
plt.ylabel("Yield (tons/ha)")
plt.xlabel("Crop")
plt.show()

mean_error = comparison["Absolute_Error"].mean()
print(f"\nAverage Absolute Error: {mean_error:.3f} tons/ha")
