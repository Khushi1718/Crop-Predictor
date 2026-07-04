# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# def load_data(file="crop_yield.csv", return_mapping=False):
#     df = pd.read_csv(file)
#     le = LabelEncoder()
#     for col in df.select_dtypes(include='object').columns:
#         if col != 'Crop':
#             df[col] = le.fit_transform(df[col])
#     df['Crop_encoded'] = le.fit_transform(df['Crop'])
#     crop_mapping = dict(zip(df['Crop'], df['Crop_encoded']))

#     # Convert datatypes
#     df['Fertilizer_Used'] = df['Fertilizer_Used'].astype(int)
#     df['Irrigation_Used'] = df['Irrigation_Used'].astype(int)

#     # Create binary target
#     df['High_Yield'] = np.where(df['Yield_tons_per_hectare'] > df['Yield_tons_per_hectare'].mean(), 1, 0)

#     # Features & Target
#     X = df[['Rainfall_mm', 'Temperature_Celsius', 'Irrigation_Used', 'Fertilizer_Used','Crop_encoded']]
#     y = df['High_Yield']

#     # Split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#     if return_mapping:
#         return X_train, X_test, y_train, y_test, df, crop_mapping
#     else:
#         return X_train, X_test, y_train, y_test
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(file="crop_yield.csv", return_mapping=False):
    df = pd.read_csv(file)
# Convert datatypes
    df['Fertilizer_Used'] = df['Fertilizer_Used'].astype(int)
    df['Irrigation_Used'] = df['Irrigation_Used'].astype(int)

    encoders = {}
    for col in ['Crop', 'Region']:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    q1 = df['Yield_tons_per_hectare'].quantile(0.33)
    q2 = df['Yield_tons_per_hectare'].quantile(0.66)

    def categorize(y):
        if y <= q1:
            return 0  # low
        elif y <= q2:
            return 1  # medium
        else:
            return 2  # high

    df['Yield_Class'] = df['Yield_tons_per_hectare'].apply(categorize)

    features = [
    'Rainfall_mm',
    'Temperature_Celsius',
    'Irrigation_Used',
    'Fertilizer_Used',
    'Crop_enc',
    'Region_enc'
]


    X = df[features]
    y_class = df['Yield_Class']
    y_reg = df['Yield_tons_per_hectare']

    # -------------------------
    # TRAIN–TEST SPLIT
    # -------------------------
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.25, random_state=42)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.25, random_state=42)

    if return_mapping:
        return (X_train_c, X_test_c, y_train_c, y_test_c,
                X_train_r, X_test_r, y_train_r, y_test_r,
                encoders)
    else:
        return (X_train_c, X_test_c, y_train_c, y_test_c,
                X_train_r, X_test_r, y_train_r, y_test_r)
