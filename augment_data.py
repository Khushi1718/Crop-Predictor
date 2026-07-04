import pandas as pd
import numpy as np

df = pd.read_csv("crop_yield.csv")

# Create augmented copies
augmented = pd.DataFrame()
for i in range(3):  
    temp = df.copy()
    temp['Rainfall_mm'] += np.random.randint(-5, 6, size=len(df))
    temp['Temperature_Celsius'] += np.random.randint(-2, 3, size=len(df))
    temp['Fertilizer_Used'] += np.random.randint(-2, 3, size=len(df))
    temp['Irrigation_Used'] = temp['Irrigation_Used']  
    augmented = pd.concat([augmented, temp])

df_aug = pd.concat([df, augmented])
df_aug.to_csv("crop_yield_augmented.csv", index=False)
print("Augmented dataset saved as crop_yield_augmented.csv")
