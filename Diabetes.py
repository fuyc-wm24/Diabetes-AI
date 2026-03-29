#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ===============================
# 1. Import Required Libraries
# ===============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Display settings
pd.set_option('display.max_columns', None)


# In[2]:


# ===============================
# 2. Load Dataset
# ===============================
# Make sure diabetes.csv is in the same directory
df = pd.read_csv('diabetes.csv')

# Preview data
print("First 5 rows:")
display(df.head())

print("\nDataset Info:")
df.info()

print("\nStatistical Summary:")
display(df.describe())


# In[3]:


# ===============================
# 3. Check Missing Values
# ===============================
print("Missing values count:")
print(df.isnull().sum())


# In[4]:


# ===============================
# 4. Handle Invalid Zero Values
# ===============================
# In this dataset, 0 is invalid for some columns
columns_with_zero_as_missing = [
    'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'
]

# Replace 0 with NaN
df[columns_with_zero_as_missing] = df[columns_with_zero_as_missing].replace(0, np.nan)

print("After replacing 0 with NaN:")
print(df.isnull().sum())


# In[5]:


# ===============================
# 5. Fill Missing Values
# ===============================
# Use median (robust to outliers)
# Compute medians
medians = df[columns_with_zero_as_missing].median()

# Fill missing values (NaN) with the median
df[columns_with_zero_as_missing] = df[columns_with_zero_as_missing].fillna(medians)

# Check
print("Remaining missing values:")
print(df[columns_with_zero_as_missing].isnull().sum())


# In[7]:


# ===============================
# 6.Full Preprocessing + Feature Engineering
# ===============================

columns_zero_as_nan = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
epsilon = 1e-5  # small value to avoid division by zero

for col in columns_zero_as_nan:
    # Replace invalid 0s with NaN and fill with median
    df[col] = df[col].replace(0, np.nan).fillna(df[col].median())

# ===============================
# 2. Categorical Features (safe)
# ===============================
# Convert pd.cut() to object type to avoid Categorical fillna errors

# BMI Category
df['BMI_Category'] = pd.cut(
    df['BMI'], bins=[-1,18.5,25,30,float('inf')],
    labels=['Underweight','Normal','Overweight','Obese']
).astype(object).fillna('Unknown')

# Age Group
df['Age_Group'] = pd.cut(
    df['Age'], bins=[0,30,40,50,60,float('inf')],
    labels=['<30','30-40','40-50','50-60','60+']
).astype(object).fillna('Unknown')

# Glucose Level
df['Glucose_Level'] = pd.cut(
    df['Glucose'], bins=[-1,100,125,float('inf')],
    labels=['Normal','Prediabetes','Diabetes']
).astype(object).fillna('Unknown')

# ===============================
# 3. Interaction / Ratio Features
# ===============================
df['BMI_Age'] = df['BMI'] * df['Age']  # BMI multiplied by Age
df['Glucose_BMI'] = df['Glucose'] * df['BMI']  # Glucose multiplied by BMI
df['Insulin_Glucose_Ratio'] = df['Insulin'] / (df['Glucose'] + epsilon)  # Insulin efficiency
df['Pregnancy_Age_Ratio'] = (df['Pregnancies'] / (df['Age'] + 1)).clip(upper=0.5)  # capped ratio
df['Pregnancy_Risk'] = pd.cut(
    df['Pregnancy_Age_Ratio'], bins=[0,0.1,0.3,0.5],
    labels=['Low','Medium','High']
).astype(object).fillna('Unknown')
df['Insulin_log'] = np.log1p(df['Insulin'])  # log transform to reduce skewness

# ===============================
# 4. Encode categorical features for ML
# ===============================
for col in ['BMI_Category','Age_Group','Glucose_Level','Pregnancy_Risk']:
    df[col] = pd.factorize(df[col])[0]  # convert categories to integer codes

# ===============================
# 5. Final Cleanup
# ===============================
df = df.fillna(df.median(numeric_only=True))  # ensure no missing values remain

# ===============================
# 6. Verification
# ===============================
print("Missing values after full feature engineering:")
print(df.isnull().sum())
display(df.head())


# In[8]:


# ===============================
# 7. Feature & Target Separation
# ===============================
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']               # Target

print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# In[9]:


# ===============================
# 8. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)


# In[10]:


# ===============================
# 9. Data Normalization (Scaling)
# ===============================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame (optional, for readability)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print("Scaled training data preview:")
display(X_train_scaled.head())


# In[11]:


# ===============================
# 10. Final Output Check
# ===============================
print("Final datasets ready for model training:")
print("X_train:", X_train_scaled.shape)
print("X_test:", X_test_scaled.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# In[ ]:




