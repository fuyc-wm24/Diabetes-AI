#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title("Diabetes Data Preprocessing App")


# In[2]:


# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv('diabetes.csv')

st.subheader("First 5 Rows")
st.dataframe(df.head())


# In[ ]:


# ===============================
# 2. Dataset Info
# ===============================
st.subheader("Dataset Info")
import io
buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())


# In[ ]:


# ===============================
# 3. Statistical Summary
# ===============================
st.subheader("Statistical Summary")
st.dataframe(df.describe())


# In[ ]:


# ===============================
# 4. Missing Values
# ===============================
st.subheader("Missing Values Count")
st.write(df.isnull().sum())


# In[ ]:


# ===============================
# 5. Handle Invalid Zero Values
# ===============================
columns_with_zero_as_missing = [
    'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'
]

df[columns_with_zero_as_missing] = df[columns_with_zero_as_missing].replace(0, np.nan)

st.subheader("After Replacing 0 with NaN")
st.write(df.isnull().sum())


# In[ ]:


# ===============================
# 6. Fill Missing Values
# ===============================
medians = df[columns_with_zero_as_missing].median()
df[columns_with_zero_as_missing] = df[columns_with_zero_as_missing].fillna(medians)

st.subheader("After Filling Missing Values")
st.write(df[columns_with_zero_as_missing].isnull().sum())


# In[ ]:


# ===============================
# 7. Feature Engineering
# ===============================
epsilon = 1e-5

# Categorical Features
df['BMI_Category'] = pd.cut(
    df['BMI'], bins=[-1,18.5,25,30,float('inf')],
    labels=['Underweight','Normal','Overweight','Obese']
).astype(object).fillna('Unknown')

df['Age_Group'] = pd.cut(
    df['Age'], bins=[0,30,40,50,60,float('inf')],
    labels=['<30','30-40','40-50','50-60','60+']
).astype(object).fillna('Unknown')

df['Glucose_Level'] = pd.cut(
    df['Glucose'], bins=[-1,100,125,float('inf')],
    labels=['Normal','Prediabetes','Diabetes']
).astype(object).fillna('Unknown')

# Interaction Features
df['BMI_Age'] = df['BMI'] * df['Age']
df['Glucose_BMI'] = df['Glucose'] * df['BMI']
df['Insulin_Glucose_Ratio'] = df['Insulin'] / (df['Glucose'] + epsilon)
df['Pregnancy_Age_Ratio'] = (df['Pregnancies'] / (df['Age'] + 1)).clip(upper=0.5)

df['Pregnancy_Risk'] = pd.cut(
    df['Pregnancy_Age_Ratio'], bins=[0,0.1,0.3,0.5],
    labels=['Low','Medium','High']
).astype(object).fillna('Unknown')

df['Insulin_log'] = np.log1p(df['Insulin'])

# Encode categorical
for col in ['BMI_Category','Age_Group','Glucose_Level','Pregnancy_Risk']:
    df[col] = pd.factorize(df[col])[0]

# Final cleanup
df = df.fillna(df.median(numeric_only=True))

st.subheader("After Feature Engineering")
st.write(df.isnull().sum())
st.dataframe(df.head())


# In[ ]:


# ===============================
# 8. Feature & Target
# ===============================
X = df.drop('Outcome', axis=1)
y = df['Outcome']

st.subheader("Feature & Target Shape")
st.write("X:", X.shape)
st.write("y:", y.shape)


# In[ ]:


# ===============================
# 9. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

st.subheader("Train-Test Split")
st.write("Training set:", X_train.shape)
st.write("Testing set:", X_test.shape)


# In[ ]:


# ===============================
# 10. Scaling
# ===============================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

