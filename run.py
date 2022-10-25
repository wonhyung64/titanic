#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# %% data import and checking
data_dir = "/Users/wonhyung64/data/titanic"
os.listdir(data_dir)
train_df = pd.read_csv(f"{data_dir}/train.csv")
test_df = pd.read_csv(f"{data_dir}/test.csv")
submit_df = pd.read_csv(f"{data_dir}/gender_submission.csv")

train_df.head()
test_df.head()
submit_df.head()

train_df.info()
test_df.info()
submit_df.info()

fig_na, ax = plt.subplots(figsize=(10, 10))
msno.matrix(train_df, ax=ax)

print("========== train Missing Rate ==========")
for col in train_df.columns:
    print(
        f"{col}: {sum(train_df[col].isna()) / len(train_df)}"
    )

print("========== test Missing Rate ==========")
for col in test_df.columns:
    print(
        f"{col}: {sum(test_df[col].isna()) / len(test_df)}"
    )

fig_dist, ax = plt.subplots(figsize=(10, 10))
sns.histplot(data=train_df, x="Survived", stat="density", discrete=True, ax=ax)

#%% eda
fig_bar, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
sns.barplot(data=train_df, x="Pclass", y="Survived", ax=axes[0][0])
sns.barplot(data=train_df, x="Sex", y="Survived", ax=axes[0][1])
sns.barplot(data=train_df, x="SibSp", y="Survived", ax=axes[1][0])
sns.barplot(data=train_df, x="Parch", y="Survived", ax=axes[1][1])
sns.barplot(data=train_df, x="Embarked", y="Survived", ax=axes[2][0])

fig_corr, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(train_df.corr(), annot=True, square=True)

# %%
