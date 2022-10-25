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
from sklearn.metrics import accuracy_score


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
train_df["Age_group"] = train_df["Age"].map(lambda x: int(x)%10 * 10 if not np.isnan(x) else -10)

fig_bar, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
sns.barplot(data=train_df, x="Pclass", y="Survived", ax=axes[0][0])
sns.barplot(data=train_df, x="Sex", y="Survived", ax=axes[0][1])
sns.barplot(data=train_df, x="SibSp", y="Survived", ax=axes[1][0])
sns.barplot(data=train_df, x="Parch", y="Survived", ax=axes[1][1])
sns.barplot(data=train_df, x="Embarked", y="Survived", ax=axes[2][0])
sns.barplot(data=train_df, x="Age_group", y="Survived", ax=axes[2][1])


fig_count, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
sns.countplot(data=train_df, x="Pclass", hue="Survived", ax=axes[0][0])
sns.countplot(data=train_df, x="Sex", hue="Survived", ax=axes[0][1])
sns.countplot(data=train_df, x="SibSp", hue="Survived", ax=axes[1][0])
sns.countplot(data=train_df, x="Parch", hue="Survived", ax=axes[1][1])
sns.countplot(data=train_df, x="Embarked", hue="Survived", ax=axes[2][0])
sns.countplot(data=train_df, x="Age_group", hue="Survived", ax=axes[2][1])


fig_scatter, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(data=train_df, x="Age", y="Fare", hue="Survived", ax=ax)


train_df["Sex"] = train_df["Sex"].map(lambda x: {"male":0, "female":1}[x])
train_df["Embarked"] = train_df["Embarked"].map(lambda x: {"S":0, "C":1, "Q":2}[x] if type(x)==str else -1)
corr_df = train_df[~train_df["Embarked"].isna()].corr()
fig_corr, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr_df, annot=True, square=True)

# %% feature_engineering
fig_na
columns = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]
X = train_df[train_df["Embarked"] != -1]
Y = X["Survived"]
X = X[columns]
X = pd.get_dummies(X, columns=["Pclass", "SibSp", "Parch", "Embarked"])
X.loc[:,"Parch_9"] = 0
columns_order = X.columns

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

test_X = test_df[columns]
msno.matrix(test_X)
test_X["Fare"] = test_X["Fare"].fillna(0.)
test_X["Sex"] = test_X["Sex"].map(lambda x: {"male":0, "female":1}[x])
test_X["Embarked"] = test_X["Embarked"].map(lambda x: {"S":0, "C":1, "Q":2}[x] if type(x)==str else -1)
test_X = pd.get_dummies(test_X, columns=["Pclass", "SibSp", "Parch", "Embarked"])
test_X = test_X[columns_order]
test_X = scaler.transform(test_X)

#%% modeling
from tqdm import tqdm
res = {}
for max_depth in tqdm(range(2,10)):
    for n_estimators in [50, 100, 150]:
        metrics = []
        for seed in range(10):
            x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=seed)
            gboost = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators)
            gboost.fit(x_train, y_train)
            y_pred = gboost.predict(x_test)
            metric = accuracy_score(y_test, y_pred)
            metrics.append(metric)
        res[np.mean(metrics)] = [max_depth, n_estimators]

optimal_max_depth, optimal_n_estimators = res[max(res.keys())]
gboost = GradientBoostingClassifier(max_depth=optimal_max_depth, n_estimators=optimal_max_depth)
gboost.fit(X, Y)
submit_df["Survived"] = gboost.predict(test_X)
submit_df.to_csv("/Users/wonhyung64/Downloads/submission.csv", index=False)


# %% feedback
fig_import, ax = plt.subplots(figsize=(10, 10))
sns.barplot(x=gboost.feature_importances_, y=columns_order, ax=ax)

#%% feature_engnieering 2
fig_na
columns = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked", "Age"]
X = train_df[train_df["Embarked"] != -1]
X = X[~X["Age"].isna()]
Y = X["Survived"]
X = X[columns]
X = pd.get_dummies(X, columns=["Pclass", "SibSp", "Parch", "Embarked"])
X.loc[:,"Parch_9"] = 0
columns_order = X.columns

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

test_X = test_df[columns]
msno.matrix(test_X)
test_X["Fare"] = test_X["Fare"].fillna(0.)
test_X["Age"] = test_X["Age"].fillna(0.)
test_X["Sex"] = test_X["Sex"].map(lambda x: {"male":0, "female":1}[x])
test_X["Embarked"] = test_X["Embarked"].map(lambda x: {"S":0, "C":1, "Q":2}[x] if type(x)==str else -1)
test_X = pd.get_dummies(test_X, columns=["Pclass", "SibSp", "Parch", "Embarked"])
test_X = test_X[columns_order]
test_X = scaler.transform(test_X)

#%% modeling 2
from tqdm import tqdm
res = {}
for max_depth in tqdm(range(2,10)):
    for n_estimators in [50, 100, 150]:
        metrics = []
        for seed in range(10):
            x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=seed)
            gboost = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators)
            gboost.fit(x_train, y_train)
            y_pred = gboost.predict(x_test)
            metric = accuracy_score(y_test, y_pred)
            metrics.append(metric)
        res[np.mean(metrics)] = [max_depth, n_estimators]

optimal_max_depth, optimal_n_estimators = res[max(res.keys())]
gboost = GradientBoostingClassifier(max_depth=optimal_max_depth, n_estimators=optimal_max_depth)
gboost.fit(X, Y)
submit_df["Survived"] = gboost.predict(test_X)
submit_df.to_csv("/Users/wonhyung64/Downloads/submission.csv", index=False)


# %% featture_engineering 3
columns = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked", "Age"]
X = train_df[train_df["Embarked"] != -1]
X["Age"] = X["Age"].fillna(X["Age"].mean())
Y = X["Survived"]
X = X[columns]
X = pd.get_dummies(X, columns=["Pclass", "SibSp", "Parch", "Embarked"])
X.loc[:,"Parch_9"] = 0
columns_order = X.columns

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

test_X = test_df[columns]
msno.matrix(test_X)
test_X["Fare"] = test_X["Fare"].fillna(0.)
test_X["Age"] = test_X["Age"].fillna(test_X["Age"].mean())
test_X["Sex"] = test_X["Sex"].map(lambda x: {"male":0, "female":1}[x])
test_X["Embarked"] = test_X["Embarked"].map(lambda x: {"S":0, "C":1, "Q":2}[x] if type(x)==str else -1)
test_X = pd.get_dummies(test_X, columns=["Pclass", "SibSp", "Parch", "Embarked"])
test_X = test_X[columns_order]
test_X = scaler.transform(test_X)

#%% modeling 3
from tqdm import tqdm
res = {}
for max_depth in tqdm(range(2,10)):
    for n_estimators in [50, 100, 150]:
        metrics = []
        for seed in range(10):
            x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=seed)
            gboost = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators)
            gboost.fit(x_train, y_train)
            y_pred = gboost.predict(x_test)
            metric = accuracy_score(y_test, y_pred)
            metrics.append(metric)
        res[np.mean(metrics)] = [max_depth, n_estimators]

optimal_max_depth, optimal_n_estimators = res[max(res.keys())]
gboost = GradientBoostingClassifier(max_depth=optimal_max_depth, n_estimators=optimal_max_depth)
gboost.fit(X, Y)
submit_df["Survived"] = gboost.predict(test_X)
submit_df.to_csv("/Users/wonhyung64/Downloads/submission.csv", index=False)


# %% feature engineering 4
columns = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked", "Age"]
X = train_df[train_df["Embarked"] != -1]
Y = X["Survived"]
X = X[columns]
X = pd.get_dummies(X, columns=["Pclass", "SibSp", "Parch", "Embarked"])
X.loc[:,"Parch_9"] = 0
columns_order = X.columns

scaler = MinMaxScaler()
scaler.fit(X)

X_1 = X[~X["Age"].isna()]
X_2 = X[X["Age"].isna()]

age_y = X_1["Age"]
age_columns = list(columns_order) 
age_columns.remove("Age")
age_x = X_1[age_columns]

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(age_x, age_y)
X_2["Age"] = np.round(rf.predict(X_2[age_columns]))
X = pd.concat([X_1, X_2])


X = scaler.transform(X)

test_X = test_df[columns]
msno.matrix(test_X)
test_X["Fare"] = test_X["Fare"].fillna(test_X["Fare"].mean())
test_X["Sex"] = test_X["Sex"].map(lambda x: {"male":0, "female":1}[x])
test_X["Embarked"] = test_X["Embarked"].map(lambda x: {"S":0, "C":1, "Q":2}[x] if type(x)==str else -1)
test_X = pd.get_dummies(test_X, columns=["Pclass", "SibSp", "Parch", "Embarked"])
test_X = test_X[columns_order]
test_age = test_X[test_X["Age"].isna()]
test_ = test_X[~test_X["Age"].isna()]

test_age["Age"] = np.round(rf.predict(test_age[age_columns]))
test_X = pd.concat([test_, test_age])
test_X = scaler.transform(test_X)

#%% modeling 4
from tqdm import tqdm
res = {}
for max_depth in tqdm(range(2,10)):
    for n_estimators in [50, 100, 150]:
        metrics = []
        for seed in range(10):
            x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=seed)
            gboost = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators)
            gboost.fit(x_train, y_train)
            y_pred = gboost.predict(x_test)
            metric = accuracy_score(y_test, y_pred)
            metrics.append(metric)
        res[np.mean(metrics)] = [max_depth, n_estimators]

optimal_max_depth, optimal_n_estimators = res[max(res.keys())]
gboost = GradientBoostingClassifier(max_depth=optimal_max_depth, n_estimators=optimal_max_depth)
gboost.fit(X, Y)
submit_df["Survived"] = gboost.predict(test_X)
submit_df.to_csv("/Users/wonhyung64/Downloads/submission.csv", index=False)


# %%
