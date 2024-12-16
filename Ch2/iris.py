import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("Iris Species Predictor")
st.write("""
This app predicts the Iris flower...
         """)

iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]
print(df.head)

st.sidebar.header("Input Parmeters")
def user_input_features():
    sepal_length = st.sidebar.slider("Sepal length (cm)",
                                     float(df['sepal length (cm)'].min()),
                                     float(df['sepal length (cm)'].max()),
                                     float(df['sepal length (cm)'].mean())
                                     )
    sepal_width = st.sidebar.slider("Sepal width (cm)",
                                     float(df['sepal width (cm)'].min()),
                                     float(df['sepal width (cm)'].max()),
                                     float(df['sepal width (cm)'].mean())
                                     )
    petal_length = st.sidebar.slider("Petal length (cm)",
                                     float(df['petal length (cm)'].min()),
                                     float(df['petal length (cm)'].max()),
                                     float(df['petal length (cm)'].mean())
                                     )
    petal_width = st.sidebar.slider("Petal width (cm)",
                                     float(df['petal width (cm)'].min()),
                                     float(df['petal width (cm)'].max()),
                                     float(df['petal width (cm)'].mean())
                                     )
    data = {'sepal length (cm)':sepal_length,
            'sepal width (cm)':sepal_width,
            'petal length (cm)':petal_length,
            'petal width (cm)':petal_width
            }
    features = pd.DataFrame(data, index = [0])
    return features

input_df = user_input_features()

st.subheader("User Input Parameters")
st.write(input)

model = RandomForestClassifier(random_state= 42)
model.fit(X,y)

prediction = model.predict(input_df.to_numpy())
prediction_proba = model.predict_proba(input_df.to_numpy())
st.subheader("Predcition")
st.write(iris.target_names[prediction])
st.subheader("Prediction Probability")
st.write(prediction_proba)

st.subheader('Histogram of Features')
fig, axes = plt.subplots(2,2, figsize=(12,18))
axes = axes.flatten()
for i, ax in enumerate(axes):
    sns.histplot(df[iris.feature_names[i]], kde=True,ax=ax)
    ax.set_title(iris.feature_names[i])
plt.tight_layout()
st.pyplot

plt.figure(figsize=(10,8))
st.subheader('Correlation Matrix')
numerical_df = df.drop('species', axis = 1)
courr_matrix=numerical_df.corr()
sns.heatmap(courr_matrix, annot=True,cmap='coolwarm',fmt='.2f',linewidths=0.5)
plt.tight_layout()
st.pyplot(plt)

st.subheader('Pairplot')
fig = sns.pairplot(df, hue="species").fig
plt.tight_layout()
st.pyplot(fig)

st.subheader('Feature Importance')
importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,4))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]),[iris.feature_names[i] for i in indices])
plt.xlim([-1, X.shape[1]])
st.pyplot(plt)

# 히스토그램
st.subheader("Histogram of Features")
fig, axes = plt.subplots(2, 2, figsize = (12,8))
axes = axes.flatten()
for i, ax in enumerate(axes):
    sns.histplot(df[iris.feature_names[i]], kde=True, ax=ax)
    ax.set_title(iris.feature_names[i])
plt.tight_layout()
st.pyplot(fig)

# 상관 행렬
plt.figure(figsize = (10, 8))
st.subheader("Correlation Matrix")
numerical_df = df.drop("species", axis = 1)
corr_matrix = numerical_df.corr()
sns.heatmap(corr_matrix, annot = True, cmap = "coolwarm", fmt = '.2f', linewidths = 0.5 )
plt.tight_layout()
st.pyplot(plt)

# 페어플롯
st.subheader('Pairplot')
fig = sns.pairplot(df, hue = "species")
plt.tight_layout()
st.pyplot(fig)