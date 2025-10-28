# ML Algorithim-KNN: How to Train a Model?

This project was completed as part of the iCodeGuru workshop **“Discussion on Machine Learning Libraries & Mini-Project: How to Train a Model?”** conducted by **Sir Abdullah Sajid**.

## 📘 Overview
The session covered:
- Introduction to **Machine Learning**, its types, and core algorithms.
- Exploration of Python libraries: `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`.
- Implementation of the **K-Nearest Neighbors (KNN)** algorithm on the **Iris dataset**.

## 🧠 Dataset
We used the classic **Iris dataset**, containing 150 samples with 4 features:
- `sepal length`
- `sepal width`
- `petal length`
- `petal width`

## ⚙️ Steps Performed
1. **Load the dataset**
   ``` python
   from sklearn.datasets import load_iris
   import pandas as pd
   iris = load_iris()
   df = pd.DataFrame(iris.data, columns=iris.feature_names)
   df['target'] = iris.target
```
```
## 2. Explore data using Pandas

```python
df.info()
df.describe()
df.head()
```

## 3. Visualize relationships
```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df, hue='target')
plt.show()
```

## 4. Train & Evaluate Model
```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

## 📊 Output

- Model Accuracy: ~1.0 (100%) on the test split

- Clear visualization of class clusters using Seaborn pair plots.

   
