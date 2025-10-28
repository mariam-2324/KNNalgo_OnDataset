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
   ```python
   from sklearn.datasets import load_iris
   import pandas as pd
   iris = load_iris()
   df = pd.DataFrame(iris.data, columns=iris.feature_names)
   df['target'] = iris.target



   
