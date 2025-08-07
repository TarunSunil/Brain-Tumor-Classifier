🧠 Brain Tumor Classifier (Streamlit App)

This Streamlit application allows users to upload a dataset with extracted features from brain MRI images and interactively explore it through data visualizations and classification model training.

## 🚀 Features

- 📊 **Interactive Data Visualization**
  - Target variable (`Class`) distribution
  - Correlation heatmap
  - Histograms of numerical features

- 🤖 **Train ML Classifiers with Hyperparameter Tuning**
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest

- 🧪 **Model Evaluation**
  - Accuracy score
  - Classification report
  - Confusion matrix

---

## 📂 Dataset Format

This project assumes you upload a **CSV file** with the following:

- **Feature columns**: e.g., `Area`, `Perimeter`, `MajorAxisLength`, `Solidity`, etc.
- A **target column** named `Class` representing the tumor type (e.g., benign vs malignant)

Example:

```csv
Area,Perimeter,MajorAxisLength,MinorAxisLength,ConvexArea,Extent,Solidity,Class
321.0,70.12,20.1,15.2,322.0,0.85,0.95,0
...
