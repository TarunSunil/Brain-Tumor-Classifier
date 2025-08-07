# File path: app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Streamlit app title
st.title("Machine Learning Model Explorer")

# Step 1: Upload Dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    dataset = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", dataset.head())

    if 'Eccentricity' in dataset.columns:
        dataset = dataset.drop(columns=['Eccentricity'])

    # Visualization Options
    st.header("Step 1: Data Visualization")
    if st.checkbox("Show Target Variable Distribution"):
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Class', data=dataset)
        plt.title('Distribution of Target Variable (Class)')
        st.pyplot(plt)

    if st.checkbox("Show Correlation Heatmap"):
        plt.figure(figsize=(12, 8))
        correlation_matrix = dataset.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix of Numerical Features')
        st.pyplot(plt)

    if st.checkbox("Show Histograms of Numerical Features"):
        numerical_features = dataset.drop(columns=['Class']).columns
        dataset[numerical_features].hist(bins=15, figsize=(15, 10), layout=(4, 4))
        plt.suptitle('Histograms of Numerical Features')
        st.pyplot(plt)

    # Step 2: Model Training
    st.header("Step 2: Model Training")

    # Drop unnecessary columns
    dataset_cleaned = dataset.drop(columns=['Unnamed: 0'], errors='ignore')
    X = dataset_cleaned.drop(columns=['Class'])
    y = dataset_cleaned['Class']

    # Handle complex numbers if any
    if 'Eccentricity' in X.columns:
        X['Eccentricity'] = X['Eccentricity'].apply(lambda x: complex(x).real)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model selection
    model_choice = st.selectbox("Choose Model", ["Logistic Regression", "SVM", "Random Forest"])

    # Logistic Regression
    if model_choice == "Logistic Regression":
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'],
            'penalty': ['l1', 'l2']
        }
        log_reg = LogisticRegression(max_iter=1000)
        grid_search = GridSearchCV(log_reg, param_grid, cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_

    # SVM
    elif model_choice == "SVM":
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        svm_model = SVC()
        grid_search = GridSearchCV(svm_model, param_grid, cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_

    # Random Forest
    elif model_choice == "Random Forest":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf_model = RandomForestClassifier()
        grid_search = GridSearchCV(rf_model, param_grid, cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_

    # Make predictions
    y_pred = best_model.predict(X_test_scaled)

    # Evaluation
    st.header("Step 3: Model Evaluation")
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write("**Classification Report:**")
    st.dataframe(pd.DataFrame(classification_rep).transpose())

    # Option to show confusion matrix
    if st.checkbox("Show Confusion Matrix"):
        from sklearn.metrics import ConfusionMatrixDisplay
        cm_display = ConfusionMatrixDisplay.from_estimator(best_model, X_test_scaled, y_test)
        st.pyplot(cm_display.figure_)
