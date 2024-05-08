# Credit Card Costumer Churn Analysis and Prediction
[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

## 📝 Description 
This project aims to predict customer churn for a financial institution using machine learning models. The dataset used for this project is the "Credit Card Customers" dataset available on Kaggle. The primary objectives include building classification and clustering models, selecting appropriate performance metrics, tuning model parameters, and describing the results from unsupervised learning.

![Alt text](images/tree1.png)

## 📦 Repo structure
    main_folder/
    ├── .gitignore
    ├── Readme.md
    ├── classification_models/
    │   ├── decision_tree_classifier.pkl.gz
    │   ├── knn_classifier.pkl.gz
    │   ├── naive_bayes_classifier.pkl.gz
    │   ├── random_forest_classifier.pkl.gz
    │   └── xgboost_classifier.pkl.gz
    ├── data/
    │   ├── BankChurners.csv
    │   └── NewClients.csv
    ├── Feature_names/
    │   ├── encoded_feature_names.csv
    │   └── original_feature_names.csv
    ├── images/
    │   └── tree1.png
    ├── output/
    │   └── feature_importance_plot.png
    ├── Preprocess_models/
    │   ├── encoder.pkl.gz
    │   ├── imputer_cat.pkl.gz
    │   ├── imputer_num.pkl.gz
    │   ├── scaler.pkl.gz
    │   └── standardizer.pkl.gz
    └── src/
        ├── __init__.py
        ├── churn_prediction.py
        ├── classifiers.py
        ├── preprocessing.py
        └── notebook/
            ├── __init__.py
            ├── 01-EDA.ipynb
            ├── 02-exploring_clustering.ipynb
            └── notebook.py
