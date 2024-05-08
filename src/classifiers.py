import os
import pandas as pd
from preprocessing import DataPreprocessor
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pickle
import gzip

# Define functions for each classifier
def random_forest_classifier(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
    rfc.fit(X_train, y_train)
    print("Random Forest Classifier:")
    print(classification_report(y_test, rfc.predict(X_test)))
    return rfc

def decision_tree_classifier(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    print("Decision Tree Classifier:")
    print(classification_report(y_test, dt.predict(X_test)))
    return dt

def xgboost_classifier(X_train, X_test, y_train, y_test, feature_names_original):
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)  # Train XGBoost on preprocessed data
    print("XGBoost Classifier:")
    print(classification_report(y_test, xgb.predict(X_test)))
    feature_importance_dict = dict(zip(feature_names_original, xgb.feature_importances_))
    return xgb, feature_importance_dict

def naive_bayes_classifier(X_train, X_test, y_train, y_test):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    print("Naive Bayes Classifier:")
    print(classification_report(y_test, nb.predict(X_test)))
    return nb

def knn_classifier(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    print("K-Nearest Neighbors Classifier:")
    print(classification_report(y_test, knn.predict(X_test)))
    return knn

def roc_c(X_test, y_test, clf, clf_name):
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"{clf_name} ROC AUC score: {roc_auc}")

def confusion_m(X_test, y_test, clf, clf_name):
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"{clf_name} Confusion matrix:")
    print(cm)
    print()

def cross_val(X_train, y_train, clf, clf_name):
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"{clf_name} Cross-validation scores:", scores)
    print(f"{clf_name} Mean cross-validation score:", scores.mean())
    print()

def plot_feature_importance(feature_importance):
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_importance)
    num_features = len(features)
    
    # Calculate the appropriate figure size based on the number of features
    figsize_height = min(0.5 * num_features, 10)  # Maximum height of 10 inches
    plt.figure(figsize=(10, figsize_height))
    
    plt.barh(range(num_features), importance, align='center')
    plt.yticks(range(num_features), features)  # Set the feature names as y-axis ticks
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('XGBoost Feature Importance')
    
    # Adjust the layout to avoid clipping feature names
    plt.tight_layout()
    
    # Save the plot as an image file
    output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    plot_file_path = os.path.join(output_folder, 'feature_importance_plot.png')
    plt.savefig(plot_file_path)
    plt.close()
    
    print(f"Feature importance plot saved to {plot_file_path}")

# Function to export each model to a separate file
def export_model(model, model_name, export_folder):
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    
    filename = os.path.join(export_folder, f"{model_name}.pkl.gz")
    with gzip.open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model {model_name} exported to {filename}")

current_directory = os.path.dirname(__file__)
csv_file_path = os.path.join(current_directory, "..", "data", "BankChurners.csv")
churners_df = pd.read_csv(csv_file_path, sep=',')

# Instantiate the class with the dataframe name
preprocessor = DataPreprocessor(df_name=churners_df)

# Use the preprocess method to get preprocessed data
X_train, X_test, y_train, y_test = preprocessor.preprocess()


# Train and evaluate each classifier
rfc = random_forest_classifier(X_train, X_test, y_train, y_test)
dt = decision_tree_classifier(X_train, X_test, y_train, y_test)
xgb_classifier, xgb_feature_importance = xgboost_classifier(X_train, X_test, y_train, y_test, preprocessor.get_original_feature_names())  # Fixed
nb = naive_bayes_classifier(X_train, X_test, y_train, y_test)
knn = knn_classifier(X_train, X_test, y_train, y_test)

# Compare ROC curves
roc_c(X_test, y_test, rfc, "Random Forest Classifier")
roc_c(X_test, y_test, dt, "Decision Tree Classifier")
roc_c(X_test, y_test, xgb_classifier, "XGBoost Classifier")  # Use xgb_classifier here
roc_c(X_test, y_test, nb, "Naive Bayes Classifier")
roc_c(X_test, y_test, knn, "K-Nearest Neighbors Classifier")

# Compare confusion matrices
confusion_m(X_test, y_test, rfc, "Random Forest Classifier")
confusion_m(X_test, y_test, dt, "Decision Tree Classifier")
confusion_m(X_test, y_test, xgb_classifier, "XGBoost Classifier")  # Use xgb_classifier here
confusion_m(X_test, y_test, nb, "Naive Bayes Classifier")
confusion_m(X_test, y_test, knn, "K-Nearest Neighbors Classifier")

# Compare cross-validation scores
cross_val(X_train, y_train, rfc, "Random Forest Classifier")
cross_val(X_train, y_train, dt, "Decision Tree Classifier")
cross_val(X_train, y_train, xgb_classifier, "XGBoost Classifier")  # Use xgb_classifier here
cross_val(X_train, y_train, nb, "Naive Bayes Classifier")
cross_val(X_train, y_train, knn, "K-Nearest Neighbors Classifier")

# Plot feature importance
print("XGBoost Feature Importance:")
plot_feature_importance(xgb_feature_importance)

# Export each model to a separate file
export_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'classification_models'))

export_model(rfc, "random_forest_classifier", export_folder)
export_model(dt, "decision_tree_classifier", export_folder)
export_model(xgb_classifier, "xgboost_classifier", export_folder)
export_model(nb, "naive_bayes_classifier", export_folder)
export_model(knn, "knn_classifier", export_folder)