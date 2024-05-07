import pandas as pd
from preprocessing import DataPreprocessor
import os
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

'''
Random Forest Classifier was chosen for the following reasons:
    - Random Forest combines multiple decision trees to improve performance and generalization.
    - By averaging multiple decision trees, Random Forest tends to be more robust to overfitting compared to individual decision trees.
    - provides a feature importance measure, which can be useful for feature selection and understanding the importance of different features in the classification task.
    - can capture nonlinear relationships between features and target variables, making it suitable for classification tasks with complex decision boundaries.
    - is relatively scalable and can handle large datasets with high dimensionality efficiently.
'''
    #Higher precision indicates a lower false positive rate.
    #Higher recall indicates a lower false negative rate.
    #F1 Score is the weighted average of Precision and Recall. It takes both false positives and false negatives into account.
def random_forest_classifier(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
    rfc.fit(X_train, y_train)
    print(classification_report(y_test, rfc.predict(X_test)))
    return rfc


def roc_c(X_test, y_test, rfc):

    # Make predictions on the test set
    y_pred_prob = rfc.predict_proba(X_test)[:, 1]  # Probability of positive class

    # Compute ROC curve and AUC score
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.show()

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC AUC score: {roc_auc}")

def confusion_m(X_test, y_test, rfc):
    y_pred = rfc.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

def cross_val(X_train, y_train,rfc):
    scores = cross_val_score(rfc, X_train, y_train, cv=5)
    print("Cross-validation scores:", scores)
    print("Mean cross-validation score:", scores.mean())

current_directory = os.path.dirname(__file__)
csv_file_path = os.path.join(current_directory, "..", "data", "BankChurners.csv")
churners_df = pd.read_csv(csv_file_path, sep=',')
# Instantiate the class with the dataframe name
preprocessor = DataPreprocessor(df_name=churners_df)

# Use the preprocess method to get preprocessed data
X_train, X_test, y_train, y_test = preprocessor.preprocess()

rfc = random_forest_classifier(X_train, X_test, y_train, y_test)
roc_c(X_test, y_test, rfc)
# true neg     false pos
# false neg    true pos
confusion_m(X_test, y_test, rfc)
cross_val(X_train, y_train, rfc)

