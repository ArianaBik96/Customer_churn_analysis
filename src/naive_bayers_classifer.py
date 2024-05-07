import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from preprocessing import DataPreprocessor

def naive_bayes_classifier(X_train, X_test, y_train, y_test):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    print(classification_report(y_test, nb.predict(X_test)))
    return nb

def roc_c(X_test, y_test, nb):
    # Make predictions on the test set
    y_pred_prob = nb.predict_proba(X_test)[:, 1]  # Probability of positive class

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

def confusion_m(X_test, y_test, nb):
    y_pred = nb.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

def cross_val(X_train, y_train, nb):
    scores = cross_val_score(nb, X_train, y_train, cv=5)
    print("Cross-validation scores:", scores)
    print("Mean cross-validation score:", scores.mean())

current_directory = os.path.dirname(__file__)
csv_file_path = os.path.join(current_directory, "..", "data", "BankChurners.csv")
churners_df = pd.read_csv(csv_file_path, sep=',')

# Instantiate the class with the dataframe name
preprocessor = DataPreprocessor(df_name=churners_df)

# Use the preprocess method to get preprocessed data
X_train, X_test, y_train, y_test = preprocessor.preprocess()

nb = naive_bayes_classifier(X_train, X_test, y_train, y_test)
roc_c(X_test, y_test, nb)
confusion_m(X_test, y_test, nb)
cross_val(X_train, y_train, nb)