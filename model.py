import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Models 
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import neighbors # knn
from sklearn import cluster # k-means clustering
from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from matplotlib.pylab import RandomState
from sklearn.calibration import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from numpy import array
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelBinarizer
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def getModel(draft):
    file_path = "e7_data/drafts_dataset.csv"
    df = pd.read_csv(file_path)
    draft_df = pd.DataFrame(draft)

    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    label_encoders = {}
    
    y_draft = draft_df.iloc[:, -1]
    X_draft = draft_df.iloc[:, :-1]
    draft_prediction_encoder = {}

    for column in X.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()

        # To remove zero from the categories
        X[column] = label_encoders[column].fit_transform(X[column]) + 1

    # For input Draft for win prob
    for column_draft in X_draft.select_dtypes(include=['object']).columns:
        draft_prediction_encoder[column_draft] = LabelEncoder()

        # To remove zero from the categories
        X_draft[column_draft] = draft_prediction_encoder[column_draft].fit_transform(X_draft[column_draft]) + 1

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{conf_matrix}")

    X_sample = X_test[0].reshape(1, -1)
    pred = model.predict(X_sample)
    
    # model prediction 
    X_draft_reshape = X_test[0].reshape(1, -1)
    win_prob = model.predict(X_draft_reshape)

    return win_prob