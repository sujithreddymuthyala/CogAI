
from unittest import TestCase

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import IRIS_DATASET



class Test_iris_model(TestCase):
    """
    Class used for unit testing
    Methods
    -----
    setUp()
        sets up a trained model from iris dataset
    test_accuracy()
        tests the accuracy of both models by some dummy data
    test_precision_recall_f1score_support()
        test precision, recall, f1 score, support of the models
    """

    def setUp(self):
        data = load_iris()

        X = pd.DataFrame(data.data, columns=(data.feature_names))
        y = pd.DataFrame(data.target, columns=["target"])

        X_train, X_test, y_train, self.y_test = train_test_split(X, y, random_state=42, test_size=0.2)

        lr= LogisticRegressionCV()
        model= lr.fit(X_train, y_train)

        #taking dummy data for testing
        self.dummy_data = X_test[10:20]
        self.pred_value =model.predict(self.dummy_data)
        

    def accuracy(self):
        a1 = accuracy_score(self.y_test[10:20], self.pred_value)
        a2 = accuracy_score(self.y_test[10:20], IRIS_DATASET.log_reg().predict(self.dummy_data))

        self.assertTrue(a1,a2)
        print(a1,a2)

    def classification_report(self):
        r1 = classification_report(self.y_test[10:20], self.prediction_value)
        r2 = classification_report(
            self.y_test[10:30], IRIS_DATASET.log_reg().predict(self.dummy_data)
        )

        self.assertTrue(r1,r2)
    
Test_iris_model()