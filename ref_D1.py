# Tensorflow / Keras
from tensorflow import keras # for building Neural Networks
from keras.models import Sequential # for creating a linear stack of layers for our Neural Network
from keras import Input # for instantiating a keras tensor
from keras.layers import Dense # for creating regular densely-connected NN layers.
import tensorflow as tf
from keras.models import load_model

# Data manipulation
import pandas as pd # for data manipulation
import numpy as np # for data manipulation

# Sklearn
import sklearn # for model evaluation
from sklearn.model_selection import train_test_split # for splitting data into train and test samples
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Erstellung eigener Aktivierungsfunktion
from keras import backend as K


import os
import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.impute import KNNImputer

from missingpy import MissForest


class inpute_values:
    def __init__(self):
        self.p = 0.6
        self.test_size = 0.3

    def import_data(self):
        csv_files = []
        for filename in os.listdir():
            if filename == "Datasets":
                for csv_file in os.listdir(filename):
                    if csv_file.endswith('.csv'):
                        csv_files.append(csv_file)

        self.raw_data = pd.read_csv(os.path.join("Datasets", csv_files[1]))
        raw_data = pd.read_csv(os.path.join("Datasets", csv_files[1]))
        self.raw_data.drop(["column_a"], axis=1, inplace=True)
        self.Y = self.raw_data["y"] - 1
        self.X = self.raw_data.drop(["y"], axis=1)
        return raw_data, self.Y, self.X, self.raw_data

    def gen_miss_values(self, p):
        self.shape = self.X.shape
        self.new_df = self.X.copy().astype(np.float64)
        self.missing = np.random.binomial(1, self.p, self.shape)
        self.X[self.missing.astype('bool')] = np.nan
        # self.Y = self.raw_data["Class"]
        return self.X

    def inpute_data(self, model):
        if model == "mean":
            self.data_inpute = self.new_df.fillna(self.new_df.mean())
            self.data_inpute = pd.concat([self.data_inpute, self.Y], axis=1, sort=False)
            columns = self.data_inpute.columns.tolist()

            for i in range(len(columns) - 1):
                columns[i] = "col_" + str(i + 1)
            self.data_inpute.columns = columns
            self.data_inpute.columns = [*self.data_inpute.columns[:-1], 'Y']

            self.Y = self.data_inpute["Y"]
            self.X = self.data_inpute.drop(["Y"], axis=1)
            return self.data_inpute

        elif model == "MICE":
            imputer = IterativeImputer()
            self.data_inpute = pd.DataFrame(imputer.fit_transform(self.new_df), columns=self.new_df.columns)
            self.data_inpute = pd.concat([self.data_inpute, self.Y], axis=1, sort=False)

            columns = self.data_inpute.columns.tolist()

            for i in range(len(columns) - 1):
                columns[i] = "col_" + str(i + 1)
            self.data_inpute.columns = columns
            self.data_inpute.columns = [*self.data_inpute.columns[:-1], 'Y']

            self.Y = self.data_inpute["Y"]
            self.X = self.data_inpute.drop(["Y"], axis=1)

            return self.data_inpute

        elif model == "kNN":
            imputer = KNNImputer()
            self.data_inpute = pd.DataFrame(imputer.fit_transform(self.new_df), columns=self.new_df.columns)
            self.data_inpute = pd.concat([self.data_inpute, self.Y], axis=1, sort=False)

            columns = self.data_inpute.columns.tolist()

            for i in range(len(columns) - 1):
                columns[i] = "col_" + str(i + 1)
            self.data_inpute.columns = columns
            self.data_inpute.columns = [*self.data_inpute.columns[:-1], 'Y']

            self.Y = self.data_inpute["Y"]
            self.X = self.data_inpute.drop(["Y"], axis=1)

            return self.data_inpute

        elif model == "RF":
            imputer = MissForest()
            self.data_inpute = pd.DataFrame(imputer.fit_transform(self.new_df), columns=self.new_df.columns)
            self.data_inpute = pd.concat([self.data_inpute, self.Y], axis=1, sort=False)

            columns = self.data_inpute.columns.tolist()

            for i in range(len(columns) - 1):
                columns[i] = "col_" + str(i + 1)
            self.data_inpute.columns = columns
            self.data_inpute.columns = [*self.data_inpute.columns[:-1], 'Y']

            self.Y = self.data_inpute["Y"]
            self.X = self.data_inpute.drop(["Y"], axis=1)

            return self.data_inpute

    def model(self, model):
        if model == 0:
            return load_model("model_D1_01.h5")
        elif model == 1:
            return load_model("model_D1_02.h5")
        elif model == 2:
            return load_model("model_D1_03.h5")

    def evaluate(self, y_test, y_nan):
        y_true = y_test
        y_pred = y_nan
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Compute the accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Compute the precision
        precision_scores = precision_score(y_true, y_pred, labels=range(len(conf_matrix)), average=None)

        # Compute the recall
        recall = recall_score(y_true, y_pred, labels=range(len(conf_matrix)), average=None)

        # Compute the F1-score
        f1 = f1_score(y_true, y_pred, labels=range(len(conf_matrix)), average=None)

        # Compute the average accuracy
        avg_accuracy = sum(precision_scores) / len(conf_matrix)

        # Compute the predicted probabilities
        y_scores = np.random.rand(len(y_true), len(np.unique(y_true)))

        # Compute the AUC for each class
        n_classes = conf_matrix.shape[0]
        auc_list = []
        for i in range(n_classes):
            auc_list.append(roc_auc_score(y_true == i, y_scores[:, i]))

        # Define the result dictionary
        result = {
            "confusion_matrix": conf_matrix,
            "accuracy": accuracy,
            "precision": precision_scores,
            "recall": recall,
            "f1_score": f1,
            "avg_accuracy": avg_accuracy,
            "auc": auc_list
        }
        return result

    def save_txt(self, filename, evaluate):
        with open(filename, 'w') as f:
            f.write(str(evaluate))


inpute_mean = inpute_values()
load_data = inpute_mean.import_data()
test_size = 0.3  # Konstante Testgröße

for model_number in range(0, 3):  # Modelle 1-3 durchlaufen
    model_name = f"Model_{model_number + 1}"
    output_prefix = f"prediction_ref_mean_D1_{model_name}_"
    model = inpute_mean.model(model_number)

    for missing_rate in [0.3, 0.6, 0.9]:
        miss_data = inpute_mean.gen_miss_values(missing_rate)
        inpute_values = inpute_mean.inpute_data("mean")

        Y = inpute_values["Y"]
        X = inpute_values.drop(["Y"], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

        y_pred = model.predict(X_test)
        y_pred_class = np.argmax(y_pred, axis=1)
        y_test = y_test[X_test.index].values

        evaluate = inpute_mean.evaluate(y_test, y_pred_class)
        filename = output_prefix + str(missing_rate).replace(".", "_")
        inpute_mean.save_txt(filename, evaluate)

inpute_MICE = inpute_values()
load_data = inpute_MICE.import_data()
test_size = 0.3  # Konstante Testgröße

for model_number in range(0, 3):  # Modelle 1-3 durchlaufen
    model_name = f"Model_{model_number + 1}"
    output_prefix = f"prediction_ref_MICE_D1_{model_name}_"
    model = inpute_MICE.model(model_number)

    for missing_rate in [0.3, 0.6, 0.9]:
        miss_data = inpute_MICE.gen_miss_values(missing_rate)
        inpute_values = inpute_MICE.inpute_data("MICE")

        Y = inpute_values["Y"]
        X = inpute_values.drop(["Y"], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

        y_pred = model.predict(X_test)
        y_pred_class = np.argmax(y_pred, axis=1)
        y_test = y_test[X_test.index].values

        evaluate = inpute_MICE.evaluate(y_test, y_pred_class)
        filename = output_prefix + str(missing_rate).replace(".", "_")
        inpute_MICE.save_txt(filename, evaluate)

inpute_kNN = inpute_values()
load_data = inpute_kNN.import_data()
test_size = 0.3  # Konstante Testgröße

for model_number in range(0, 3):  # Modelle 1-3 durchlaufen
    model_name = f"Model_{model_number + 1}"
    output_prefix = f"prediction_ref_kNN_D1_{model_name}_"
    model = inpute_kNN.model(model_number)

    for missing_rate in [0.3, 0.6, 0.9]:
        miss_data = inpute_kNN.gen_miss_values(missing_rate)
        inpute_values = inpute_kNN.inpute_data("kNN")

        Y = inpute_values["Y"]
        X = inpute_values.drop(["Y"], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

        y_pred = model.predict(X_test)
        y_pred_class = np.argmax(y_pred, axis=1)
        y_test = y_test[X_test.index].values

        evaluate = inpute_kNN.evaluate(y_test, y_pred_class)
        filename = output_prefix + str(missing_rate).replace(".", "_")
        inpute_kNN.save_txt(filename, evaluate)

inpute_RF = inpute_values()
load_data = inpute_RF.import_data()
test_size = 0.3  # Konstante Testgröße

for model_number in range(0, 3):  # Modelle 1-3 durchlaufen
    model_name = f"Model_{model_number + 1}"
    output_prefix = f"prediction_ref_RF_D1_{model_name}_"
    model = inpute_RF.model(model_number)

    for missing_rate in [0.3, 0.6, 0.9]:
        miss_data = inpute_RF.gen_miss_values(missing_rate)
        inpute_values = inpute_RF.inpute_data("RF")

        Y = inpute_values["Y"]
        X = inpute_values.drop(["Y"], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

        y_pred = model.predict(X_test)
        y_pred_class = np.argmax(y_pred, axis=1)
        y_test = y_test[X_test.index].values

        evaluate = inpute_RF.evaluate(y_test, y_pred_class)
        filename = output_prefix + str(missing_rate).replace(".", "_")
        inpute_RF.save_txt(filename, evaluate)