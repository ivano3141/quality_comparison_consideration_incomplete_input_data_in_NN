# Tensorflow / Keras
from tensorflow import keras # for building Neural Networks
from keras.models import Sequential # for creating a linear stack of layers for our Neural Network
from keras import Input # for instantiating a keras tensor
from keras.layers import Dense # for creating regular densely-connected NN layers.
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

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
import scipy
from sklearn.mixture import GaussianMixture


class gmm_model():

    def __init__(self):
        self.test_size = 0.3
        self.p = 0.3

    def import_data(self):
        csv_files = []
        for filename in os.listdir():
            if filename == "Datasets":
                for csv_file in os.listdir(filename):
                    if csv_file.endswith('.csv'):
                        csv_files.append(csv_file)
        raw_data = pd.read_csv(os.path.join("Datasets", csv_files[1]))
        raw_data.drop(["column_a"], axis=1, inplace=True)
        Y = raw_data["y"].astype('float32') - 1
        X = raw_data.drop(["y"], axis=1).astype('float32')

        return X, Y, raw_data

    def gen_gmm(self, data):
        data.drop(["y"], axis=1, inplace=True)
        # erstellen GMM über gesamtes
        gmm = GaussianMixture(n_components=3, covariance_type="diag")
        gmm.fit(data)
        return gmm

    def gen_miss_values(self, data, p):
        shape = data.shape
        new_df = data.copy().astype(np.float64)
        missing = np.random.binomial(1, self.p, shape)
        data[missing.astype('bool')] = np.nan

        return data

    def train_test_split(self, X, Y, test_size):
        # train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=self.test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def gmm_activation_smieja(self, data_missing, gmm, model):
        w = model.layers[0].get_weights()[0]  # Gewichte von erster Schicht
        b = model.layers[0].get_weights()[1][:, np.newaxis]  # Bias erste Schicht
        output_list = []
        # man muss nur durch spalten iterieren
        for s in range(0, len(pd.DataFrame(w).columns)):
            for i in range(0, len(data_missing)):  # iteriere DF mit NaN-Values
                list_nan = np.array(data_missing.iloc[i])

                result = 0
                for ii in range(0, gmm.n_components):  # iteriere GMM mit n_komp
                    list_gmm_cov = gmm.covariances_[ii]
                    list_gmm_mean = gmm.means_[ii]
                    p_i = gmm.weights_[ii]

                    # Erstellen von GMM-Vektoren
                    # create cov_gmm: in gmm_covariances_vektor werte 0 setzen die in der jeweiligen Spalte von data_missing nicht NaN sind
                    global s_i
                    s_i = []
                    for y in range(len(list_gmm_cov)):
                        if np.isnan(list_nan[y]):
                            s_i.append(list_gmm_cov[y])
                        else:
                            s_i.append(0)

                    # create mean_gmm: in gmm_mean_vektor werden alle festen werte aufgenommen aus data_missing und NaN-Werte durch list_gmm_mean ersetzt
                    m_i = []
                    for y in range(len(list_gmm_mean)):
                        if not np.isnan(list_nan[y]):
                            m_i.append(list_nan[y])
                        else:
                            m_i.append(list_gmm_mean[y])

                    # Berechnen ReLUw,b(F)
                    ft = p_i * np.sqrt(np.transpose(np.array(w[:, s]) * w[:, s]) @ s_i)
                    z = np.transpose(np.array(w[:, s])) @ m_i + b[s]  # Zähler für NR w
                    n = np.sqrt(np.transpose(np.array(w[:, s]) * w[:, s]) @ s_i)  # Nenner für NR w
                    nr_w = z / n  # Input in NR
                    nr = 1 / np.sqrt(2 * np.pi) * np.exp(-(np.square(nr_w) / 2)) + (nr_w / 2) * (
                                1 + scipy.special.erf(nr_w / np.sqrt(2)))  # NR-Funktion
                    ReLUwb = ft * nr  # Summe aus i-GMM-Komponenten
                    # print(f'ReLUwb:{ReLUwb:} ft:{ft:} z:{z:} n:{n:} nr_w:{nr_w:} nr:{nr:}')
                    result += ReLUwb
                    # print(f'resutl{result}')
            output_list.append(result)
        return output_list

    def model_01(self, X_train, X_test, y_train, y_test):
        model = tf.keras.Sequential(name="Model")
        model.add(tf.keras.layers.Dense(units=32, activation=tf.nn.relu, input_shape=[len(X_train.columns)]))
        model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(units=5, activation=tf.nn.softmax))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=5)

        # Fit model with early stopping
        model.fit(X_train, tf.keras.utils.to_categorical(y_train), epochs=50,
                  validation_data=(X_test, tf.keras.utils.to_categorical(y_test)),
                  callbacks=[early_stop])
        return model

    def model_02(self, X_train, X_test, y_train, y_test):
        model = tf.keras.Sequential(name="Model")
        model.add(tf.keras.layers.Dense(units=32, activation=tf.nn.relu, input_shape=[len(X_train.columns)]))
        model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(units=5, activation=tf.nn.softmax))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=5)

        # Fit model with early stopping
        model.fit(X_train, tf.keras.utils.to_categorical(y_train), epochs=50,
                  validation_data=(X_test, tf.keras.utils.to_categorical(y_test)),
                  callbacks=[early_stop])
        return model

    def model_03(self, X_train, X_test, y_train, y_test):
        model = tf.keras.Sequential(name="Model")
        model.add(tf.keras.layers.Dense(units=100, activation=tf.nn.relu, input_shape=[len(X_train.columns)]))
        model.add(tf.keras.layers.Dense(units=400, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(units=600, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(units=5, activation=tf.nn.softmax))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=5)

        # Fit model with early stopping
        model.fit(X_train, tf.keras.utils.to_categorical(y_train), epochs=50,
                  validation_data=(X_test, tf.keras.utils.to_categorical(y_test)),
                  callbacks=[early_stop])
        return model

    def predict_model(self, X_test_m, model):
        if len(model.layers) == 4:
            predictions = []
            for index, row in X_test_m.iterrows():
                row_df = pd.DataFrame([row])
                row_df.columns = [f"x{i + 1}" for i in range(len(row_df.columns))]

                gmm_output = self.gmm_activation_smieja(row_df, gmm, model)

                input_data = np.array(gmm_output).reshape((1, len(gmm_output)))
                get_layer_output = K.function([model.layers[1].input], [model.layers[1].output])
                output = get_layer_output([input_data])[0]
                get_layer_output = K.function([model.layers[2].input], [model.layers[2].output])
                output = get_layer_output([output])[0]
                get_layer_output = K.function([model.layers[3].input], [model.layers[3].output])
                output = get_layer_output([output])[0]
                index_of_max = np.argmax(output)
                print(f'output:{output} index_of_max:{index_of_max}')
                predictions.append(index_of_max)

            return predictions

        elif len(model.layers) == 5:
            predictions = []
            for index, row in X_test_m.iterrows():
                row_df = pd.DataFrame([row])
                row_df.columns = [f"x{i + 1}" for i in range(len(row_df.columns))]

                gmm_output = self.gmm_activation_smieja(row_df, gmm, model)

                input_data = np.array(gmm_output).reshape((1, len(gmm_output)))
                get_layer_output = K.function([model.layers[1].input], [model.layers[1].output])
                output = get_layer_output([input_data])[0]
                get_layer_output = K.function([model.layers[2].input], [model.layers[2].output])
                output = get_layer_output([output])[0]
                get_layer_output = K.function([model.layers[3].input], [model.layers[3].output])
                output = get_layer_output([output])[0]
                get_layer_output = K.function([model.layers[4].input], [model.layers[4].output])
                output = get_layer_output([output])[0]
                index_of_max = np.argmax(output)
                print(f'output:{output} index_of_max:{index_of_max}')
                predictions.append(index_of_max)

            return predictions

        else:
            print("Something wrong")

    def evaluate(self, y_test, y_gmm_nan):
        y_true = y_test
        y_pred = y_gmm_nan
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
        avg_accuracy = sum(precision_scores) / 5

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
# Model_01-_03 Test
# p = 0.3,  0.6, 0,9

# List of models to test
models = [("Model_01", "predictions_D1_01_"), ("Model_02", "predictions_D1_02_"), ("Model_03", "predictions_D1_03_")]

# List of missing values percentages to test
p_values = [0.3, 0.6, 0.9]

for model_name, output_prefix in models:
    # Initialize model and prepare data
    model = gmm_model()
    X_raw, Y_raw, data_raw = model.import_data()
    X_train, X_test, y_train, y_test = model.train_test_split(X_raw, Y_raw, 0.33)

    # Train model and create gmm
    if model_name == "Model_01":
        trained_model = model.model_01(X_train, X_test, y_train, y_test)
        trained_model.save('model_D1_01.h5'')
    elif model_name == "Model_02":
        trained_model = model.model_02(X_train, X_test, y_train, y_test)
        trained_model.save('model_D1_02.h5'')
    elif model_name == "Model_03":
        trained_model = model.model_03(X_train, X_test, y_train, y_test)
        trained_model.save('model_D1_03.h5'')

    gmm = model.gen_gmm(data_raw)

    for p in p_values:
        # Generate missing values data and predict values
        data_miss = model.gen_miss_values(X_test, p)
        predictions = model.predict_model(data_miss, trained_model)

        # Evaluate model and save results to file
        evaluate = model.evaluate(y_test, predictions)
        output_filename = output_prefix + str(p).replace(".", "_")
        model.save_txt(output_filename, evaluate)