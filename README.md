# quality_comparison_consideration_incomplete_input_data_in_NN
code-repository for the master thesis Consideration of incomplete input data in neural networks - a quality comparison 

Description: 
The corresponding code to the master thesis "Berücksichtigung unvollständige Inputdaten in
neuronalen Netzen – ein Gütevergleich" by Ivan Nosov. 

The masterthesis compares  a general, theoretically justified mechanism for processing missing data by neural networks with common methods for missing values.
Ref.: "Processing of missing data by neural networks
" Marek Smieja, Łukasz Struski, Jacek Tabor, Bartosz Zieliński, Przemysław Spurek | https://arxiv.org/abs/1805.07405




Content:
Link for data to downlod:
https://drive.google.com/drive/folders/1wWl7oByt3Ny8R2oZWyOC5a02iBS6MoIb?usp=sharing

- D = Dataset
- 8 Similar .py scripts 
 - 4 Scripts for 4 datasets to predict values with Smieja.
 - 4 Scripts for 4 datasets to predict values with common methods.

Description Example gmm_D0.py:

1. The necessary libraries are imported, including Tensorflow/Keras for building the neural network, Pandas and NumPy for data manipulation, and Sklearn for model evaluation.

2. The gmm_model class is defined, which contains methods for importing the dataset, generating a GMM, generating missing values in the dataset, and training and evaluating the neural network.

3. The import_data() method reads in a dataset from a CSV file, drops the Time and Amount columns, and separates the target variable (Class) from the input variables (X).

4. The gen_gmm() method creates a GMM with three components using the GaussianMixture() function from Sklearn.

5. The gen_miss_values() method generates missing values in the input data by randomly setting a proportion of the values to NaN.

6. The train_test_split() method splits the data into training and testing sets using the train_test_split() function from Sklearn.

7. The gmm_activation_smieja() method calculates the activation function using the GMM and neural network. It takes the GMM, a dataframe with missing values, and the trained model as inputs, and returns a list of activation values.

8. The model_01(), model_02(), and model_03() methods define the neural network models with different layer sizes and activations.

9. The predict_model() method uses the trained model and the gmm_activation_smieja() function to predict the class labels of a dataframe with missing values.

10. The evaluate() method calculates various metrics (e.g. accuracy, precision, recall, F1 score, and AUC) to evaluate the performance of the model.

11. The save_txt() method saves the evaluation results to a text file.

12. The models list contains the names of the models to test (Model_01, Model_02, and Model_03) and their output file prefixes.

13. The p_values list contains the percentages of missing values to test (0.3, 0.6, and 0.9).

14. For each model and percentage of missing values, the script imports the data, trains the model, generates missing values, predicts class labels, evaluates the model, and saves the results to a text file.



Description Example ref_D0.py:

This code defines a class called inpute_values which is used to import, generate missing values and impute the credit card transaction dataset, as well as train and evaluate machine learning models. The inpute_values class has four methods: import_data, gen_miss_values, inpute_data, and model. It also has an evaluate function that takes in predicted and actual labels and computes various classification metrics like accuracy, precision, recall, F1-score, AUC score, etc. The class also defines a save_txt function that saves the evaluation results as a text file.

After defining the inpute_values class, the code creates four instances of the class to impute missing values using four different imputation methods: mean imputation, MICE, kNN, and random forest imputation. Each of these instances is used to train and evaluate three different machine learning models (Model_1, Model_2, and Model_3) using a range of missing values rates (0.3, 0.6, and 0.9).

The steps to perform this process are:

1. Import libraries like TensorFlow/Keras, pandas, numpy, sklearn, etc.

2. Define the inpute_values class with its methods and functions.

3. Create four instances of the inpute_values class (inpute_mean, inpute_MICE, inpute_kNN, inpute_RF) to impute missing values using four different imputation methods: mean imputation, MICE, kNN, and random forest imputation.

4. Load the credit card transaction dataset using the import_data function from the inpute_values class

5. Train and evaluate three different machine learning models (Model_1, Model_2, and Model_3) using each of the four instances of the inpute_values class (inpute_mean, inpute_MICE, inpute_kNN, inpute_RF) for different missing value rates (0.3, 0.6, and 0.9).

6. Save the evaluation results for each combination of model, imputation method, and missing value rate as a text file.
