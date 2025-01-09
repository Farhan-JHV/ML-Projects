import os
import sys
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException

# Function to save any object (like a trained model) to a file
def save_object(file_path, obj):
    try:
        # Ensure the directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write-binary mode and use dill to serialize the object
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)

# Function to evaluate multiple models
def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report = {}  # Dictionary to store model scores
        
        # Loop through each model and evaluate it
        for model_name, model in models.items():
            model.fit(x_train, y_train)  # Fit the model on the training data
            y_train_pred = model.predict(x_train)  # Predict on training data
            y_test_pred = model.predict(x_test)  # Predict on testing data

            # Calculate R2 score for both training and test datasets
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Add the test R2 score to the report
            report[model_name] = test_model_score
            
        # Return the dictionary of models and their test scores
        return report
    
    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)
