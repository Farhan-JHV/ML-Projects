import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    # Paths for train, test, and raw datasets, stored in 'artifacts' folder
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")

class DataIngestion:
    def __init__(self):
        # Initialize ingestion config with default paths
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Step 1: Read the raw CSV file
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info('Read the dataset as dataframe')

            # Step 2: Ensure the directory for 'train.csv', 'test.csv', 'raw.csv' exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)  # Corrected

            # Step 3: Save the raw data as CSV (to be used later in the pipeline)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")

            # Step 4: Split the data into training and testing sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Step 5: Save the training and testing data as separate CSVs
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data has been completed")

            # Return the paths of the processed train and test data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # In case of any exception, raise a custom exception with traceback information
            logging.error(f"An error occurred: {e}")
            raise CustomException(e, sys)
        
# Entry point for initiating data ingestion when running the script
if __name__ == "__main__":
    try:
        # Initialize the DataIngestion object and start the ingestion process
        obj = DataIngestion()  
        train_data, test_data = obj.initiate_data_ingestion()  # Start the data ingestion process

        # Proceed to data transformation
        data_transformation = DataTransformation()
        train_array, test_array, additional_info = data_transformation.initate_data_transformation(train_data, test_data)
        
        # Train the model using the transformed data
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_array, test_array))
    
    except Exception as e:
        logging.error(f"An error occurred in data ingestion pipeline: {e}")
        raise CustomException(e, sys)
