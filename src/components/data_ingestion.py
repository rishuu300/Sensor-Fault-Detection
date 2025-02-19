import os
import sys
import numpy as np
import pandas as pd
from zipfile import Path
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
from src.utils.main_utils import MainUtils
from pymongo.mongo_client import MongoClient
from src.constant import MONGO_DB_URL, MONGO_COLLECTION_NAME, MONGO_DATABASE_NAME, artifact_folder


@dataclass
class DataIngestionConfig:
    artifact_folder: str = os.path.join(artifact_folder)


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.mongo_client = MongoClient(MONGO_DB_URL)
        self.utils = MainUtils()

    def export_collection_as_dataframe(self, collection_name, db_name) -> pd.DataFrame:
        try:
            collection = self.mongo_client[db_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise CustomException(e, sys) from e

    def export_data_into_feature_store_file_path(self) -> pd.DataFrame:
        try:
            logging.info("Exporting data from mongodb")
            raw_file_path = self.data_ingestion_config.artifact_folder

            os.makedirs(raw_file_path, exist_ok=True)

            sensor_data = self.export_collection_as_dataframe(
                collection_name = MONGO_COLLECTION_NAME, db_name = MONGO_DATABASE_NAME
            )

            logging.info(
                f"Saving the exported data into feature store file path : {raw_file_path}"
            )

            feature_store_file_path = os.path.join(raw_file_path, "wafer_fault.csv")

            sensor_data.to_csv(feature_store_file_path, index=False)

            return feature_store_file_path

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> Path:
        logging.info(
            "Entered initiated_data_ingestion method of data_integration class"
        )

        try:
            feature_store_file_path = self.export_data_into_feature_store_file_path()

            logging.info("Got the data from mongodb")

            logging.info(
                "Exited initiated_data_ingestion methods of data ingestion class"
            )

            return feature_store_file_path

        except Exception as e:
            raise CustomException(e, sys) from e