from pymongo.mongo_client import MongoClient
import pandas as pd
import json

uri = "mongodb+srv://rishuu300:Rishurock300@cluster0.chcy3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri)

# Database and cluster
DATABASE_NAME = "ml_projects"
COLLECTION_NAME = "sensor-fault"

df = pd.read_csv(
    "C:\\Users\\91808\\Documents\\PW\\6. Machine Learning\\15. Projects\\1. Sensor\\notebooks\\wafer_23012020_041211.csv"
)

df = df.drop("Unnamed: 0", axis=1)

json_record = list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)