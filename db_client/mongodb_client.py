from pymongo import MongoClient
from dotenv import load_dotenv
import json
import os

load_dotenv()
JWT_DURATION = os.getenv("JWT_DURATION")
JWT_SECRET = os.getenv("JWT_SECRET")
MONGODB_SRV = os.getenv("MONGODB_SRV")

# Instantiate database object and collection object
client = MongoClient(MONGODB_SRV)
database = client.dev
devices_collection = database.devices
dist_collection = database.distMeasure
color_collection = database.colorMeasure
reso_collection = database.resoMeasure
