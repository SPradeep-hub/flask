from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')

if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env file — check your .env file exists and has the correct URI")

print(f"Connecting to MongoDB Atlas...")
client = MongoClient(MONGO_URI)

db          = client['deepverify']
users_col   = db['users']
history_col = db['scan_history']

# Test connection before creating index
try:
    client.admin.command('ping')
    print("MongoDB Atlas connected successfully!")
    users_col.create_index('email', unique=True)
except Exception as e:
    raise ConnectionError(f"MongoDB Atlas connection failed: {e}")