
import key_param
from pymongo import MongoClient

client = MongoClient(
    key_param.mongodb_uri,
    serverSelectionTimeoutMS=5000
)

try:
    client.admin.command("ping")
    print("Connexion MongoDB réussie !")
finally:
    client.close()
