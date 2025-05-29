from pymongo import MongoClient


client = MongoClient("localhost:27017")

db = client.get_database("tecsci")

pcbs = db.get_collection("pcb")

for pcb in pcbs.find():
    print(pcb)