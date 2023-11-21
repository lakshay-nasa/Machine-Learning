import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "/"
})

ref = db.reference('Users')

data = {
    "678655":
        {
            "name": "Sam",
            "role": "Assistant",
            "status": "Online",
            "starting_year": 2024,
            "access_level": 3,
            "scl": "TS",
            "year": 1,
            "last_grant_time": "2022-04-13 00:54:34"
        },
    "456787":
        {
            "name": "Lakshay",
            "role": "Creator",
            "status": "Online",
            "starting_year": 2021,
            "access_level": 1,
            "scl": "A",
            "year": 3,
            "last_grant_time": "2022-04-13 00:54:34"
        },
    "232479":
        {
            "name": "Elan",
            "role": "Assistant",
            "status": "Online",
            "starting_year": 2022,
            "access_level": 4,
            "scl": "R",
            "year": 2,
            "last_grant_time": "2024-04-13 00:54:34"
        }
}

for key, value in data.items():
    ref.child(key).set(value)