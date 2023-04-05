REDIS_CONFIG = {
     "host": "0.0.0.0",
     "port": 6379
}


NEO4J_CONFIG = {
    "uri": "bolt://127.0.0.1:7687",
    "auth": ("neo4j", "123456"),
    "encrypted": False
}

model_serve_url = "http://0.0.0.0:5001/root/ai_docotor/doctor_online/main_serve"

TIMEOUT = 2

reply_path = "./reply.json"

ex_time = 36000


