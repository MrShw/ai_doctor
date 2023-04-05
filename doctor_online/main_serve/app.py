from flask import Flask
from flask import request
app = Flask(__name__)

import requests
import redis
import json
from doctor_online.main_serve.test_unit import unit_chat
from neo4j import GraphDatabase

from config import NEO4J_CONFIG
from config import REDIS_CONFIG
from config import model_serve_url
from config import TIMEOUT
from config import reply_path
from config import ex_time

# init Redis and Neo4j
pool = redis.ConnectionPool(**REDIS_CONFIG)
_driver = GraphDatabase.driver(**NEO4J_CONFIG)

def query_neo4j(text):
    with _driver.session() as session:
        cypher = "MATCH(a:Symptom) WHERE(%r contains a.name) WITH" \
                 " a MATCH(a)-[r:dis_to_sym]-(b:Disease) RETURN b.name LIMIT 5" % text
        record = session.run(cypher)
        result = list(map(lambda x: x[0], record))

    return result


class Handler(object):
    def non_first_sentence(self, previous):
        try:
            print("准备请求句子相关模型服务!")
            data = {'text1':previous, 'text2':self.text}
            result = requests.post(model_serve_url, data=data, timeout=TIMEOUT)
            print("句子相关模型服务请求成功, 返回结果为:", result.text)
            if not result.text:
                return unit_chat(self.text)
            # print("non_first_sentence, unit_chat")
            # return unit_chat(self.text)
        except Exception as e:
            print("模型异常", e)
            return unit_chat(self.text)

        s = query_neo4j(self.text)
        print(s)
        if not s:
            return unit_chat(self.text)

        old_disease = self.r.hget(str(self.uid), "previous_d")
        if old_disease:
            new_disease = list(set(s)|set(old_disease))
            res = list(set(s)-set(old_disease))
        else:
            res = list(set(s))
            new_disease = res

        self.r.hset(str(self.uid), "previous_d", str(new_disease))
        self.r.expire(str(self.uid), ex_time)

        if not res:
            return self.reply['4']
        else:
            res = '，'.join(res)
            return self.reply['2'] % res

    def __init__(self, uid, text, r, reply):
        self.uid = uid
        self.r = r
        self.text = text
        self.reply = reply

    def first_sentence(self):
        s = query_neo4j(self.text)

        if not s:
            return unit_chat(self.text)

        self.r.hset(str(self.uid), 'previous_d', str(s))
        self.r.expire(str(self.uid), ex_time) #

        res = "，".join(s)

        return self.reply['2'] % res


@app.route("/main_serve/", methods=["POST"])
def main_serve():
    # receive werobot
    uid = request.form['uid']
    text = request.form['text']

    r = redis.StrictRedis(connection_pool=pool)
    previous = r.hget(str(uid), 'previous')
    print("main_serve previous:", previous)

    r.hset(str(uid), 'previous', text)
    reply = json.load(open(reply_path, 'r'))
    handler = Handler(uid, text, r, reply)

    if previous:
        return handler.non_first_sentence(previous)
    else:
        return handler.first_sentence()

if __name__ == '__main__':
    text = "我最近腹痛!"
    print(query_neo4j(text))

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
