import os
import fileinput
from neo4j import GraphDatabase


NEO4J_CONFIG = {
    "uri": "bolt:localhost:7687",
    "auth": ("neo4j", "123456"),
    "encrypted": False
}

driver = GraphDatabase.driver(**NEO4J_CONFIG)


def _load_data(path):
    disesase_csv_list = os.listdir(path)
    disesase_list = list(map(lambda x: x.split('.')[0], disesase_csv_list))
    symptom_list = []

    for disesase_csv in disesase_csv_list:
        f = fileinput.FileInput(os.path.join(path, disesase_csv))
        symptom = list(map(lambda x: x.strip(), f))
        symptom = list(filter(lambda x: 0 < len(x) < 100, symptom))
        symptom_list.append(symptom)

    result = dict(zip(disesase_list, symptom_list))

    return result


def neo4j_write(path):
    disease_symptom_dict = _load_data(path)

    with driver.session() as session:
        for key, values in disease_symptom_dict.items():
            # key 就是疾病名称
            # values 是一个列表，key这个疾病对应的多种症状
            # 创建疾病节点
            cypher = "MERGE (a:Disease{name:%r}) RETURN a" % key
            session.run(cypher)

            for value in values:
                # 创建症状节点
                cypher = "MERGE (b:Symptom{name:%r}) RETURN b" % value
                session.run(cypher)

                # 创建疾病和症状的关系
                cypher = "MATCH (a:Disease{name:%r}) MATCH(b:Symptom{name:%r}) WITH a, b MERGE(a)-[r:dis_to_sym]-(b)" % (
                key, value)
                session.run(cypher)

        cypher = "CREATE INDEX ON:Disease(name)"
        session.run(cypher)
        cypher = "CREATE INDEX ON:Symptom(name)"
        session.run(cypher)


if __name__ == '__main__':
    path = "./structured/reviewed/"
    neo4j_write(path)
