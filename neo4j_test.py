from neo4j import GraphDatabase

uri = "bolt://182.92.83.160:7687"

driver = GraphDatabase.driver(uri, auth=("neo4j", "123456") # 密码自己修改的
                              , max_connection_lifetime=1000)

with driver.session() as session:
    cypher = "create (c:Company) SET c.name='company' return c.name"
    record = session.run(cypher)
    print(type(record), record)
    result = list(map(lambda x: x[0], record))
    print("result: ", result)
