
from flask import Flask

# 创建要给app对象
app = Flask(__name__)

# when user access to /hello, hello_world method will be called.
@app.route("/hello")
def hello_world():
    return "Hello World!"



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) # 5000 port
