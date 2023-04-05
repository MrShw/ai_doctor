import werobot
import requests

# 主要逻辑服务请求地址, ip+端口
url = "http://182.92.83.160:5000/main_serve/"
robot = werobot.WeRoBot(token="sean86")
# 服务超时时间
TIMEOUT = 3

# 处理请求入口
@robot.handler
def doctor(message, sesssion):
    try:
        uid = message.source
        try:
            if sesssion.get(uid, None) != '1':
                sesssion[uid] = '1'
                return "您好, 我是Mr.S, 有什么需要帮忙的吗?"

            text = message.content
        except:
            return "您好, 我是Mr.S, 有什么需要帮忙的吗?"

        data = {'uid': uid, 'text': text}
        res = requests.post(url, data=data, timeout=TIMEOUT)

        return res.text

    except Exception as e:
        print(e)
        return "wr.py 捕捉错误，机器人正在休息..."


robot.config['HOST'] = '0.0.0.0'
robot.config['PORT'] = 80
robot.run()
