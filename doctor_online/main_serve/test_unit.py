import json
import random
import requests

client_id = "xjm19awSW7q"
client_secret = "lPA8KANbTopaiNEvFZUFuMMGC"
service_id = "W81959"


def unit_chat(chat_input, terminal_id="88888"):
    chat_reply = "正在休息，稍后回复."
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s" % (client_id, client_secret)
    res = requests.get(url)
    access_token = eval(res.text)["access_token"]

    unit_chatbot_url = "https://aip.baidubce.com/rpc/2.0/unit/service/v3/chat?access_token=" + access_token
    # 拼装data数据
    post_data = {
                "log_id": str(random.random()),
                "request": {
                    "query": chat_input,
                    "terminal_id": terminal_id
                },
                "session_id": "",
                "service_id": service_id,
                "version": "3.0"
            }

    res = requests.post(url=unit_chatbot_url, json=post_data)
    unit_chat_obj = json.loads(res.content)
    # print(unit_chat_obj)

    if unit_chat_obj["error_code"] != 0:
        return chat_reply

    # 返回内容 result -> responses -> schema -> intent_confidence(>0) -> actions -> say
    unit_chat_obj_result = unit_chat_obj["result"]
    unit_chat_response_list = unit_chat_obj_result["responses"]

    response_list = []
    for each_response in unit_chat_response_list:
        if each_response["schema"]["intents"][0]["intent_confidence"] > 0.0:
            response_list.append(each_response)
    unit_chat_response_obj = random.choice(response_list)

    unit_chat_response_action_list = unit_chat_response_obj["actions"]
    unit_chat_response_action_obj = random.choice(unit_chat_response_action_list)
    unit_chat_response_say = unit_chat_response_action_obj["say"]
    return unit_chat_response_say


if __name__ == '__main__':
    # chat_reply = unit_chat("你好")
    while True:
        chat_input = input(" >>>：")
        if chat_input == 'Q' or chat_input == 'q' or chat_input == 'bye':
            break
        # print(chat_input)
        chat_reply = unit_chat(chat_input)
        # print("用户输入 >>>", chat_input)
        # print("Unit回复 >>>", chat_reply)
        print(" >>>", chat_reply)
