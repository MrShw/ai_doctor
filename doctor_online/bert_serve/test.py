import requests

url = "http://0.0.0.0:5001/root/ai_docotor/doctor_online/main_serve/"
data = {"text1":"我试试", "text2": "凑合用"}
res = requests.post(url, data=data)

print("预测样本:", data["text1"], "|", data["text2"])
print("预测结果:", res.text)
