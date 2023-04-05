import requests

url = "http://182.92.83.160/"
# data = {"uid":"1888254", "text": "头疼"}
data = {"uid":"1888254", "text": "鼻出血"}


res = requests.post(url, data=data)

print(res.text)
