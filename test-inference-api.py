import requests

res = requests.post("http://localhost:8080/predictions/yolov5s", 
        files={'img1': open('perro-robot.jpg', 'rb')})
print(res.status_code)
print(res.json())
