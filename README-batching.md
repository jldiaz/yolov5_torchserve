The docker image created with Docker-gpu-smaller contains a yolov5s.mar model enabled for batch_size=32
but it can be used with any batch_size, depending on how the model is registered in torchserve.

By default, the endpoint /predictions/yolov5s is registered with a batch_size=1, but you can
register aditional endpoints which use a different bach_size by issuing the appropiate POST
command to the container's IP, port 8081, as for example:

$ curl -X POST "172.17.0.2:8081/models?model_name=yolov5sb4&url=yolov5s.mar&batch_size=4&initial_workers=1&max_batch_delay=5000&synchronous=true"

This example would create the endpoint /predictions/yolov5sb4 which uses a batch size of 4, and batch_delay of 5 seconds.

This means that torchserve will equeue the requests until 4 are collected (or no more arrive in 5 seconds). 
Then it will perform the inference with a batch size of 4, and finally it will deliver each result
to the appropiate client.

You can test this with the images in `sample_images`, doing for example:

```bash
http --multipart POST 172.17.0.2:8080/predictions/yolov5sb4 img1@perro-robot-resized.jpg > perro-results.json &
http --multipart POST 172.17.0.2:8080/predictions/yolov5sb4 img1@bus-resized.jpg > bus-results.json &
http --multipart POST 172.17.0.2:8080/predictions/yolov5sb4 img1@horses-resized.jpg > horses-results.json &
http --multipart POST 172.17.0.2:8080/predictions/yolov5sb4 img1@zidane-resized.jpg > zidane-results.json &
```
