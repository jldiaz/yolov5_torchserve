yolov5s.mar: ressources/yolov5s.torchscript ressources/torchserve_handler.py
	torch-model-archiver --force --model-name yolov5s --version 0.1\
  --serialized-file ressources/yolov5s.torchscript --handler ressources/torchserve_handler.py\
  --extra-files ressources/index_to_name.json,ressources/torchserve_handler.py

ressources/yolov5s.torchscript: ressources/yolov5s.pt
	python ressources/yolov5/export.py --weights ressources/yolov5s.pt --include torchscript --batch 1 --img 640

restar-server:
	-killall java
	-rm -rf logs
	torchserve --start --model-store . --models yolov5s=yolov5s.mar --ts-config config.properties