## How to deploy model using Triton Inference Server

```bash
# EXPORT [first terminal]
source scripts/docker_names.sh
sh scripts/deploy/build+run.sh
# next command will take a lot of time
python export.py --weights yolov5s.pt --include onnx engine --device 0 --simplify --opset 17 --workspace 24
mv yolov5s.engine utils/triton/yolov5s_trt/1/model.plan

# RUN TRITON [second terminal]
docker exec -it yolo-deploy /bin/bash
/opt/tritonserver/bin/tritonserver \
    --model-repository=/usr/src/app/utils/triton/ \
    --backend-directory=/opt/tritonserver/backends/ \
    --backend-config=tensorrt,coalesce-request-input=true \
    --model-control-mode=none \
    --allow-grpc=true \
    --grpc-port=8000 \
    --allow-http=true

# EXAMPLE REQUIEST [first terminal]
python detect.py --weights grpc://0.0.0.0:8000 --source data/cat/images/IMG_20211226_111130.jpg --data data/coco128.yaml
```