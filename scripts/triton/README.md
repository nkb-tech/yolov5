## How to deploy model using Triton Inference Server

```bash
# EXPORT [first terminal]
source scripts/docker_names.sh
sh scripts/deploy/build+run.sh # OR docker exec -it yolo-deploy /bin/bash
# next command will take a lot of time
python export.py --weights yolov5s.pt --include onnx --device 0 --simplify --opset 15 --workspace 24
mv yolov5s.onnx utils/triton/yolov5s_onnx/1/model.onnx

# RUN TRITON [second terminal]
docker exec -it yolo-deploy /bin/bash
/opt/tritonserver/bin/tritonserver \
    --model-repository=/usr/src/app/utils/triton/ \
    --backend-directory=/opt/tritonserver/backends/ \
    --backend-config=tensorrt,coalesce-request-input=true \
    --model-control-mode=none \
    --allow-grpc=true \
    --grpc-port=8000 \
    --allow-grpc=false

# EXAMPLE REQUIEST [first terminal]
python detect.py --weights grpc://0.0.0.0:8000 --source data/cat/images/IMG_20211226_111130.jpg --data data/coco128.yaml

# TRITON+DEEPSTREAM
cd utils/deepstream_triton/pysrc/yolov5s_trt
# script now supports only one stream with file-based input decoded in `h264` format
# for example ffmpeg -i input.mp4 -an -vcodec libx264 -crf 23 output.h264
python3 deepstream.py /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264
```