## How to deploy model using Triton Inference Server with DeepStream Python bindings

To run the test app:
```bash
python3 main.py <h264_elementary_stream>
```

This document shall describe the sample deepstream-ssd-parser application.

It is meant for simple demonstration of how to make a custom neural network
output parser and use it in the pipeline to extract meaningful insights
from a video stream.

This example:
- Uses YOLOv5 neural network running on Triton Inference Server
- Selects custom post-processing in the Triton Inference Server config file
- Parses the inference output into bounding boxes
- Performs post-processing on the generated boxes with NMS (Non-maximum Suppression)
- Adds detected objects into the pipeline metadata for downstream processing
- Encodes OSD output and saves to MP4 file. Note that there is no visual output on screen.

## Useful links
https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/apps/deepstream-ssd-parser