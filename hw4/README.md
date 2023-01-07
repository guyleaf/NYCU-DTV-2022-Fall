# Live Streaming Server

## Prerequisite

### Hardware

1. Intel CPU
2. Webcam

### Software

1. Anaconda or miniconda
2. Git
3. [Git LFS](https://git-lfs.com/)
4. FFmpeg

## Installation

```bash
git lfs install
# git clone git@github.com:guyleaf/NYCU-DTV-2022-Fall.git
git clone https://github.com/guyleaf/NYCU-DTV-2022-Fall.git
git switch hw4

cd NYCU-DTV-2022-Fall/hw4

conda create -n live_streaming_server python=3.9
conda activate live_streaming_server

pip install -U pip
pip install -e .
pip install -r requirements.txt
```

## Demo

```bash
conda activate live_streaming_server

# nireq: based on you cpu, 2/4/8
# tip: use --streaming_debug to check fps
# tip: use --show_model_output to preview frame
python scripts/demo.py --model "yolov7.yolov7-tiny_int8" --pre_api --grid --nireq 4 --infer_device CPU --ffmpeg_path "your ffmpeg path"

# Check more options
python scripts/demo.py -h
```
