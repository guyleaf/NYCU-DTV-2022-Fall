from setuptools import setup

setup(
    name="live_streaming_server",
    packages=["live_streaming_server"],
    version="0.0.1",
    install_requires=[
        "openvino==2022.3.0",
        "numpy",
        "typed-argument-parser~=1.7.2",
        "Flask~=2.2.2",
        "python-ffmpeg-video-streaming~=0.1.15",
    ],
)
