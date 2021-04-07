#!/bin/sh

# apt update & install necessary package
sudo apt update -y
sudo apt install -y xvfb python-opengl

# nnabla-cuda
pip install nnabla-ext-cuda110

# virtualdisplay
pip install pyvirtualdisplay
