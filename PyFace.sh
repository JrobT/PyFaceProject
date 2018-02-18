#!/bin/bash

python3 face_recogniser.py
python3 pick_params.py
python3 svms.py  # compare support vector machines with different kernels
python3 cnn.py  # compare convolutional neural network (also with dropout)
# python3 music_player.py  # run music player application demo
