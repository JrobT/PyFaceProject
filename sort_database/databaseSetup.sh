#!/bin/bash

python3 cleanMyDataset.py  # Clean the destination folders
python3 sortDataset.py  # Sort through each dataset I've collected
python3 extractFaces.py  # Extract faces from the collected images
# At this point, `database' should be full of facial images representing each
# emotion I have listed in my emotions list.
