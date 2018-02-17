#!/usr/bin/env python3

"""A script to ensure `combined_dataset' and `database' are clean.

During testing, I was required to test each stage of dataset setup. This script
iterates through the emotion list, and then builds two strings for the paths
to the emotion folder in `combined_dataset' and in `database'. Each file in
the path is then silently deleted.

This is a simple removal script. There should be no problems once it is run,
and should finish fairly quickly. It is normally ran before `sortDataset.py'
and `extractFaces.py' as part of `databaseSetup.sh'.
"""

# Import packages.
import os
import errno
import glob

# My imports.
from utils import EMOTIONS_8


def silent_remove(fname):
    """Silently remove a file that may or may not exist."""
    try:
        os.remove(fname)
    except OSError as e:
        if e.errno != errno.ENOENT:  # 'No such file or directory'
            raise  # re-raise exception if a different error occurred


for emotion in EMOTIONS_8:
    folder1 = glob.glob("combined_dataset//{0!s}//*".format(emotion))
    folder2 = glob.glob("database//{0!s}//*".format(emotion))
    folder3 = glob.glob("database2//{0!s}//*".format(emotion))
    for f in folder1:
        silent_remove(f)
    for f in folder2:
        silent_remove(f)
    for f in folder3:
        silent_remove(f)

# Return a list of all the asf files.
files = glob.glob("IMM_dataset//*.asf")
for fname in files:
    silent_remove(fname)
