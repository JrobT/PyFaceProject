#!/usr/bin/env python3

"""Python script to sort the IMM Database.

Cited in my report.
"""

# Import packages.
import glob
from shutil import copyfile

# My imports.
from utils import standardise_image


dataset = "IMM"


print("***> Processing IMM Face Database...")

# Return a list of all the jpg image files.
files = glob.glob("IMM_dataset//*.jpg")

cnt = 0  # iterator for naming the saved image
dest = ""  # destination for the image
for image in files:
    # Filename format = <person number>-<image type><gender>.
    boolean_check = image.split("-")
    boolean_check[0] = boolean_check[0][12:]  # <person number>
    boolean_check[1] = boolean_check[1][:-5]  # <image type>

    cnt = cnt + 1
    if int(boolean_check[1]) in (1, 3, 4):
        dest = "combined_dataset//neutral//{}_{}_{}.png".format(dataset, cnt, "frontal")
    elif int(boolean_check[1]) == 2:
        dest = "combined_dataset//happy//{}_{}_{}.png".format(dataset, cnt, "frontal")
    else:
        continue

    try:
        # All images are kept to a standardised format.
        standardise_image(image)

        # Finally, copy the files to the combined dataset.
        copyfile(image, dest)

        print("Successful copy number {} for {} dataset."
              .format(cnt, dataset))
    except OSError as e:
        print('***> Some IO error occurred!!')
        continue
