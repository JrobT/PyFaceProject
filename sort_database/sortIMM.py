#!/usr/bin/env python3
"""Python script to sort the IMM Database.

Cited in my report.
"""
from PIL import Image
from shutil import copyfile

import PIL
import glob


dataset = "IMM"


def standardise_image(pic):
    """Save image in resized, standard format."""
    img = Image.open(open(pic, 'rb')).convert('LA')  # Grayscale

    basewidth = 480  # Resize
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

    img.save(pic, "PNG")  # Save as .png


print("***> Processing IMM Face Database...")

# Return a list of all the jpg image files.
files = glob.glob("IMM_dataset//*.jpg")

cnt = 0  # iterator for naming the saved image
dest = ""  # destination for the image
for image in files:
    cnt = cnt + 1

    # Filename format = <person number>-<image type><gender>.
    boolean_check = image.split("-")
    boolean_check[0] = boolean_check[0][12:]  # <person number>
    boolean_check[1] = boolean_check[1][:-5]  # <image type>

    if int(boolean_check[1]) in (1, 3, 4):
        dest = "combined_dataset//neutral//IMM{}.png".format(cnt)
    elif int(boolean_check[1]) == 2:
        dest = "combined_dataset//happy//IMM{}.png".format(cnt)
    else:
        continue

    try:
        # All images are kept to a standardised format.
        standardise_image(image)

        # Finally, copy the files to the combined dataset.
        copyfile(image, dest)

        print("{}: Copied {} into {}."
              .format(dataset, image.replace("//", "/"),
                      dest.replace("//", "/")))
    except OSError as e:
        print('***> Some IO error occurred!!')
        continue
