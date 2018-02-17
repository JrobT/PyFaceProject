#!/usr/bin/env python3

"""Python script to sort the Google Images Dataset.

I filled out my database by using the Google images search engine. By typing
each emotion into the search, I used the ZIG Lite Chrome extension
(chrome-extension://bedbigoemkinkepgmcmgnapjcahnedmn/out.html?get) to scrape
the page for suitable images.

This may not be strictly useful. These images come without proper permission.
"""

# Import packages.
import glob
from shutil import copyfile

# My imports.
from utils import EMOTIONS_8, standardise_image


dataset = "Google"


print("***> Processing my `Google Images' Dataset...".format(dataset))

num = 0
for emotion in EMOTIONS_8:
    for f in glob.glob("Google_dataset//{0!s}//*".format(emotion)):
        num = num + 1
        dest = "combined_dataset//{}//{}_{}_{}.png".format(EMOTIONS_8[emotion],
                                                           dataset, num,
                                                           "frontal")

        try:
            # All images are kept to a standardised format.
            standardise_image(f)

            # Finally, copy the files to the combined dataset.
            copyfile(f, dest)

            print("Successful copy number {} for {} dataset."
                  .format(num, dataset))
        except OSError as e:
            print('***> Some IO error occurred!!')
            continue
