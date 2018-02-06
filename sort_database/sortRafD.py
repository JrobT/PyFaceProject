#!/usr/bin/env python3

"""Python script to sort the RafD Database. Cited in my report."""

import PIL.Image
import PIL.ImageFile
import glob
from shutil import copyfile


dataset = "RafD"
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


def standardise_image(pic):
    """Save image in resized, standard format."""
    img = PIL.Image.open(open(pic, 'rb')).convert('LA')  # Grayscale

    basewidth = 480  # Resize
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

    img.save(pic, "PNG")  # Save as .png


print("***> Processing RafD Dataset...")

# Return a list of all the jpg image files.
files = glob.glob("RafD_dataset//*.jpg")

cnt = 0  # iterator for naming the saved image
dest = ""  # destination for the image
for image in files:
    cnt = cnt + 1

    # Filename format = RafD_dataset/<subject number>_<num>_<person>_<gender>_<emotion>_<direction>.jpg
    boolean_check = image.split("_")
    img_emotion = boolean_check[5]  # <emotion>
    img_direction = boolean_check[6][:-4]  # <direction>

    if img_emotion == "neutral":
        dest_emot = "combined_dataset//{}//{}{}_{}.png".format("neutral",
                                                               dataset, cnt,
                                                               img_direction)
    elif img_emotion == "angry":
        dest_emot = "combined_dataset//{}//{}{}_{}.png".format("anger",
                                                               dataset, cnt,
                                                               img_direction)
    elif img_emotion == "happy":
        dest_emot = "combined_dataset//{}//{}{}_{}.png".format("happy",
                                                               dataset, cnt,
                                                               img_direction)
    elif img_emotion == "fearful":
        dest_emot = "combined_dataset//{}//{}{}_{}.png".format("fear",
                                                               dataset, cnt,
                                                               img_direction)
    elif img_emotion == "contemptuous":
        dest_emot = "combined_dataset//{}//{}{}_{}.png".format("contempt",
                                                               dataset, cnt,
                                                               img_direction)
    elif img_emotion == "surprised":
        dest_emot = "combined_dataset//{}//{}{}_{}.png".format("surprise",
                                                               dataset, cnt,
                                                               img_direction)
    elif img_emotion == "disgusted":
        dest_emot = "combined_dataset//{}//{}{}_{}.png".format("disgust",
                                                               dataset, cnt,
                                                               img_direction)
    elif img_emotion == "sad":
        dest_emot = "combined_dataset//{}//{}{}_{}.png".format("sad",
                                                               dataset, cnt,
                                                               img_direction)
    else:
        continue

    try:
        # All images are kept to a standardised format.
        standardise_image(image)

        # Finally, copy the files to the combined dataset.
        copyfile(image, dest_emot)

        print("{}: Copied {} into {}."
              .format(dataset, image.replace("//", "/"),
                      dest_emot.replace("//", "/")))
    except OSError as e:
        print(e)
        continue
