import cv2
import os

from matplotlib import image as mpimg

from rain_add import add_rain
from PIL import Image

import glob
import json
source_path = "images"
save_path = "rain_images/"
def export_rain_images():
    images = []
    for filename in os.listdir(source_path):
        imgPath = os.path.join(source_path,filename)
        imgRe = add_rain(mpimg.imread(imgPath))
        im = Image.fromarray(imgRe)
        im.save(os.path.join(save_path,filename))

    return images

export_rain_images()