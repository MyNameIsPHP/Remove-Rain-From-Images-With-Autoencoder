import pickle
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2


import os
import keras
from rain_add import *
import numpy as np

# Dinh nghia bien
image_size = 64

def load_normal_images(data_path):
    normal_images_path = os.listdir(data_path)
    normal_images = []
    for img_path  in normal_images_path:
        full_img_path = os.path.join(data_path, img_path)
        img = keras.utils.load_img(full_img_path)
        img = keras.utils.img_to_array(img)
        img = img/255
        # Dua vao list
        normal_images.append(img)
    normal_images = np.array(normal_images)
    return normal_images

def make_rain_images(normal_images):
    rain_images = []
    for img in normal_images:
        rain_image = add_rain(img)
        rain_images.append(rain_image)
    rain_images = np.array(rain_images)
    return rain_images

# Doc du lieu train, test tu file
normal_path = 'images'
rain_path = 'rain_images'

normal_images = load_normal_images(normal_path)
rain_images = load_normal_images(rain_path)

# Load model
model = load_model("rain_remove_model.h5")

# Chon random 5 anh de khu nhieu
s_id = 25
e_id = 35

pred_images = model.predict(rain_images[s_id: e_id])

# Ve len man hinh de kiem tra
for i in range(s_id, e_id):
    new_image = cv2.blur(rain_images[i], (3,3))
    new_image_1 = cv2.blur(rain_images[i], (5, 5))
    plt.figure(figsize=(8,3))
    plt.subplot(141)
    plt.imshow(pred_images[i-s_id], cmap='gray')
    plt.title('Model')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(142)
    plt.imshow(new_image, cmap='gray')
    plt.title('Blur OpenCV (K3)')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(143)
    plt.imshow(new_image_1, cmap='gray')
    plt.title('Blur OpenCV (K5)')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(144)
    plt.imshow(rain_images[i], cmap='gray')
    plt.title('Noise image')
    plt.xticks([])
    plt.yticks([])

    plt.show()