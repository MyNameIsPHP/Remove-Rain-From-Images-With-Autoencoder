import os
import keras
from sklearn.model_selection import train_test_split

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import EarlyStopping

from rain_add import *
import pickle
import numpy as np

n_epochs = 20
n_batchsize = 32
normal_path = 'images'
rain_path = 'rain_images'

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



# normal_images = load_normal_images(normal_path)
# rain_images = make_rain_images(normal_images)
# print(normal_images.shape)
# print(rain_images.shape)


# Tao model Auto Encoder
def make_ae_model():
    input_img = Input(shape=(120, 160, 3), name='image_input')

    # encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv1')(input_img)
    x = MaxPooling2D((2, 2), padding='same', name='pool1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv2')(x)
    x = MaxPooling2D((2, 2), padding='same', name='pool2')(x)

    # decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv3')(x)
    x = UpSampling2D((2, 2), name='upsample1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv4')(x)
    x = UpSampling2D((2, 2), name='upsample2')(x)
    x = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='Conv5')(x)

    # model
    model = Model(inputs=input_img, outputs=x)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    return model

# Tao model
my_model = make_ae_model()
my_model.summary()


if not os.path.exists('data.dat'):
    normal_images = load_normal_images(normal_path)
    rain_images = load_normal_images(rain_path)
    # Chia du lieu train test
    rain_train, rain_test, normal_train, normal_test = train_test_split(rain_images, normal_images, test_size=0.2)
    with open("data.dat", "wb") as f:
        pickle.dump([rain_train, rain_test, normal_train, normal_test], f)
else:
    with open("data.dat", "rb") as f:
        arr = pickle.load(f)
        rain_train, rain_test, normal_train, normal_test = arr[0], arr[1], arr[2], arr[3]

# Train model
early_callback = EarlyStopping(monitor="val_loss", min_delta= 0 , patience=10, verbose=1, mode="auto")
history = my_model.fit(rain_train, normal_train, epochs=n_epochs, batch_size=n_batchsize,
                       validation_data=(rain_test, normal_test),
                       callbacks=[early_callback])

my_model.save("rain_remove_model.h5")
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


