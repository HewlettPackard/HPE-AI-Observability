######################################################################
# (C)Copyright 2022 Hewlett Packard Enterprise Development LP
######################################################################


from hpeai.adl.functions.utils.plugin_utils import register_plugin

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
inChannel = 1
x, y = 128, 128
input_img = Input(shape=(x, y, inChannel))


def auto_encoder(input_img):
    # encoder
    # input = 28 x 28 x 1 (wide and thin)
    print('Custom AE: creating custom autoencoder')
    conv1 = Conv2D(32, (3, 3), activation='relu',
                   padding='same')(input_img)  # 28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu',
                   padding='same')(pool1)  # 14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(
        pool2)  # 7 x 7 x 128 (small and thick)
    # decoder
    conv4 = Conv2D(128, (3, 3), activation='relu',
                   padding='same')(conv3)  # 7 x 7 x 128
    up1 = UpSampling2D((2, 2))(conv4)  # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu',
                   padding='same')(up1)  # 14 x 14 x 64
    up2 = UpSampling2D((2, 2))(conv5)  # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid',
                     padding='same')(up2)  # 28 x 28 x 1
    print('Custom AE: custom autoencoder created successfully')
    return decoded


def create():
    autoencoder = Model(input_img, auto_encoder(input_img))
    autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())
    autoencoder.summary()
    return autoencoder


class CustomAE:
    def __init__(self, arg1=1, arg2=2) -> None:
        pass

    def get_auto_encoder(self):
        return create()


def register():
    register_plugin('customae', CustomAE)
