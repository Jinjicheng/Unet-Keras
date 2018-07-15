from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,Activation
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from aug_data import dataProcess


class myUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        # imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train  #, imgs_test

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 1))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        print("conv1_1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print("conv1_2 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:", pool1.shape)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print("conv2_1 shape:", conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print("conv2_2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:", pool2.shape)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print("conv3_1 shape:", conv3.shape)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print("conv3_2 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:", pool3.shape)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        print("conv4_1 shape:",conv4.shape)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        print("conv4_2 shape:",conv4.shape)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        print("pool4 shape:",pool4.shape)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        print("conv5_1 shape:",conv5.shape)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        print("conv5_2 shape:",conv5.shape)
        drop5 = Dropout(0.5)(conv5)
        print("drop5 shape:",drop5.shape)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        print("up6 shape:",up6.shape)
        merge6 = merge([drop4, up6], mode='concat', concat_axis=3) # 合并这两层，从第3维度合并，这要求待合并层最后一个维度不同
        print("merge6 shape:",merge6.shape)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        print("conv6 shape:",conv6.shape)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        print("up7 shape:",up7.shape)
        merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        print("merge7 shape:",merge7.shape)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        print("conv7 shape:",conv7.shape)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        print("up8 shape:",up8.shape)
        merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        print("merge8 shape:",merge8.shape)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        print("conv8 shape:",conv8.shape)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        print("up9 shape:",up9.shape)
        merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
        print("merge9.shape:",merge9.shape)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        print("conv9 shape:",conv9.shape)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        print("conv 2 class shape:",conv9.shape)
        conv10 = Conv2D(1, 1, activation='softmax')(conv9)
        print("last conv 1*1 shape:",conv10.shape)

        model = Model(input=inputs, output=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        print('model compile')
        return model

    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")

        # 保存的是模型和权重,
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=10, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint])

        # print('predict test data')
        # imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        # np.save('imgs_mask_test.npy', imgs_mask_test)


if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()