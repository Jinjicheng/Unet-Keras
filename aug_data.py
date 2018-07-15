# 这个文件是用于将增强后的train和label分别合并成npy文件, 模仿原作者的data.py
import os, glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
from data_augmentation import get_file

class dataProcess(object):
    def __init__(self, out_rows, out_cols, aug_merge_path="./Unet_data/augmentation", aug_train_path="./Unet_data/merge_split/train/",
                 aug_label_path="./Unet_data/merge_split/label/", npy_path="./Unet_data/npydata",
                 test_path = "./Unet_data/split_image/test/",img_type="tif"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.aug_merge_path = aug_merge_path
        self.aug_train_path = aug_train_path
        self.aug_label_path = aug_label_path
        self.npy_path = npy_path
        self.test_path = test_path
        self.img_type = img_type

    def create_train_data(self):
        i = 0
        print('-' * 30)
        print('creating train image')
        print('-' * 30)
        count = len(os.listdir(self.aug_merge_path))
        imgdatas = np.ndarray((count, self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imglabels = np.ndarray((count, self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for imagename in (os.listdir(self.aug_train_path)):
                img = load_img(self.aug_train_path + imagename, grayscale=True)
                label = load_img(self.aug_label_path + imagename, grayscale=True)
                img = img_to_array(img)
                label = img_to_array(label)
                imgdatas[i] = img
                imglabels[i] = label
                i += 1
                print(i)
        print('loading done', imgdatas.shape)
        np.save(self.npy_path + '/augimgs_train.npy', imgdatas)            # 将30张训练集和30张label生成npy数据
        np.save(self.npy_path + '/augimgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

    def creat_test_data(self):
        i = 0
        print('-'*30)
        print('Creating test images...')
        print('-'*30)
        imgs = os.listdir(self.test_path)
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1),dtype=np.uint8)
        for imgname in imgs:
            img = load_img(self.test_path + imgname,grayscale=True)
            img = img_to_array(img)
            imgdatas[i] = img
            print(i)
            i += 1
        print('loading done',imgdatas.shape)
        np.save(self.npy_path + '/imgs_test.npy',imgdatas)
        print('Saving to .npy files done.')

    def load_train_data(self):
        print('-' * 30)
        print('loading train data')
        print('-' * 30)
        augimgs_train = np.load(self.npy_path + '/augimgs_train.npy')
        augimgs_mask_train = np.load(self.npy_path + '/augimgs_mask_train.npy')
        augimgs_train = augimgs_train.astype('float32')
        augimgs_mask_train = augimgs_mask_train.astype('float32')
        augimgs_train /= 255
        mean = augimgs_train.mean(axis=0)
        augimgs_train -= mean
        augimgs_mask_train /= 255
        augimgs_mask_train[augimgs_mask_train > 0.5] = 1
        augimgs_mask_train[augimgs_mask_train <= 0.5] = 0
        print(augimgs_mask_train.shape)
        return augimgs_train, augimgs_mask_train

    def load_test_data(self):
        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        mean = imgs_test.mean(axis=0)
        imgs_test -= mean
        return imgs_test
if __name__ == '__main__':
    mydata = dataProcess(512, 512)
    # mydata.create_train_data()
    # train,label = mydata.load_train_data()
    # binary = label[12]
    # binary = array_to_img(binary)
    # binary.show()
    # print(binary.size)
    mydata.creat_test_data()
