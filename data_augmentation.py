from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
import numpy as np
import os
import cv2

def get_file(file_path):
    file = []
    name = []
    for file_name in os.listdir(file_path):
        all_name = os.path.join(file_path,file_name)
        file.append(all_name)
        name.append(file_name)
    return file,name

def merge_mask():
    save_path = "./Unet_data/merge/"
    image_path = "./Unet_data/split_image/train/"
    label_path = "./Unet_data/split_image/label/"
    for i in range(30):
        img_path = image_path+str(i)+".tif"
        img_t = load_img(img_path)
        lab_path = label_path+str(i)+".tif"
        img_l = load_img(lab_path)
        x_t = img_to_array(img_t)
        x_l = img_to_array(img_l)
        x_t[:,:,2]=x_l[:,:,0]
        img_tmp = array_to_img(x_t)
        img_tmp.save(save_path+"/"+str(i)+".tif")

def data_aug():
    data_aug_path = "./Unet_data/augmentation/"
    merge_path = "./Unet_data/merge/"
    merge_file,merge_name = get_file(merge_path)
    n_sample = len(merge_file)

    datagen = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest')
    for i in range(n_sample):
        merge_dir = merge_path+str(i)+".tif"
        merge = load_img(merge_dir)
        merge = img_to_array(merge)
        merge = merge.reshape((1,)+merge.shape)
        j = 0
        for batch in datagen.flow(merge,batch_size=1,
                                  save_to_dir=data_aug_path,
                                  save_format='tif',
                                  save_prefix=str(i)):
            j += 1
            if j>10:
                break

def merge_split():
    data_aug_path = "./Unet_data/augmentation/"
    split_train_path = "./Unet_data/merge_split/train/"
    split_label_path = "./Unet_data/merge_split/label/"
    train_image,train_name = get_file(data_aug_path)
    n_sample = len(train_image)
    for i in range(n_sample):
        img = cv2.imread(train_image[i])
        img_train = img[:,:,2]
        img_label = img[:,:,0]
        cv2.imwrite(split_train_path+train_name[i],img_train)
        cv2.imwrite(split_label_path+train_name[i],img_label)

def show():
    image_path = "./Unet_data/merge/"
    train_image, image_name = get_file(image_path)
    n_sample = len(train_image)
    for i in range(n_sample):
        print(train_image[i])

if __name__ == "__main__":
    merge_split()
