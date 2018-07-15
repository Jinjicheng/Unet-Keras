'''
split 30 single images from an array of images : train-volume.tif label-volume.tif test-volume.tif
'''
from libtiff import TIFF3D, TIFF


def split_img():
    '''split a tif volume into single tif'''
    path = './Unet_data/data/'
    imgdir = TIFF.open(path + "test-volume.tif",mode='r')
    # imgarr = imgdir.read_image()
    i = 0
    for image in imgdir.iter_images():
        imgname = "./Unet_data/split_image/test/"  + str(i) + ".tif"
        i = i + 1
        img = TIFF.open(imgname, 'w')
        img.write_image(image)

def merge_img():
    '''merge single tif into a tif volume'''
    path = './Unet_data/data/'
    # imgdir = TIFF3D.open("test_mask_volume_server2.tif", 'w')
    imgarr = []
    for i in range(30):
        img = TIFF.open(path + str(i) + ".tif")
        imgarr.append(img.read_image())
    # imgdir.write_image(imgarr)


if __name__ == "__main__":
    split_img()