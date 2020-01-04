from random import shuffle
import glob
import matplotlib.pyplot as plt
import matplotlib._png as png
import h5py
import numpy as np
shuffle_data = True  # shuffle the addresses before saving
hdf5_path_cyst = '/home/mak/Desktop/Projects/Stroboscopy/Cyst.hdf5'  # address to where you want to save the hdf5 file
hdf5_path_nodule = '/home/mak/Desktop/Projects/Stroboscopy/Nodule.hdf5'
hdf5_path_polyp = '/home/mak/Desktop/Projects/Stroboscopy/Polyp.hdf5'


train_path_cyst = '/home/mak/Desktop/Projects/Stroboscopy/VocalCystCropped1/*.png'
train_path_nodule = '/home/mak/Desktop/Projects/Stroboscopy/VocalNoduleCropped1/*.png'
train_path_polyp = '/home/mak/Desktop/Projects/Stroboscopy/VocalPolypCropped1/*.png'
# read addresses and labels from the 'train' folder
addrs_cyst = glob.glob(train_path_cyst)
addrs_nodule = glob.glob(train_path_nodule)
addrs_polyp = glob.glob(train_path_polyp)


##First Cysts 
print "Creating cysts dataset"


hdf5_file = h5py.File(hdf5_path_cyst, mode='w')
train_shape = (len(addrs_cyst), 256, 256, 3)
hdf5_file.create_dataset("train_img", train_shape, np.uint8)

for i in range(len(addrs_cyst)):
    # print how many images are saved every 1000 images
    if i % 100 == 0 and i > 1:
        print 'Train data: {}/{}'.format(i, len(addrs_cyst))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = addrs_cyst[i]
    img =  png.read_png_int(addr)
    print img.dtype
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    hdf5_file["train_img"][i,:,:,:] = img
#
hdf5_file.close()


print "Creating nodules dataset"

hdf5_file = h5py.File(hdf5_path_nodule, mode='w')
train_shape = (len(addrs_nodule), 256, 256, 3)
hdf5_file.create_dataset("train_img", train_shape, np.uint8)

for i in range(len(addrs_nodule)):
    # print how many images are saved every 1000 images
    if i % 100 == 0 and i > 1:
        print 'Train data: {}/{}'.format(i, len(addrs_nodule))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = addrs_nodule[i]
    img =  png.read_png_int(addr)
    print img.dtype
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    hdf5_file["train_img"][i,:,:,:] = img
#
hdf5_file.close()



print "Creating polyps dataset"

hdf5_file = h5py.File(hdf5_path_polyp, mode='w')
train_shape = (len(addrs_polyp), 256, 256, 3)
hdf5_file.create_dataset("train_img", train_shape, np.uint8)

for i in range(len(addrs_polyp)):
    # print how many images are saved every 1000 images
    if i % 100 == 0 and i > 1:
        print 'Train data: {}/{}'.format(i, len(addrs_polyp))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = addrs_polyp[i]
    img =  png.read_png_int(addr)
    print img.dtype
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    hdf5_file["train_img"][i,:,:,:] = img
#
hdf5_file.close()

