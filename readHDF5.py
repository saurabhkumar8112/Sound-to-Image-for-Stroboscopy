import h5py
import numpy as np
import matplotlib.pyplot as plt
cyst = '/home/mak/Desktop/Projects/Stroboscopy/Cyst.hdf5'
nodule = '/home/mak/Desktop/Projects/Stroboscopy/Nodule.hdf5'
polyp = '/home/mak/Desktop/Projects/Stroboscopy/Polyp.hdf5'
hdf5_file_cyst = h5py.File(cyst, "r")
hdf5_file_nodule = h5py.File(nodule, "r")
hdf5_file_polyp = h5py.File(polyp, "r")
# subtract the training mean
cyst_num = hdf5_file_cyst["train_img"].shape[0]
print cyst_num
nodule_num = hdf5_file_nodule["train_img"].shape[0]
print nodule_num
polyp_num = hdf5_file_polyp["train_img"].shape[0]
print polyp_num



imageC = hdf5_file_cyst["train_img"][0:10,:,:,:]
imageN = hdf5_file_nodule["train_img"][0:10,:,:,:]
imageP = hdf5_file_polyp["train_img"][0:10,:,:,:]

imgC = imageC[0]
imgP = imageP[0]
imgN = imageN[0]

f = plt.figure()
sp1 = f.add_subplot(1,3,1)
plt.imshow(imgC)
sp1.set_title("Cyst")
sp2 = f.add_subplot(1,3,2)
plt.imshow(imgN)
sp2.set_title("Nodule")
sp3 = f.add_subplot(1,3,3)
plt.imshow(imgP)
sp3.set_title("Polyp")
plt.show()