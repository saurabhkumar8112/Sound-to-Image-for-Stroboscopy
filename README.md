# WGAN
To train follow these steps
1. Move all images to 'AllImages' folder. 'moveFiles.py' is a utility if data involves sub directories
2. Create hdf5 dataset using 'createHDF5.py', need to give appropriate paths in the file.
3. Check the dataset using 'readHDF5.py'
4. Run 'myTrain.py' to start training. Can modify the epochs and all in this file
5. After training images can be generated using 'GenerateImages.py'. Number of images can be specified.
6. While training checkpoints are saved every 100 iters and images produced every 50 iters.


