from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
# from keras.datasets import cifar10
import glob
import utils
import traceback
import numpy as np
import tensorflow as tf
import models_64x64 as models
import h5py


z_dim = 100
imgNo = 100


generator = models.generator
discriminator = models.discriminator_wgan_gp

z = tf.placeholder(tf.float32, shape=[None, z_dim])

f_sample = generator(z, training=False,reuse = None)

sess = tf.Session()

ckpt_dir = './checkpoints/checkpoints'
utils.mkdir(ckpt_dir + '/')
if not utils.load_checkpoint(ckpt_dir, sess):
    sess.run(tf.global_variables_initializer())

save_dir  = "./"
print ("Loaded Checkpoint #$$@$")
for i in range(0,5):
	print (i)
	z_ipt_sample = np.random.normal(size=[imgNo, z_dim])
	f_sample_opt = sess.run(f_sample, feed_dict={z: z_ipt_sample})
	epoch = i
	it_epoch = 0
	batch_epoch = 0
	utils.imwrite(utils.immerge(f_sample_opt, 10, 10), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_epoch, batch_epoch))


