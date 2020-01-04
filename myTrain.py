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

""" param """
epoch = 10000
batch_size = 64
lr = 0.0002
z_dim = 100
n_critic = 7
gpu_id = 1
labelNames = ["Cyst","Nodule","Polyp"]
''' data '''
# you should prepare your own data in ./data/faces
# cartoon faces original size is [96, 96, 3]
print ("Loading dataset")
hdf5_file = h5py.File('/home/ee/btech/ee3150521/Stroboscopy/DatasetWithLabels.hdf5', "r")
train_dataset = hdf5_file["train_img"][()]
label_dataset = hdf5_file["labels"][()]
hdf5_file.close()
print ("Dataset Loaded")


# (x_train, y_train), (x_test, y_test) = cifar10.load_data()


def preprocess_fn(img):
    print("Inside Preprocess function")
    re_size = 128
    print (tf.shape(img))
    img = tf.to_float(tf.image.resize_images(images = img,size =  [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)) / 127.5 - 1
    return img

img_paths = glob.glob('./data/faces/*.jpg')

def trainable_variables(filters=None, combine_type='or'):
    t_var = tf.trainable_variables()
    if filters is None:
        return t_var
    else:
        return tensors_filter(t_var, filters, combine_type)


def tensors_filter(tensors, filters, combine_type='or'):
    assert isinstance(tensors, (list, tuple)), '`tensors` shoule be a list or tuple!'
    assert isinstance(filters, (str, list, tuple)), \
        '`filters` should be a string or a list(tuple) of strings!'
    assert combine_type == 'or' or combine_type == 'and', "`combine_type` should be 'or' or 'and'!"

    if isinstance(filters, str):
        filters = [filters]

    f_tens = []
    for ten in tensors:
        if combine_type == 'or':
            for filt in filters:
                if filt in ten.name:
                    f_tens.append(ten)
                    break
        elif combine_type == 'and':
            all_pass = True
            for filt in filters:
                if filt not in ten.name:
                    all_pass = False
                    break
            if all_pass:
                f_tens.append(ten)
    return f_tens


""" graphs """

 
generator = models.generator
discriminator = models.discriminator_wgan_gp

''' graph '''
# inputs
real = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
realP = preprocess_fn(real)
# realP = real
z = tf.placeholder(tf.float32, shape=[None, z_dim])
lab_placeholder = tf.placeholder(tf.float32,shape=[None,3])
# generate
fake = generator(z,lab_placeholder, reuse=False)

# dicriminate
r_logit = discriminator(realP,lab_placeholder, reuse=False)
f_logit = discriminator(fake,lab_placeholder)

# losses



def gradient_penalty(real,lab,fake, f):
    def interpolate(a, b):
        shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
        alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
        inter = a + alpha * (b - a)
        inter.set_shape(a.get_shape().as_list())
        return inter

    x = interpolate(realP, fake)
    pred = f(x,lab)
    gradients = tf.gradients(pred, x)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=range(1, x.shape.ndims)))
    gp = tf.reduce_mean((slopes - 1.)**2)
    return gp

wd = tf.reduce_mean(r_logit) - tf.reduce_mean(f_logit)
gp = gradient_penalty(realP,lab_placeholder,fake, discriminator)
d_loss = -wd + gp * 10.0
g_loss = -tf.reduce_mean(f_logit)
d_var = utils.trainable_variables('discriminator')
g_var = utils.trainable_variables('generator')
d_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_var)
g_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g_var)

f_sample = generator(z,lab_placeholder, training=False)


""" train """
''' init '''
# session
sess = tf.Session()
# iteration counter
it_cnt, update_cnt = utils.counter()
# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer

''' initialization '''
ckpt_dir = './checkpoints/checkpoints'
utils.mkdir(ckpt_dir + '/')
if not utils.load_checkpoint(ckpt_dir, sess):
    sess.run(tf.global_variables_initializer())

print ("Loaded Checkpoint #$$@$")
''' train '''
try:
    z_ipt_sample = np.random.normal(size=[100, z_dim])
    batch_epoch = 1847// (batch_size * n_critic)
    max_it = epoch * batch_epoch
    for it in range(sess.run(it_cnt), max_it):
        sess.run(update_cnt)

        # which epoch
        epoch = it // batch_epoch
        it_epoch = it % batch_epoch + 1

        # train D
        for i in range(n_critic):
            print ("Feeding Real placeholder")
            idx_ipt = np.random.randint(0, 880, batch_size)
            real_ipt= train_dataset[idx_ipt]
            lab_ipt = label_dataset[idx_ipt]
            print (np.shape(real_ipt))
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            # realT = sess.run([real],feed_dict={real:real_ipt})
            _ = sess.run([d_step], feed_dict={real: real_ipt, z: z_ipt,lab_placeholder:lab_ipt})
            print ("Feeded")
            break

        # train G
        z_ipt = np.random.normal(size=[batch_size, z_dim])
        _ = sess.run([g_step], feed_dict={z: z_ipt,lab_placeholder:lab_ipt})

        # display
        if it % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

        # save
        if (it + 1) % 100 == 0:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
            print('Model saved in file: % s' % save_path)

        # sample
        if (it + 1) % 50 == 0:
            lab_cyst_sample = np.ones((100,3))*np.asarray([1,0,0])
            lab_nod_sample = np.ones((100,3))*np.asarray([0,1,0])
            lab_polyp_sample = np.ones((100,3))*np.asarray(([0,0,1]))
            f_sample_opt_c = sess.run(f_sample, feed_dict={z: z_ipt_sample,lab_placeholder:lab_cyst_sample})
            f_sample_opt_n = sess.run(f_sample, feed_dict={z: z_ipt_sample,lab_placeholder:lab_nod_sample})
            f_sample_opt_p = sess.run(f_sample, feed_dict={z: z_ipt_sample,lab_placeholder:lab_polyp_sample})

            save_dir = './sample_images_while_training/images'
            utils.mkdir(save_dir + '/')
            utils.imwrite(utils.immerge(f_sample_opt_c, 10, 10), '%s/Epoch_(%d)_(%dof%d)_Cyst.jpg' % (save_dir, epoch, it_epoch, batch_epoch))
            utils.imwrite(utils.immerge(f_sample_opt_n, 10, 10), '%s/Epoch_(%d)_(%dof%d)_Nodule.jpg' % (save_dir, epoch, it_epoch, batch_epoch))
            utils.imwrite(utils.immerge(f_sample_opt_p, 10, 10), '%s/Epoch_(%d)_(%dof%d)_Polyp.jpg' % (save_dir, epoch, it_epoch, batch_epoch))

except Exception ,e:
    print ("Exception encountered")
    traceback.print_exc()
finally:
    print(" [*] Close main session!")
    sess.close()
