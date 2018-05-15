import tensorflow as tf
import numpy as np
import imageio as iio



#import image
im = iio.imread("image.jpg", as_gray=True)

#take real fft
f_im = np.fft.rfft2(im)

#generate random 50% mask
l = lambda : np.random.randint(2,size=f_im.shape)

#now it's a 25% mask
mask = l()*l()


l_f_im = mask*f_im

#take inv fft
l_im = np.fft.irfft2(l_f_im)

l_im = np.reshape(l_im,[l_im.shape[0],l_im.shape[1],1])

l_im = l_im.astype(np.float32)

#convert to tensorflow variable
loss_img_var = tf.Variable(l_im, trainable=True)

#use total variation loss
loss = tf.image.total_variation(loss_img_var)

sess = tf.Session()



#apply ProximalAdagradOptimizer with 0.01 learning rate and 0.1 l1 regularization.
#documentation: https://www.tensorflow.org/api_docs/python/tf/train/ProximalAdagradOptimizer
solver = tf.train.ProximalAdagradOptimizer(0.01, l1_regularization_strength=0.1).minimize(loss)

min = tf.Variable(tf.reduce_min(loss_img_var),trainable=False)

max = tf.Variable(tf.reduce_max(loss_img_var), trainable=False)

update_min = tf.assign(min, tf.reduce_min(loss_img_var))
update_max = tf.assign(max, tf.reduce_max(loss_img_var))
reduce_img = tf.assign(loss_img_var, 256*(loss_img_var-min)/(max-min))

sess.run(tf.global_variables_initializer())

i = 0
while i < 100000:
	# increase contrast
	sess.run(update_min)
	sess.run(update_max)
	sess.run(reduce_img)
	
	#export image
	if i % 5000 == 0:
		print('Step ' + str(i))
		u = sess.run(loss_img_var)
		iio.imwrite("img_" + str(i) + ".jpg",u)
	i = i + 1
	
	# compute gradients, apply them to loss_img_var
	sess.run(solver)
	
		 