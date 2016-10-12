from menpofit.dpm import FeaturePyramid
from ibugnet import resnet_model
from ibugnet.utils import caffe_preprocess, rescale_image
import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

class ResnetFeaturePyramid(FeaturePyramid):
    def __init__(self, vgg=None):
        self.feature_size = 69
        self.spacial_scale = 1
        self.shift = None
        self.CHECKPOINT_PATH = '/vol/atlas/homes/gt108/pretrained_models/keypoints/69_classes'

        self.sess = tf.Session()
        with tf.variable_scope('net'):
            self.menpo_image = tf.placeholder(tf.float32, shape=(3, None, None))
            with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=False):
                image = tf.expand_dims(tf.transpose(self.menpo_image, [1, 2, 0]), 0)
                prediction, _ = resnet_model.multiscale_kpts_net(image, scales=(1, 2, 4))
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        model_path = slim.evaluation.tf_saver.get_checkpoint_state(self.CHECKPOINT_PATH).model_checkpoint_path
        saver.restore(self.sess, model_path)
        self.softmax = tf.nn.softmax(prediction)


    def extract_feature(self, img):
        original_shape = img.shape
        if img.n_channels == 1:
            img.pixels = np.vstack([img.pixels]*3)
        import time
        start = time.time()
        img = rescale_image(img)
        img_pixels = caffe_preprocess(img)
        stop = time.time()
        print('preprocess take:', stop-start)
        feature = tf.image.resize_images(self.softmax, (tf.to_int32(original_shape[0]), tf.to_int32(original_shape[1])))
        transpose_feature = tf.transpose(tf.squeeze(feature, [0]), [2, 0, 1])
        result = self.sess.run(transpose_feature, feed_dict={self.menpo_image: img_pixels})
        start = time.time()
        print('extracting feature take:', start-stop)
        return result

    def extract_pyramid(self, img, interval=0, pyramid_pad=(0, 0)):
        spacial_scale = self.spacial_scale
        shift = self.shift
        fea = self.extract_feature(img)
        import time
        start = time.time()
        fea = np.pad(fea, ((0, 0), (int(pyramid_pad[1]), int(pyramid_pad[1])),
                                                 (int(pyramid_pad[0]), int(pyramid_pad[0]))), 'constant')
        stop = time.time()
        print('padding take:', stop-start)
        return {0:fea}, {0:self.spacial_scale}, self.shift