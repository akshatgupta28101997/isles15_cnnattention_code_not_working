
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import cv2

#%%
a = []
k=256
path = "E:/btp/btp_dataset/final_images/siss - Copy/"
for i in range(1):
   print(i)
   im1 = cv2.imread(path + "Flair/"+ str(i) + ".jpg",0)
   im1 = cv2.resize(im1,(k,k))
   im1=im1/255
   im1 = np.expand_dims(im1,axis=2)

#
  #  im2 = cv2.imread(path + "DWI/"+ str(i) + ".jpg",0)
  #  im2 = cv2.resize(im2,(k,k))
  #  im2=im2/255
  #  im2 = np.expand_dims(im2,axis=2)
   
  #  im3 = cv2.imread(path + "T1/"+ str(i) + ".jpg",0)
  #  im3 = cv2.resize(im3,(k,k))
  #  im3=im3/255
  #  im3 = np.expand_dims(im3,axis=2)
  
  
  #  im4 = cv2.imread(path+"T2/"+ str(i) + ".jpg",0)
  #  im4 = cv2.resize(im4,(k,k))
  #  im4 = im4/255
  #  im4= np.expand_dims(im4,axis=2)
   
  #  ans = np.concatenate((im1,im2,im3,im4),axis=2)

   ans = np.expand_dims(im1,axis=0)
   
   
   print(ans.shape)
   
   #%%
for i in range(1,100,1):
   print(i)
   im1 = cv2.imread(path + "Flair/"+ str(i) + ".jpg",0)
   im1 = cv2.resize(im1,(k,k))
   im1=im1/255
   im1 = np.expand_dims(im1,axis=2)


  #  im2 = cv2.imread(path + "DWI/"+ str(i) + ".jpg",0)
  #  im2 = cv2.resize(im2,(k,k))
  #  im2=im2/255
  #  im2 = np.expand_dims(im2,axis=2)
   
  #  im3 = cv2.imread(path + "T1/"+ str(i) + ".jpg",0)
  #  im3 = cv2.resize(im3,(k,k))
  #  im3=im3/255
  #  im3 = np.expand_dims(im3,axis=2)
  
  
  #  im4 = cv2.imread(path+"T2/"+ str(i) + ".jpg",0)
  #  im4 = cv2.resize(im4,(k,k))
  #  im4 = im4/255
  #  im4= np.expand_dims(im4,axis=2)
   
   
   
   
   
  #  ans2 = np.concatenate((im1,im2,im3,im4),axis=2)
   ans2 = np.expand_dims(im1,axis=0)
  
   
   ans = np.concatenate((ans,ans2),axis=0)
   


   #%% preparing labels
   
   
   
   
for i in range(1):
    print(i)
    im1 = cv2.imread(path + "OT/"+str(i)+".jpg",0)
    
    im1 = cv2.resize(im1,(k,k))
    for i in range(k):
      for j in range(k):
        if(im1[i][j]>128):
          im1[i][j]=1
        else:
          im1[i][j]=0
          
   
    print(np.unique(im1))
    
    im1 = np.expand_dims(im1,axis=2)
    labels = np.expand_dims(im1,axis=0)
   
for i in range(1,100,1):
    print(i)
    im1 = cv2.imread(path + "OT/"+str(i)+".jpg",0)
    im1 = cv2.resize(im1,(k,k))
    for i in range(k):
      for j in range(k):
        if(im1[i][j]>128):
          im1[i][j]=1
        else:
          im1[i][j]=0
   
    im1 = np.expand_dims(im1,axis=0)
    im1 = np.expand_dims(im1,axis=3)
    labels = np.concatenate((labels,im1),axis=0)

   #%%
from keras.layers import Layer
from keras.layers import Conv2D
from keras.layers import concatenate

from keras import initializers
from keras import backend as K

import tensorflow as tf
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import cv2

def _conv_layer(filters, kernel_size, strides=(1, 1), padding='same', name=None):
    return Conv2D(filters, kernel_size, strides=strides, padding=padding,
                  use_bias=True, kernel_initializer='he_normal', name=name)


def _normalize_depth_vars(depth_k, depth_v, filters):
    """
    Accepts depth_k and depth_v as either floats or integers
    and normalizes them to integers.
    Args:
        depth_k: float or int.
        depth_v: float or int.
        filters: number of output filters.
    Returns:
        depth_k, depth_v as integers.
    """

    if type(depth_k) == float:
        depth_k = int(filters * depth_k)
    else:
        depth_k = int(depth_k)

    if type(depth_v) == float:
        depth_v = int(filters * depth_v)
    else:
        depth_v = int(depth_v)

    return depth_k, depth_v


class AttentionAugmentation2D(Layer):

  

    def build(self, input_shape):
        self._shape = input_shape

        # normalize the format of depth_v and depth_k
        self.depth_k, self.depth_v = _normalize_depth_vars(self.depth_k, self.depth_v,
                                                           input_shape)

        if self.axis == 1:
            _, channels, height, width = input_shape
        else:
            _, height, width, channels = input_shape

        if self.relative:
            dk_per_head = self.depth_k // self.num_heads

            if dk_per_head == 0:
                print('dk per head', dk_per_head)

            self.key_relative_w = self.add_weight('key_rel_w',
                                                  shape=[2 * width - 1, dk_per_head],
                                                  initializer=initializers.RandomNormal(
                                                      stddev=dk_per_head ** -0.5))

            self.key_relative_h = self.add_weight('key_rel_h',
                                                  shape=[2 * height - 1, dk_per_head],
                                                  initializer=initializers.RandomNormal(
                                                      stddev=dk_per_head ** -0.5))

        else:
            self.key_relative_w = None
            self.key_relative_h = None

    def call(self, inputs, **kwargs):
        if self.axis == 1:
            # If channels first, force it to be channels last for these ops
            inputs = K.permute_dimensions(inputs, [0, 2, 3, 1])

        q, k, v = tf.split(inputs, [self.depth_k, self.depth_k, self.depth_v], axis=-1)

        q = self.split_heads_2d(q)
        k = self.split_heads_2d(k)
        v = self.split_heads_2d(v)

        # scale query
        depth_k_heads = self.depth_k / self.num_heads
        q *= (depth_k_heads ** -0.5)

        # [Batch, num_heads, height * width, depth_k or depth_v] if axis == -1
        qk_shape = [self._batch, self.num_heads, self._height * self._width, self.depth_k // self.num_heads]
        v_shape = [self._batch, self.num_heads, self._height * self._width, self.depth_v // self.num_heads]
        flat_q = K.reshape(q, K.stack(qk_shape))
        flat_k = K.reshape(k, K.stack(qk_shape))
        flat_v = K.reshape(v, K.stack(v_shape))

        # [Batch, num_heads, HW, HW]
        logits = tf.matmul(flat_q, flat_k, transpose_b=True)

        # Apply relative encodings
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits

        weights = K.softmax(logits, axis=-1)
        attn_out = tf.matmul(weights, flat_v)

        attn_out_shape = [self._batch, self.num_heads, self._height, self._width, self.depth_v // self.num_heads]
        attn_out_shape = K.stack(attn_out_shape)
        attn_out = K.reshape(attn_out, attn_out_shape)
        attn_out = self.combine_heads_2d(attn_out)
        # [batch, height, width, depth_v]

        if self.axis == 1:
            # return to [batch, depth_v, height, width] for channels first
            attn_out = K.permute_dimensions(attn_out, [0, 3, 1, 2])

        return attn_out

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = self.depth_v
        return tuple(output_shape)

    def split_heads_2d(self, ip):
        tensor_shape = K.shape(ip)

        # batch, height, width, channels for axis = -1
        tensor_shape = [tensor_shape[i] for i in range(len(self._shape))]

        batch = tensor_shape[0]
        height = tensor_shape[1]
        width = tensor_shape[2]
        channels = tensor_shape[3]

        # Save the spatial tensor dimensions
        self._batch = batch
        self._height = height
        self._width = width

        ret_shape = K.stack([batch, height, width,  self.num_heads, channels // self.num_heads])
        split = K.reshape(ip, ret_shape)
        transpose_axes = (0, 3, 1, 2, 4)
        split = K.permute_dimensions(split, transpose_axes)

        return split

    def relative_logits(self, q):
        shape = K.shape(q)
        # [batch, num_heads, H, W, depth_v]
        shape = [shape[i] for i in range(5)]

        height = shape[2]
        width = shape[3]

        rel_logits_w = self.relative_logits_1d(q, self.key_relative_w, height, width,
                                               transpose_mask=[0, 1, 2, 4, 3, 5])

        rel_logits_h = self.relative_logits_1d(
            K.permute_dimensions(q, [0, 1, 3, 2, 4]),
            self.key_relative_h, width, height,
            transpose_mask=[0, 1, 4, 2, 5, 3])

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, transpose_mask):
        rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = K.reshape(rel_logits, [-1, self.num_heads * H, W, 2 * W - 1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = K.reshape(rel_logits, [-1, self.num_heads, H, W, W])
        rel_logits = K.expand_dims(rel_logits, axis=3)
        rel_logits = K.tile(rel_logits, [1, 1, 1, H, 1, 1])
        rel_logits = K.permute_dimensions(rel_logits, transpose_mask)
        rel_logits = K.reshape(rel_logits, [-1, self.num_heads, H * W, H * W])
        return rel_logits

    def rel_to_abs(self, x):
        shape = K.shape(x)
        shape = [shape[i] for i in range(3)]
        B, Nh, L, = shape
        col_pad = K.zeros(K.stack([B, Nh, L, 1]))
        x = K.concatenate([x, col_pad], axis=3)
        flat_x = K.reshape(x, [B, Nh, L * 2 * L])
        flat_pad = K.zeros(K.stack([B, Nh, L - 1]))
        flat_x_padded = K.concatenate([flat_x, flat_pad], axis=2)
        final_x = K.reshape(flat_x_padded, [B, Nh, L + 1, 2 * L - 1])
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

    def combine_heads_2d(self, inputs):
        # [batch, num_heads, height, width, depth_v // num_heads]
        transposed = K.permute_dimensions(inputs, [0, 2, 3, 1, 4])
        # [batch, height, width, num_heads, depth_v // num_heads]
        shape = K.shape(transposed)
        shape = [shape[i] for i in range(5)]

        a, b = shape[-2:]
        ret_shape = K.stack(shape[:-2] + [a * b])
        # [batch, height, width, depth_v]
        return K.reshape(transposed, ret_shape)

    def get_config(self):
        config = {
            'depth_k': self.depth_k,
            'depth_v': self.depth_v,
            'num_heads': self.num_heads,
            'relative': self.relative,
        }
        base_config = super(AttentionAugmentation2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




def augmented_conv2d(ip, filters, kernel_size=(3, 3), strides=(1, 1),
                     depth_k=0.2, depth_v=0.2, num_heads=8, relative_encodings=True):
    """
    Builds an Attention Augmented Convolution block.
    Args:
        ip: keras tensor.
        filters: number of output filters.
        kernel_size: convolution kernel size.
        strides: strides of the convolution.
        depth_k: float or int. Number of filters for k.
            Computes the number of filters for `v`.
            If passed as float, computed as `filters * depth_k`.
        depth_v: float or int. Number of filters for v.
            Computes the number of filters for `k`.
            If passed as float, computed as `filters * depth_v`.
        num_heads: int. Number of attention heads.
            Must be set such that `depth_k // num_heads` is > 0.
        relative_encodings: bool. Whether to use relative
            encodings or not.
    Returns:
        a keras tensor.
    """
    # input_shape = K.int_shape(ip)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    depth_k, depth_v = _normalize_depth_vars(depth_k, depth_v, filters)

    conv_out = _conv_layer(filters - depth_v, kernel_size, strides)(ip)

    # Augmented Attention Block
    qkv_conv = _conv_layer(2 * depth_k + depth_v, (1, 1), strides)(ip)
    attn_out = AttentionAugmentation2D(depth_k, depth_v, num_heads, relative_encodings)(qkv_conv)
    attn_out = _conv_layer(depth_v, kernel_size=(1, 1))(attn_out)

    output = concatenate([conv_out, attn_out], axis=channel_axis)
    return output
#%%
input_size = (256,256,1)
inputs = Input(input_size) #returns a tensor
conv1 = Convolution2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
#conv1 = augmented_conv2d(inputs,  filters=32, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#conv1 = LeakyReLU(alpha=0.2)(conv1)
#conv1 = BatchNormalization()(conv1)
#conv1 =augmented_conv2d( conv1, filters=32, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#conv1 = LeakyReLU(alpha=0.2)(conv1)
#conv1 = BatchNormalization()(conv1)
#pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#conv2 = augmented_conv2d( pool1, filters=64, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#conv2= LeakyReLU(alpha=0.2)(conv2)
#conv2 = BatchNormalization()(conv2)
#conv2 = augmented_conv2d(conv2,  filters=64, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#conv2 = LeakyReLU(alpha=0.2)(conv2)
#conv2 = BatchNormalization()(conv2)
#pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#conv3 = augmented_conv2d(pool2,  filters=128, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#conv3 = LeakyReLU(alpha=0.2)(conv3)
#conv3 = BatchNormalization()(conv3)
#conv3 = augmented_conv2d(conv3,  filters=128, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#conv3 = LeakyReLU(alpha=0.2)(conv3)
#conv3 = BatchNormalization()(conv3)
#pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#conv4 = augmented_conv2d(pool3,  filters=256, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#conv4 = LeakyReLU(alpha=0.2)(conv4)
#conv4 = BatchNormalization()(conv4)
#conv4 = augmented_conv2d(conv4,  filters=256, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#conv4 = BatchNormalization()(conv4)
#drop4 = Dropout(0.5)(conv4)
#pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
##    
#conv5 = augmented_conv2d( pool4, filters=512, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#conv5 = LeakyReLU(alpha=0.2)(conv5)
##    conv5 = BatchNormalization()(conv5)
#conv5 = augmented_conv2d(conv5,  filters=512, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#conv5 = LeakyReLU(alpha=0.2)(conv5)
##    conv5 = BatchNormalization()(conv5)
#drop5 = Dropout(0.5)(conv5)
##    
#up6 = augmented_conv2d(UpSampling2D(size = (2,2))(drop5),  filters=256, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#up6 = LeakyReLU(alpha=0.2)(up6)
#merge6 = concatenate([drop4,up6],axis=3)
#conv6 = augmented_conv2d( merge6, filters=256, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#conv6 = LeakyReLU(alpha=0.2)(conv6)
##    conv6 = BatchNormalization()(conv6)
#conv6 = augmented_conv2d(conv6,  filters=256, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#conv6 = LeakyReLU(alpha=0.2)(conv6)
##    conv6 = BatchNormalization()(conv6)
##    
#up7 = augmented_conv2d(UpSampling2D(size = (2,2))(conv6),  filters=128, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#up7 = LeakyReLU(alpha=0.2)(up7)
#merge7 = concatenate([conv3,up7], axis = 3)
#conv7 = augmented_conv2d(merge7,  filters=128, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#conv7 = LeakyReLU(alpha=0.2)(conv7)
##    conv7 = BatchNormalization()(conv7)
#conv7 = augmented_conv2d(conv7,  filters=128, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#conv7 = LeakyReLU(alpha=0.2)(conv7)
##    conv7 = BatchNormalization()(conv7)
##    
#up8 = augmented_conv2d(UpSampling2D(size = (2,2))(conv7),  filters=64, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
##conv5 = LeakyReLU(alpha=0.2)(up8)
#up8 = LeakyReLU(alpha=0.2)(up8)
#merge8 = concatenate([conv2,up8], axis = 3)
#conv8 = augmented_conv2d( merge8, filters=64, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
##conv8 = BatchNormalization()(conv8)
#conv8 = augmented_conv2d(conv8,  filters=64, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#conv8 = LeakyReLU(alpha=0.2)(conv8)
##    #conv8 = BatchNormalization()(conv8)
##    
#up9 = augmented_conv2d( UpSampling2D(size = (2,2))(conv8), filters=32, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#up9 = LeakyReLU(alpha=0.2)(up9)
#merge9 = concatenate([conv1,up9],axis = 3)
#conv9 = augmented_conv2d(merge9 , filters=32, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#conv9 = LeakyReLU(alpha=0.2)(conv9)
##    conv9 = BatchNormalization()(conv9)
#conv9 = augmented_conv2d(conv9,  filters=32, kernel_size=(3, 3), depth_k=8, depth_v=8, num_heads=4, relative_encodings=True)
#conv9 = LeakyReLU(alpha=0.2)(conv9)
###    #conv9 = BatchNormalization()(conv9)
#
#
#conv9 = augmented_conv2d(conv9, filters=2, kernel_size=(3, 3), depth_k=1, depth_v=1, num_heads=1, relative_encodings=True)
#conv9 = LeakyReLU(alpha=0.2)(conv9)
###    #conv9 = BatchNormalization()(conv9)
#conv10 = augmented_conv2d(conv9,  filters=1, kernel_size=(3, 3), depth_k=1, depth_v=1, num_heads=1, relative_encodings=True)
#conv10  = Activation('sigmoid')(conv10)
model = Model(input = inputs, output = conv1)

   
print(model.summary())
#%%
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

#%%

callbacks_list= [EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)]
#%%
hist = model.fit(ans[:600],labels[:600],epochs=30,batch_size=2,callbacks=callbacks_list,validation_data=[ans2[600:],labels2[600:]],shuffle=True)   
