import streamlit as st
import os
import io
import PIL
import time
import shutil
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import glob


def get_radial_mat(n, m, normalized = False):
        radial_mat = np.fromfunction(lambda i,j:(i-(n-1)/2)**2+(j-(n-1)/2)**2, (n, m))
        if not normalized:
            return radial_mat
        else:
            radial_mat_norm = radial_mat.copy()
            return (radial_mat_norm-np.mean(radial_mat_norm))/np.std(radial_mat_norm)
def make_model_large2():
    pre_trained_model = tf.keras.applications.Xception(include_top = False, input_shape = (512, 512, 3), weights = 'imagenet')
    for layer in pre_trained_model.layers:
        layer.trainable = False
    last_layer = pre_trained_model.get_layer('block13_sepconv2_bn')
    last_output = last_layer.output
    radial_mat = get_radial_mat(32, 32, normalized = False)
    x = tf.keras.layers.Lambda(lambda x: tf.multiply(x, radial_mat[None, ..., None]))(last_output)
    x = tf.keras.layers.Conv2D(512, (3, 3), strides = 2, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), strides = 2, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides = 2, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation = 'relu')(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(pre_trained_model.input, x)

@st.cache(show_spinner = False)
def load_upload_model():
    # shutil.unpack_archive("bin/savedmodel.tar.gz", "bin/")
    # model = tf.keras.models.load_model("bin/modelsave")
    try:
        del model
    except:
        pass
    weights = glob.glob("bin/weights/*.h5")
    if len(weights) == 0:
        st.error("Please add model weights to bin/weights/")
    else:
        tf.keras.backend.clear_session()
        model = make_model_large2()
        model.load_weights(weights[0])
        return model



def idxarr(sx,sy):
    yidx, xidx = tf.meshgrid(tf.range(0,sy), tf.range(0,sx))
    idx = tf.stack([xidx,yidx], axis = -1)
    idx = tf.cast(idx, tf.float32)
    return idx
  
def scalemapinv_vect(idx, c):
    height = tf.shape(idx)[0]
    width = tf.shape(idx)[1]
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)

    mapidx_xnorm = idx[..., 0]/height-0.5
    mapidx_ynorm = idx[..., 1]/width-0.5

    scale = 1+c/2
    r2 = tf.square(mapidx_xnorm)+tf.square(mapidx_ynorm)
    mcr2 = scale-c*r2

    mapidx_xscale = tf.divide(mapidx_xnorm, mcr2)
    mapidx_yscale = tf.divide(mapidx_ynorm, mcr2)

    mapidx_x = (mapidx_xscale+0.5)*height
    mapidx_y = (mapidx_yscale+0.5)*width
    mapidx = tf.stack([mapidx_x, mapidx_y], axis = -1)
    return mapidx

def scalemapfor_vect(idx, c):
    height = tf.shape(idx)[0]
    width = tf.shape(idx)[1]
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)

    mapidx_xnorm = idx[..., 0]/height-0.5
    mapidx_ynorm = idx[..., 1]/width-0.5

    scale = 1-c/2
    r2 = tf.square(mapidx_xnorm)+tf.square(mapidx_ynorm)
    rcm2 = 2/(tf.sqrt(scale**2+4*c*r2)+scale)

    mapidx_xscale = tf.multiply(mapidx_xnorm, rcm2)
    mapidx_yscale = tf.multiply(mapidx_ynorm, rcm2)

    mapidx_x = (mapidx_xscale+0.5)*height
    mapidx_y = (mapidx_yscale+0.5)*width
    mapidx = tf.stack([mapidx_x, mapidx_y], axis = -1)
    return mapidx

def warp_undistort(image, c, height, width):
    image = tf.cast(image, tf.float32)
    idx = idxarr(height, width)
    flow = idx-scalemapfor_vect(idx, c)
    tf.assert_rank(image, 3)
    return tfa.image.dense_image_warp(image = image[None, ...], flow = flow[None, ...])[0]

def loaddat_norm(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels = 3, dtype = tf.uint8)
    image = tf.image.resize_with_pad(image, 512, 512)
    image = tf.reshape(image, [512,512,3])
    image = tf.cast(image, tf.float32)
    image = image/127.5 - 1
    return image

def loaddat_unnorm(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels = 3, dtype = tf.uint8)
    return image

def det_correct(model):
    prog = st.progress(0.0)
    with st.spinner("Detecting"):
        image = loaddat_norm('tmp'+os.sep+'autodetector'+os.sep+'input.png')
        prog.progress(1/5)
        c = np.squeeze(model(image[None, ...], training = False).numpy())
        prog.progress(2/5)
    st.write("Found Distortion Coefficient: " + str(c))
    with st.spinner("Undistorting"):
        image  = loaddat_unnorm('tmp'+os.sep+'autodetector'+os.sep+'input.png')
        prog.progress(3/5)
        undist = warp_undistort(image, c, image.shape[-3], image.shape[-2])
        prog.progress(4/5)
        undist = PIL.Image.fromarray(undist.numpy().astype(np.uint8))
        prog.progress(1.0)
        undist.save('tmp'+os.sep+'autodetector'+os.sep+'output.png')
        st.write("Undistorted:")
        st.image(undist, width = 700)

def main():
    st.title("Radial Distortion Correction")
    with st.spinner("Loading Model, Please wait..."):
        model = load_upload_model()

    if os.path.exists('tmp'+os.sep+'autodetector'+os.sep+'input.png') :
        st.write("Added Image:")
        img = PIL.Image.open('tmp'+os.sep+'autodetector'+os.sep+'input.png')
        st.image(img, width = 700)
        if st.button('Reset'):
            try:
                os.remove('tmp'+os.sep+'autodetector'+os.sep+'input.png')
                st.rerun()
            except:
                pass
        if st.button('Detect and Correct'):
            det_correct(model)
    else:
        try:
            if not img:
                img = None
        except:
            img = None

        if not img:
            try:
                del data
            except:
                pass
            data = st.file_uploader("Add image")
            try:
                if data:
                    bts = io.BytesIO(data.read())
                    img = PIL.Image.open(bts)
                    img.save('tmp'+os.sep+'autodetector'+os.sep+'input.png')
                    st.image(img, width = 700)
                    if st.button('Detect and Correct'):
                        det_correct(model)
            except:
                pass