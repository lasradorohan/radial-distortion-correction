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

def warp_distort(image, c, height, width):
  image = tf.cast(image, tf.float32)
  idx = idxarr(height, width)
  flow = idx-scalemapinv_vect(idx, c)
  # tf.assert_rank(image, 3)
  return tfa.image.dense_image_warp(image = image[None, ...], flow = flow[None, ...])[0]

def warp_undistort(image, c, height, width):
    image = tf.cast(image, tf.float32)
    idx = idxarr(height, width)
    flow = idx-scalemapfor_vect(idx, c)
    tf.assert_rank(image, 3)
    return tfa.image.dense_image_warp(image = image[None, ...], flow = flow[None, ...])[0]

def loaddat_unnorm(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels = 3, dtype = tf.uint8)
    return image

def distort():
    st.write("Input c value: ")
    c = st.slider('', -1.0, 1.0, value = 0.0)
    # with st.spinner("Distorting"):
    image = loaddat_unnorm('tmp'+os.sep+'distorter'+os.sep+'input.png')
    dist = warp_distort(image, c, image.shape[-3], image.shape[-2])
    dist = PIL.Image.fromarray(dist.numpy().astype(np.uint8))
    dist.save('tmp'+os.sep+'distorter'+os.sep+'output.png')
    st.write("Distorted:")
    st.image(dist, width = 700)

def main():
    st.title("Distort Images")
    if os.path.exists('tmp'+os.sep+'distorter'+os.sep+'input.png') :
        st.write("Added Image:")
        img = PIL.Image.open('tmp'+os.sep+'distorter'+os.sep+'input.png')
        st.image(img, width = 700)
        distort()
        if st.button('Reset'):
            try:
                os.remove('tmp'+os.sep+'distorter'+os.sep+'input.png')
                os.remove('tmp'+os.sep+'distorter'+os.sep+'output.png')
                st.rerun()
            except:
                pass
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
                    img.save('tmp'+os.sep+'distorter'+os.sep+'input.png')
                    st.image(img, width = 700)
                    distort()
            except:
                pass