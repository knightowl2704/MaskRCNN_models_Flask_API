

# Importing flask module in the project is mandatory 
# An object of Flask class is our WSGI application. 
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import numpy as np 
import pandas as pd
from flask import jsonify
from flask import render_template
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from util import base64_to_pil, pil2datauri

import os
import sys

from PIL import Image
import io
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from linkedin import lkn
# import pickle
# Flask constructor takes the name of  
# current module (__name__) as argument. 
app = Flask(__name__)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


DEVICE = "/gpu:0"
# The route() function of the Flask class is a decorator,  
# which tells the application which URL should call  
# the associated function. 
config = lkn.LinkedinBar()


MODEL_PATH = './models/mask_rcnn_lkdbar_0030.h5'
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    ALLOW_GROWTH = True
    PER_PROCESS_GPU_MEMORY_FRACTION = 0.9


config = InferenceConfig()
# config.display()
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_PATH,config = config)

model.load_weights(MODEL_PATH, by_name=True)
model.keras_model._make_predict_function()
print('Loading Weights')

# model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

def model_predict(img, model):
    # img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)
    # If has an alpha channel, remove it for consistency
    if x.shape[-1] == 4:
        x = x[..., :3]

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!

    preds = model.detect([x], verbose = 1)
    r = preds[0]

    return r['rois']


@app.route('/predict',methods=['GET', 'POST']) 
def predict():
    if request.method == 'POST':
        q = dict(request.files)
        img = q["image"]
        img = Image.open(io.BytesIO(img.read()))
        # img = Image.open(io.BytesIO(img))
        # img = base64_to_pil(request.json)
        preds = model_predict(img,model)
        if preds.size>0:
            print(preds)
            responsejson = {
            'x1': int(preds[0][0]),
            'x2': int(preds[0][1]),
            'y1': int(preds[0][2]),
            'y2': int(preds[0][3]),
            }
            json_resp = jsonify(responsejson)
            json_resp.status_code = 200
            print(json_resp)
            return json_resp
        else:
            json_resp = {
                'x1':'None',
                'x2':'None',
                'y1':'None',
                'y2':'None',
            }
            json_resp = jsonify(json_resp)
            json_resp.status_code = 200
            return json_resp
# main driver function 
if __name__ == '__main__': 
  
    # run() method of Flask class runs the application  
    # on the lo
    app.run()