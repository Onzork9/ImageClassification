from flask import Flask, redirect, url_for, render_template, Response
from flask import request, jsonify
import os
import time
import pandas as pd
import requests
import json
import random
import datetime
import numpy as np
from datetime import datetime


from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import glob


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D,
                          Dense,
                          LeakyReLU,
                          BatchNormalization, 
                          MaxPooling2D, 
                          Dropout,
                          Flatten)
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import PIL.Image
from datetime import datetime as dt



app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def index():   
    data=[]
    if len(request.files)>0:
        file1 = request.files['file1']
        img = Image.open(file1)
        data = loadPrediction(img)
        return render_template('index.html',data=data)
    
    return render_template('index.html',)


@app.route('/train', methods=['GET','POST'])
def train():  
    loadTrain()
    return  "New Data trained" 

@app.route('/resize', methods=['GET','POST'])
def resize():  
    Resize_img()
    return  "Image Resized done" 

dataset_list = ["Macro","Monti", "Pocill","Porit","Sand","Turf"]
#dataset_list = ["flower","fruit"]


def loadPrediction(image):

    img = image
    model = load_model("keras_model.h5")
    
    image=image.resize((100, 100))
    image=np.expand_dims(image,axis=0)
    image=np.array(image)
    # prediction=model.predict([image])[0]    
    # prediction = np.argmax(prediction)
    # ret = dataset_list[prediction]

    result = model.predict([image]) 
    rr = result[0]
    max = rr[np.argmax(rr)]       
    percentage = "{:.2f}".format(float(max)*100)

    ans = np.argmax(rr)
    ans = dataset_list[ans]
    
    return  {'img': img,'r': ans,'p':percentage} 


def Resize_img():
    root_dir = r'iimg/'

    for filename in glob.iglob(root_dir + '**/*.jpg', recursive=True):       
        im = Image.open(filename)
        imResize = im.resize((100,100), Image.ANTIALIAS)
        imResize.save(filename , 'JPEG', quality=90)
    
    return "Data image Resize" 

def loadTrain():
    root_dir = r'iimg/'

    # for filename in glob.iglob(root_dir + '**/*.jpg', recursive=True):       
    #     im = Image.open(filename)
    #     imResize = im.resize((100,100), Image.ANTIALIAS)
    #     imResize.save(filename , 'JPEG', quality=90)


    start = dt.now()

    # number of output classes (i.e. fruits)
    output_n = len(dataset_list)
    # image size to scale down to (original images are 100 x 100 px)
    img_width = 100
    img_height = 100
    target_size = (img_width, img_height)
    # image RGB channels number
    channels = 3
    # path to image folders
    path = root_dir
    train_image_files_path = path + "trin"
    valid_image_files_path = path + "val"

    ## input data augmentation/modification
    # training images
    train_data_gen = ImageDataGenerator(
    rescale = 1./255
    )
    # validation images
    valid_data_gen = ImageDataGenerator(
    rescale = 1./255
    )

    ## getting data
    # training images
    train_image_array_gen = train_data_gen.flow_from_directory(train_image_files_path,                                                            
                                                            target_size = target_size,
                                                            classes = dataset_list, 
                                                            class_mode = 'categorical',
                                                            seed = 42)

    # validation images
    valid_image_array_gen = valid_data_gen.flow_from_directory(valid_image_files_path, 
                                                            target_size = target_size,
                                                            classes = dataset_list,
                                                            class_mode = 'categorical',
                                                            seed = 42)

    ## model definition
    # number of training samples
    train_samples = train_image_array_gen.n
    # number of validation samples
    valid_samples = valid_image_array_gen.n
    # define batch size and number of epochs
    batch_size = 32
    epochs = 35

    # initialise model
    model = Sequential()

    # add layers
    # input layer
    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', input_shape = (img_width, img_height, channels), activation = 'relu'))
    # hiddel conv layer
    model.add(Conv2D(filters = 16, kernel_size = (3,3), padding = 'same'))
    model.add(LeakyReLU(.5))
    model.add(BatchNormalization())
    # using max pooling
    model.add(MaxPooling2D(pool_size = (2,2)))
    # randomly switch off 25% of the nodes per epoch step to avoid overfitting
    model.add(Dropout(.25))
    # flatten max filtered output into feature vector   
    model.add(Flatten())
    # output features onto a dense layer
    model.add(Dense(units = 100, activation = 'relu'))
    # randomly switch off 25% of the nodes per epoch step to avoid overfitting
    model.add(Dropout(.5))
    # output layer with the number of units equal to the number of categories
    model.add(Dense(units = output_n, activation = 'softmax'))

    # compile the model
    model.compile(loss = 'categorical_crossentropy', 
                metrics = ['accuracy'], 
                optimizer = RMSprop(lr = 1e-4, decay = 1e-6))

    # train the model
    model.fit_generator(
    # training data
    train_image_array_gen,

    # epochs
    steps_per_epoch = train_samples // batch_size, 
    epochs = epochs, 

    # validation data
    validation_data = valid_image_array_gen,
    validation_steps = valid_samples // batch_size,

    # print progress
    verbose = 2

    )

    _, acc = model.evaluate_generator(train_image_array_gen, steps=len(train_image_array_gen), verbose=0)
    print('> %.3f' % (acc * 100.0))
    model.save("keras_model.h5")

    return (acc * 100.0)




if __name__ == "__main__":    
    app.run()
    #app.run(host='192.168.43.46', port=8080)