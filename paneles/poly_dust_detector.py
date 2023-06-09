# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 12:49:06 2023

@author: crat2

Its important to check two things before using this code:
    1. We trained the model using RGB images, so the input image must be in RGB.
    2. We trained the model using 256 x 256 patches, the input image must have
       a dimension being multiple of 256 (like 512 x 512, 1024 x 512, 1792 x 1024). 
       We trained using 1792 x 1024 images, so we added a resizing step with this
       dimension, but it can be edited for other 256 multiple.
       
How to use this code?
1. Define the path of the pre-trained weights once and the image path as needed.
2. Run once the LOAD_MODEL function from keras.
3. Run as needed the UNET_PREDICTION function.

Modifications for robotic system:
@author: blackzafiro
"""

#LIBRARIES
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
import cv2

import argparse
from termcolor import cprint
MC = 'yellow'

image_size = (1792, 1024)


#LOAD_IMAGE FUNCTION
def load_image(path, import_type='bgr_img'):
    """
    Loads image from specified file path.
    """
    img=cv2.imread(path)
    res_img = cv2.resize(img, image_size, interpolation = cv2.INTER_LINEAR)
    if import_type == 'gray_img':
        load_img=cv2.cvtColor(res_img,cv2.COLOR_BGR2GRAY)
    elif import_type == 'bgr_img':
        load_img=res_img
    elif import_type == 'rgb_img':
        load_img=cv2.cvtColor(res_img,cv2.COLOR_BGR2RGB)
    else:
        load_img=False
        print("Incorrect import type.")
    return load_img
    
#READ_IMAGE FUNCTION
def read_image(url):
    """
    Reads an image from the video stream
    url: of the video stream
    returns an rgb image
    """
    cap = cv2.VideoCapture(url)
    ret, img = cap.read()
    if not ret:
        cprint("Failed to capture frame from stream", MC)
        return # break if no next frame
    cv2.waitKey(0)
    img = cv2.resize(img, image_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cap.release()
    return img
        
#UNET_PREDICTION FUNCTION
def unet_prediction(image, model=None):
    """
    Receives an RGB image and returns a 1D numpy array of ints where
    0 - Panel
    1 - Background
    2 - Dust
    """
    patch_size = 256
    unet_model = model

    #Patches preprocessing
    cprint("Creating patches...", MC)
    small_patches = []
    patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size) ##perform patchify
    for i in range(patches_img.shape[0]): 
        for j in range(patches_img.shape[1]): 
            single_patch_img = patches_img[i,j,:,:]
            small_patches.append(single_patch_img)
    small_patches =  np.array(small_patches)
    
    #Unet4 SM preprocessing
    cprint("Preprocessing with resnet50...", MC)
    BACKBONE = 'resnet50' #define the backbone
    preprocess_input = sm.get_preprocessing(BACKBONE)
    norm_img = preprocess_input(small_patches)
    norm_img =  np.array(norm_img)
    norm_img = np.squeeze(norm_img, 1)
    
    #Prediction
    cprint("Predicting...", MC)
    y_pred_unet = unet_model.predict(norm_img) #make the prediction
    prediction_unet = np.argmax(y_pred_unet, axis=3)[:,:,:] #from prob to int
    
    #Reconstructed image
    cprint("Reconstructing image...", MC)
    patched_prediction = np.reshape(prediction_unet, [patches_img.shape[0], patches_img.shape[1], 
                                                      patches_img.shape[3], patches_img.shape[4]])
    reconstructed_image = unpatchify(patched_prediction, (image.shape[0], image.shape[1]))
    
    return reconstructed_image
    
#OUTPUT: Plotting original and GT and prediction
def show_images(original_image, dust_image):
    """
    Displays the original image and image with detected regions.
    """
    cprint("Plotting image...", MC)
    scale = 5
    new_size = (image_size[0]//scale, image_size[1]//scale)
    original = cv2.resize(original_image.astype(np.uint8), new_size)
    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    
    reconstructed = cv2.resize(dust_image.astype(np.uint8), new_size) * 100
    reconstructed = cv2.merge((reconstructed,reconstructed,reconstructed))
    twins = np.concatenate((original, reconstructed), axis=1)
    
    ## Funciona en Ubuntu, debe funcionar también en Windows
    cprint('Max value: ' + str(np.max(dust_image)), MC)
    cprint("Press any key on the image to exit", MC)
    cv2.namedWindow("Image vs Unet prediction", cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Image vs Unet prediction', twins)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    ## Sólo funciona en Windows
    #import matplotlib as mpl
    #mpl.rcParams['figure.dpi'] = 330
    
    #plt.figure(figsize=(10, 6))
    #plt.subplot(231)
    #plt.title('Image')
    #plt.imshow(original_image, cmap='gray')
    #plt.subplot(232)
    #plt.title('Unet prediction')
    #plt.imshow(dust_image, cmap='gray')
    #plt.show()



if __name__ == '__main__':
    ##### EXCECUTION MODE
    parser = argparse.ArgumentParser()
    parser.add_argument("-ip", help="ip of the images stream")
    args = parser.parse_args()
    
    ##### PATHS
    path_models="sm_unet4_03.hdf5" #path for hdf5 file with trained weights
    if args.ip:
        cprint("Reading image from stream...", MC)
        url = f"http://{args.ip}:8000/stream.mjpg"
        img = read_image(url)
    else:
        cprint("Reading image from file...", MC)
        path_image = "IMG_20221012_132227_DRO.png" #path for image
        img = load_image(path_image, import_type='rgb_img') #load the image

    #### LOAD KERAS
    cprint("Loading keras...", MC)
    import segmentation_models as sm
    from keras.models import load_model
    
    ##### LOAD THE MODEL (just run once because it takes too long due to load the pre-trained weights)
    cprint("Loading model...", MC)
    unet_model = load_model(path_models, compile=False)

    ##### UNET_PREDICTION (run as needed, edit the path_image to predict in different images)  
    #cprint(type(img), MC)
    img_dust = unet_prediction(img, model=unet_model)
    show_images(img, img_dust)

