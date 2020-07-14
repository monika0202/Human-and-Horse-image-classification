#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 18:20:12 2020

@author: dell
"""

import streamlit as st
import tensorflow as tf
import os
from keras.preprocessing import image
from keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image,ImageEnhance 
import pandas as pd
import numpy as np
import cv2
import io
from tensorflow.keras import Model
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img


#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg


def main():
    
    train_horse_dir = os.path.join('horse-or-human/horses')
    train_human_dir = os.path.join('horse-or-human/humans')
    
  
        
        
    def display_images():
        train_horse_names = os.listdir(train_horse_dir)
        train_human_names = os.listdir(train_human_dir)
        
        nrows = 4
        ncols = 4

# Index for iterating over images
        pic_index = 0

        fig = plt.gcf()
        fig.set_size_inches(ncols * 4, nrows * 4)

        pic_index += 8
        next_horse_pix = [os.path.join(train_horse_dir, fname) 
                       for fname in train_horse_names[pic_index-8:pic_index]]
        next_human_pix = [os.path.join(train_human_dir, fname) 
                       for fname in train_human_names[pic_index-8:pic_index]]

        for i, img_path in enumerate(next_horse_pix+next_human_pix):
            sp = plt.subplot(nrows, ncols, i + 1)
            sp.axis('Off') # Don't show axes (or gridlines)

            img = mpimg.imread(img_path)
            st.image(img)

        
        
    
    
    def fit():
        model,_=model_data()
        
        model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001), metrics=['accuracy','mean_absolute_error', 'mean_squared_error'])
        train_datagen = ImageDataGenerator(rescale=1/255)
#validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
        train_generator = train_datagen.flow_from_directory(
        '/home/dell/tensor_flow/project/horse-or-human',  # This is the source directory for training images
       target_size=(300, 300) # All images will be resized to 300x300
      
        # Since we use binary_crossentropy loss, we need binary labels
       ,class_mode='binary')

        history = model.fit(train_generator,
          epochs=8)   
        return history
    
    def model_data():
        model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300,3)),
        tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
        tf.keras.layers.Dense(1, activation='sigmoid')
         ])
        return model,model.summary()
    
    
    def get_model_summary(model):
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()
        stream.close()
        return summary_string
    
    def visualize():
        model,_=model_data()
        train_horse_names = os.listdir(train_horse_dir)
        train_human_names = os.listdir(train_human_dir)
        successive_outputs = [layer.output for layer in model.layers[1:]]
        visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
# Let's prepare a random input image from the training set.
        horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
        human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
        img_path = random.choice(horse_img_files + human_img_files)

        img = load_img(img_path, target_size=(300, 300))  # this is a PIL image
        x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
        x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
        x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
        successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
        layer_names = [layer.name for layer in model.layers[1:]]

# Now let's display our representations
        for layer_name, feature_map in zip(layer_names, successive_feature_maps):
            if len(feature_map.shape) == 4:
                n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
                size = feature_map.shape[1]
    # We will tile our images in this matrix
                display_grid = np.zeros((size, size * n_features))
                for i in range(n_features):
                    x = feature_map[0, :, :, i]
                    x -= x.mean()
                    #x/= x.std()
                    x *= 64
                    x += 128
                    x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
                    display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
                scale = 20. / n_features
                
                plt.figure(figsize=(scale * n_features, scale*10))
                
                plt.title("\n\n"+layer_name,fontsize=30)
                plt.grid(False)
                st.write('\n\n')
                plt.imshow(display_grid, aspect='auto', cmap='viridis')
                st.write('\n')
                st.pyplot()

    
    st.title("Detection ")
    #st.sidebar("horse and human")
    
    activities=['detection','prediction']
    
    choice= st.sidebar.selectbox("select activity",activities)
    
    if choice=='detection':
        st.subheader("horse and human detection")
        image=st.file_uploader('Upload image',type=['jpg','png','jpeg'])
        
        if image is not None:
            our_image=Image.open(image)
            st.text('image')
            st.image(our_image)
        
        enhance_type = st.sidebar.radio("Enhance Type",['original','Gray-Scale','Contrast','Brightness','Blurring'])
        if  enhance_type == 'Gray-Scale':
            new_img=np.array(our_image.convert('RGB'))
            img=cv2.cvtColor(new_img,1)
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            st.image(gray)
            
        if  enhance_type == 'Contrast':
            c_rate=st.sidebar.slider("Contrast",0.5,3.5)
            enhancer=ImageEnhance.Contrast(our_image)
             
            img_output=enhancer.enhance(c_rate)
            st.image(img_output)   
            
        if  enhance_type == 'Brightness':
            c_rate=st.sidebar.slider("Brightness",0.5,3.5)
            enhancer=ImageEnhance.Brightness(our_image)
             
            img_output=enhancer.enhance(c_rate)
            st.image(img_output)       
            
        
        if  enhance_type == 'Blurring':
            new_img=np.array(our_image.convert('RGB'))
            blu_rate=st.sidebar.slider("Blurring",0.5,3.5)
            img=cv2.cvtColor(new_img,1)
            gray=cv2.GaussianBlur(img,(11,11),blu_rate)
            st.image(gray)
            
        #else:
         #   st.image(our_image,width=300)
            
        
        if st.button("detect"):
            new=np.float32(our_image)[:,:,:3]
            x=np.array(new)
            x=np.expand_dims(x,axis=0)
            
            images=np.vstack([x])
            model, _ =model_data()
            h=fit()
            classes=model.predict(images)
            print(classes[0])
            if classes[0]>0.5:
                st.subheader("image is a human" )
                print("image is a human")
                
            else:
                st.subheader("image is horse")
                print("is horse")
            
    elif choice == 'prediction':
        st.subheader('prediction')
        
        st.write("Total training horse images: ",len(os.listdir(train_horse_dir)))
        st.write("Total training human images: ",len(os.listdir(train_human_dir)))
        
        if st.sidebar.checkbox("show images"):
            display_images()
            
        if st.sidebar.button("Bulid model"):
            model,x=model_data()
            s = io.StringIO()
            model.summary(print_fn=lambda x: s.write('\t'+x + '\n\n'))
            model_summary = s.getvalue()
            s.close()

            print("The model summary is:\n\n{}".format(model_summary))
            #model,x=model_data()
           # model_summary_string = get_model_summary(model)
            #st.text(model_summary_string)
            #print(model_summary_string)
            st.write(model_summary)
            
        if st.sidebar.button("fit model"):
            #number = st.sidebar.number_input('Enter the epoch no')
            history=fit()
            st.write(history)
            
            hist = pd.DataFrame(history.history)
            hist['epoch'] = history.epoch
            
            #matrics=['mean_absolute_error','accuracy', 'mean_squared_error']
    
            #choice= st.selectbox("select metrics",matrics)
            
            #if choice=='mean_absolute_error':
            st.subheader("Mean Absolute Error : ")
            st.write(hist['mean_absolute_error'])
            #st.subheader("Loss : ",hist['loss'])
            #st.subheader("mean_absolute_error : ",hist['accuracy'])
            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('Mean Absolute Error')
            plt.plot(hist['epoch'],hist['mean_absolute_error'], label = 'train error')
        
            st.pyplot() 
            
            #elif choice=='accuracy':
            st.subheader("Acuuracy : ")
            st.write(hist['accuracy'])
            #st.subheader("Loss : ",hist['loss'])
            #st.subheader("mean_absolute_error : ",hist['accuracy'])
            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.plot(hist['epoch'],hist['accuracy'], label = 'accuracy')
            
            st.pyplot()
                
           
                
            #elif choice=='mean_absolute_error':
            
                
           # elif choice=='mean_squared_error':
            st.subheader("Mean Squared Error : ")
            st.write(hist['mean_squared_error'])
            #st.subheader("Loss : ",hist['loss'])
            #st.subheader("mean_absolute_error : ",hist['accuracy'])
            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('Mean Squared Error')
            plt.plot(hist['epoch'],hist['mean_squared_error'], label = 'train error')
            
            st.pyplot()  
            
            st.balloons()
        if st.sidebar.button("visualization"):
            visualize()
            
            
            
            
            
        
        
        
    
    
    #@st.cache(persit-True)
    
    
    def print_data():    
        train_horse_names = os.listdir(train_horse_dir)
        train_human_names = os.listdir(train_human_dir)
        print(train_horse_names[:10])
        print(train_human_names[:10])
        
    #@st.cache(persit-True)
    
    
            
        #return (model.summary())

   
         
    
    
    
    
if __name__ == '__main__':
        main()
    
    