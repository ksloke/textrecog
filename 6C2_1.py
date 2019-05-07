#8 July 2018 corrected : used a-1,b-1 twice
#25 Jan 2019 average the surround pixel values

import tensorflow 
from tensorflow.python import keras
from tensorflow.python.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation, BatchNormalization
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras import utils
from tensorflow.python.keras import backend as Backend
from tensorflow import set_random_seed
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.callbacks import Callback

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
#from scipy.misc import imread, imresize
from sklearn.model_selection  import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from functools import reduce

from io import BytesIO

import matplotlib.image as matplotImage
import PIL.Image as Image
import os
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline

from mymetrics import F1Metrics

directory = './nikonD700/cropped'
#directory = './MiMax_2/cropped'

def replace_all(text, dic):
  for i, j in dic.items():
      text = text.replace(i, j)
  return text


width, height = 50, 50
files = os.listdir(directory)
index = 0

images = []
labels = []
  
for file in files:
  img = Image.open(directory + '/' + file)
  img = img.convert("RGB") # for 4 channel PNG file
  imgWidth, imgHeight = img.size
  print(file)
  mrr,mrg,mrb,mgg,mgb,mbb=0,0,0,0,0,0  #per image
  
  for h in range(0, imgHeight-height, height):
    for w in range(0, imgWidth-width, width):
      #mrr,mrg,mrb,mgg,mgb,mbb=0,0,0,0,0,0  #per patch
      box = (w, h, w+width, h+height)
      croppedImage = np.array(img.crop(box)).astype(int)
      labels.append(np.array(replace_all(file, {".jpg" : "", ".png" : "", ".jpeg": ""})))
      temp_arr = np.zeros(shape=(height - 2, width - 2, 6))
      i = 0
      for a, temp_height in enumerate(croppedImage):
        j = 0
        is_skipping = False
        for b, temp_width in enumerate(temp_height):
          if(a == 0 or a == (len(croppedImage) - 1)):
            is_skipping = True
            break
          if(b == 0 or b == (len(temp_height) - 1)):
            continue
          #version 2
          rr = (temp_width[0] * (croppedImage[a-1][b-1][0] + croppedImage[a-1][b][0] + croppedImage[a-1][b+1][0] + croppedImage[a][b-1][0] + 
                                croppedImage[a][b+1][0] +  croppedImage[a+1][b-1][0] + croppedImage[a+1][b][0] + croppedImage[a+1][b+1][0]))/8.0
          rg = (temp_width[0] * (croppedImage[a-1][b-1][1] + croppedImage[a-1][b][1] + croppedImage[a-1][b+1][1] + croppedImage[a][b-1][1] + 
                                croppedImage[a][b+1][1]  + croppedImage[a+1][b-1][1] + croppedImage[a+1][b][1] + croppedImage[a+1][b+1][1]))/8.0
          rb = (temp_width[0] * (croppedImage[a-1][b-1][2] + croppedImage[a-1][b][2] + croppedImage[a-1][b+1][2] + croppedImage[a][b-1][2] + 
                                croppedImage[a][b+1][2]  + croppedImage[a+1][b-1][2] + croppedImage[a+1][b][2] + croppedImage[a+1][b+1][2]))/8.0
          gg = (temp_width[1] * (croppedImage[a-1][b-1][1] + croppedImage[a-1][b][1] + croppedImage[a-1][b+1][1] + croppedImage[a][b-1][1] + 
                                croppedImage[a][b+1][1]  + croppedImage[a+1][b-1][1] + croppedImage[a+1][b][1] + croppedImage[a+1][b+1][1]))/8.0
          gb = (temp_width[1] * (croppedImage[a-1][b-1][2] + croppedImage[a-1][b][2] + croppedImage[a-1][b+1][2] + croppedImage[a][b-1][2] + 
                                croppedImage[a][b+1][2] +  croppedImage[a+1][b-1][2] + croppedImage[a+1][b][2] + croppedImage[a+1][b+1][2]))/8.0
          bb = (temp_width[2] * (croppedImage[a-1][b-1][2] + croppedImage[a-1][b][2] + croppedImage[a-1][b+1][2] + croppedImage[a][b-1][2] + 
                                croppedImage[a][b+1][2] + croppedImage[a+1][b-1][2] + croppedImage[a+1][b][2] + croppedImage[a+1][b+1][2]))/8.0

          if(rr>mrr):mrr=rr
          if(rg>mrg):mrg=rg
          if(rb>mrb):mrb=rb
          if(gg>mgg):mgg=gg
          if(gb>mgb):mgb=gb
          if(bb>mbb):mbb=bb
          
          temp_arr[i][j] = np.array([rr, rg, rb, gg, gb, bb])
          j += 1
          
        if not is_skipping:
          i += 1
          
      temp_arr[:,:,0]=temp_arr[:,:,0]/mrr
      temp_arr[:,:,1]=temp_arr[:,:,1]/mrg
      temp_arr[:,:,2]=temp_arr[:,:,2]/mrb
      temp_arr[:,:,3]=temp_arr[:,:,3]/mgg
      temp_arr[:,:,4]=temp_arr[:,:,4]/mgb
      temp_arr[:,:,5]=temp_arr[:,:,5]/mbb      
      images.append(temp_arr)
      del temp_arr
      del croppedImage
  index += 1

images = np.array(images)
labels = np.array(labels)
print(images.shape)
print(labels.shape)

#encode labels

le = LabelEncoder()
labels = le.fit_transform(labels)
seed = 1
set_random_seed(seed)
np.random.seed(seed)

## Split the training and test set
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=seed)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

### Number of classes, 26 different wood species (for image, image_lowres); 27 for nikonD600.
num_classes=27

y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)


#Model 6 channel
num_channels=6
h=height-2
w=width-2



model = Sequential()
model.add(Conv2D(32, (3, 3),input_shape=(h , w , num_channels), data_format="channels_last"))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


num_epochs = 120
num_batch = 100

### ONLY USE EITHER Model.fit or Model.fit_generator. DO NOT RUN BOTH!

metrics = F1Metrics()

## Normal Fit to model without augmentation on the input
history = model.fit(x_train, y_train, validation_data=[x_test, y_test], epochs= num_epochs, batch_size= num_batch, verbose=2, shuffle=True, callbacks=[metrics])
## Fit to model with augmentation on the input data. Note that steps_per_epoch should be the data length/batch_size.
#datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
#datagen.fit(x_train)
#history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=num_batch), validation_data=(x_test, y_test), steps_per_epoch=len(x_train) / num_batch, epochs=num_epochs, verbose=2, shuffle=True)

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=1)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

filename='channel-6_2a-Nikon.h5'
model.save(filename)


#Graph
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()