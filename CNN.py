import tensorflow as tf
#for preprocessing of images
from keras.preprocessing.image import ImageDataGenerator

tf.__version__ 

#Data PreProcessing
#1) Preprocessing the training set
#applying transformations to modify images = Image Augmentation. We do this so that we don't over train our model to avoid overfitting
#code snippet copied from keras API docs, image datapreprocessing
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
#rescale- feature scaling of all pixels
#shear_range = geometrical transformation
#zoom_range = zoom in/out of images
#horizontal_flip = to flip images horizontally

#Connecting train_datagen to our training set images
training_set = train_datagen.flow_from_directory(
        '/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/Python/dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
#target_size= size of images to be fed to the CNN. lesser the size, lesser pixels and faster performance
#batch_size = # images in each batch
#class_mode = binary/categorical depending on need. Here binary because cat/dog

#2) Preprocessing test set
#not applying same transformation as train to avoid information leakage, but only feature scaling on pixels. 
test_datagen = ImageDataGenerator(rescale=1./255)

testing_set = test_datagen.flow_from_directory(
        '/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/Python/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#part 2: Building the CNN
#initializing cnn as sequence of layers similar to ANN
cnn = tf.keras.models.Sequential()
#Step1: Convolution. Adding convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))
#filter= # of feature dectectors/filters. kernel_size is the size of that filter (3 = 3X3 matrix). activation is the relu func. input_shape = resize of image, 3 for RGB, 1 for BW

#Step2: Pooling. Applying max pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
#pool_size = size of pixels used for maxpooling  | strides= number of pixels the frame is shifted. | padding = valid for missing pixels(eg-last column with moving the strides)[deault], 'same'-add missing pixels with 0 value

#Adding second convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu')) #no input_shape becasue that is only for the layer connected to the input
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Step3: Flattening
cnn.add(tf.keras.layers.Flatten())

#Step4: Full Connection || Similar steps as ANN
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))

#Step 5: Output Layer
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid')) #units=1 because only cat/dog binary classification. 
#for binary classification we use sigmoid activation
#for multi class classification we use softmax activation


#Part 3 - Training the CNN

#Compiling CNN ||Similar to ANN
cnn.compile(optimizer='adam' ,loss='binary_crossentropy' ,metrics=['accuracy'])
#optimizer = stochastic gradient descent (adam) #for binary ouput classification we must use binary_crossentropy, non binary = categorical_crossentropy

#Training CNN on the training set and evaluating it on the test set
cnn.fit(x=training_set,validation_data=testing_set, epochs=25)

#Part 4: making a single prediction
import numpy as np
from keras.preprocessing import image #for loading the image and processing it
test_image = image.load_img('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/Python/dataset/single_prediction/cat_or_dog_2.jpg',target_size=(64,64))
#target_size = resizing the image to match with what we modelled
#predict method expects a 2D array. So we covert the image into a 2D array
test_image = image.img_to_array(test_image)
#adding an extra (fake) dimension batch_size since predict expects this and we modelled to take 32 images at a time i.e., bacth_size=32
test_image = np.expand_dims(test_image,axis=0) #axis=0 - dimension of the batch we are adding will be the first dimension(argument)

#now we are ready to use predict method
result = cnn.predict(test_image)

#for getting the indices for cats and dogs from the training_set classifications
training_set.class_indices
if result[0][0] == 1: #result consists of batch size and the 1 predicted element result[0] is the batch dimension [0] is the 1st value
    prediction ='Dog'
else:
    prediction = 'Cat'
    

print(prediction)
