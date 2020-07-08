from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.datasets import mnist
from keras.utils import np_utils
import keras
import sys

sys.stdin = open('/loc1/initial.txt','r')

# loads the MNIST dataset
(x_train, y_train), (x_test, y_test)  = mnist.load_data()

# Lets store the number of rows and columns
img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

# Getting our data in the right 'shape' needed for Keras
# We need to add a 4th dimension to our data thereby changing our original image shape of (60000,28,28) to (60000,28,28,1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# store the shape of a single image 
input_shape = (img_rows, img_cols, 1)

# change our image type to float32 data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize our data by changing the range from (0 to 255) to (0 to 1)
x_train /= 255
x_test /= 255

# Now we one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]


# ### Now let's create our layers to replicate LeNet
# create model
model = Sequential()

# first set of CRP (Convolution, RELU, Pooling)

convlayers = int(input())
first_layer_nfilter = int(input())
first_layer_filter_size = int(input())
first_layer_pool_size = int(input())

this_layer = 'No. of convolve layers : ' + str(convlayers)
this_layer = this_layer + '\nLayer 1'
this_layer = this_layer + '\nNo of filters : ' + str(first_layer_nfilter) + '\nFilter Size : ' + str(first_layer_filter_size) + '\nPool Size : ' + str(first_layer_pool_size)

model.add(Conv2D(first_layer_nfilter, (first_layer_filter_size, first_layer_filter_size),
                 padding = "same", 
                 input_shape = input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (first_layer_pool_size, first_layer_pool_size)))

#Subsequent CRP sets
for i in range(1,convlayers):
	nfilters = int(input())
	filter_size = int(input())
	pool_size = int(input())
	this_layer = this_layer + '\nLayer ' + str(i+1) + ': '
	this_layer = this_layer + '\nNo of filters : ' + str(nfilters) + '\nFilter Size : ' + str(filter_size) + '\nPool Size : ' + str(pool_size)
	model.add(Conv2D(nfilters, (filter_size, filter_size),padding = "same"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size = (pool_size, pool_size)))

# Fully connected layers
model.add(Flatten())

fc_input = int(input())

this_layer = this_layer + '\nNo. of FC Layers : ' + str(fc_input+1) 

for i in range(0,fc_input):
	no_neurons = int(input())
	this_layer = this_layer + '\nNeurons in Layer ' + str(i+1) + ' : ' + str(no_neurons)
	model.add(Dense(no_neurons))
	model.add(Activation("relu"))

# Softmax (for classification)
model.add(Dense(num_classes))
model.add(Activation("softmax"))
           
this_layer = this_layer + '\nNeurons in Layer ' + str(fc_input + 1) + ' : ' + str(num_classes)

model.compile(loss = 'categorical_crossentropy',
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])
    
print(model.summary())

### Training LeNet on our MNIST Dataset
# Training Parameters
batch_size = 128
epochs = 5

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

# Evaluate the performance of our trained model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss : %.2f%%' % (scores[0]*100))
print("Test accuracy : %.2f%%" % (scores[1]*100))

accuracy = open('/loc1/accuracy.txt','w')
accuracy.write(str(scores[1]))
accuracy.close()

show = open('/loc1/show.html','r+')
show.read()
show.write('<pre>\n---------------------------------------------\n')
show.write(this_layer)
show.write('\nAccuracy achieved : ' + str(scores[1])+'\n</pre>')
show.close()