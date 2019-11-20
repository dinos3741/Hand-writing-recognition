# THIS IS THE VERSION WHERE I IMPLEMENT MY OWN NN AND TRAIN IT WITH THE MNIST SET.

# Divide the window to 28x28 squares. When the mouse passes over them, they become blue. When user hits the control
# key, the 28x28 data gets dumped to an input numpy array.

from tkinter import *
import numpy as np
import math
import mnist
import os.path

# the grid dimensions (28x28) of the handwriting pad. This will be the set of features of one sample, inserted in
# one row of the input matrix
GRID_X = 28
GRID_Y = 28

# size of the writing pad:
WIDTH = 280
HEIGHT = 280

# create the empty writing pad array to store the features (28x28 - 784) of one sample:
writing_pad = np.zeros((GRID_X, GRID_Y))

# flag to draw in the trackpad window:
drawing_active = False

# download (if not already) the MNIST samples and turn them into the numpy arrays according to this:
# https://github.com/hsjeong5/MNIST-for-Numpy/blob/master/README.md
if not os.path.exists("mnist.pkl"):
    mnist.init()

# x_train : 60,000x784 numpy array that each row contains flattened version of training images.
# t_train : 1x60,000 numpy array that each component is true label of the corresponding training images.
# x_test : 10,000x784 numpy array that each row contains flattened version of test images.
# t_test : 1x10,000 numpy array that each component is true label of the corresponding test images.
x_train, t_train, x_test, t_test = mnist.load()

# the x_test and x_train are grayscale images, not binary. I transform to binary:
threshold = 200
bin_x_test = 1.0 * (x_test > threshold)
bin_x_train = 1.0 * (x_train > threshold)

# SOS: I need to reshape the t_test and t_train as follows because the initial dimensions are (1000,) and I turn them
# to (1000,1) and (60000,1)
t_test = t_test.reshape(t_test.shape[0], 1)
t_train = t_train.reshape(t_train.shape[0], 1)


# This is my implementation of a Neural Network class and functions
#------------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x  # input is the two-dimensional array of features for all samples. dimension [0] (rows) is
        # the samples, dimension [1] (columns) is the features.
        self.weights1 = np.random.rand(self.input.shape[1], 4)  # weights 2-D matrix of first layer. input.shape[1]
        # is the columns size of the input, ie. the features, so this is a random array 0-1 of 784x4
        # The dimension [0] (rows) is the features, the [1] (columns) is the weights to connect to the next layer.
        # We use 4 neurons for the hidden layer.
        self.weights2 = np.random.rand(4, 1)  # weights matrix of second layer. It outputs one number.
        self.y = y  # this is the array of the expected values of the outputs (1x60000)
        self.output = np.zeros(y.shape)  # this is the prediction

    # forward propagation:
    def feed_forward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))  # 2-D matrix multiplication (60000x784)x(784x4)=(60000x4)
        self.output = sigmoid(np.dot(self.layer1, self.weights2)) # (60000x4)x(4x1)=(60000x1)

        print("output: " + str(self.output))

    def back_propagation(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        # (4x60000)x(60000x1)=4x1

        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                self.weights2.T) * sigmoid_derivative(self.layer1)))
        # (784x1000)xx ( (1000x1)x(1x4) )=(784x1000)x(1000x4)=784x4

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


# initialize the NN and train with the MNIST dataset:
neural = NeuralNetwork(bin_x_train, t_train)
for i in range(10):
    #print("before iteration " + str(i) + ": weights1 = " +str(neural.weights1) + ", weights2 = " + str(neural.weights2))
    neural.feed_forward()
    neural.back_propagation()


#------------------------------------------------------------------
# create the main window using Tk:
top_window = Tk()
top_window.title("Handwriting recognition")
top_window.resizable(False, False)  # turn off resizing of the top window

# create canvas to draw inside with the dimensions defined above:
canvas = Canvas(top_window, width=WIDTH, height=HEIGHT, bg="white")
# packs the elements in the window:
canvas.pack()

#------------------------------------------------------------------
# define mouse move callback inside canvas => get the current mouse position:
def mouse_motion(event):
    global GRID_X
    global GRID_Y
    global WIDTH
    global HEIGHT

    # quantize the inputs to 0-19:
    x_input = math.floor(GRID_X * event.x/WIDTH-1)
    y_input = math.floor(GRID_Y * event.y/HEIGHT-1)
    if x_input < 0:
        x_input = 0
    if x_input > GRID_X - 1:
        x_input = GRID_X - 1
    if y_input < 0:
        y_input = 0
    if y_input > GRID_Y - 1:
        y_input = GRID_Y - 1

    if drawing_active == True:  # if mouse left button clicked:
        # draw the squares so we see on the screen what was written:
        canvas.create_rectangle(x_input*10, y_input*10, x_input*10+10, y_input*10+10, fill="blue")
        # dump the array inputs of one sample hand-written number, y and then x:
        writing_pad[y_input][x_input] = 1

# bind mouse left button press with callback routine:
canvas.bind('<Motion>', mouse_motion)

#----------------------------------------------
# define control key press callback inside canvas: it sends the data for processing
def control_key_callback(event):
    global GRID_X
    global GRID_Y
    global sample_num
    global clf
    global writing_pad

    # clear the canvas:
    canvas.delete("all")

    # predict result:

    writing_pad = np.zeros((GRID_X, GRID_Y))

# bind control key press with callback routine:
top_window.bind('<Control-Key>', control_key_callback)  # bound to top_window because canvas does not get focus immediately

#----------------------------------------------
# define shift key press callback inside canvas:
def shift_key_callback(event):
    # clear the canvas:
    canvas.delete("all")

# bind shift key press with callback routine:
top_window.bind('<Shift-Key>', shift_key_callback)

#----------------------------------------------
# define left mouse button press callback inside canvas: it activates drawing in the trackpad
def button1_press_callback(event):
    global drawing_active
    drawing_active = True

# bind mouse button press with callback routine:
top_window.bind('<ButtonPress-1>', button1_press_callback)

#----------------------------------------------
# define left button release callback inside canvas: it de-activates drawing in the trackpad:
def button1_release_callback(event):
    global drawing_active
    drawing_active = False

# bind mouse button release with callback routine:
top_window.bind('<ButtonRelease-1>', button1_release_callback)
#------------------------------------------------------------------
# main event loop of tkinter:
top_window.mainloop()
