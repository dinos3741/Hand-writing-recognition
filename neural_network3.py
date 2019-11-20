# THIS IS THE VERSION THAT I LOADED THE MNIST HANDWRITTEN NUMBER SET AND CONVERTED TO A NUMPY ARRAY AND TRAINED
# THE NN. IT WORKS SUFFICIENTLY WELL.

# Divide the window to 28x28 squares. When the mouse passes over them, they become blue. When user hits the control
# key, the 28x28 data gets dumped to an input numpy array.

from tkinter import *
import numpy as np
import math
from sklearn.neural_network import MLPClassifier
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

# print a complete numpy array:
np.set_printoptions(threshold=np.inf)

# the x_test and x_train are grayscale images, not binary. I transform to binary:
threshold = 200
bin_x_test = 1.0 * (x_test > threshold)
bin_x_train = 1.0 * (x_train > threshold)

# Initialize NN with default options: hidden layer sizes is the nuber of layers and nr of neurons each
clf = MLPClassifier(verbose=True, max_iter=30, hidden_layer_sizes=(10, 10, 10))

# train the NN with the MNIST samples and labels (answers):
clf.fit(bin_x_train, t_train)
print("finished fitting model")
#print(clf.predict(bin_x_test), t_test)

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
        # dump the array inputs of one sample hand-written number:
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

    print(clf.predict(writing_pad.reshape(1, GRID_X * GRID_Y)))
    # reset writing pad after prediction
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
