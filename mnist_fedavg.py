
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from tensorflow.keras.preprocessing import sequence
import tensorflow as tfs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import LSTM

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D,LeakyReLU
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2, random

height=28
width=28
depth=1

inputShape = (height, width, depth)

# Prepare the train and test dataset.
from tensorflow import keras
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train.astype("float32") / 255.0
x_train = np.reshape(x_train, (-1, 28, 28, 1))

x_test = x_test.astype("float32") / 255.0
x_test = np.reshape(x_test, (-1, 28, 28, 1))

from  keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train)
# y_test= np_utils.to_categorical(y_test)
# print(y_train[0])
print(y_train.shape,y_test.shape)
print(x_train.shape)
# print(y_train[0],x_train[0])

from sklearn.utils import shuffle
x_train, y_train= shuffle(x_train, y_train, random_state=22)
X=[]
Y=[]
j=0
for i in range(40):
  X.append(x_train[j:j+3000])
  Y.append(y_train[j:j+3000])
  j+=1450

X=np.array(X)
Y=np.array(Y)
print(X.shape,Y.shape)


from tensorflow.keras.layers import BatchNormalization
#For teacher in each cluster
class MODEL:
    @staticmethod
    def build():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",input_shape=inputShape) )
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same") )
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same") )
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same") )
        model.add(Flatten())
        model.add(Dense(10,activation='softmax'))#for output layer
        return model
#For student in each cluster
class MODEL1:
    @staticmethod
    def build():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",input_shape=inputShape) )
        model.add(Conv2D(16, (3, 3), strides=(2, 2), padding="same") )
        model.add(Conv2D(16, (3, 3), strides=(2, 2), padding="same") )
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same") )
        model.add(Flatten())
        model.add(Dense(10,activation='softmax'))#for output layer
        return model

from sklearn.metrics import accuracy_score
def test_model(X_test, Y_test,  model, comm_round):
    model.compile(optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
    # loss,acc=model.evaluate(x_test,y_test)
    acc,loss=model.evaluate(x_test,y_test,verbose=0)
    return acc, loss
def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    return avg_grad


class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=1.0,
        temperature=3,
    ):

        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
        

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from tensorflow import keras
from sklearn.cluster import KMeans
import random
import time

beta =0.5
comms_round = 100
epoch_teacher = 8
epoch_student = 7
total_clusters = 4

batch_size = 64

# Assuming other necessary functions and the MODEL, MODEL1, Distiller definitions are above this code

def create_dirichlet_distributed_data(X, Y, num_clients, beta):
    if len(Y.shape) > 1 and Y.shape[1] > 1:  # Check if Y is one-hot encoded
        Y_int = np.argmax(Y, axis=1)  # Convert from one-hot to integer labels
    else:
        Y_int = Y.ravel()  # Simply flatten the array

    num_classes = len(np.unique(Y_int))

    client_data = [[] for _ in range(num_clients)]
    label_distribution = np.random.dirichlet([beta] * num_classes, num_clients)

    for i in range(len(Y_int)):
        target_label = Y_int[i]
        client_probabilities = label_distribution[:, target_label]
        client = np.random.choice(np.arange(num_clients), p=client_probabilities/sum(client_probabilities))
        client_data[client].append((X[i], Y[i]))

    client_data = [(np.array([t[0] for t in client]), np.array([t[1] for t in client])) for client in client_data]
    return client_data

# Define the total number of clients and beta for Dirichlet distribution
total_clients = 40


# Apply the Dirichlet distribution to create non-IID data
clients_data = create_dirichlet_distributed_data(x_train, y_train, total_clients, beta)




# Continue with your original code:

global1 = MODEL()
global_model = global1.build()



teacher = MODEL()
teacher_model = teacher.build()

from tensorflow.keras import backend as K


for comm_round in range(comms_round):
    f1= open("FLbasic_mnist0.5.txt", "a+") 
    scaled_local_weight_list = list()	
    global_weights = global_model.get_weights()
    print("\nCommunication Round:", comm_round)
    for ind in range(total_clients):
        client_data, client_labels = clients_data[ind]
        if ind == 0:
            teacher_model.compile(optimizer=keras.optimizers.Adam(),
                                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
            teacher_model.set_weights(global_weights)
            history = teacher_model.fit(client_data, client_labels, epochs=epoch_teacher, verbose=0)
            scaling_factor = 1.0 / total_clients
            scaled_weights = scale_model_weights(teacher_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)
            K.clear_session()
        else:
            student = MODEL1().build()
            student_model = Distiller(student=student, teacher=teacher_model)
            student_weights = student_model.get_weights()
            student_model.compile(optimizer=keras.optimizers.Adam(),
                                  metrics=[keras.metrics.SparseCategoricalAccuracy()],
                                  student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                  distillation_loss_fn=keras.losses.KLDivergence(),
                                  alpha=1.0,
                                  temperature=5)
            student_model.set_weights(student_weights)
            history = student_model.fit(client_data, client_labels, epochs=epoch_student, verbose=0)
            #print("success")
            scaling_factor = 1.0 / total_clients
            scaled_weights = scale_model_weights(student_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)
            K.clear_session()
        print(ind)
    average_weights = sum_scaled_weights(scaled_local_weight_list)
    global_model.set_weights(average_weights)
    global_loss, global_acc = test_model(x_test, y_test, global_model, comm_round)
    print("GLOBAL ACCURACY", global_acc, "GLOBAL LOSS", global_loss)

    model_size = global_model.count_params()
    print("Model Size (Parameters):", model_size)

    start_time = time.time()
    global_loss, global_acc = test_model(x_test, y_test, global_model, comm_round)
    end_time = time.time()
    time_to_test = end_time - start_time
    #print("Time to Test:", time_to_test, "seconds")
    f1.write("\nCommunication Round: %d GLOBAL MODEL ACCURACY: %f LOSS: %f \n" %(comm_round ,global_acc ,global_loss))

