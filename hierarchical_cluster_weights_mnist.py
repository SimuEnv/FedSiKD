

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
#(40, 3000, 28, 28, 1) (40, 3000)

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
    acc,loss=model.evaluate(x_test,y_test)
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
        alpha=0.1,
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

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import time

import numpy as np
from tensorflow import keras
from sklearn.cluster import KMeans
import random
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming other necessary functions and the MODEL, MODEL1, Distiller definitions are above this code

def create_dirichlet_distributed_data(X, Y, num_clients, alpha):
    if len(Y.shape) > 1 and Y.shape[1] > 1:  # Check if Y is one-hot encoded
        Y_int = np.argmax(Y, axis=1)  # Convert from one-hot to integer labels
    else:
        Y_int = Y.ravel()  # Simply flatten the array

    num_classes = len(np.unique(Y_int))

    client_data = [[] for _ in range(num_clients)]
    label_distribution = np.random.dirichlet([alpha] * num_classes, num_clients)

    for i in range(len(Y_int)):
        target_label = Y_int[i]
        client_probabilities = label_distribution[:, target_label]
        client = np.random.choice(np.arange(num_clients), p=client_probabilities/sum(client_probabilities))
        client_data[client].append((X[i], Y[i]))

    client_data = [(np.array([t[0] for t in client]), np.array([t[1] for t in client])) for client in client_data]
    return client_data

# Define the total number of clients and alpha for Dirichlet distribution
total_clients = 40
alpha = 2.0

# Apply the Dirichlet distribution to create non-IID data
clients_data = create_dirichlet_distributed_data(x_train, y_train, total_clients, alpha)

# Continue with your original code:

global1 = MODEL()
global_model = global1.build()
comms_round = 75
epoch_teacher = 10
epoch_student = 6
total_clusters = 4
#total_clients = 40
client_stats = []
clients_per_cluster = int(total_clients / total_clusters)

# clients_per_cluster = int(total_clients / total_clusters)



# from scipy.stats import skew

# # Initialize the client statistics list
# client_stats = []

# # Loop over each client's data
# for client_data, _ in clients_data:
#     # Flatten the images
#     client_data_flat = client_data.reshape(client_data.shape[0], -1)

#     # Compute mean, standard deviation, and skewness
#     mean = np.array([np.mean(client_data_flat)])
#     std_dev = np.array([np.std(client_data_flat)])
#     data_skewness = np.array([skew(client_data_flat, axis=None)])

#     # Compute normalized histogram bins
#     hist_bins = np.histogram(client_data_flat, bins=5)[0]
#     hist_bins = np.array(hist_bins / np.sum(hist_bins))

#     # Concatenate the features
#     features = np.concatenate((mean, std_dev, data_skewness, hist_bins))
#     client_stats.append(features)

# # ... Proceed with clustering code


# client_stats = np.array(client_stats)  # Convert the list to a numpy array


# # Determine the optimal number of clusters
# X = np.array(client_stats)
# distortions = []
# K_range = range(2, 6)  # Setting range from 2 to 6
# for k in K_range:
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(X)
#     distortions.append(kmeans.inertia_)

# optimal_k = distortions.index(min(distortions)) + 2


# # Cluster clients based on data distribution similarity
# kmeans = KMeans(n_clusters=optimal_k, random_state=0)
# cluster_assignments = kmeans.fit_predict(client_stats)
# print("Cluster Assignments:", cluster_assignments)

import tensorflow.keras.backend as K

import resource
import logging
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.cluster import Birch
#from scipy.spatial.distance import euclidean_distances

"""**Similarties-Based Approach**"""

import numpy as np
cluster_assignments = None  # Initialize cluster_assignments outside the loop

teacher = MODEL()
teacher_model = teacher.build()

for comm_round in range(comms_round):
    K.clear_session()
    if comm_round == 0:
        global_weights = global_model.get_weights()
        client_updates = []
              # Collect client weights after round zero
        for client_tuple in clients_data:
            if len(client_tuple) == 2:
                client_data, client_labels = client_tuple
            else:
                # Handle the case where the length is not 2 (e.g., print a warning)
                print(f"Unexpected tuple length: {len(client_tuple)}")
                continue  # Skip this iteration or handle it according to your needs

            client_model = MODEL().build()
            client_model.compile(optimizer=keras.optimizers.Adam(),
                                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
            client_model.set_weights(global_weights)
            history = client_model.fit(client_data, client_labels, verbose=2)

            # Collect the weights without clearing the session
            client_weights = client_model.get_weights()
            client_updates.append(client_weights)


        # Convert the list of arrays to a single flat array
        client_updates_array = np.concatenate([np.concatenate([arr.flatten() for arr in client_weights]) for client_weights in client_updates])

        # Reshape the flat array into a 2D array
        weights = client_updates_array.reshape(len(client_updates), -1)




        # Perform hierarchical clustering using Euclidean distances
        distance_matrix = euclidean_distances(weights)


        # Determine the number of clusters automatically using hierarchical clustering
        Z = linkage(euclidean_distances(weights), method='average', metric='euclidean')
        cluster_assignments = fcluster(Z, t=5, criterion='distance')  # Use 'maxclust' criterion to specify the number of clusters
        total_clusters = len(np.unique(cluster_assignments))
        clients_per_cluster = int(total_clients / total_clusters)
        print("\ntotal_clusters:", total_clusters)
        print("\nclients_per_cluster:", clients_per_cluster)


    else:
        # Subsequent rounds - use clustering information obtained in round zero
        # Apply the cluster assignments to client data
        for i in range(len(clients_data)):
            clients_data[i] = (clients_data[i][0], clients_data[i][1], cluster_assignments[i])

        # Update global weights using clustered client data
        # global_weights = update_global_weights(clients_data, cluster_assignments, global_model)

    scaled_cluster_weight_list = []

    print("\nCommunication Round:", comm_round)
    for clstr in range(total_clusters):


        cluster_clients = [client_stats[i] for i in range(len(client_stats)) if cluster_assignments[i] == clstr]
        cluster_mean = [stat[0] for stat in cluster_clients]
        cluster_std_dev = [stat[1] for stat in cluster_clients]

        scaled_local_weight_list = []

        # Assuming you have a function to create and compile cluster model
        cluster_model = MODEL().build()
        cluster_model.compile(optimizer=keras.optimizers.Adam(),
                              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                              metrics=[keras.metrics.SparseCategoricalAccuracy()])
        cluster_model.set_weights(global_weights)
        #cluster_model.summary()
        # Print clients_data for debugging
        #print("clients_data structure:", clients_data)\


        for client_tuple in clients_data[clstr * clients_per_cluster : (clstr + 1) * clients_per_cluster]:
            if len(client_tuple) == 3:
                # Unpack three elements
                client_data, client_labels, cluster_assignment = client_tuple
            elif len(client_tuple) == 2:
                # Unpack two elements
                client_data, client_labels = client_tuple
                cluster_assignment = None  # Or assign a default value if cluster_assignment is not present
            else:
                # Handle the case where the length is neither 2 nor 3
                print(f"Unexpected tuple length: {len(client_tuple)}")
                continue  # Skip this iteration or handle it according to your needs


            teacher_model.compile(optimizer=keras.optimizers.Adam(),
                                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
            teacher_model.set_weights(cluster_model.get_weights())  # Set teacher weights
            history = teacher_model.fit(client_data, client_labels, verbose=2)

            scaling_factor = 1.0 / clients_per_cluster
            scaled_weights = scale_model_weights(teacher_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)

            # After updating teacher model
            #print("Memory usage after updating teacher model:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            K.clear_session()

            # Student model update
            student = MODEL1().build()
            student_model = Distiller(student=student, teacher=teacher_model)
            student_model.compile(optimizer=keras.optimizers.Adam(),
                                  metrics=[keras.metrics.SparseCategoricalAccuracy()],
                                  student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                  distillation_loss_fn=keras.losses.KLDivergence(),
                                  alpha=0.1,
                                  temperature=3)

            # Train the student model
            history = student_model.fit(client_data, client_labels, verbose=2)

            # Directly use the trained student_model for predictions
            scaling_factor = 1.0 / clients_per_cluster
            scaled_weights = scale_model_weights(student_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)

            K.clear_session()


        # Update global model weights based on clustered weights
        average_cluster_weights = sum_scaled_weights(scaled_local_weight_list)
        cluster_model.set_weights(average_cluster_weights)

        scaling_cluster_factor = 1.0 / total_clusters
        scaled_cluster_weights = scale_model_weights(cluster_model.get_weights(), scaling_cluster_factor)
        scaled_cluster_weight_list.append(scaled_cluster_weights)
        cluster_loss, cluster_acc = test_model(x_test, y_test, cluster_model, comm_round)
        print("CLUSTER ", clstr, "ACC: ", cluster_acc, "CLUSTER ", clstr, "LOSS: ", cluster_loss)

    global_model.set_weights(sum_scaled_weights(scaled_cluster_weight_list))
    global_loss, global_acc = test_model(x_test, y_test, global_model, comm_round)
    print("GLOBAL ACCURACY", global_acc, "GLOBAL LOSS", global_loss)

    model_size = global_model.count_params()
    print("Model Size (Parameters):", model_size)

    start_time = time.time()
    global_loss, global_acc = test_model(x_test, y_test, global_model, comm_round)
    end_time = time.time()
    time_to_test = end_time - start_time
    print("Time to Test:", time_to_test, "seconds")

print(global_loss)
print(global_acc)


