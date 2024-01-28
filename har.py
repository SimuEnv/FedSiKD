import tensorflow as tf
import pandas as pd
from os import listdir
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D,LeakyReLU,Conv2D,MaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Conv1D, MaxPooling1D
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2, random
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from  keras.utils import np_utils
import warnings
TF_MIN_GPU_MULTIPROCESSOR_COUNT=4
warnings.filterwarnings("ignore", category=FutureWarning)

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	x_train, y_train = load_dataset_group('train', prefix + 'HAR/')
	print(x_train.shape, y_train.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'HAR/')
	print(testX.shape, testy.shape)
	# zero-offset class values
	y_train = y_train - 1
	testy = testy - 1
	print(x_train.shape, y_train.shape, testX.shape, testy.shape)
	return x_train, y_train, testX, testy

#load the data
x_train, y_train, testX, testy = load_dataset()
n_timesteps, n_features = x_train.shape[1], x_train.shape[2]
n_outputs=6
x_train=x_train.reshape(x_train.shape[0],n_timesteps, n_features)
testX=testX.reshape(testX.shape[0],n_timesteps, n_features)

y_train=y_train.reshape(-1)
testy=testy.reshape(-1)
print(y_train.shape,testy.shape)

from sklearn.utils import shuffle
x_train,y_train = shuffle(x_train, y_train, random_state=0)
testX,testy=shuffle(testX,testy, random_state=0)
ix=0
X=list()
Y=list()
for ind11 in range(40):
  X.append(x_train[ix:ix+375])
  Y.append(y_train[ix:ix+375])
  ix=ix+169
  

X=np.array(X)
Y=np.array(Y)
print(X.shape,Y.shape)

x_test = testX

y_test = testy

beta =0.5
comms_round = 100
epoch_teacher = 7
epoch_student = 3
total_clusters = 4

batch_size = 64


from tensorflow import keras

from tensorflow.keras.layers import BatchNormalization
#For teacher in each cluster
class MODEL:
    @staticmethod
    def build():
        model = Sequential()
        model.add(Conv1D(128, 3,strides=2, padding="same",input_shape=(n_timesteps, n_features)))
        # model.add(Conv1D(256, 3,strides=2, padding="same"))#new
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))#new
        model.add(MaxPooling1D(pool_size=2, strides=1, padding="same"))
        model.add(Dropout(0.25))


        model.add(Conv1D(256, 3, strides=2, padding="same") )
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(6,activation='softmax'))#for output layer
        return model


#For student in each cluster
class MODEL1:
    @staticmethod
    def build():
        model = Sequential()
        model.add(Conv1D(64, 3,strides=2, padding="same",input_shape=(n_timesteps, n_features)))
        # model.add(Conv1D(256, 3,strides=2, padding="same"))#new
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))#new
        model.add(MaxPooling1D(pool_size=2, strides=1, padding="same"))
        model.add(Dropout(0.25))
        model.add(Conv1D(256, 3, strides=2, padding="same") )
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(6,activation='softmax'))#for output layer
        return model

from sklearn.metrics import accuracy_score
def test_model(x_test, y_test,  model, comm_round):
    model.compile(optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],)   
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
        temperature=5,
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

"""**Similarties-Based Approach**"""

from tensorflow import keras
from sklearn.cluster import KMeans
import random
import time

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

#total_clients = 40
clients_per_cluster = int(total_clients / total_clusters)



from scipy.stats import skew

# Initialize the client statistics list
client_stats = []

# Loop over each client's data
for client_data, _ in clients_data:
    # Flatten the images
    client_data_flat = client_data.reshape(client_data.shape[0], -1)

    # Compute mean, standard deviation, and skewness
    mean = np.array([np.mean(client_data_flat)])
    std_dev = np.array([np.std(client_data_flat)])
    data_skewness = np.array([skew(client_data_flat, axis=None)])

    # Compute normalized histogram bins
    hist_bins = np.histogram(client_data_flat, bins=5)[0]
    hist_bins = np.array(hist_bins / np.sum(hist_bins))

    # Concatenate the features
    features = np.concatenate((mean, std_dev, data_skewness, hist_bins))
    client_stats.append(features)

# ... Proceed with clustering code


client_stats = np.array(client_stats)  # Convert the list to a numpy array




# Cluster clients based on data distribution similarity
kmeans = KMeans(n_clusters=4, random_state=0)
cluster_assignments = kmeans.fit_predict(client_stats)
print("Cluster Assignments:", cluster_assignments)

teacher = MODEL()
teacher_model = teacher.build()

from tensorflow.keras import backend as K


for comm_round in range(comms_round):
    global_weights = global_model.get_weights()
    scaled_cluster_weight_list = list()
    index = list({0, 1, 2, 3, 4})

    print("\nCommunication Round:", comm_round)
    for clstr in range(total_clusters):
        cluster_clients = [client_stats[i] for i in range(len(client_stats)) if cluster_assignments[i] == clstr]
        cluster_mean = [stat[0] for stat in cluster_clients]
        cluster_std_dev = [stat[1] for stat in cluster_clients]

        scaled_local_weight_list = list()
        cluster_model = MODEL().build()
        cluster_model.compile(optimizer=keras.optimizers.Adam(),
                              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                              metrics=[keras.metrics.SparseCategoricalAccuracy()],
                              )
        cluster_model.set_weights(global_weights)
        cluster_weights = cluster_model.get_weights()

        for ind in range(clients_per_cluster):
            client_data, client_labels = clients_data[clstr * clients_per_cluster + ind]
            if ind == 0:
                teacher_model.compile(optimizer=keras.optimizers.Adam(),
                                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                      metrics=[keras.metrics.SparseCategoricalAccuracy()])
                teacher_model.set_weights(cluster_weights)
                history = teacher_model.fit(client_data, client_labels, epochs=epoch_teacher, verbose=0)




                scaling_factor = 1.0 / clients_per_cluster
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
                print("success")
                scaling_factor = 1.0 / clients_per_cluster
                scaled_weights = scale_model_weights(student_model.get_weights(), scaling_factor)
                scaled_local_weight_list.append(scaled_weights)
                K.clear_session()

        average_cluster_weights = sum_scaled_weights(scaled_local_weight_list)
        cluster_model.set_weights(average_cluster_weights)

        scaling_cluster_factor = 1.0 / total_clusters
        scaled_cluster_weights = scale_model_weights(cluster_model.get_weights(), scaling_cluster_factor)
        scaled_cluster_weight_list.append(scaled_cluster_weights)
        K.clear_session()
        cluster_loss, cluster_acc = test_model(x_test, y_test, cluster_model, comm_round)
        print("CLUSTER ", clstr, "ACC: ", cluster_acc, "CLUSTER ", clstr, "LOSS: ", cluster_loss)

    average_weights = sum_scaled_weights(scaled_cluster_weight_list)
    global_model.set_weights(average_weights)
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

del global_model
del global_loss
del global_acc
del student_model
del teacher_model 
del cluster_model




"""**non-similarties Random**"""
print("non-similarities Random")

teacher = MODEL()
teacher_model = teacher.build()

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
#total_clients = 40
clients_per_cluster = int(total_clients / total_clusters)

client_stats = []
for client_data, client_labels in clients_data:
    mean = np.mean(client_data)
    std_dev = np.std(client_data)
    client_stats.append((mean, std_dev))

def similarity_metric(stat1, stat2):
    distance = np.sqrt((stat1[0] - stat2[0])**2 + (stat1[1] - stat2[1])**2)
    return distance

random_cluster_assignments = [random.randint(0, total_clusters - 1) for _ in range(total_clients)]
print("Random Cluster Assignments:", random_cluster_assignments)

teacher = MODEL()
teacher_model = teacher.build()

from tensorflow.keras import backend as K

for comm_round in range(comms_round):
    global_weights = global_model.get_weights()
    scaled_cluster_weight_list = list()
    index = list({0, 1, 2, 3, 4})

    print("\nCommunication Round:", comm_round)
    for clstr in range(total_clusters):
        cluster_clients = [client_stats[i] for i in range(len(client_stats)) if random_cluster_assignments[i] == clstr]
        cluster_mean = [stat[0] for stat in cluster_clients]
        cluster_std_dev = [stat[1] for stat in cluster_clients]

        scaled_local_weight_list = list()
        cluster_model = MODEL().build()
        cluster_model.compile(optimizer=keras.optimizers.Adam(),
                              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                              metrics=[keras.metrics.SparseCategoricalAccuracy()],
                              )
        cluster_model.set_weights(global_weights)
        cluster_weights = cluster_model.get_weights()

        for ind in range(clients_per_cluster):
            client_data, client_labels = clients_data[clstr * clients_per_cluster + ind]
            if ind == 0:
                teacher_model.compile(optimizer=keras.optimizers.Adam(),
                                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                      metrics=[keras.metrics.SparseCategoricalAccuracy()])
                teacher_model.set_weights(cluster_weights)
                history = teacher_model.fit(client_data, client_labels, epochs=epoch_teacher, verbose=0)




                scaling_factor = 1.0 / clients_per_cluster
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

                scaling_factor = 1.0 / clients_per_cluster
                scaled_weights = scale_model_weights(student_model.get_weights(), scaling_factor)
                scaled_local_weight_list.append(scaled_weights)
                K.clear_session()

        average_cluster_weights = sum_scaled_weights(scaled_local_weight_list)
        cluster_model.set_weights(average_cluster_weights)

        scaling_cluster_factor = 1.0 / total_clusters
        scaled_cluster_weights = scale_model_weights(cluster_model.get_weights(), scaling_cluster_factor)
        scaled_cluster_weight_list.append(scaled_cluster_weights)
        K.clear_session()
        cluster_loss, cluster_acc = test_model(x_test, y_test, cluster_model, comm_round)
        print("CLUSTER ", clstr, "ACC: ", cluster_acc, "CLUSTER ", clstr, "LOSS: ", cluster_loss)

    average_weights = sum_scaled_weights(scaled_cluster_weight_list)
    global_model.set_weights(average_weights)
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


import numpy as np
from scipy.stats import skew, kurtosis, entropy
import numpy as np
from tensorflow import keras
from sklearn.cluster import KMeans
import random
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Define a function to compute all features for a given client dataset
def compute_features(data):
    # Using the first channel for grayscale or RGB images
    data = data[:,:,:,0] if len(data.shape) == 4 else data

    # 1. Mean and standard deviation
    data_mean = np.mean(data)
    data_std = np.std(data)

    # 2. Skewness and kurtosis
    data_skewness = skew(data.ravel())
    data_kurtosis = kurtosis(data.ravel())

    # 3. Histogram
    hist, _ = np.histogram(data, bins=10, range=(0, 255))
    hist = hist / np.sum(hist)  # normalize

    # 4. Entropy
    data_entropy = entropy(hist)

    return [data_mean, data_std, data_skewness, data_kurtosis, data_entropy] + hist.tolist()

# Compute features for each client
features = np.array([compute_features(client_data[0]) for client_data in clients_data])

# Print statisticsde
# Print statistics
feature_names = ['Mean', 'Std Dev', 'Skewness', 'Kurtosis', 'Entropy'] + [f'Hist_Bin_{i}' for i in range(10)]
feature_stats = []

print("Statistics of the computed features:")
print("-" * 50)
for i, name in enumerate(feature_names):
    mean_value = np.mean(features[:, i])
    std_value = np.std(features[:, i])
    feature_stats.append((mean_value, std_value))

    print(f"{name}: Mean = {mean_value:.4f}, Std Dev = {std_value:.4f}")

print("\nNumber of clients where feature value is above the mean:")
print("-" * 50)
for i, name in enumerate(feature_names):
    above_mean = np.sum(features[:, i] > feature_stats[i][0])
    print(f"{name}: {above_mean} clients")

# Compute and print correlations
correlations = np.corrcoef(features, rowvar=False)
print("\nCorrelation matrix of the features:")
print("-" * 50)
print("\t" + "\t".join(feature_names))
for i, name in enumerate(feature_names):
    correlation_values = "\t".join([f"{val:.2f}" for val in correlations[i]])
    print(f"{name}:\t{correlation_values}")

import numpy as np
from scipy.stats import skew, kurtosis, entropy
import numpy as np
from tensorflow import keras
from sklearn.cluster import KMeans
import random
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Define a function to compute all features for a given client dataset
def compute_features(data):
    # Using the first channel for grayscale or RGB images
    data = data[:,:,:,0] if len(data.shape) == 4 else data

    # 1. Mean and standard deviation
    data_mean = np.mean(data)
    data_std = np.std(data)

    # 2. Skewness and kurtosis
    data_skewness = skew(data.ravel())
    data_kurtosis = kurtosis(data.ravel())

    # 3. Histogram
    hist, _ = np.histogram(data, bins=10, range=(0, 255))
    hist = hist / np.sum(hist)  # normalize

    # 4. Entropy
    data_entropy = entropy(hist)

    return [data_mean, data_std, data_skewness, data_kurtosis, data_entropy] + hist.tolist()

# Compute features for each client
features = np.array([compute_features(client_data[0]) for client_data in clients_data])

# Print statistics
feature_names = ['Mean', 'Std Dev', 'Skewness', 'Kurtosis', 'Entropy'] + [f'Hist_Bin_{i}' for i in range(10)]
feature_stats = []

print("Statistics of the computed features:")
print("-" * 50)
for i, name in enumerate(feature_names):
    mean_value = np.mean(features[:, i])
    std_value = np.std(features[:, i])
    feature_stats.append((mean_value, std_value))

    print(f"{name}: Mean = {mean_value:.4f}, Std Dev = {std_value:.4f}")

print("\nNumber of clients where feature value is above the mean:")
print("-" * 50)
for i, name in enumerate(feature_names):
    above_mean = np.sum(features[:, i] > feature_stats[i][0])
    print(f"{name}: {above_mean} clients")

# Compute and print correlations
correlations = np.corrcoef(features, rowvar=False)
print("\nCorrelation matrix of the features:")
print("-" * 50)
print("\t" + "\t".join(feature_names))
for i, name in enumerate(feature_names):
    correlation_values = "\t".join([f"{val:.2f}" for val in correlations[i]])
    print(f"{name}:\t{correlation_values}")

from scipy.stats import skew
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np

# Initialize the client statistics list
client_stats = []

# Loop over each client's data
for client_data, _ in clients_data:
    # Flatten the images
    client_data_flat = client_data.reshape(client_data.shape[0], -1)

    # Compute mean, standard deviation, and skewness
    mean = np.array([np.mean(client_data_flat)])
    std_dev = np.array([np.std(client_data_flat)])
    data_skewness = np.array([skew(client_data_flat, axis=None)])

    # Compute normalized histogram bins
    hist_bins = np.histogram(client_data_flat, bins=5)[0]
    hist_bins = np.array(hist_bins / np.sum(hist_bins))

    # Concatenate the features
    features = np.concatenate((mean, std_dev, data_skewness, hist_bins))
    client_stats.append(features)

# Convert the list to a numpy array
client_stats = np.array(client_stats)

# Determine the optimal number of clusters
X = np.array(client_stats)
distortions = []
K_range = range(2, 7)  # Setting range from 2 to 6
for k in K_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)

optimal_k = distortions.index(min(distortions)) + 2

# Cluster clients based on data distribution similarity
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
cluster_assignments = kmeans.fit_predict(client_stats)

print("Cluster Assignments:", cluster_assignments)

# Silhouette Coefficient
score = silhouette_score(client_stats, cluster_assignments)
print("Silhouette Coefficient:", score)

# Calinski-Harabasz Index
score = calinski_harabasz_score(client_stats, cluster_assignments)
print("Calinski-Harabasz Index:", score)

# Davies-Bouldin Index
score = davies_bouldin_score(client_stats, cluster_assignments)
print("Davies-Bouldin Index:", score)

# Inspecting clusters
client_labels_list = [labels for _, labels in clients_data]
clusters = np.unique(cluster_assignments)

for cluster in clusters:
    # Get indices of clients in this cluster
    client_indices_in_cluster = np.where(cluster_assignments == cluster)[0]

    # Get labels of these clients
    labels_in_cluster = [client_labels_list[i] for i in client_indices_in_cluster]
    flattened_labels = [label for sublist in labels_in_cluster for label in sublist]  # Flatten the list

    # Count the occurrence of each unique label
    unique_labels, counts = np.unique(flattened_labels, return_counts=True)

    # Print the counts for each label in this cluster
    print(f"Cluster {cluster}:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} samples")
    print("--------------------------")

client_stats = []
for client_data, client_labels in clients_data:
    mean = np.mean(client_data)
    std_dev = np.std(client_data)
    client_stats.append((mean, std_dev))

def similarity_metric(stat1, stat2):
    distance = np.sqrt((stat1[0] - stat2[0])**2 + (stat1[1] - stat2[1])**2)
    return distance

random_cluster_assignments = [random.randint(0, total_clusters - 1) for _ in range(total_clients)]
print("Random Cluster Assignments:", random_cluster_assignments)

from sklearn.metrics import silhouette_score
score1 = silhouette_score(client_stats, random_cluster_assignments)
print("Silhouette Coefficient:", score)
from sklearn.metrics import calinski_harabasz_score
score1 = calinski_harabasz_score(client_stats, random_cluster_assignments)
print("Calinski-Harabasz Index:", score)
from sklearn.metrics import davies_bouldin_score
score1 = davies_bouldin_score(client_stats, random_cluster_assignments)
print("Davies-Bouldin Index:", score)

# Inspecting clusters
client_labels_list = [labels for _, labels in clients_data]
clusters = np.unique(random_cluster_assignments)

for cluster in clusters:
    # Get indices of clients in this cluster
    client_indices_in_cluster = np.where(random_cluster_assignments == cluster)[0]

    # Get labels of these clients
    labels_in_cluster = [client_labels_list[i] for i in client_indices_in_cluster]
    flattened_labels = [label for sublist in labels_in_cluster for label in sublist]  # Flatten the list

    # Count the occurrence of each unique label
    unique_labels, counts = np.unique(flattened_labels, return_counts=True)

    # Print the counts for each label in this cluster
    print(f"Cluster {cluster}:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} samples")
    print("--------------------------")


