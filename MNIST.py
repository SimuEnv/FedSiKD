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
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Conv2D, MaxPooling2D,LeakyReLU
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
print(y_train.shape,y_test.shape)
print(x_train.shape)

from sklearn.utils import shuffle

x_train, y_train = shuffle(x_train, y_train, random_state=22)

num_clients = 80
samples_per_client = 3000

X = np.array_split(x_train, num_clients)  # Split the data into 80 subsets
Y = np.array_split(y_train, num_clients)  # Split the labels into 80 subsets

# Ensure that each sample has the desired shape (28, 28, 1)
X = np.array([x.reshape(-1, 28, 28, 1) for x in X])

X = np.array(X)
Y = np.array(Y)
print(X.shape, Y.shape)

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
    metrics=[keras.metrics.SparseCategoricalAccuracy()],)
    acc,loss=model.evaluate(x_test,y_test,verbose=0)
    return acc, loss
def scale_model_weights(weight, scalar):
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

"""**non-similarties Random**"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import time
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

global1 = MODEL()
global_model = global1.build()
comms_round = 2
epoch_teacher = 10
epoch_student = 6
total_clusters = 4
total_clients = 80
clients_per_cluster = int(total_clients / total_clusters)
X = np.reshape(X, (total_clusters, clients_per_cluster, 750, 28, 28, 1))
Y = np.reshape(Y, (total_clusters, clients_per_cluster, 750))
#print(X.shape, Y.shape)

# Step 1: Perform data augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=45,  # Increase the rotation range to [-45, 45] degrees
    width_shift_range=0.5,  # Increase the horizontal shift range to [-0.3, 0.3] of the total width
    height_shift_range=0.5,  # Increase the vertical shift range to [-0.3, 0.3] of the total height
    zoom_range=[0.6, 1.4],  # Increase the zoom range to [0.6, 1.4]
    horizontal_flip= True  # Allow horizontal flipping
)

# ...

# Apply augmented data generation with increased diversity
augmented_X = []
for clstr in range(total_clusters):
    for ind in range(clients_per_cluster):
        client_data = X[clstr][ind]  # Retrieve client data
        augmented_data = data_augmentation.flow(client_data, batch_size=len(client_data), shuffle=False)
        augmented_X.append(augmented_data.next())

client_stats = []  # List to store client statistics
for i in range(total_clients):
    client_data = augmented_X[i]  # Retrieve augmented client data
    # Calculate statistics (mean, standard deviation, etc.)
    mean = np.mean(client_data)
    std_dev = np.std(client_data)
    # Add statistics to the list
    client_stats.append((mean, std_dev))

# ...


# Step 2: Define similarity metric
def similarity_metric(stat1, stat2):
    # Define a simple similarity metric based on statistics (e.g., Euclidean distance)
    distance = np.sqrt((stat1[0] - stat2[0])**2 + (stat1[1] - stat2[1])**2)
    return distance

# Step 3: Cluster clients based on data distribution similarity
from sklearn.cluster import KMeans
# Step 3: Randomly assign clients to clusters
random_cluster_assignments = [random.randint(0, total_clusters - 1) for _ in range(total_clients)]
print("Random Cluster Assignments:", random_cluster_assignments)

# ...

teacher = MODEL()
teacher_model = teacher.build()

from tensorflow.keras import backend as K

for comm_round in range(comms_round):
    global_weights = global_model.get_weights()
    scaled_cluster_weight_list = list()
    index = list({0, 1, 2, 3, 4})
    # random.shuffle(index)
    # print(index)
    print("\nCommunication Round:", comm_round)
    #plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
    for clstr in range(total_clusters):




            # Get the clients and their statistics in the current cluster
        cluster_clients = [client_stats[i] for i in range(len(client_stats)) if random_cluster_assignments[i] == clstr]
        cluster_mean = [stat[0] for stat in cluster_clients]
        cluster_std_dev = [stat[1] for stat in cluster_clients]


        # Show the plot
        #plt.show()
        scaled_local_weight_list = list()
        cluster_model = MODEL().build()
        cluster_model.compile(optimizer=keras.optimizers.Adam(),
                              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                              metrics=[keras.metrics.SparseCategoricalAccuracy()],
                              )

        cluster_model.set_weights(global_weights)
        cluster_weights = cluster_model.get_weights()
        for ind in range(clients_per_cluster):
            if ind == 1:
                teacher_model.compile(optimizer=keras.optimizers.Adam(),
                                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                      metrics=[keras.metrics.SparseCategoricalAccuracy()])
                teacher_model.set_weights(cluster_weights)
                history=teacher_model.fit(X[clstr][ind],Y[clstr][ind], epochs=epoch_teacher,verbose=0)#,validation_data=(x_test,y_test))
                #history = teacher_model.fit(augmented_X[clstr * clients_per_cluster + ind], Y[clstr][ind], epochs=epoch_teacher, verbose=2)

                scaling_factor = 1.0 / clients_per_cluster  # 1/no.ofclients
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
                                      alpha=0.1,
                                      temperature=3)
                student_model.set_weights(student_weights)
                #history = student_model.fit(augmented_X[clstr * clients_per_cluster + ind], Y[clstr][ind], epochs=epoch_student, verbose=2)
                history=student_model.fit(X[clstr][ind],Y[clstr][ind], epochs=epoch_student,verbose=0)#,validation_data=(x_test,y_test))
                scaling_factor = 1.0 / clients_per_cluster  # 1/no.ofclients
                scaled_weights = scale_model_weights(student_model.get_weights(), scaling_factor)
                scaled_local_weight_list.append(scaled_weights)
                K.clear_session()

        average_cluster_weights = sum_scaled_weights(scaled_local_weight_list)
        cluster_model.set_weights(average_cluster_weights)

        scaling_cluster_factor = 1.0 / total_clusters  # 1/no.ofclients
        scaled_cluster_weights = scale_model_weights(cluster_model.get_weights(), scaling_cluster_factor)
        scaled_cluster_weight_list.append(scaled_cluster_weights)
        K.clear_session()
        cluster_loss, cluster_acc = test_model(x_test, y_test, cluster_model, comm_round)
        print("CLUSTER ", clstr, "ACC: ", cluster_acc, "CLUSTER ", clstr, "LOSS: ", cluster_loss)

    average_weights = sum_scaled_weights(scaled_cluster_weight_list)
    print(len(scaled_cluster_weight_list))
    global_model.set_weights(average_weights)
    global_loss, global_acc = test_model(x_test, y_test, global_model, comm_round)
    print("GLOBAL ACCURACY", global_acc, "GLOBAL LOSS", global_loss)
     # Calculate and print the model size
    model_size = global_model.count_params()
    print("Model Size (Parameters):", model_size)

    # Calculate and print the communication efficiency (model update size)
    #model_updates_sizes = [get_model_update_size(model_weights, global_weights) for model_weights in scaled_cluster_weight_list]
    #avg_update_size = sum(model_updates_sizes) / len(model_updates_sizes)
    #print("Average Model Update Size:", avg_update_size)

    global_predictions = global_model.predict(x_test)
    global_predictions_classes = np.argmax(global_predictions, axis=1)

    # Calculate global precision, recall, F1 score, AUC-ROC, and AUC-PR
    global_precision = precision_score(y_test, global_predictions_classes, average='weighted')
    global_recall = recall_score(y_test, global_predictions_classes, average='weighted')
    global_f1_score = f1_score(y_test, global_predictions_classes, average='weighted')
    global_auc_roc = roc_auc_score(y_test, global_predictions, multi_class='ovr')
    #global_auc_pr = average_precision_score(y_test, global_predictions, average='weighted')

    # Print the metrics
    print("GLOBAL PRECISION:", global_precision)
    print("GLOBAL RECALL:", global_recall)
    print("GLOBAL F1 SCORE:", global_f1_score)
    print("GLOBAL AUC-ROC:", global_auc_roc)
    #print("GLOBAL AUC-PR:", global_auc_pr)

    start_time = time.time()
    global_loss, global_acc = test_model(x_test, y_test, global_model, comm_round)
    end_time = time.time()
    time_to_test = end_time - start_time
    print("Time to Test:", time_to_test, "seconds")

print(global_loss)
print(global_acc)

import matplotlib.pyplot as plt

# Assuming you already have the cluster assignments in 'cluster_assignments'
# Assuming you have the client statistics in 'client_stats'
# Assuming 'total_clusters' is defined

# Create a list of colors for each cluster
colors = ['red', 'blue', 'green', 'purple']

"""**Similarity-based Assigenment federated edge learning and knowledge distillation **"""

global1 = MODEL()
global_model = global1.build()
comms_round = 2
epoch_teacher = 10
epoch_student = 6
total_clusters = 4
total_clients = 80
clients_per_cluster = int(total_clients / total_clusters)
X = np.reshape(X, (total_clusters, clients_per_cluster, 750, 28, 28, 1))
Y = np.reshape(Y, (total_clusters, clients_per_cluster, 750))
print(X.shape, Y.shape)

# Step 1: Perform data augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=45,  # Increase the rotation range to [-45, 45] degrees
    width_shift_range=0.5,  # Increase the horizontal shift range to [-0.3, 0.3] of the total width
    height_shift_range=0.5,  # Increase the vertical shift range to [-0.3, 0.3] of the total height
    zoom_range=[0.6, 1.4],  # Increase the zoom range to [0.6, 1.4]
    horizontal_flip= True  # Allow horizontal flipping
)

# Apply augmented data generation with increased diversity
augmented_X = []
for clstr in range(total_clusters):
    for ind in range(clients_per_cluster):
        client_data = X[clstr][ind]  # Retrieve client data
        #augmented_data = data_augmentation.flow(client_data, batch_size=len(client_data), shuffle=False)
        #augmented_X.append(augmented_data.next())

client_stats = []  # List to store client statistics
for i in range(total_clients):
    client_data = X[:, i // clients_per_cluster, i % clients_per_cluster]  # Retrieve client data
    # Calculate statistics (mean, standard deviation, etc.)
    mean = np.mean(client_data)
    std_dev = np.std(client_data)
    # Add statistics to the list
    client_stats.append((mean, std_dev))

# Step 2: Define similarity metric
def similarity_metric(stat1, stat2):
    # Define a simple similarity metric based on statistics (e.g., Euclidean distance)
    distance = np.sqrt((stat1[0] - stat2[0])**2 + (stat1[1] - stat2[1])**2)
    return distance

# Step 3: Cluster clients based on data distribution similarity
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=total_clusters, random_state=0)
cluster_assignments = kmeans.fit_predict(client_stats)

# Print cluster assignments for verification
print("Cluster Assignments:", cluster_assignments)

# ...

teacher = MODEL()
teacher_model = teacher.build()

from tensorflow.keras import backend as K

for comm_round in range(comms_round):
    global_weights = global_model.get_weights()
    scaled_cluster_weight_list = list()
    index = list({0, 1, 2, 3, 4})
    # random.shuffle(index)
    # print(index)
    print("\nCommunication Round:", comm_round)
    plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
    for clstr in range(total_clusters):




            # Get the clients and their statistics in the current cluster
        cluster_clients = [client_stats[i] for i in range(len(client_stats)) if cluster_assignments[i] == clstr]
        cluster_mean = [stat[0] for stat in cluster_clients]
        cluster_std_dev = [stat[1] for stat in cluster_clients]

            # Plot the cluster points with a unique color and label
        #plt.scatter(cluster_mean, cluster_std_dev, color=colors[clstr], label=f'Cluster {clstr}')

        # Add labels and legend
        '''
        plt.xlabel('Mean')
        plt.ylabel('Standard Deviation')
        plt.legend()
        plt.title('Client Clustering based on Mean and Standard Deviation')
        plt.grid(True)
        '''
        # Show the plot
        #plt.show()


        scaled_local_weight_list = list()
        cluster_model = MODEL().build()
        cluster_model.compile(optimizer=keras.optimizers.Adam(),
                              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                              metrics=[keras.metrics.SparseCategoricalAccuracy()],
                              )

        cluster_model.set_weights(global_weights)
        cluster_weights = cluster_model.get_weights()
        for ind in range(clients_per_cluster):
            if ind == 1:
                teacher_model.compile(optimizer=keras.optimizers.Adam(),
                                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                      metrics=[keras.metrics.SparseCategoricalAccuracy()])
                teacher_model.set_weights(cluster_weights)
                #history = teacher_model.fit(augmented_X[clstr * clients_per_cluster + ind], Y[clstr][ind], epochs=epoch_teacher, verbose=2)
                history=teacher_model.fit(X[clstr][ind],Y[clstr][ind], epochs=epoch_teacher,verbose=0)#,validation_data=(x_test,y_test))
                scaling_factor = 1.0 / clients_per_cluster  # 1/no.ofclients
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
                                      alpha=0.1,
                                      temperature=3)
                student_model.set_weights(student_weights)
                #history = student_model.fit(augmented_X[clstr * clients_per_cluster + ind], Y[clstr][ind], epochs=epoch_student, verbose=2)
                history=student_model.fit(X[clstr][ind],Y[clstr][ind], epochs=epoch_student,verbose=0)#,validation_data=(x_test,y_test))
                scaling_factor = 1.0 / clients_per_cluster  # 1/no.ofclients
                scaled_weights = scale_model_weights(student_model.get_weights(), scaling_factor)
                scaled_local_weight_list.append(scaled_weights)
                K.clear_session()

        average_cluster_weights = sum_scaled_weights(scaled_local_weight_list)
        cluster_model.set_weights(average_cluster_weights)

        scaling_cluster_factor = 1.0 / total_clusters  # 1/no.ofclients
        scaled_cluster_weights = scale_model_weights(cluster_model.get_weights(), scaling_cluster_factor)
        scaled_cluster_weight_list.append(scaled_cluster_weights)
        K.clear_session()
        cluster_loss, cluster_acc = test_model(x_test, y_test, cluster_model, comm_round)
        print("CLUSTER ", clstr, "ACC: ", cluster_acc, "CLUSTER ", clstr, "LOSS: ", cluster_loss)

    average_weights = sum_scaled_weights(scaled_cluster_weight_list)
    print(len(scaled_cluster_weight_list))
    # global_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    global_model.set_weights(average_weights)
    # val_acc,val_loss=global_model.evaluate(x_train,y_train)
    # print(val_loss,val_acc)
    global_loss, global_acc = test_model(x_test, y_test, global_model, comm_round)
    print("GLOBAL ACCURACY", global_acc, "GLOBAL LOSS", global_loss)
     # Calculate and print the model size
    model_size = global_model.count_params()
    print("Model Size (Parameters):", model_size)



    global_predictions = global_model.predict(x_test)
    global_predictions_classes = np.argmax(global_predictions, axis=1)

    # Calculate global precision, recall, F1 score, AUC-ROC, and AUC-PR
    global_precision = precision_score(y_test, global_predictions_classes, average='weighted')
    global_recall = recall_score(y_test, global_predictions_classes, average='weighted')
    global_f1_score = f1_score(y_test, global_predictions_classes, average='weighted')
    global_auc_roc = roc_auc_score(y_test, global_predictions, multi_class='ovr')
    #global_auc_pr = average_precision_score(y_test, global_predictions, average='weighted')

    # Print the metrics
    print("GLOBAL PRECISION:", global_precision)
    print("GLOBAL RECALL:", global_recall)
    print("GLOBAL F1 SCORE:", global_f1_score)
    print("GLOBAL AUC-ROC:", global_auc_roc)
    #print("GLOBAL AUC-PR:", global_auc_pr)

    # Calculate and print the communication efficiency (model update size)
    #model_updates_sizes = [get_model_update_size(model_weights, global_weights) for model_weights in scaled_cluster_weight_list]
    #avg_update_size = sum(model_updates_sizes) / len(model_updates_sizes)
    #print("Average Model Update Size:", avg_update_size)

    start_time = time.time()
    global_loss, global_acc = test_model(x_test, y_test, global_model, comm_round)
    end_time = time.time()
    time_to_test = end_time - start_time
    print("Time to Test:", time_to_test, "seconds")

"""**FedAVG**

**FedProx**
"""

print(global_loss)

print(global_acc)

"""**FedAVG**"""





print(global_acc)
