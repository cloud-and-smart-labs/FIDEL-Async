import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import json
import paho.mqtt.client as mqtt
import time
import yaml
import ast

import tensorflow as tf  
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv1D, Dropout, Reshape, MaxPooling1D, Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
# from collections import deque

"""
The server need to be executed first so that it can take care of no of connected 
devices based on connect status
"""
borker_address = "192.168.0.174" #"192.168.0.174"
borker_port = 1883
keep_alive = 8000
topic_train = "train"
topic_aggregate = "aggregate"
topic_initilize = "initlize"
client_name = "server"
client_queue = []
client_queue_size_per_iteration = []
no_of_node = 1
no_connected_device = 0
no_iteration = 60
iter = 0 



#Utility function to convert model(h5) into string
def encode_file(file_name):
    with open(file_name,'rb') as file:
        encoded_string = base64.b64encode(file.read())
    return encoded_string

def getTestDataset():

    # print("Inside testdata")

    database = sio.loadmat('../IEEE for federated learning/data_base_all_sequences_random.mat')
    
    X = database['Data_test_2']
    y = database['label_test_2']

    return X, y

def saveLearntMetrices(modelName):
     
    # print("Inside save paramertes")
    model = load_model(modelName)
    X_test, y_test = getTestDataset()
    y_test = to_categorical(y_test)
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Agreegated model with all data loss : {} and accuracy : {}".format(score[0], score[1]))

    with open('Models/globalMetrics.txt','r+') as f:
        trainMetrics = json.load(f)
        trainMetrics['accuracy'].append(score[1])
        trainMetrics['loss'].append(score[0])
        f.seek(0) 
        f.truncate()
        f.write(json.dumps(trainMetrics))


# This fucntion aggregates all models parmeters and create new optimized model 
def optimiseModels():
    global client_queue
    global iter
    clients_list = len(client_queue)
    print("lenght of queue ", clients_list)
    if clients_list == 0 :
        iter = iter - 1
        return 

    models = list()
    for param in client_queue:
        with open("Models/temp_model.h5","wb") as file:
            file.write(base64.b64decode(param))
        model = load_model("Models/temp_model.h5")
        models.append(model)
    client_queue = []
    client_queue_size_per_iteration.append(clients_list)
    print(client_queue_size_per_iteration)
    weights = [model.get_weights() for model in models]

    new_weights = list()
    for weights_list_tuple in zip(*weights):
        new_weights.append(
            np.array([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)]))

    new_model = models[0]
    new_model.set_weights(new_weights)
    new_model.save("Models/model.h5")

    print("Averaged over all models - optimised model saved!")
    saveLearntMetrices("Models/model.h5")

def createInitialModel():

    # K.clear_session()
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(512,)))
    # model.add(tf.keras.layers.Dense(8, activation='softmax'))
    # model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    # model.save('Models/model.h5')

    # K.clear_session()
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(512,)))
    # model.add(tf.keras.layers.Dense(16, activation='relu'))
    # model.add(tf.keras.layers.Dense(8, activation='softmax'))
    # model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    # model.save('Models/model.h5')

    K.clear_session()
    model = Sequential()
    model.add(Reshape((512,1), input_shape=(512,1)))
    model.add(Conv1D(filters=8, kernel_size=3,padding='same', activation='relu', input_shape=(512,1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=4, kernel_size=3, padding='same',  activation='relu'))
    model.add(Flatten())
    model.add(Dense(8, activation='softmax'))
    model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model.save('Models/model.h5')

    model.summary()


def initlizeGlobalMetrics():
    metric = {'accuracy' : [], 'loss' : []}
    
    with open('Models/globalMetrics.txt', "w") as f:
        f.write(json.dumps(metric))

def visualizeTraining():
    path = "/home/aditya/Desktop/ANN_Sync_result_1_soft/"
    print("Inside visualize")    
    fp =  open(path + 'server/globalMetrics.txt','r')
    gloablMetrics = json.load(fp)

    f = plt.figure(1)
    plt.plot(gloablMetrics['accuracy'], label='Test')
    plt.title('Global model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Rounds')
    plt.legend()
    f.show()
    
    g = plt.figure(2)
    plt.plot(gloablMetrics['loss'], label='Test')
    plt.title('Global loss')
    plt.ylabel('Loss')
    plt.xlabel('Rounds')
    plt.legend()
    g.show()


    h = plt.figure(3)
    for i in range(0, n):
        with open(path + "client" + str(i) + '/metrics.txt', 'r') as f:
            trainMetrics = json.load(f)
            plt.plot(trainMetrics['accuracy'], label='Device ' + str(i) )
    plt.plot(gloablMetrics['accuracy'], '--b', label='Server')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Rounds')
    plt.legend()
    h.show()

    s = plt.figure(4)
    for i in range(0, n):
        with open(path + "client" + str(i) + '/metrics.txt', 'r') as f:
            trainMetrics = json.load(f)
            plt.plot(trainMetrics['loss'], label='Device ' + str(i) )
    plt.plot(gloablMetrics['loss'], '--b', label='Server')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Rounds')
    plt.legend()
    s.show()


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(topic_aggregate, 2)
        client.subscribe("connected", 2)
        client.subscribe("disconnected", 2)
        
    else:
        print("Failed to connect, return code ", rc)

def on_message(client, userdata, msg):
    global no_connected_device
    if msg.topic == topic_aggregate :
        print("Training completed Client ", client)
        client_queue.append(msg.payload)
    if msg.topic == "connected" :
        no_connected_device = no_connected_device + 1
        print("Connected, devices are ", no_connected_device)
    if msg.topic == "disconnected":
        no_connected_device = no_connected_device - 1
        print("Disconnected, devices are ", no_connected_device)

def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscribe to aggregate", client)



mqttc = mqtt.Client(client_name)       

# Assign event callbacks
mqttc.on_connect = on_connect
mqttc.on_message = on_message
mqttc.on_subscribe = on_subscribe
mqttc.username_pw_set("server","cloud")
mqttc.connect(borker_address, borker_port, keep_alive)

createInitialModel()
initlizeGlobalMetrics()
saveLearntMetrices("Models/model.h5")
mqttc.loop_start()
input("Press any key to start training")
print("Total no of conntected devices are ", no_connected_device)
mqttc.publish(topic_initilize, "I am initlizing")
while True:
    try :
        send_message = {'model_file' : encode_file("Models/model.h5"), 
                        'epochs' : 16,
                        'batch' : 60}
        if iter < no_iteration : 
            mqttc.publish(topic_train, payload = str(send_message))
            print("Current iteration : ", iter)
            iter = iter + 1
            time.sleep(60)
            optimiseModels()

    except KeyboardInterrupt:
        print("Keyboard intruupt")
        with open("client_q", "w") as fp:
            json.dump(client_queue_size_per_iteration, fp)
        break
mqttc.loop_stop()
mqttc.disconnect()
