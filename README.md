# FIDEL-Async: A Federated learning framework for neural network training

The framework is designed to train a neural network task on various fog nodes asynchronously. The fog nodes consist of resources constrained devices such as Raspberry Pi.

The standard federated learning paradigm with fedAvg implementation is done on continuously generating datasets. 

Firstly, the server starts training by initializing the model and asking connected nodes to train the model on their local datasets. The server aggregates the locally trained models with or without complete arrival of updates. The training continues till the global model converges, or the expected model is created. 

## Structure and prerequisite

The project contains code for the client and server that need to be run on respective devices. The clients are resource-constrained devices such as Raspberry Pi, and a server can be computers/laptops.  

All participating nodes should have docker installed so that appropriate images can be downloaded. Otherwise, the user has to create a docker image with the code given in the project. 

The server and clients should have access to the ip address of the MQTT broker for the subscription. 

Additionally, every client should have data in the data folder that will be used for local training. 

# How to run the code

## On the server side
1. Run requirement.txt
2. Update broker id/port and authentication credentials for MQTT in server.py
3. Define the neural network in the createInitialModel() function.
4. Run server.py


## On the client side 

### To run directly on the client 
1. Check/change docker image in docker-compose.yaml file
2. Update broker id/port and authentication credentials for MQTT in client.py
3. Run docker-compose.yaml 

The compose will download and execute images on the fog device. 


### To create a new image

In case of a given image is not working, then the image can be created with docker-compose.yaml file. 

Edit compose file on the image tag with build code. Then run docker-compose.yaml.
```
build:
    context: .
    dockerfile: Dockerfile
```

As the client gets connected to the server, the server will prompt the connection. Thereafter, the server can start training by pressing any key.

## Result  and report

The server will have evaluation results(accuracy, loss) and latest global model in Models folder.  

Similarly, every client will have locally trained results and latest model in the data folder. 





