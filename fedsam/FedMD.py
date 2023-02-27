import numpy as np
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from train_resnet20 import train_resnet20
from data_utils import generate_alignment_data
from Neural_Networks import remove_last_layer
from torch.utils.data import random_split, DataLoader
from copy import deepcopy
from torchvision import transforms
import torch
import torch.nn as nn
import sklearn.model_selection

"""_summary_
Algoritmo FedMD:
1) ogni partecipante pretraina il suo modello sul dataset pubblico (cifar10)
2) ogni partecipante traina il suo modello sul proprio dataset privato (subset di 120 samples diversi di cifar100 su 6 classi definite per tutti)
3) calcola l'upper bound per ogni modello (meglio di cosi non si può ottenere)

"""

def get_prediction_on_train(resnet20, alignment_data, batch_size=32):
  dataloader = DataLoader(alignment_data, batch_size=batch_size, shuffle=False, num_workers=2)
  #print(f"dataloader = {dataloader}")
  net = resnet20.to('cuda') # this will bring the network to GPU if DEVICE is cuda
  net.train(False) # Set Network to evaluation mode

  total_values = torch.tensor([])
  for images, labels in dataloader:
    images = images.to('cuda')
    labels = labels.to('cuda')

      # Forward Pass
    images=images.float()
    outputs = net(images)
    # Sample 1 -> [2.3, 4.1, 4.3, ..., ]
    # Sample .. ->[3.1, 1.3, 2.4, ..., ]

    # Get predictions
    total_values = torch.cat((total_values.to('cuda'), outputs.data))
  # Calculate Accuracy
  return total_values

def get_prediction_on_testing(resnet20, alignment_data, batch_size=32):
  dataloader = DataLoader(alignment_data, batch_size=batch_size, shuffle=False, num_workers=2)
  #print(f"dataloader = {dataloader}")
  net = resnet20.to('cuda') # this will bring the network to GPU if DEVICE is cuda
  net.train(False) # Set Network to evaluation mode

  running_corrects = 0
  for images, labels in dataloader:
      images = images.to('cuda')
      labels = labels.to('cuda')

      # Forward Pass
      images=images.float()
      outputs = net(images)
      # Sample 1 -> [2.3, 4.1, 4.3, ..., ]
      # Sample .. ->[3.1, 1.3, 2.4, ..., ]
      _,preds = torch.max(outputs.data, 1)
      running_corrects +=torch.sum(preds==labels.data).data.item()

      # Get predictions
      acc = running_corrects /float(len(alignment_data))

  # Calculate Accuracy
  return acc

class FedMD():
    def __init__(self, parties, public_dataset, 
                 private_data, total_private_data,  
                 private_test_data, N_alignment,
                 N_rounds, 
                 N_logits_matching_round, logits_matching_batchsize, 
                 N_private_training_round, private_training_batchsize):
        
        self.N_parties = len(parties)
        self.public_dataset = public_dataset
        self.private_data = private_data
        self.private_test_data = private_test_data
        self.N_alignment = N_alignment
        
        self.N_rounds = N_rounds
        self.N_logits_matching_round = N_logits_matching_round
        self.logits_matching_batchsize = logits_matching_batchsize
        self.N_private_training_round = N_private_training_round
        self.private_training_batchsize = private_training_batchsize
        
        self.collaborative_parties = []
        self.init_result = []

        
        print("start model initialization: ")
        for i in range(self.N_parties):
          model_A_twin = None
          model_A_twin = deepcopy(parties[i])
          print(f"parties = {i}")
          #print(f"model = {model_A_twin.__class__.__name__}")

          if model_A_twin.__class__.__name__ == "Resnet20":

            #print(f' {np.shape(private_data[0]["X"])}') #(120,32,32,3)
            trainset = []
            for i, img in enumerate(private_data[0]["X"]):
              convert_tensor = transforms.ToTensor()
              trainset.append((convert_tensor(img).type(dtype=torch.double), private_data[0]["y"][i]))
              
            val_size = 20
            train_size = len(trainset) - val_size
            trainset, validset = random_split(trainset, [train_size, val_size])

            testset = []
            for i, img in enumerate(private_test_data["X"]):
              convert_tensor = transforms.ToTensor()
              testset.append((convert_tensor(img).type(dtype=torch.double), private_test_data["y"][i]))

            print("inizio train sul private data")
            train_accuracies, train_losses, val_accuracies, val_losses = train_resnet20(model_A_twin, trainset, validset, testset, epochs = 1, batch_size= 8)
            print("fine train sul private data")
              
              #print(f"modello completo = {model_A_twin}")
              
              #remove last layer
            model_A = deepcopy(model_A_twin)
            del model_A.linear
            self.collaborative_parties.append({"model_logits": model_A, 
                                              "model_classifier": model_A_twin,
                                              "model_weights": model_A.state_dict()})
            self.init_result.append({ "val_acc": val_accuracies,
                                            "train_acc": train_accuracies,
                                            "val_loss": val_losses,
                                            "train_loss": train_losses
                                            })    
          else: 
            model_A_twin = None
            model_A_twin = clone_model(parties[i])
            #print("set weights")
            model_A_twin.set_weights(parties[i].get_weights())
            #print("compile")
            model_A_twin.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3), 
                                  loss = "sparse_categorical_crossentropy",
                                  metrics = ["accuracy"])
            #print("fit")
            model_A_twin.fit(private_data[i]["X"], private_data[i]["y"],
                              batch_size = 32, epochs = 25, shuffle=True, verbose = 0,
                              validation_data = [private_test_data["X"], private_test_data["y"]],
                              callbacks=[EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=10)]
                              )
          
              #si rimuove l'ultimo layer per essere più generali possibili e adattarsi alle specifiche del problema
              #se non lo togliessi avrei un numero di classi definito che sarebbe diverso per ogni modello 
            model_A = remove_last_layer(model_A_twin, loss="mean_absolute_error")

            self.collaborative_parties.append({"model_logits": model_A, 
                                              "model_classifier": model_A_twin,
                                              "model_weights": model_A_twin.get_weights()})

            self.init_result.append({"val_acc": model_A_twin.history.history['val_accuracy'],
                                      "train_acc": model_A_twin.history.history['accuracy'],
                                      "val_loss": model_A_twin.history.history['val_loss'],
                                      "train_loss": model_A_twin.history.history['loss'],
                                      })
            print(f'train_acc = {model_A_twin.history.history["accuracy"]}' )
              
          del model_A, model_A_twin
        #END FOR LOOP
        
        #total private data = somma di tutti i samples di tutti i partecipanti ( ma le labels coinvolte sono sempre le stesse 6 private)
        
       
        print("calculate the theoretical upper bounds for participants: ")
        
        self.upper_bounds = []
        self.pooled_train_result = []

        for model in parties:
          model_ub = deepcopy(model)
          if model.__class__.__name__ == "Resnet20":
            print(f'size tot_private_data = {np.shape(total_private_data["X"])}') #(240,32,32,3)
            trainset = []
            for i, img in enumerate(total_private_data["X"]):
              convert_tensor = transforms.ToTensor()
              trainset.append((convert_tensor(img).type(dtype=torch.double), total_private_data["y"][i]))
  
            val_size = 40
            train_size = len(trainset) - val_size
            trainset, validset = random_split(trainset, [train_size, val_size])
      
            testset = []
            print(f'size tot_private_labels = {np.shape(private_test_data["X"])}') #(600,32,32,3)
            for i, img in enumerate(private_test_data["X"]):
              convert_tensor = transforms.ToTensor()
              testset.append((convert_tensor(img).type(dtype=torch.double), private_test_data["y"][i]))

            print("inizio train upper bound su total_private_data")
            train_accuracies, train_losses, val_accuracies, val_losses = train_resnet20(model, trainset, validset, testset, epochs = 1, batch_size= 32)
            print("fine train upper bound su total_private_data")

            self.upper_bounds.append({ "val_acc": val_accuracies})
            self.pooled_train_result.append({ "val_acc": val_accuracies,
                                                 "acc": train_accuracies })
          else:
            model_ub.set_weights(model.get_weights())
            model_ub.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3),
                            loss = "sparse_categorical_crossentropy", 
                            metrics = ["accuracy"])
            
            model_ub.fit(total_private_data["X"], total_private_data["y"],
                        batch_size = 32, epochs = 50, shuffle=True, verbose = 0, 
                        validation_data = [private_test_data["X"], private_test_data["y"]],
                        callbacks=[EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=10)])
            
            self.upper_bounds.append(model_ub.history.history["val_accuracy"][-1])
            self.pooled_train_result.append({"val_acc": model_ub.history.history["val_accuracy"], 
                                            "acc": model_ub.history.history["accuracy"]})
            
          del model_ub   

        print("the upper bounds are:", self.upper_bounds)
    
    def collaborative_training(self):
      # start collaborating training    
      collaboration_performance = {i: [] for i in range(self.N_parties)}
      r = 0
      while True:
          # At beginning of each round, generate new alignment dataset
          #prende un sottoinsieme (5000) del dataset pubblico (CIFAR10)
        alignment_data = generate_alignment_data(self.public_dataset["X"], 
                                                    self.public_dataset["y"],
                                                    self.N_alignment)
          
        print("round ", r)
          
          #print("update logits ... ")
          #shape logits = (5000x16)
          # [0.156,0.145,0.1564] + [0.1,..,..] + .... + [0.1,0.2,0.3] -> fa la media per ogni sample
          # (tot samples = 5000 = alignment_data)
          #considerando le label delle classi pubbliche (10) + private (6)
          # update logits
        logits = 0
        for d in self.collaborative_parties:
          #print(f' modello = {d["model_logits"]}')

          if d["model_logits"].__class__.__name__ == "Resnet20":
            d["model_logits"].load_state_dict(d["model_weights"])

            print(f'alignment data = {np.shape(alignment_data["X"])}') #(5000,32,32,3)
            trainset = []
            for i, img in enumerate(alignment_data["X"]):
              convert_tensor = transforms.ToTensor()
              trainset.append((convert_tensor(img).type(dtype=torch.double), alignment_data["y"][i]))
            
            print("predizioni sul training")
            preds = get_prediction_on_train(d["model_logits"], trainset)
            #preds, _ = torch.max(preds, 1)
            preds = preds.to('cpu')
            preds = preds.numpy()
            print(f"preds = {np.shape(preds)}")
            print(f"logits = {np.shape(logits)}")
            logits += preds
            print(f"logits = {np.shape(logits)}")
          else:
            d["model_logits"].set_weights(d["model_weights"])
            logits += d["model_logits"].predict(alignment_data["X"], verbose = 0)
            #print(f"preds = {np.shape(preds)}")
            print(f"logits = {np.shape(logits)}")
            
       
        logits /= self.N_parties
        print(f"logits tot = {np.shape(logits)}")
          
          
          # test performance
          #print("test performance ... ")
          
        for index, d in enumerate(self.collaborative_parties):
          if d["model_logits"].__class__.__name__ == "Resnet20":
            print(f' {np.shape(self.private_test_data["X"])}')
            testset = []
            for i, img in enumerate(self.private_test_data["X"]):
              convert_tensor = transforms.ToTensor()
              testset.append((convert_tensor(img).type(dtype=torch.double), self.private_test_data["y"][i]))
            
            print("predizioni sul testing")
            accuracy = get_prediction_on_testing(d["model_classifier"], testset)
            print(f" acc = {accuracy}")
            collaboration_performance[index].append(accuracy)
              
          else:
            y_pred = d["model_classifier"].predict(self.private_test_data["X"], verbose = 0).argmax(axis = 1)
            collaboration_performance[index].append(np.mean(self.private_test_data["y"] == y_pred))
          
              
        r+= 1
        if r > self.N_rounds:
          break
              
              
        print("updates models ...")
        for index, d in enumerate(self.collaborative_parties):
              #print("model {0} starting alignment with public logits... ".format(index))
              
          weights_to_use = None
          weights_to_use = d["model_weights"]
          if d["model_logits"].__class__.__name__ == "Resnet20":
            print(f' {np.shape(alignment_data["X"])}')
            d["model_logits"].load_state_dict(weights_to_use)
            trainset = []
            for i, img in enumerate(alignment_data["X"]):
              convert_tensor = transforms.ToTensor()
              trainset.append((convert_tensor(img).type(dtype=torch.double), alignment_data["y"][i]))

            val_size = 500
            train_size = len(trainset) - val_size
            trainset, validset = random_split(trainset, [train_size, val_size])

            print("inizio train sugli alignment data")
            train_accuracies, train_losses, val_accuracies, val_losses = train_resnet20(d["model_logits"], trainset, validset, logits, epochs = 1, batch_size= 8)
            print("fine train sugli alignment data")

            weights_to_use = None
            weights_to_use = d["model_weights"]
            d["model_logits"].load_state_dict(weights_to_use)

            print(f' {np.shape(self.private_data[index]["X"])}')
            trainset = []
            for i, img in enumerate(self.private_data[index]["X"]):
              convert_tensor = transforms.ToTensor()
              trainset.append((convert_tensor(img).type(dtype=torch.double), self.private_data[index]["y"][i]))
            val_size = 20
            train_size = len(trainset) - val_size
            trainset, validset = random_split(trainset, [train_size, val_size])
       
            print("inizio train su private data")
            train_accuracies, train_losses, val_accuracies, val_losses = train_resnet20(d["model_classifier"], trainset,validset, self.private_data[index]["y"], epochs = 1, batch_size= 8)
            print("fine train su private data")

            d["model_weights"] = d["model_classifier"].state_dict()

          else:
            d["model_logits"].set_weights(weights_to_use)
            d["model_logits"].fit(alignment_data["X"], logits, 
                                    batch_size = self.logits_matching_batchsize,  
                                    epochs = self.N_logits_matching_round, 
                                    shuffle=True, verbose = 0)
                                    
            d["model_weights"] = d["model_logits"].get_weights()
              # print("model {0} done alignment".format(index))

              #print("model {0} starting training with private data... ".format(index))
            weights_to_use = None
            weights_to_use = d["model_weights"]
            d["model_classifier"].set_weights(weights_to_use)
            d["model_classifier"].fit(self.private_data[index]["X"], 
                                        self.private_data[index]["y"],       
                                        batch_size = self.private_training_batchsize, 
                                        epochs = self.N_private_training_round, 
                                        shuffle=True, verbose = 0)

            d["model_weights"] = d["model_classifier"].get_weights()
              #print("model {0} done private training. \n".format(index))
          #END FOR LOOP
      
      #END WHILE LOOP
      return collaboration_performance

