import torch 
import torch.nn.init as init
import torchvision
import torch.nn as nn
from torch.utils.data import random_split
from torch.autograd import Variable
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn
from torchvision import transforms
from torchvision.models import alexnet, vgg16, resnet18, resnet50
from PIL import Image
import numpy as np
from tqdm import tqdm
from train_resnet20 import train_resnet20
from tensorflow.keras.models import Model, Sequential, clone_model, load_model
from tensorflow.keras.layers import Input, Dense, add, concatenate, Conv2D,Dropout,\
BatchNormalization, Flatten, MaxPooling2D, AveragePooling2D, Activation, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

NUM_GROUP = 8
# GroupNorm takes number of groups to divide the channels in and the number of channels to expect
# in the input. 

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
    
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(NUM_GROUP, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(NUM_GROUP, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                  nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                  nn.GroupNorm(NUM_GROUP, self.expansion * planes)  )
          
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out



class Resnet20(nn.Module):
    """implementation of ResNet20 with GN layers"""
    def __init__(self, lr, device, n_classes, input_shape = (28,28)):
    #def __init__(self, num_classes=100):
      super(Resnet20, self).__init__()
      block = BasicBlock
      num_blocks = [3,3,3]
      self.num_classes = n_classes
      self.device = device
      self.lr = lr
      self.in_planes = 16
      self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
      self.gn1 = nn.GroupNorm(NUM_GROUP, 16)
      self.relu = nn.ReLU()
      
      self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
      self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
      self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
      self.linear = nn.Linear(64, n_classes)

      self.apply(_weights_init)
      #self.weights = self.apply(_weights_init)
      self.size = self.model_size()
      print(f"size definito {self.size}")

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = torch.nn.functional.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        try:
            out = self.linear(out)
        except:
            out = out
            
        return out
      
    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size
        
    def summary(self):
        return "summary"



def cnn_3layer_fc_model(n_classes,n1 = 128, n2=192, n3=256, dropout_rate = 0.2,input_shape = (28,28)):
    model_A, x = None, None
     
    x = Input(input_shape)
    if len(input_shape)==2: 
        y = Reshape((input_shape[0], input_shape[1], 1))(x)
    else:
        y = Reshape(input_shape)(x)
    y = Conv2D(filters = n1, kernel_size = (3,3), strides = 1, padding = "same", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size = (2,2), strides = 1, padding = "same")(y)

    y = Conv2D(filters = n2, kernel_size = (2,2), strides = 2, padding = "valid", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Conv2D(filters = n3, kernel_size = (3,3), strides = 2, padding = "valid", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    #y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Flatten()(y)
    y = Dense(units = n_classes, activation = None, use_bias = False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-3))(y)
    y = Activation("softmax")(y)


    model_A = Model(inputs = x, outputs = y)

    model_A.compile(optimizer=tf.keras.optimizers.Adam(learning_rate  = 1e-3), 
                        loss = "sparse_categorical_crossentropy",
                        metrics = ["accuracy"])
    return model_A
  
def cnn_2layer_fc_model(n_classes,n1 = 128, n2=256, dropout_rate = 0.2,input_shape = (28,28)):
    model_A, x = None, None
    
    x = Input(input_shape)
    if len(input_shape)==2: 
        y = Reshape((input_shape[0], input_shape[1], 1))(x)
    else:
        y = Reshape(input_shape)(x)
    y = Conv2D(filters = n1, kernel_size = (3,3), strides = 1, padding = "same", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size = (2,2), strides = 1, padding = "same")(y)


    y = Conv2D(filters = n2, kernel_size = (3,3), strides = 2, padding = "valid", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    #y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Flatten()(y)
    y = Dense(units = n_classes, activation = None, use_bias = False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-3))(y)
    y = Activation("softmax")(y)


    model_A = Model(inputs = x, outputs = y)

    model_A.compile(optimizer=tf.keras.optimizers.Adam(learning_rate  = 1e-3), 
                        loss = "sparse_categorical_crossentropy",
                        metrics = ["accuracy"])
    return model_A


def remove_last_layer(model, loss = "mean_absolute_error"):
    """
    Input: Keras model, a classification model whose last layer is a softmax activation
    Output: Keras model, the same model with the last softmax activation layer removed,
        while keeping the same parameters 
    """
    
    new_model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
    new_model.set_weights(model.get_weights())
    new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate  = 1e-3), 
                      loss = loss)
    
    return new_model


def train_models(models, X_train, y_train, X_test, y_test, 
                 save_dir = "./", save_names = None,
                 early_stopping = True, min_delta = 0.001, patience = 3, 
                 batch_size = 128, epochs = 20, is_shuffle=True, verbose = 1
                ):
    '''
    Train an array of models on the same dataset. 
    We use early termination to speed up training. 
    '''
    
    print(f"{np.shape(X_train)}") 
    print(f"{np.shape(y_train)}") 

    trainset = []
    for i, img in enumerate(X_train):
        convert_tensor = transforms.ToTensor()
        trainset.append((convert_tensor(img).type(dtype=torch.double), y_train[i]))
    
    val_size = 5000
    train_size = len(trainset) - val_size
    trainset, validset = random_split(trainset, [train_size, val_size])
        

    testset = []
    for i, img in enumerate(X_test):
        convert_tensor = transforms.ToTensor()
        testset.append((convert_tensor(img).type(dtype=torch.double), y_test[i]))
        
    print(f"train = {np.shape(trainset)}") 
    print(f"test = {np.shape(testset)}") 

    resulting_val_acc = []
    record_result = []
    for n, model in enumerate(models):
        print("Training model ", n)
        if early_stopping:
            if model.__class__.__name__ == "Resnet20":
                train_accuracies, train_losses, val_accuracies, val_losses = train_resnet20(model, trainset, validset, testset, epochs = epochs, batch_size= batch_size)
            else:
                model.fit(X_train, y_train, 
                        validation_data = [X_test, y_test],
                        callbacks=[EarlyStopping(monitor='val_accuracy', min_delta=min_delta, patience=patience)],
                        batch_size = batch_size, epochs = epochs, shuffle=is_shuffle, verbose = verbose
                        )
        else:
            model.fit(X_train, y_train, 
                      validation_data = [X_test, y_test],
                      batch_size = batch_size, epochs = epochs, shuffle=is_shuffle, verbose = verbose
                     )
        
        if model.__class__.__name__ == "Resnet20":
            resulting_val_acc.append(val_accuracies[-1])
            record_result.append({"train_acc": train_accuracies, 
                              "val_acc": val_accuracies,
                              "train_loss": train_losses, 
                              "val_loss": val_losses})
        else:
            resulting_val_acc.append(model.history.history["val_accuracy"][-1])
            record_result.append({"train_acc": model.history.history["accuracy"], 
                                "val_acc": model.history.history["val_accuracy"],
                                "train_loss": model.history.history["loss"], 
                                "val_loss": model.history.history["val_loss"]})
        
        if save_dir is not None:
            save_dir_path = os.path.abspath(save_dir)
            #make dir
            try:
                os.makedirs(save_dir_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise    

            if save_names is None:
                file_name = save_dir + "model_{0}".format(n) + ".h5"
            else:
                file_name = save_dir + save_names[n] + ".h5"
            model.save(file_name)
    
    print("pre-train accuracy: ")
    print(resulting_val_acc)
        
    return record_result