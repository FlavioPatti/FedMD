import os
import argparse
import sys
import torch
import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset
from utils.data_utils import load_CIFAR_data, generate_partial_data, generate_bal_private_data
from FedMD import FedMD
from utils.Neural_Networks import cnn_2layers, cnn_3layers, train_and_eval, evaluate
from utils.Resnet20Batch import Resnet20_batchNorm
from utils.Resnet20Group import Resnet20_groupNorm
from utils.Resnet50 import ResNet50
from utils.vision_transformer import VisionTransformer, CONFIGS
import wandb

os.environ["WANDB_API_KEY"] = "a4771dae610ad0306d4c407ecca287079d83d0d9"
os.environ["WANDB_MODE"] = "online"

def parseArg():
    parser = argparse.ArgumentParser(description='FedMD, a federated learning framework. \
    Participants are training collaboratively. ')
    parser.add_argument('-conf', metavar='conf_file', nargs=1,
                        help='the config file for FedMD.'
                        )

    conf_file = os.path.abspath("conf/CIFAR_conf.json")

    if len(sys.argv) > 1:
        args = parser.parse_args(sys.argv[1:])
        if args.conf:
            conf_file = args.conf[0]
    return conf_file


CANDIDATE_MODELS = {"2_layer_CNN": cnn_2layers,
                    "3_layer_CNN": cnn_3layers,
                    "resnet20_group": Resnet20_groupNorm,
                    "resnet20_batch": Resnet20_batchNorm,
                    "resnet50": ResNet50}

def save_checkpoint_pretrain(state, filename1="checkpoint_model_pretrain"):
  print("=> Saving checkpoint")
  filename = 'checkpoint/'+ filename1
  torch.save(state,filename)



def init_wandb(args, alpha=None, run_id=None, group = 0):
        
        

        configuration = args

        job_name = ''
        if alpha is not None:
            job_name = 'alpha' + str(alpha) 
        
        

        run = wandb.init(
                    id = run_id,
                    # Set entity to specify your username or team name
                    entity = "aml-2022", 
                    # Set the project where this run will be logged
                    project="FedMD_IID_15",
                    group=f'{group}',
                    # Track hyperparameters and run metadata
                    config=configuration,
                    resume="allow")

        if os.environ["WANDB_MODE"] != "offline" and not wandb.run.resumed:
            random_number = wandb.run.name.split('-')[-1]
            wandb.run.name = job_name + '-' + random_number
            wandb.run.save()

        return run, job_name

if __name__ == "__main__":
    conf_file = parseArg()
    with open(conf_file, "r") as f:
        conf_dict = eval(f.read())

        # n_classes = conf_dict["n_classes"]
        model_config = conf_dict["models"]
        pre_train_params = conf_dict["pre_train_params"]
        model_saved_dir = conf_dict["model_saved_dir"]
        model_saved_names = conf_dict["model_saved_names"]
        is_early_stopping = conf_dict["early_stopping"]
        public_classes = conf_dict["public_classes"]
        private_classes = conf_dict["private_classes"]
        n_epochs = conf_dict["epochs"]
        n_classes = len(public_classes) + len(private_classes)
        alpha = conf_dict["alpha"]
        manualseed = conf_dict["manualseed"]
        
        N_parties = conf_dict["N_parties"]
        N_samples_per_class = conf_dict["N_samples_per_class"]

        checkpoint = conf_dict["checkpoint"]
        N_rounds = conf_dict["N_rounds"]
        N_alignment = conf_dict["N_alignment"]
        N_private_training_round = conf_dict["N_private_training_round"]
        private_training_batchsize = conf_dict["private_training_batchsize"]
        N_logits_matching_round = conf_dict["N_logits_matching_round"]
        logits_matching_batchsize = conf_dict["logits_matching_batchsize"]

        result_save_dir = conf_dict["result_save_dir"]
        args_wandb = conf_dict["args_wandb"]

    del conf_dict, conf_file
    
    # Load public and private datasets
    
    X_train_CIFAR10, y_train_CIFAR10, X_test_CIFAR10, y_test_CIFAR10 \
        = load_CIFAR_data(data_type="CIFAR10",
                          standarized=True, verbose=True)

    public_dataset = {"X": X_train_CIFAR10, "y": y_train_CIFAR10}

    X_train_CIFAR100, y_train_CIFAR100, X_test_CIFAR100, y_test_CIFAR100 \
        = load_CIFAR_data(data_type="CIFAR100",
                          standarized=True, verbose=True)

    # only use those CIFAR100 data whose y_labels belong to private_classes
    X_train_CIFAR100, y_train_CIFAR100 \
        = generate_partial_data(X=X_train_CIFAR100, y=y_train_CIFAR100,
                                class_in_use=private_classes,
                                verbose=True)

    X_test_CIFAR100, y_test_CIFAR100 \
        = generate_partial_data(X=X_test_CIFAR100, y=y_test_CIFAR100,
                                class_in_use=private_classes,
                                verbose=True)

    # relabel the selected CIFAR100 data for future convenience
    for index, cls_ in enumerate(private_classes):
        y_train_CIFAR100[y_train_CIFAR100 == cls_] = index + len(public_classes)
        y_test_CIFAR100[y_test_CIFAR100 == cls_] = index + len(public_classes)
    del index, cls_

    print(pd.Series(y_train_CIFAR100).value_counts())
    mod_private_classes = np.arange(len(private_classes)) + len(public_classes)  # [10,11,12,13,14,15]

    private_data, total_private_data\
    =generate_bal_private_data(X_train_CIFAR100, y_train_CIFAR100,      
                               N_parties = N_parties,           
                               classes_in_use = mod_private_classes, 
                               N_samples_per_class = N_samples_per_class, 
                               data_overlap = False)

    X_tmp, y_tmp = generate_partial_data(X=X_test_CIFAR100, y=y_test_CIFAR100,
                                         class_in_use=mod_private_classes,
                                         verbose=True)
    private_test_data = {"X": X_tmp, "y": y_tmp}

    del X_tmp, y_tmp

    # convert dataset to train_loader and test_loader
    train_dataset = (TensorDataset(torch.from_numpy(X_train_CIFAR10).float(), torch.from_numpy(y_train_CIFAR10).long()))
    test_dataset = (TensorDataset(torch.from_numpy(X_test_CIFAR10).float(), torch.from_numpy(y_test_CIFAR10).long()))

    pre_models_dir = "./pretrained_CIFAR10/"

    
    parties = []
    
    #Load pre-trained models
    #Model 0-4: CNN
    #Model 5: our implementation of ResNet20 with group norm
    #Model 6: out implementation of ResNet20 with batch norm
    #Model 7: ResNet50
    #Model 8: vit-small
    #Model 9: vit-base 
    
    for i, item in enumerate(model_config):
        
        if model_saved_names[i].startswith('ViT'):
          config = CONFIGS[model_saved_names[i]]
          client_model = VisionTransformer(config, 32, zero_head=True, num_classes=n_classes)
          client_model.device = 'cuda'
          print("model {0} : {1}".format(i, model_saved_names[i]))

          try:
            checkpoint1 = torch.load('checkpoint/' + model_saved_names[i])
            client_model.load_state_dict(checkpoint1['state_dict'])

            device = torch.device("cuda:0")
            
            eval_metrics = evaluate(client_model.to(device), test_dataset, cuda = True, name = model_saved_names[i])
            print('checkpoint loaded: ')
            print(f"acc: {eval_metrics['acc']}")
            print(f"loss: {eval_metrics['loss']}")

          except:
            client_model, train_acc, train_loss, val_acc, val_loss = train_and_eval(client_model, train_dataset,
                                                                                  test_dataset,num_epochs=20, batch_size=512, name = model_saved_names[i])
            save_checkpoint_pretrain({'state_dict': client_model.state_dict()}, filename1=model_saved_names[i])
            
          parties.append(client_model)

        else:
          model_name = item["model_type"]
          model_params = item["params"]
          client_model = CANDIDATE_MODELS[model_name](n_classes=n_classes, **model_params)
          print("model {0} : {1}".format(i, model_saved_names[i]))

          try:
            checkpoint1 = torch.load('checkpoint/' + model_saved_names[i])
            client_model.load_state_dict(checkpoint1['state_dict'])

            device = torch.device("cuda:0")
       
            eval_metrics = evaluate(client_model.to(device), test_dataset, cuda = True, name = model_saved_names[i])
            print('checkpoint loaded: ')
            print(f"acc: {eval_metrics['acc']}")
            print(f"loss: {eval_metrics['loss']}")

          except:
            client_model, train_acc, train_loss, val_acc, val_loss = train_and_eval(client_model, train_dataset,
                                                                                  test_dataset,num_epochs=20, batch_size=128, name = model_saved_names[i])
            save_checkpoint_pretrain({'state_dict': client_model.state_dict()}, filename1=model_saved_names[i])

          parties.append(client_model)

        
        
        

    del model_params, client_model

    del X_train_CIFAR10, y_train_CIFAR10, X_test_CIFAR10, y_test_CIFAR10, \
        X_train_CIFAR100, y_train_CIFAR100, X_test_CIFAR100, y_test_CIFAR100,

    #Run FedMD with public and private data on pretrained models 

    fedmd = FedMD(parties,
                       public_dataset=public_dataset,
                       private_data=private_data,
                       model_saved_name=model_saved_names,
                       total_private_data=total_private_data,
                       private_test_data=private_test_data,
                       manualseed=manualseed,
                       checkpoint=checkpoint,
                       N_rounds=N_rounds,
                       N_alignment=N_alignment,
                       N_logits_matching_round=N_logits_matching_round,
                       logits_matching_batchsize=logits_matching_batchsize,
                       N_private_training_round=N_private_training_round,
                       private_training_batchsize=private_training_batchsize,
                       names = model_saved_names)

    initialization_result = fedmd.init_result
    upper_bounds = fedmd.upper_bounds

    collaboration_performance = fedmd.collaborative_training(names = model_saved_names)

    print('upper bounds')
    print(upper_bounds)

    for i in collaboration_performance:
          
        run, job_name = init_wandb(args_wandb, alpha, run_id=args_wandb['wandb_run_id'], group=i)

        for i, val in enumerate(collaboration_performance[i]):
          wandb.log({'accuracy2': val, 'step2': i})
        
        run.finish()
    
    for i, d in enumerate(initialization_result):
      run, job_name = init_wandb(args_wandb, alpha, run_id=args_wandb['wandb_run_id'], group=f'lower_bound{i}')

      for x in range(3):
        wandb.log({'accuracy2': d['val_acc'], 'step2': x})
      
      run.finish()
    
    for i, val in enumerate(upper_bounds):
      run, job_name = init_wandb(args_wandb, alpha, run_id=args_wandb['wandb_run_id'], group=f'upper_bound{i}')

      for x in [18,19,20]:
        wandb.log({'accuracy2': val, 'step2': x})
      
      run.finish()
