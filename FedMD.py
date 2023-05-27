import numpy as np
import copy
from torch.utils.data import TensorDataset, DataLoader
import torch
from utils.data_utils import generate_alignment_data
from utils.Neural_Networks import train_and_eval, evaluate, train
import torch.nn as nn
from utils.Sia import SIA
from utils.logger import Logger, mkdir_p
import os

def get_logits(model, data_loader, names, cuda):
    model.eval()
    # logits for of the unlabeld public dataset
    logits = []
    data_loader = DataLoader(data_loader, batch_size=128, shuffle=False)
    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch in data_loader:

            if cuda:
                data_batch = data_batch.cuda()  # (B,3,32,32)

            # compute model output
            if names.startswith('ViT'):
              output_batch = model(data_batch)[0]
            else:
              output_batch = model(data_batch)
            
            #print(f"names = {names}, output_batch = {output_batch}")

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()

            # append logits
            logits.append(output_batch)

    # get 2-D logits array
    logits = np.concatenate(logits)

    return logits


class FedMD():
    def __init__(self, parties, public_dataset,
                 private_data, total_private_data,
                 private_test_data, N_alignment,
                 N_rounds,
                 manualseed,
                 checkpoint,
                 model_saved_name,
                 N_logits_matching_round, logits_matching_batchsize,
                 N_private_training_round, private_training_batchsize,
                 names):
        self.N_parties = len(parties)
        self.public_dataset = public_dataset
        self.model_saved_name = model_saved_name
        self.private_data = private_data
        self.private_test_data = private_test_data
        self.total_private_data = total_private_data
        self.checkpoint = checkpoint
        self.manualseed = manualseed
        self.N_alignment = N_alignment

        self.N_rounds = N_rounds
        self.N_logits_matching_round = N_logits_matching_round
        self.logits_matching_batchsize = logits_matching_batchsize
        self.N_private_training_round = N_private_training_round
        self.private_training_batchsize = private_training_batchsize

        self.collaborative_parties = []
        self.init_result = []

        print('\n **************************************************************************** \n')
        print("start model initialization: \n")

        test_dataset = (TensorDataset(torch.from_numpy(private_test_data["X"]).float(),
                                      torch.from_numpy(private_test_data["y"]).long()))
        n_init_training = 0
        for i in range(self.N_parties):
            print("model: ", names[i])
            model_A_twin = copy.deepcopy(parties[i])

            print("init values:  ")
            # get train_loader and test_loader
            train_dataset = (TensorDataset(torch.from_numpy(private_data[i]["X"]).float(),
                                           torch.from_numpy(private_data[i]["y"]).long()))
            
            if names[i] in ["RESNET20_G"]:
              n_init_training  = 200
            else:
              n_init_training  = 50

            #print(f"len train = {len(train_dataset)}") 300
            #print(f"len test = {len(test_dataset)}")  600
            model_A, train_acc, train_loss, val_acc, val_loss = train_and_eval(model_A_twin, train_dataset,
                                                                               test_dataset, n_init_training, batch_size=32, name = names[i], tqdm_v=False)

            print("full stack training done\n\n")

            self.collaborative_parties.append({"model": model_A})

            self.init_result.append({"val_acc": val_acc,
                                     "train_acc": train_acc,
                                     "val_loss": val_loss,
                                     "train_loss": train_loss,
                                     })

            del model_A, model_A_twin
        
        train_dataset = (TensorDataset(torch.from_numpy(total_private_data["X"]).float(),
                                      torch.from_numpy(total_private_data["y"]).long()))
        

        print('**************************************************************************** \n')
        print("calculate the theoretical upper bounds for participants: \n")

        self.upper_bounds = []
        self.pooled_train_result = []
        n_ub_train = 0
        for i in range(self.N_parties):
            model_ub = copy.deepcopy(parties[i])

            if names[i] in ["RESNET20_G"]:
              n_ub_train = 150
            else:
              n_ub_train = 50

            print("model: ", names[i])
            print("UB values:  ")
            model_A, train_acc, train_loss, val_acc, val_loss = train_and_eval(model_ub, train_dataset,
                                                                               test_dataset, n_ub_train, batch_size=32, name = names[i], tqdm_v = False)
            self.upper_bounds.append(val_acc)
            self.pooled_train_result.append({"val_acc": val_acc, 
                                             "acc": train_acc})
            
            print('\n')
            
            del model_ub    

        print('end upper bounds phase.. ')
        # END FOR LOOP

    def collaborative_training(self, names):
        # start collaborating training

        print('start collaborative training')
        if not os.path.isdir(self.checkpoint):
            mkdir_p(self.checkpoint)

        logger = Logger(os.path.join(self.checkpoint, 'log_seed{}.txt'.format(self.manualseed)))
        logger.set_names(['alpha', 'comm. round', 'ASR'])

        test_dataset = (TensorDataset(torch.from_numpy(self.private_test_data["X"]).float(),
                                      torch.from_numpy(self.private_test_data["y"]).long()))
        collaboration_performance = {i: [] for i in range(self.N_parties)}
       
       
        r = 0
        while True:
            # At beginning of each round, generate new alignment dataset
            alignment_data = generate_alignment_data(self.public_dataset["X"],
                                                     self.public_dataset["y"],
                                                     self.N_alignment)
            alignment_dataloader = (torch.from_numpy(alignment_data["X"]).float())
            #print(f"len alignmet = {len(alignment_data)}") 5000

            print("round ", r)

            print("update logits ... ")

            # update logits
            logits = 0

            i = 0
            for d in self.collaborative_parties:
              print(f"nome = {names[i]}")
              logits += get_logits(d["model"], alignment_dataloader, names[i], cuda=True)
              i = i + 1

            logits /= self.N_parties

            # test performance
            print("test performance ... ")

            private_test_dataloader = (TensorDataset(torch.from_numpy(self.private_test_data["X"]).float(),
                                                     torch.from_numpy(self.private_test_data["y"]).long()))
            #print(f"len private = {len(private_test_dataloader)}") 600

            for index, d in enumerate(self.collaborative_parties):
                metrics_mean = evaluate(d["model"], private_test_dataloader, cuda = True,name = names[index])
                collaboration_performance[index].append(metrics_mean["acc"])

                print(collaboration_performance[index][-1])
            
            r += 1
            if r > self.N_rounds:
                break

            print("updates models ...")
            for index, d in enumerate(self.collaborative_parties):
                print("model {0} starting alignment with public logits... ".format(index))

                public_dataloader = (
                    TensorDataset(torch.from_numpy(alignment_data["X"]).float(), torch.from_numpy(logits).float()))

                #print(f"len public dataloader = {len(public_dataloader)}") 5000
            
                model_alignment = train(d["model"], public_dataloader, epochs=self.N_logits_matching_round, cuda=True,
                                        batch_size=self.logits_matching_batchsize,
                                        loss_fn=nn.L1Loss(), name = names[index])

                print("model {0} done alignment".format(index))

                print("model {0} starting training with private data... ".format(index))

                private_dataloader = (TensorDataset(torch.from_numpy(self.private_data[index]["X"]).float(),
                                                    torch.from_numpy(self.private_data[index]["y"]).long()))
                
                #print(f"len private dataloader = {len(private_dataloader)}") 300
                model_local = train(model_alignment, private_dataloader, epochs=self.N_private_training_round,
                                    cuda=True, batch_size=self.private_training_batchsize,
                                    loss_fn=nn.CrossEntropyLoss(), name = names[index])

                d["model"] = model_local
                print("model {0} done private training. \n".format(index))
            # END FOR LOOP

        # END WHILE LOOP
       

        return collaboration_performance
