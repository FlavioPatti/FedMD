import wandb
import os


os.environ["WANDB_API_KEY"] = "a4771dae610ad0306d4c407ecca287079d83d0d9"
os.environ["WANDB_MODE"] = "online"

wandb.login()


script = 'python CIFAR_Imbalanced.py -conf conf/CIFAR_imbalance_conf.json'
esecuzione = os.popen(script)
o = esecuzione.read()