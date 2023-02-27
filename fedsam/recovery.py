import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from models.cifar100.resnet20 import ClientModel as res20
from train_resnet20 import evaluate
import inversefed
import os
import torchvision
import datetime
import time

start_time = time.time()
num_images = 1
trained_model = True
target_id = -1
image_path = 'images/'

setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative')

loss_fn, trainloader, validloader =  inversefed.construct_dataloaders('CIFAR100', defs)

model = res20(lr=0.1, num_classes=100, device='cuda')
model.to(**setup)
if trained_model:
    checkpoint = torch.load('./checkpoint')
    model.load_state_dict(checkpoint['model_state_dict'])

model.eval();

accuracy = evaluate(model, validloader)[0]
print('\nTest Accuracy: {}'.format(accuracy))


dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
def plot(tensor):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    if tensor.shape[0] == 1:
        return plt.imshow(tensor[0].permute(1, 2, 0).cpu());
    else:
        fig, axes = plt.subplots(1, tensor.shape[0], figsize=(12, tensor.shape[0]*12))
        for i, im in enumerate(tensor):
            axes[i].imshow(im.permute(1, 2, 0).cpu());



if num_images == 1:
    if target_id == -1:  # demo image
        # Specify PIL filter for lower pillow versions
        ground_truth = torch.as_tensor(
            np.array(Image.open("auto.jpg").resize((32, 32), Image.BICUBIC)) / 255, **setup
        )
        ground_truth = ground_truth.permute(2, 0, 1).sub(dm).div(ds).unsqueeze(0).contiguous()
        
        labels = torch.as_tensor((1,), device=setup["device"])
        target_id = -1
    else:
        #Se il target è none prende un immagine a caso altrimenti quella dell'id scelto
        if target_id is None:
            target_id = np.random.randint(len(validloader.dataset))
        else:
            target_id = target_id
        ground_truth, labels = validloader.dataset[target_id]

        ground_truth, labels = (
            ground_truth.unsqueeze(0).to(**setup),
            torch.as_tensor((labels,), device=setup["device"]),
        )
else:
    ground_truth, labels = [], []
    idx = 25 # choosen randomly ... just whatever you want
    while len(labels) < num_images:
        img, label = validloader.dataset[idx]
        idx += 1
        if label not in labels:
            labels.append(torch.as_tensor((label,), device=setup['device']))
            ground_truth.append(img.to(**setup))
    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels)



model.zero_grad()
target_loss, _, _ = loss_fn(model(ground_truth), labels)
input_gradient = torch.autograd.grad(target_loss, model.parameters())
input_gradient = [grad.detach() for grad in input_gradient]

config = dict(signed=True,
              boxed=True,
              cost_fn='sim',
              indices='def',
              weights='equal',
              lr=0.1,
              optim='adam',
              restarts=1,
              max_iterations=4000,
              total_variation=1e-6,
              init='randn',
              filter='none',
              lr_decay=True,
              scoring_choice='loss')

rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=num_images)
output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=(3, 32, 32))

test_mse = (output.detach() - ground_truth).pow(2).mean()
feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()  
test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1/ds)

print('output')

os.makedirs(image_path, exist_ok=True)
output_denormalized = torch.clamp(output * ds + dm, 0, 1)
rec_filename = (
    f'{validloader.dataset.classes[labels][0]}'
    f"res20_{target_id}.png"
)
torchvision.utils.save_image(output_denormalized, os.path.join(image_path, rec_filename))

gt_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
gt_filename = f"{validloader.dataset.classes[labels][0]}_ground_truth-{target_id}.png"
torchvision.utils.save_image(gt_denormalized, os.path.join(image_path, gt_filename))

print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |")

 # Print final timestamp
print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
print("---------------------------------------------------")
print(f"Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}")
print("-------------Job finished.-------------------------")