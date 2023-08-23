import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import wandb
import argparse
import random

from model import Net


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Tutorial from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

def init_logging(args):
    now = datetime.now() # current date and time
    date_time = now.strftime("%d-%m-%Y_%H-%M-%S")

    os.environ['WANDB_PROJECT']= args.wandb_project
    wandb.init(config = args, reinit=True, group = args.wandb_group, mode='online')
    wandb.run.name = date_time+"_"+str(wandb.run.id)

    log_path = "./output/"+args.wandb_project+"/"+args.wandb_group+"/"+str(wandb.run.name)+"/"
    wandb.log({"Logging Path": log_path})
    os.makedirs(log_path, exist_ok=True)    
    return log_path

def log_img(imgs, preds, labels, title):
    fig,axs = plt.subplots(1,len(imgs))
    for i,img in enumerate(imgs): 
        img = img / 2 + 0.5     # unnormalize
        img = img.numpy()
        axs[i].imshow(img.transpose(1,2,0))
        axs[i].set_title("Label: "+str(labels[i]+"\nPred: "+str(preds[i])))
        axs[i].set_axis_off()
    plt.tight_layout()
    wandb.log({title: wandb.Image(plt)})
    plt.close(fig)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def deterministic(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #True
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(True)
    print("INFO::Deterministic true with seed="+str(seed))
    return

if __name__ == '__main__':

    print("******************************")
    print("TRAIN CLASSIFIER ON CIFAR10")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Binary')

    # General Parameters
    parser.add_argument('--batch_size', type = int, default=4, help='Batch Size')
    parser.add_argument('--epochs', type = int, default=2, help='Epochs')
    parser.add_argument('--wandb_project', type = str, default="Debug", help='wandb Project')
    parser.add_argument('--wandb_group', type = str, default="Collection", help='wandb Group')
    parser.add_argument('--seed', type = int, default=42, help='Deterministic Seed')
    parser.add_argument('--deterministic', type = str, choices=["true", "false"], default="false", help='Deterministic Mode')



    args = parser.parse_args()
    args.deterministic = bool(args.deterministic == "true")

    if(args.deterministic):
        deterministic(args.seed)

    LOG_PATH = init_logging(args)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = args.batch_size

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # log images
    class_labels = [f'{classes[labels[j]]:5s}' for j in range(batch_size)]
    preds = ['' for j in range(batch_size)]
    log_img(images, preds, class_labels, "Training Examples")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Net().to(device)

    # define loss + optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # training loop
    for epoch in range(args.epochs): 

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # log statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    
                wandb.log({'Training Loss': running_loss / 2000})
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = LOG_PATH+'/cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # test
    net.eval()
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    # log images
    class_labels = [f'{classes[labels[j]]:5s}' for j in range(batch_size)]
    outputs = net(images.to(device))
    _, predicted = torch.max(outputs, 1)
    preds = [f'{classes[predicted[j]]:5s}' for j in range(batch_size)]
    log_img(images, preds, class_labels, "Test Examples")

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    wandb.log({'Test Accuracy': 100 * correct // total})