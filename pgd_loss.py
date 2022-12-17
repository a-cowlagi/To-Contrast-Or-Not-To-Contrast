
import sys
sys.path.insert(0, "../")

import copy
import os
import numpy as np
import torch
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd.variable import Variable
import utils.projection as proj
from utils.data import Cifar10Dataset, Cifar100Dataset
from utils.initialization import set_seed
from nets.clr_nets import SupConResNet, SupCEResNet, LinearClassifier
import net_plotter
import utils.scheduler as scheduler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import utils.plot_2D as plot_2D
import utils.plot_1D as plot_1D
import utils.h5_util as h5_util

class model_wrapper(nn.Module):
    def __init__(self, model, classifier):
        super(model_wrapper, self).__init__()
        self.backbone = model.encoder
        self.classifier = classifier

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

def load_model(model_path, classifier_path, learning_mode, tasks):
    model_ckpt = torch.load(model_path, map_location = torch.device('cpu'))
    clf_ckpt = torch.load(classifier_path, map_location = torch.device('cpu'))
    if (learning_mode == "SimCLR") or (learning_mode == "SupCon"):
        model = SupConResNet(feat_dim = model_ckpt["model"]["head.2.bias"].shape[0])
    else:
        model = SupCEResNet(num_classes= model_ckpt["model"]["fc.bias"].shape[0])
    
    model_state_dict = model_ckpt['model']
    clf_state_dict = clf_ckpt['model']
    
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(num_classes=len(tasks))

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_model_state_dict = {}
            for k, v in model_state_dict.items():
                k = k.replace("module.", "")
                new_model_state_dict[k] = v
            model_state_dict = new_model_state_dict
        
        
        # model = model.cuda()
        # classifier = classifier.cuda()
        # criterion = criterion.cuda()

        model.load_state_dict(model_state_dict)
        classifier.load_state_dict(clf_state_dict)

    return model, classifier, criterion


def fetch_dataset(dataset, tasks):
    if dataset == "cifar10":
        dataset = Cifar10Dataset(tasks, permute=False)
    elif dataset == "cifar100":
        dataset = Cifar100Dataset(tasks, permute=False)
    return dataset

def get_accuracy(net, criterion, loader):
    correct = 0
    total = 0
    net.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            batch_size = inputs.size(0)
            total += batch_size
            inputs = Variable(inputs)
            targets = Variable(targets.long())
            outputs = net(inputs)   
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)

    return correct / total

def eval_loss(net, criterion, loader, use_cuda=True):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    total = 0 # number of samples
    num_batch = len(loader)

    if (use_cuda):
        net.cuda()
    net.eval()

    with torch.no_grad():
        if isinstance(criterion, nn.CrossEntropyLoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)
                targets = Variable(targets.long())
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)   
                loss = criterion(outputs, targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()

        elif isinstance(criterion, nn.MSELoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)

                one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
                one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
                one_hot_targets = one_hot_targets.float()
                one_hot_targets = Variable(one_hot_targets)
                if use_cuda:
                    inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
                outputs = F.softmax(net(inputs))
                loss = criterion(outputs, one_hot_targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).sum().item()

    return total_loss/total, 100.*correct/total

def generate_pgd_set(loader, epsilons, model, criterion):
    #BATCH SIZE = 1!
    print(f"Generating PGD Set")
    perturbed_images_per_eps = []
    targets_for_perturbed_images = []
    for epsilon in epsilons:
        perturbed_images_per_eps.append([])

    for batch_idx, (inputs, targets) in enumerate(loader):
        print(f"Batch idx: {batch_idx}/{len(loader)}")

        image = Variable(inputs)
        targets = Variable(targets.long())
        image.requires_grad = True   
        output = model(image)
        loss = criterion(output, targets)
        loss.backward()
        dx = image.grad.data.clone()
        dx = torch.sign(dx)
        for epsilon_idx, epsilon in enumerate(epsilons):
            cloned_image = image.clone()
            cloned_image += dx * epsilon / 255.0
            perturbed_images_per_eps[epsilon_idx].append(cloned_image[0])
        targets_for_perturbed_images.append(targets[0])

    return perturbed_images_per_eps, targets_for_perturbed_images



def evaluate_pgd_attack(dataset, tasks, model_path, classifier_path, learning_mode):
    model, classifier, criterion = load_model(model_path = "Final Model Weights/" + model_path, classifier_path= "Final Model weights/cifar100_classification_heads/" + classifier_path, learning_mode = learning_mode, tasks = tasks)
    dataset = fetch_dataset(dataset, tasks)
    loaders = dataset.fetch_data_loaders(bs=1, shuf=False)
    trainloader, testloader = loaders[0], loaders[1]

    epsilons = np.arange(1.0, 16.0, 1.0)
    net = model_wrapper(model, classifier)
    net.eval()
  
    orig_loss, orig_accuracy = eval_loss(net, criterion, trainloader, use_cuda = False)
    
    print(f"Epsilon: {0}, loss: {orig_loss}, accuracy: {orig_accuracy}")


    perturbed_images_per_eps, targets_for_perturbed_images = generate_pgd_set(trainloader, epsilons, net, criterion)


    losses_per_eps = []
    accuracies_per_eps = []

    for i, perturbed_images in enumerate(perturbed_images_per_eps): 
        perturbed_dataset = TensorDataset(torch.stack(perturbed_images), torch.stack(targets_for_perturbed_images))
        loader = DataLoader(perturbed_dataset, batch_size = 16)

        eps_loss, eps_accuracy = eval_loss(net, criterion, loader, use_cuda = True)
        print(f"Epsilon: {epsilons[i]}, loss: {eps_loss}, accuracy: {eps_accuracy}")
        losses_per_eps.append(eps_loss)
        accuracies_per_eps.append(eps_accuracy)

        print(f"Epsilon: {epsilons[i]}, accuracy: {eps_accuracy}")

    #BELOW DOES NOT WORK, ABOVE WORKS

    epsilon_and_original = np.insert(epsilons, 0, 0)
    losses_per_eps_and_original = np.insert(losses_per_eps, 0, orig_loss)
    accuracies_per_eps_and_original = np.insert(accuracies_per_eps, 0, orig_accuracy)

    plt.scatter(epsilon_and_original, losses_per_eps_and_original, label = "Losses")
    plt.xlabel("Epsilon (/255)")
    plt.ylabel("Loss")
    title_str = model_path + '_' + classifier_path
    title_str = title_str.replace('/', '_')
    plt.title(title_str)
    plt.savefig(title_str + "_loss.png")
    plt.show()

    plt.scatter(epsilon_and_original, accuracies_per_eps_and_original, label = "Accuracies")
    plt.xlabel("Epsilon (/255)")
    plt.ylabel("Accuracy")
    title_str = model_path + '_' + classifier_path
    title_str = title_str.replace('/', '_')
    plt.title(title_str)
    plt.savefig(title_str + "_accuracy.png")
    plt.show()


def main():
    evaluate_pgd_attack("cifar100", tasks = [0, 1, 2, 3, 4], model_path="simclr_final_model_seed0", classifier_path = "task_0_1_2_3_4/final_classifier_seed0", learning_mode="SimCLR")

if __name__ == '__main__':
    main()

   

    