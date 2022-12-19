
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
import pickle

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
        
        
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()

        model.load_state_dict(model_state_dict)
        classifier.load_state_dict(clf_state_dict)
    else:
        new_model_state_dict = {}
        for k, v in model_state_dict.items():
            k = k.replace("module.", "")
            new_model_state_dict[k] = v
        model_state_dict = new_model_state_dict
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
        if batch_idx % 1000 == 0:
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
    model, classifier, criterion = load_model(model_path = model_path, classifier_path= classifier_path, learning_mode = learning_mode, tasks = tasks)
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

        eps_loss, eps_accuracy = eval_loss(net, criterion, loader, use_cuda = False)
        print(f"Epsilon: {epsilons[i]}, loss: {eps_loss}, accuracy: {eps_accuracy}")
        losses_per_eps.append(eps_loss)
        accuracies_per_eps.append(eps_accuracy)

    epsilon_and_original = np.insert(epsilons, 0, 0)
    losses_per_eps_and_original = np.insert(losses_per_eps, 0, orig_loss)
    accuracies_per_eps_and_original = np.insert(accuracies_per_eps, 0, orig_accuracy)

    return epsilon_and_original, losses_per_eps_and_original, accuracies_per_eps_and_original

def pgd_across_seed(dataset, tasks, seeds, learning_mode):
    model_base_path = "Final Model Weights/cifar10_backbones/" + learning_mode + "/" + learning_mode.lower() + "_final_model"
    classifier_base_path = "Final Model Weights/cifar100_classification_heads/" + learning_mode + "/task"

    for task in tasks:
        classifier_base_path += "_"
        classifier_base_path += str(task)
    
    classifier_base_path += "/final_classifier"

    save_path = "results/" + learning_mode

    for task in tasks:
        save_path += "_"
        save_path += str(task)


    losses_per_epsilon = []
    accuracies_per_epsilon = []
    epsilons = np.arange(0.0, 16.0, 1.0)
    for epsilon in epsilons:
        losses_per_epsilon.append([])
        accuracies_per_epsilon.append([])

    for seed in seeds:
        model_path = model_base_path + "_seed" + str(seed)
        classifier_path =  classifier_base_path + "_seed" + str(seed)
        print(f"----Seed: {seed}-----")
        _, curr_losses, curr_accuracies = evaluate_pgd_attack(dataset=dataset, tasks=tasks, model_path = model_path, classifier_path = classifier_path, learning_mode=learning_mode)
        for i, curr_loss in enumerate(curr_losses):
            losses_per_epsilon[i].append(curr_loss)
            accuracies_per_epsilon[i].append(curr_accuracies[i])

    loss_means = []
    loss_stds = []
    accuracy_means = []
    accuracy_stds = []

    for i, epsilon in enumerate(epsilons):
        loss_means.append(np.mean(losses_per_epsilon[i]))
        loss_stds.append(np.std(losses_per_epsilon[i]))

        accuracy_means.append(np.mean(accuracies_per_epsilon[i]))
        accuracy_stds.append(np.std(accuracies_per_epsilon[i]))

       
    with open(save_path + '_loss_means.pkl', 'wb') as f:
        pickle.dump(loss_means, f)
    
    with open(save_path + '_loss_stds.pkl', 'wb') as f:
        pickle.dump(loss_stds, f)

    with open(save_path + '_accuracy_means.pkl', 'wb') as f:
        pickle.dump(accuracy_means, f)

    with open(save_path + '_accuracy_stds.pkl', 'wb') as f:
        pickle.dump(accuracy_stds, f)


def gen_progress_dict(base_path, base_str, learning_modes, tasks_master):
  progress_dict = {}
  for tasks in tasks_master:
    task_title = ', '.join([str(x) for x in tasks])
    progress_dict[task_title] = {}

  for learning_mode in learning_modes:
    for tasks in tasks_master:
      task_phrase = ""
      for task in tasks:
        task_phrase += "_"
        task_phrase += str(task)
      mean_phrase = base_path + learning_mode + task_phrase + "_" + base_str + "_means.pkl"
      std_phrase = base_path + learning_mode + task_phrase + "_" + base_str + "_stds.pkl"

      with open(mean_phrase, 'rb') as f:
        means = np.array(pickle.load(f))
      with open(std_phrase, 'rb') as f:
        stds = np.array(pickle.load(f))

      task_title = ', '.join([str(x) for x in tasks])
      progress_dict[task_title][learning_mode] = (means, stds)

  return progress_dict

def plot_progress(progress_dict, xlabel, ylabel, xs, ylim_low, ylim_high, save_path=None):
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    for ctr, (task_title, task_progress) in enumerate(progress_dict.items()):
        for method_name, (means, stds) in task_progress.items():
            ax[ctr // 2, ctr % 2].plot(xs, means, label=method_name)
            ax[ctr // 2, ctr % 2].fill_between(xs, means - stds, means + stds, alpha=0.2)

        # ax[ctr // 2, ctr % 2].set_title(task_title)
        ax[ctr // 2, ctr % 2].set_xlabel(xlabel)
        ax[ctr // 2, ctr % 2].set_ylabel(ylabel)
        ax[ctr // 2, ctr % 2].legend()
        ax[ctr // 2, ctr % 2].set_ylim(ylim_low, ylim_high)

    if save_path is not None:
        plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
    
    return fig, ax

def generate_and_save_pgd_results():
    tasks_master = [[0, 1, 2, 3, 4], [6, 11, 16, 21, 26], [56, 58, 62, 66, 68], [95, 96, 97, 98, 99]]
    learning_modes = ["SimCLR", "SupCE", "SupCon"]
    for learning_mode in learning_modes:
        for tasks in tasks_master:
            pgd_across_seed(dataset="cifar100", tasks = tasks, seeds = [0, 10, 20, 30], learning_mode=learning_mode)

def main():
    generated_data_already = True
    if not generated_data_already:
        generate_and_save_pgd_results()

    tasks_master = [[0, 1, 2, 3, 4], [6, 11, 16, 21, 26], [56, 58, 62, 66, 68], [95, 96, 97, 98, 99]]
    learning_modes = ["SimCLR", "SupCE", "SupCon"]
    accuracy_dict = gen_progress_dict(base_path="results/", base_str="accuracy", learning_modes=learning_modes, tasks_master=tasks_master)
    loss_dict = gen_progress_dict(base_path="results/", base_str="loss", learning_modes=learning_modes, tasks_master=tasks_master)

    fig, ax = plot_progress(progress_dict=accuracy_dict, xlabel=r'$\epsilon$', ylabel="Accuracy (%)", xs=np.arange(0.0, 16.0/255.0, 1.0/255.0), ylim_low=0, ylim_high=100, save_path = "accuracy_pgd")
    
    fig, ax = plot_progress(progress_dict=loss_dict, xlabel=r'$\epsilon$', ylabel=r'$\mathcal{L}(w: D)$', xs=np.arange(0.0, 16.0/255.0, 1.0/255.0), ylim_low=0.0, ylim_high=5.0, save_path = "loss_pgd")
if __name__ == '__main__':
    main()

   

    