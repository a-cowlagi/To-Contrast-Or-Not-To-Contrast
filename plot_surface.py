
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
import hydra
from omegaconf import OmegaConf,open_dict
import h5py
import utils.projection as proj
from utils.data import Cifar10Dataset, Cifar100Dataset
from utils.initialization import set_seed
from nets.clr_nets import SupConResNet, SupCEResNet, LinearClassifier
import net_plotter
import utils.scheduler as scheduler

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

def load_model(cfg):
    model_ckpt = torch.load(cfg.plot_loss.model_ckpt_name)
    print(cfg.plot_loss.clf_ckpt_name)
    print(cfg.plot_loss.model_ckpt_name)
    print(cfg.task.labs)
    clf_ckpt = torch.load(cfg.plot_loss.clf_ckpt_name)
    if (cfg.map.learning_mode == "SimCLR") or (cfg.map.learning_mode == "SupCon"):
        model = SupConResNet(name=cfg.plot_loss.model, feat_dim = model_ckpt["model"]["head.2.bias"].shape[0])
    else:
        model = SupCEResNet(name=cfg.plot_loss.model, num_classes= model_ckpt["model"]["fc.bias"].shape[0])
    
    model_state_dict = model_ckpt['model']
    clf_state_dict = clf_ckpt['model']
    
    criterion = torch.nn.CrossEntropyLoss()
    print(len(cfg.task.labs))
    classifier = LinearClassifier(name=cfg.plot_loss.model, num_classes=len(cfg.task.labs))

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


def fetch_dataset(cfg):
    if cfg.task.dataset == "cifar10":
        dataset = Cifar10Dataset(cfg.task.labs, permute=False)
    elif cfg.task.dataset == "cifar100":
        dataset = Cifar100Dataset(cfg.task.labs, permute=False)
    return dataset


def setup_surface_file(cfg, surf_file, dir_file):
    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (cfg.plot_loss.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print ("%s is already set up" % surf_file)
            return

    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(cfg.plot_loss.xmin, cfg.plot_loss.xmax, num=int(cfg.plot_loss.xnum))
    f['xcoordinates'] = xcoordinates

    if cfg.plot_loss.y:
        ycoordinates = np.linspace(cfg.plot_loss.ymin, cfg.plot_loss.ymax, num=int(cfg.plot_loss.ynum))
        f['ycoordinates'] = ycoordinates
    f.close()

    return surf_file


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
                targets = Variable(targets)
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


def crunch(surf_file, net, w, s, d, dataloader, loss_key, acc_key, comm, rank, cfg):
    """
        Calculate the loss values and accuracies of modified models in parallel
        using MPI reduce.
    """

    f = h5py.File(surf_file, 'r+' if rank == 0 else 'r')
    losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        if rank == 0:
            f[loss_key] = losses
            f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, comm)

    print('Computing %d values for rank %d'% (len(inds), rank))
    start_time = time.time()
    total_sync = 0.0

    criterion = nn.CrossEntropyLoss()
    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        if cfg.plot_loss.dir_type == 'weights':
            net_plotter.set_weights(net, w, d, coord)
        elif cfg.plot_loss.dir_type == 'states':
            net_plotter.set_states(net, s, d, coord)

        # Record the time to compute the loss value
        loss_start = time.time()
        loss, acc = eval_loss(net, criterion, dataloader, True)
        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        # Send updated plot data to the master node
        syc_start = time.time()
        syc_time = time.time() - syc_start
        total_sync += syc_time

        # Only the master node writes to the file - this avoids write conflicts
        if rank == 0:
            f[loss_key][:] = losses
            f[acc_key][:] = accuracies
            f.flush()

        print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
                rank, count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                acc_key, acc, loss_compute_time, syc_time))

    total_time = time.time() - start_time
    print('Rank %d done!  Total time: %.2f Sync: %.2f' % (rank, total_time, total_sync))

    f.close()





@hydra.main(config_path="./config", config_name="conf.yaml")
def main(cfg):
    set_seed(cfg.plot_loss.seed)
    cfg.seed = cfg.plot_loss.seed

    comm, rank, nproc = None, 0, 1

    try:
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.plot_loss.xmin, cfg.plot_loss.xmax, cfg.plot_loss.xnum = [float(a) for a in cfg.plot_loss.x.split(':')]
            cfg.plot_loss.ymin, cfg.plot_loss.ymax, cfg.plot_loss.ynum = (None, None, None)
            if cfg.plot_loss.y:
                cfg.plot_loss.ymin, cfg.plot_loss.ymax, cfg.plot_loss.ynum = [float(a) for a in cfg.plot_loss.y.split(':')]
                assert cfg.plot_loss.ymin and cfg.plot_loss.ymax and cfg.plot_loss.ynum, \
                'You specified some arguments for the y axis, but not all'
    except Exception as e:
        print(e)
        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')


    model, classifier, _ = load_model(cfg)
    net = model_wrapper(model, classifier)
    net.eval()
    
    w = net_plotter.get_weights(net)
    s = copy.deepcopy(net.state_dict())

    if (len(cfg.task.labs) > 20):
        task_name = f'tasks_{len(cfg.task.labs)}'
    else:
        task_name = '_'.join([str(i) for i in cfg.task.labs])

    direction_dir = f"{cfg.plot_loss.direction_dir}/pretrained_{cfg.plot_loss.pretrained_tag}_seed_{cfg.seed}/{task_name}/directions/"
    os.makedirs(direction_dir, exist_ok=True)
    dir_file = f"{direction_dir}/x_{cfg.plot_loss.x}_y_{cfg.plot_loss.y}.h5"
    net_plotter.setup_direction(cfg, dir_file, net)

    surf_dir = f"{cfg.plot_loss.surf_dir}/pretrained_{cfg.plot_loss.pretrained_tag}_seed_{cfg.seed}/{task_name}/surfaces/"
    os.makedirs(surf_dir, exist_ok=True)
    surf_file = f"{surf_dir}/x_{cfg.plot_loss.x}_y_{cfg.plot_loss.y}.h5"
    setup_surface_file(cfg, surf_file, dir_file)

    d = net_plotter.load_directions(dir_file)

    if len(d) == 2 and rank == 0:
        similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)
    
    dataset = fetch_dataset(cfg)
    loaders = dataset.fetch_data_loaders(cfg.hp.bs, cfg.workers, shuf=False)
    trainloader, testloader = loaders[0], loaders[1]

    crunch(surf_file, net, w, s, d, trainloader, 'train_loss', 'train_acc', comm, rank, cfg)

    #--------------------------------------------------------------------------
    # Plot figures
    #--------------------------------------------------------------------------
    if cfg.plot_loss.plot and rank == 0:
        if cfg.plot_loss.y and cfg.plot_loss.proj_file:
            plot_2D.plot_contour_trajectory(surf_file, dir_file, cfg.plot_loss.proj_file, 'train_loss', cfg.plot_loss.show)
        elif cfg.plot_loss.y:
            plot_2D.plot_2d_contour(surf_file, surf_dir, 'train_loss', cfg.plot_loss.vmin, cfg.plot_loss.vmax, cfg.plot_loss.vlevel, cfg.plot_loss.show)
        else:
            plot_1D.plot_1d_loss_err(surf_file, cfg.plot_loss.xmin, cfg.plot_loss.xmax, cfg.plot_loss.loss_max, cfg.plot_loss.log, cfg.plot_loss.show)

if __name__ == '__main__':
    main()

   

    