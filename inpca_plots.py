from utils.inpca import reduce_rep
from utils.inpca import reduce_rep
import numpy as np
import matplotlib.pyplot as plt
from utils.data import Cifar100Dataset
import torch
import hydra
from omegaconf import OmegaConf
import os
import warnings
warnings.filterwarnings('ignore')


def plot_inpca(reduced, num_models, labels = None, elevation=None, azimuth=None, plot_true_probs = False, save_path = None):
    if labels == None:
        labels = [f"Seed {i}" for i in range(num_models)]


    # define new colors that are easier to tell apart
    cols = ['#336699', '#993333', '#999933', '#666699']

    fig = plt.figure(figsize=(15, 15))  # set figure size
    ax = plt.axes(projection='3d')

    if elevation != None and azimuth != None:
        ax.view_init(elev=elevation, azim=azimuth)  # set viewing angle

    # plot the original points
    for i in range(num_models):
        if (not plot_true_probs):
            ax.scatter3D(reduced["inpca"][i, :, 0], reduced["inpca"][i, :, 1], reduced["inpca"][i, :, 2], c=cols[i], label=labels[i], s=25)  # reduce point size using the 's' parameter
        else:
            include_elts = np.mod(np.arange(reduced["inpca"].shape[1]), reduced["inpca"].shape[1] // 4) != (reduced["inpca"].shape[1] // 4 - 1)
            include_elts = np.argwhere(include_elts).flatten()
            ax.plot3D(reduced["inpca"][i, -1, 0], reduced["inpca"][i,  -1, 1] ,reduced["inpca"][i,  -1, 2], marker='o', markersize=5, color='k', alpha=1)
            ax.text(reduced["inpca"][i, -1, 0]+ 0.02, reduced["inpca"][i, -1, 1] + 0.01, reduced["inpca"][i, -1, 2] + 0.005, s=r'$P^*$', fontsize=20)
            ax.scatter3D(reduced["inpca"][i, include_elts, 0], reduced["inpca"][i, include_elts, 1], reduced["inpca"][i, include_elts, 2], c=cols[i], label=labels[i], s=25)
            

    # label the axes as PC1, PC2, PC3
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    

    ax.legend()  # show legend
    ax.grid(False)  # remove grid lines

    if (save_path is not None):
        plt.savefig(save_path + ".png", bbox_inches='tight')


def get_probs(fdir, model_name, task_name):
    # iterate over seeds 0, 10, 20, 30 then load the probs for each seed and stack along the first axis
    for seed in range(0, 40, 10):
        # load pretraining
        pretraining_probs = np.load(f"{fdir}/pretraining/{task_name}/{model_name}_seed_{seed}/probs.npy")

        # load finetuning
        finetuning_probs = np.load(f"{fdir}/finetuning/{task_name}/{model_name}_seed_{seed}/probs.npy")


        # stack each along a new axis -- the first axis, to make pretraining and finetuning
        # have shape (num_seeds, num_examples, num_classes)
        if seed == 0:
            pretraining = pretraining_probs[np.newaxis, ...]
            finetuning = finetuning_probs[np.newaxis, ...]
        else:
            pretraining = np.concatenate((pretraining, pretraining_probs[np.newaxis, ...]), axis=0)
            finetuning = np.concatenate((finetuning, finetuning_probs[np.newaxis, ...]), axis=0)

    return pretraining, finetuning



@hydra.main(config_path="./config", config_name="conf.yaml")
def main(cfg):
    # join task labels with underscore
    task_name = '_'.join([str(i) for i in cfg.plot_inpca.task_labs])
    # split task name by underscore and convert to int
    task_labs = cfg.plot_inpca.task_labs

    # load dataset
    dataset = Cifar100Dataset(task_labs, permute=False)
    trainloader, testloader = dataset.fetch_data_loaders(cfg.hp.bs, cfg.workers, shuf=False)

    simclr_fdir = "probs/SimCLR"
    supcon_fdir = "probs/SupCon"
    supce_fdir = "probs/SupCE"

    simclr_model_name = "cifar10_resnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_cosine"
    supcon_model_name = "cifar10_resnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_cosine"
    supce_model_name = "cifar10_resnet18_lr_0.2_decay_0.0001_bsz_256_cosine"

    simclr_pretraining, simclr_finetuning = get_probs(simclr_fdir, simclr_model_name, task_name)
    supcon_pretraining, supcon_finetuning = get_probs(supcon_fdir, supcon_model_name, task_name)
    supce_pretraining, supce_finetuning = get_probs(supce_fdir, supce_model_name, task_name)

    train_data = torch.cat([batch[1] for batch in trainloader])
    all_labels = torch.nn.functional.one_hot(train_data, num_classes=5).numpy()
    
    # stack num seeds copies of all_labels along the first axis
    all_labels = np.stack([all_labels for _ in range(4)], axis=0)
    all_labels = np.expand_dims(all_labels, axis=1)

    finetuning_simclr_with_true = np.concatenate((simclr_finetuning, all_labels), axis=1)
    finetuning_supcon_with_true = np.concatenate((supcon_finetuning, all_labels), axis=1)
    finetuning_supce_with_true = np.concatenate((supce_finetuning, all_labels), axis=1)

    pretraining_supce_all_seeds = np.concatenate(supce_pretraining, axis=0)
    pretraining_simclr_all_seeds = np.concatenate(simclr_pretraining, axis=0)
    pretraining_supcon_all_seeds = np.concatenate(supcon_pretraining, axis=0)

    # stack all pretraining probs along the first axis
    all_pretraining = np.stack((pretraining_supce_all_seeds, pretraining_simclr_all_seeds, pretraining_supcon_all_seeds), axis=0)

    finetuning_supce_all_seeds = np.concatenate(finetuning_supce_with_true, axis=0)
    finetuning_simclr_all_seeds = np.concatenate(finetuning_simclr_with_true, axis=0)
    finetuning_supcon_all_seeds = np.concatenate(finetuning_supcon_with_true, axis=0)

    # stack all finetuning probs along the first axis
    all_finetuning = np.stack((finetuning_supce_all_seeds, finetuning_simclr_all_seeds, finetuning_supcon_all_seeds), axis=0)

    # reduce representation
    pretraining_reduced = reduce_rep(all_pretraining, inpca=True, dynamic_shape=False) 
    finetuning_reduced = reduce_rep(all_finetuning, inpca=True, dynamic_shape=False)  

    # plot inpca
    save_path = f"inpca_plots/{task_name}/"
    print(save_path)
    os.makedirs(save_path, exist_ok=True)

    plot_inpca(pretraining_reduced, 3, labels = ["SupCE", "SimCLR", "SupCon"], azimuth=cfg.plot_inpca.azimuth, elevation=cfg.plot_inpca.elevation, save_path=f"{save_path}pretraining", plot_true_probs = False)
    plot_inpca(finetuning_reduced, 3, labels = ["SupCE", "SimCLR", "SupCon"], save_path=f"{save_path}finetuning", plot_true_probs = True)  


if __name__ == "__main__":
    main()

    

