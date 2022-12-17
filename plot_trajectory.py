from utils.trajectory import project
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.data import Cifar100Dataset
import torch
import warnings
import argparse
warnings.filterwarnings('ignore')


def plot_progress(progress_dict, save_path=None):
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    for ctr, (task_title, task_progress) in enumerate(progress_dict.items()):
        for method_name, (progress, std) in task_progress.items():
            ax[ctr // 2, ctr % 2].plot(progress, label=method_name)
            ax[ctr // 2, ctr % 2].fill_between(np.arange(progress.shape[0]), progress - std, progress + std, alpha=0.2)

        ax[ctr // 2, ctr % 2].set_title(task_title)
        ax[ctr // 2, ctr % 2].set_xlabel("Epoch")
        ax[ctr // 2, ctr % 2].set_ylabel("Progress")
        ax[ctr // 2, ctr % 2].legend()

    if save_path is not None:
        plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
    
    return fig, ax



def get_probs(fdir, model_name, task_name):
    # iterate over seeds 0, 10, 20, 30 then load the probs for each seed and stack along the first axis
    for seed in range(0, 40, 10):
        # load pretraining
        pretraining_probs = np.load(f"{fdir}/pretraining/{task_name}/{model_name}_seed_{seed}/probs.npy")

        # load finetuning
        finetuning_probs = np.load(f"{fdir}/finetuning/{task_name}/{model_name}_seed_{seed}/probs.npy")

        if seed == 0:
            pretraining = pretraining_probs[np.newaxis, ...]
            finetuning = finetuning_probs[np.newaxis, ...]
        else:
            pretraining = np.concatenate((pretraining, pretraining_probs[np.newaxis, ...]), axis=0)
            finetuning = np.concatenate((finetuning, finetuning_probs[np.newaxis, ...]), axis=0)

    return pretraining, finetuning

def get_progress_and_std(probs_all_seeds, ground_truths):
    start = np.sqrt(np.zeros_like(ground_truths) + 1/(ground_truths.shape[1]))
    end = np.sqrt(ground_truths)

    progresses = []

    for i in range(probs_all_seeds.shape[0]):
        traj = probs_all_seeds[i]
        progress = np.array(project(traj=traj, start=start, end = end))
        progresses.append(progress.reshape((1, -1)))
        
    progresses = np.concatenate(progresses, axis = 0)
    mean_progress = np.mean(progresses, axis = 0)
    std_progress = np.std(progresses, axis = 0)

    return mean_progress, std_progress

def process_task(task_labs):
    task_name = '_'.join([str(lab) for lab in task_labs])
    print(task_name)

    # load dataset
    dataset = Cifar100Dataset(task_labs, permute=False)
    trainloader, testloader = dataset.fetch_data_loaders(256, 10, shuf=False)

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
    
    (progress_simclr_pre, progress_simclr_pre_std) = get_progress_and_std(simclr_pretraining, all_labels)
    (progress_supcon_pre, progress_supcon_pre_std) = get_progress_and_std(supcon_pretraining, all_labels)
    (progress_supce_pre, progress_supce_pre_std) = get_progress_and_std(supce_pretraining, all_labels)

    (progress_simclr_fine, progress_simclr_fine_std) = get_progress_and_std(simclr_finetuning, all_labels)
    (progress_supcon_fine, progress_supcon_fine_std) = get_progress_and_std(supcon_finetuning, all_labels)
    (progress_supce_fine, progress_supce_fine_std) = get_progress_and_std(supce_finetuning, all_labels)


    task_finetuning = {"SimCLR": (progress_simclr_fine, progress_simclr_fine_std),
                    "SupCon": (progress_supcon_fine, progress_supcon_fine_std),
                    "SupCE": (progress_supce_fine, progress_supce_fine_std)}
    
    task_pretraining = {"SimCLR": (progress_simclr_pre, progress_simclr_pre_std),
                    "SupCon": (progress_supcon_pre, progress_supcon_pre_std),
                    "SupCE": (progress_supce_pre, progress_supce_pre_std)}

    return task_finetuning, task_pretraining


def main(plots):
    tasks = [[0,1,2,3,4], 
    [6, 11, 16, 21, 26], 
    [56, 58, 62, 66, 68], 
    [95, 96, 97, 98, 99]]
    
    pretraining_progress = {}
    finetuning_progress = {}
    for task_labs in tasks:
        task_title = str(task_labs)
        task_finetuning, task_pretraining = process_task(task_labs)
        pretraining_progress[task_title] = task_pretraining
        finetuning_progress[task_title] = task_finetuning


    if "pretraining_progress" in plots:
        save_path = f"progress_plots/pretraining/"
        os.makedirs(save_path, exist_ok=True)
        plot_progress(pretraining_progress, save_path=f"{save_path}pretraining")
    
    if "finetuning_progress" in plots:
        save_path = f"progress_plots/finetuning/"
        os.makedirs(save_path, exist_ok=True)
        plot_progress(finetuning_progress, save_path=f"{save_path}finetuning") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots", nargs="+", default=["pretraining_progress", "finetuning_progress"])
    args = parser.parse_args()
    
    main(set(args.plots))

    

    

