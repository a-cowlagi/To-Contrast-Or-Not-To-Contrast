import random
import os
import numpy as np
import torch
import json
import torch.nn as nn
import wandb
import hydra
from utils.data import Cifar10Dataset, Cifar100Dataset
from utils.runner import LinearProtoMap
from nets.wideresnet import WideResnet
from nets.simclr_net import SimclrModel

from omegaconf import OmegaConf
from glob import glob
from utils.initialization import set_seed

def fetch_dataset(cfg):
    if cfg.task.dataset == "cifar10":
        dataset = Cifar10Dataset(cfg.task.labs, permute=False)
    elif cfg.task.dataset == "cifar100":
        dataset = Cifar100Dataset(cfg.task.labs, permute=False)
    return dataset


@hydra.main(config_path="./config", config_name="conf.yaml")
def main(cfg):
    print(cfg)

    # Wandb
    if cfg.deploy:
        wandb.init(project="task_manifold")
        wandb.run.name = wandb.run.id
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg))

    set_seed(cfg.seed)
    cfg.task.labs = cfg.map.task_labs

    dataset = fetch_dataset(cfg)
    loaders = dataset.fetch_data_loaders(cfg.hp.bs, cfg.workers, shuf=False)

    all_metrics = {}

    if cfg.deploy:
        fpath = wandb.run.id
    else:
        fpath = "tmp"
        
    fdir = 'ckpt/' + cfg.tag + '/' + fpath
    os.makedirs(fdir, exist_ok=True)

    dirs = list(glob(cfg.map.ckpt_dir + "/*"))
    get_dir_ep = lambda x: int(x.split("_")[-1].split(".")[0])
    dirs = [(get_dir_ep(d), d) for d in dirs]
    dirs = sorted(dirs)

    all_probs = []
    for idx, (ep_num, ckpt_name) in enumerate(dirs):
        
        ckpt = torch.load(ckpt_name)
        if idx == 0:
            if (cfg.learning_mode == "CL"):
                net = SimclrModel()
            elif (cfg.learning_mode == "SL"):
                num_c = ckpt["fc.weight"].shape[0]
                net = WideResnet(num_cls=num_c,
                                net_cfg=cfg.net,
                                cls_per_task=None)
        
        net.load_state_dict(ckpt)

        runner = LinearProtoMap(cfg, loaders, net)
        probs, metrics = runner.get_prob()

        all_probs.append(probs)

        print("%d: %.4f, %.4f"
              % (ep_num, metrics[0],  metrics[1]))
        all_metrics[ep_num] = (metrics[0], metrics[1])

        if cfg.deploy:
            wandb.log({"epoch": ep_num, 
                       "train_acc": metrics[0],
                       "train_loss": metrics[1]})

        with open(fdir + '/metrics.json', 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=4)

    all_probs = np.array(all_probs)
    np.save(fdir + '/probs', all_probs)

    wandb.finish()


if __name__ == '__main__':
    main()
