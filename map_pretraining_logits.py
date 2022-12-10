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
from nets.clr_nets import SupConResNet, SupCEResNet

from omegaconf import OmegaConf
from glob import glob
from utils.initialization import set_seed

def fetch_dataset(cfg):
    if cfg.task.dataset == "cifar10":
        dataset = Cifar10Dataset(cfg.task.labs, permute=False)
    elif cfg.task.dataset == "cifar100":
        dataset = Cifar100Dataset(cfg.task.labs, permute=False)
    return dataset

def load_model(ckpt, cfg):
    if (cfg.map.learning_mode == "SimCLR") or (cfg.map.learning_mode == "SupCon"):
        model = SupConResNet(feat_dim = ckpt["model"]["head.2.bias"].shape[0])
    else:
        model = SupCEResNet(num_classes= ckpt["model"]["fc.bias"].shape[0])
    
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        model.load_state_dict(state_dict)
    
    return model


@hydra.main(config_path="./config", config_name="conf.yaml")
def main(cfg):
    if (len(cfg.map.task_labs) > 20):
        task_name = f'tasks_{len(cfg.map.task_labs)}'
    else:
        task_name = '_'.join([str(i) for i in cfg.map.task_labs])
    
    fdir = 'probs/' + cfg.map.learning_mode + f'/pretraining/{task_name}/' + cfg.map.ckpt_dir.split("/")[-1] + "/"
    os.makedirs(fdir, exist_ok=True)

    # Wandb
    if cfg.deploy:
        wandb.init(project="map_pretraining_logits", dir = fdir)
        wandb.run.name = f"{cfg.map.learning_mode}_{cfg.map.task_labs}_{cfg.seed}"
        wandb.config.update(OmegaConf.to_container(cfg))
        fpath = wandb.run.id
    else:
        fpath = "tmp"   

    set_seed(cfg.seed)
    cfg.task.labs = cfg.map.task_labs


    dataset = fetch_dataset(cfg)
    loaders = dataset.fetch_data_loaders(cfg.hp.bs, cfg.workers, shuf=False)

    all_metrics = {}

    dirs = list(glob(cfg.map.ckpt_dir + "/*"))
    get_dir_ep = lambda x: int(x.split("_")[-1].split(".")[0])
    dirs = [(get_dir_ep(d), d) for d in dirs]
    dirs = sorted(dirs)

    all_probs = []
    for idx, (ep_num, ckpt_name) in enumerate(dirs):
        ckpt = torch.load(ckpt_name)
        
        net = load_model(ckpt, cfg)
       
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
