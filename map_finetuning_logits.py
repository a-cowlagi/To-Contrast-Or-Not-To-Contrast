import random
import os
import numpy as np
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import wandb
import hydra
from utils.data import Cifar10Dataset, Cifar100Dataset
from utils.runner import LinearProtoMap
from nets.wideresnet import WideResnet
from nets.clr_nets import SupConResNet, SupCEResNet, LinearClassifier

from omegaconf import OmegaConf
from glob import glob
from utils.initialization import set_seed

class model_wrapper(nn.Module):
    def __init__(self, model, classifier, n_cls):
        super(model_wrapper, self).__init__()
        self.encoder = model.encoder
        self.classifier = classifier
        self.encoder_dim = n_cls

    def forward(self, x, task):
        x = self.encoder(x)
        x = self.classifier(x)
        self.rep = x
        return x


def fetch_dataset(cfg):
    if cfg.task.dataset == "cifar10":
        dataset = Cifar10Dataset(cfg.task.labs, permute=False)
    elif cfg.task.dataset == "cifar100":
        dataset = Cifar100Dataset(cfg.task.labs, permute=False)
    return dataset

def load_model(cfg, clf_ckpt_name):
    model_ckpt = torch.load(cfg.map.model_ckpt_name)
    clf_ckpt = torch.load(clf_ckpt_name)
    if (cfg.map.learning_mode == "SimCLR") or (cfg.map.learning_mode == "SupCon"):
        model = SupConResNet(name="resnet18", feat_dim = model_ckpt["model"]["head.2.bias"].shape[0])
    else:
        model = SupCEResNet(name="resnet18", num_classes= model_ckpt["model"]["fc.bias"].shape[0])
    
    model_state_dict = model_ckpt['model']
    clf_state_dict = clf_ckpt['model']
    
    classifier = LinearClassifier(name="resnet18", num_classes=len(cfg.task.labs))

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_model_state_dict = {}
            for k, v in model_state_dict.items():
                k = k.replace("module.", "")
                new_model_state_dict[k] = v
            model_state_dict = new_model_state_dict
        

        model.load_state_dict(model_state_dict)
        classifier.load_state_dict(clf_state_dict)

    net = model_wrapper(model, classifier, len(cfg.task.labs))

    if torch.cuda.is_available():
        net = net.cuda()

    return net


   

@hydra.main(config_path="./config", config_name="conf.yaml")
def main(cfg):
    set_seed(0)
    cfg.seed = cfg.map.seed

    if (len(cfg.task.labs) > 20):
        task_name = f'tasks_{len(cfg.task.labs)}'
    else:
        task_name = '_'.join([str(i) for i in cfg.task.labs])
    
    fdir = 'probs/' + cfg.map.learning_mode + f'/finetuning/{task_name}/' + cfg.map.model_ckpt_name.split("/")[-2] + "/"
    os.makedirs(fdir, exist_ok=True)

    # Wandb
    if cfg.deploy:
        wandb.init(project="map_finetuning_logits", dir = fdir)
        wandb.run.name = f"{cfg.map.learning_mode}_{cfg.map.task_labs}_{cfg.seed}"
        wandb.config.update(OmegaConf.to_container(cfg))
        fpath = wandb.run.id
    else:
        fpath = "tmp"   

    
    dataset = fetch_dataset(cfg)
    loaders = dataset.fetch_data_loaders(cfg.hp.bs, cfg.workers, shuf=False)

    all_metrics = {}

    dirs = list(glob(cfg.map.clf_ckpt_dir + "/*"))
    get_dir_ep = lambda x: int(x.split("_")[-1].split(".")[0])
    dirs = [(get_dir_ep(d), d) for d in dirs]
    dirs = sorted(dirs)

    all_probs = []
    for idx, (ep_num, clf_ckpt_name) in enumerate(dirs):
        net = load_model(cfg, clf_ckpt_name)

        cfg.map.proto = False
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
