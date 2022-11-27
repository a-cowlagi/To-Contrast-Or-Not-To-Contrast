import torch
import torch.nn as nn
import numpy as np


class LinearProtoMap():
    def __init__(self, cfg, dataloaders, net):

        self.criterion = nn.CrossEntropyLoss()
        self.cfg = cfg
        self.dataloaders = dataloaders
        self.net = net
        self.gpu = torch.cuda.is_available()
        if self.gpu:
            self.net.cuda()

    def data_prototypes(self):
        if hasattr(self.net, "encoder_dim"):
            hid_dim = self.net.encoder_dim
        else:
            hid_dim = self.net.fc.weight.shape[1]

        prototypes = torch.zeros(len(self.cfg.task.labs), hid_dim)
        lab_count = torch.zeros(len(self.cfg.task.labs))
        if self.gpu:
            prototypes = prototypes.cuda()
            lab_count = lab_count.cuda()

        num_cls = len(self.cfg.task.labs)

        with torch.inference_mode():
            for dat, labels in self.dataloaders[0]:
                dat, task, labels = self.create_batch(dat, labels)
                _ = self.net(dat, task)
                rep = self.net.rep
                prototypes.index_add_(0, labels, rep)
                lab_count += torch.bincount(labels, minlength=num_cls)

        prototypes = prototypes / lab_count.reshape(-1, 1)
        return prototypes
     
    def euclid_dist(self, proto, rep):
        if self.cfg.map.euclid:
            n = rep.shape[0]
            k = proto.shape[0]
            rep = rep.unsqueeze(1).expand(n, k, -1)
            proto = proto.unsqueeze(0).expand(n, k, -1)
            logits = -((rep - proto)**2).sum(dim=2)
        else:
            logits = rep @ proto.T
        return logits

    def create_batch(self, dat, labels):
        task = None

        labels = labels.long()
        batch_size = int(labels.size()[0])
        if self.gpu:
            dat = dat.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        return dat, task, labels

    def get_prob(self):
        all_out = []
        loss, acc, count = 0.0, 0.0, 0.0
        self.net.eval()

        if self.cfg.map.proto:
            prototypes = self.data_prototypes()

        with torch.inference_mode():
            for dat, labels in self.dataloaders[0]:
                dat, task, labels = self.create_batch(dat, labels)
                batch_size = int(labels.size()[0])

                if self.cfg.map.proto:
                    _ = self.net(dat, task)
                    rep = self.net.rep
                    out = self.euclid_dist(prototypes, rep)
                else:
                    out = self.net(dat, task)

                loss += (self.criterion(out, labels).item()) * batch_size

                labels = labels.cpu().numpy()
                out = out.cpu().detach()
                all_out.append(torch.nn.functional.softmax(out, dim=1))
                out = out.numpy()

                acc += np.sum(labels == (np.argmax(out, axis=1)))
                count += batch_size

            ret = np.array((acc/count, loss/count))

        all_out = np.concatenate(all_out)

        return all_out, ret
