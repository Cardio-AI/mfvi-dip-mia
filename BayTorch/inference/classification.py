import time
import sys
import os

import torch
import torch.nn.functional as F

import numpy as np

from .utils import uncert_classification_kwon, get_beta

# TODO: classification/regression difference mainly with accuracy
class ClassificationTrainer:
    def __init__(self,
                net,
                train_loader,
                val_loader=None, # for further training!
                gpu=1):

        # super(Trainer, self).__init__()

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.net = net.type(self.dtype)

        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self, n_epochs, criterion, net_path=None, scheduler=None,
              show_every=1, opt_kwargs={'lr': 0.01, 'weight_decay': 1e-4},
              sched_kwargs=dict()):

        optimizer = torch.optim.AdamW(self.net.parameters(), **opt_kwargs)
        if net_path is not None:
            train_data = torch.load(net_path)
            self.net.load_state_dict(train_data['state_dict'])
            optimizer.load_state_dict(train_data['optimizer'])

        if criterion == 'nll':
            criterion = F.nll_loss
        elif criterion == 'cross_entropy':
            criterion = F.cross_entropy

        if scheduler is not None:
            scheduler = scheduler(optimizer, **sched_kwargs)

        nll = []
        kl = []
        accuracy_train = []
        accuracy_val = []

        n_minibatches = np.ceil(float(len(self.train_loader.dataset)) / self.train_loader.batch_size)
        # need fct get_beta
        # beta = torch.tensor(1.0/(n_minibatches))

        start = time.time()
        for epoch in range(n_epochs):

            info_dict = self.train_epoch(criterion, optimizer, epoch, n_epochs, n_minibatches)
            if self.val_loader is not None:
                accuracy = self.evaluate(self.val_loader)
                accuracy_val.append(accuracy)
            if scheduler is not None:
                scheduler.step()

            nll.append(info_dict["nll"])
            kl.append(info_dict["kl"])
            accuracy_train.append(info_dict["accuracy"])

            if (epoch+1) % show_every == 0:
                if self.val_loader is not None:
                    print("#{:4d} | ELBO Loss: {:7.2f} | Accuracy: {:6.2f} % [{:6.2f} %] | KL: {:7.2f} | NLL: {:7.2f} |"\
                          .format(epoch+1, np.sum(info_dict["nll"]) + np.sum(info_dict["kl"]), info_dict["accuracy"], \
                                  accuracy, np.sum(info_dict["kl"]), np.sum(info_dict["nll"])))
                else:
                    print("#{:4d} | ELBO Loss: {:7.2f} | Accuracy: {:6.2f} % | KL: {:7.2f} | NLL: {:7.2f} |"\
                          .format(epoch+1, np.sum(info_dict["nll"]) + np.sum(info_dict["kl"]), info_dict["accuracy"], \
                                  np.sum(info_dict["kl"]), np.sum(info_dict["nll"])))

        end = time.time() - start
        print("\nTraining time: {} h {} min {} s".format(int(end/3600), int((end/60)%60), int(end%60)))

        self.train_data = {
            'accuracy_train': accuracy_train,
            'accuracy_val': accuracy_val,
            'nll': nll,
            'kl': kl,
            'state_dict': self.net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'time': end,
        }

    def train_epoch(self, criterion, optimizer, epoch, n_epochs, n_minibatches, beta_type='Standard', warmup_epochs=0, n=1):
        self.net.train()

        correct = 0
        kl_l = []
        nll_l = []

        for i, (input, target) in enumerate(self.train_loader):
            input = input.type(self.dtype)
            target = target.type(self.dtype)

            # compute output
            if n == 1:
                output = self.net(input)
            else:
                output = torch.zeros((input.shape[0], self.net.n_outputs)).type(self.dtype)
                for _ in range(n):
                    output += self.net(input)
                output /= n

            nll = criterion(output, target.long(), reduction='sum')
            if hasattr(self.net, 'kl'):
                kl = self.net.kl.type(self.dtype)
                beta = get_beta(i, n_minibatches, beta_type, epoch, n_epochs, warmup_epochs)
                kl *= beta
            else:
                kl = torch.tensor([0]).to(nll.device)
            ELBOloss = nll + kl

            optimizer.zero_grad()
            ELBOloss.backward()
            optimizer.step()

            _, labels = output.max(1)
            correct += labels.eq(target).sum().float().item()

            kl_l.append(kl.item())
            nll_l.append(nll.item())

        return {"nll": nll_l,
                "kl": kl_l,
                "accuracy": correct * 100 / len(self.train_loader.dataset),}

    def evaluate(self, data_loader):
        self.net.eval()

        correct = 0
        with torch.no_grad():
            for input, target in data_loader:
                input = input.type(self.dtype)
                target = target.type(self.dtype)

                # compute output
                output = self.net(input)
                _, labels = output.max(1)

                # count correct predictions
                correct += labels.eq(target).sum()

        return correct.float() * 100 / len(data_loader.dataset)

    def save(self, path, **kwargs):
        self.train_data['batch_size'] = self.train_loader.batch_size
        for key, value in kwargs.items():
            self.train_data[key] = value
        torch.save(self.train_data, path)

class Predictor:
    def __init__(self,
                 net,
                 num_classes,
                 gpu=1):
        # super(Predictor, self).__init__()

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.net = net.type(self.dtype)
        self.num_classes = num_classes

    def predict(self, data_loader, n_samples=25):
        # self.net.eval()

        logits = []
        labels = []
        with torch.no_grad():
            for i in range(n_samples):
                for batch_idx, (data, target) in enumerate(data_loader):
                    data, target = data.type(self.dtype), target.type(self.dtype)
                    output = torch.softmax(self.net(data), dim=1)
                    logits.append(output.detach().unsqueeze(0))
                    labels.append(target.detach().unsqueeze(0))
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)

        p_hat = logits.cpu().reshape(-1, len(data_loader)*data_loader.batch_size, self.num_classes)

        pred, uncert, ale, epi = uncert_classification_kwon(p_hat)

        return pred.argmax(dim=1), uncert, ale, epi
