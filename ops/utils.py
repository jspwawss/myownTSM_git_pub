import numpy as np
import torch

def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def AUC(output, target):
    print("output is ",output)
    print("target is ",target)
    #>0.8 => True   || 0.8 is my threshold
    output_cpu=output.cpu()
    target_cpu = target.cpu()
    pred = torch.where(output_cpu>0.8,torch.ones(output.size()),torch.zeros(output.size()))
    tp = torch.eq(pred,target_cpu)
    acc = torch.sum(tp)
    auc = acc/(output.size()[1]+target.size()[0]-acc)
    print("auc in auc is ",auc)
    return auc
    

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    print("output size in accuracy,",output.size())

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    print("pred=",pred)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res