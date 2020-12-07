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
    #print("output is ",output)
    #print("target is ",target)
    #>0.8 => True   || 0.8 is my threshold
    #output_cpu=output.cpu()
    #target_cpu = target.cpu()
    #output = target
    
    pred = torch.where(output>0.8,torch.ones(output.size()).cuda(),torch.zeros(output.size()).cuda()).cuda()
    #pred[0] = 1
    #print(pred)

    tp = torch.eq(pred,target).cuda()
    #print("tp=",tp)
    acc = torch.sum(tp)
    accdata = acc.float()
    #print("intersection=",accdata)
    #print(output.size())
    #print(type(target.size()))
    if not output.size():
        a = 1+1-accdata
    else:
        a = output.size()[0]+target.size()[0]-accdata
    #print("area =",a)
    auc = float(acc.data/a.data)
    #print("auc in auc is ",auc)
    return auc

def testAUC(output, target):
    print("output is ",output)
    print("target is ",target)
    #>0.8 => True   || 0.8 is my threshold
    #output_cpu=output.cpu()
    #target_cpu = target.cpu()
    #output = target
    
    pred = torch.where(output>0.8,torch.ones(output.size()),torch.zeros(output.size()))
    #pred[0] = 1
    #print(pred)

    tp = torch.eq(pred,target)
    #print("tp=",tp)
    acc = torch.sum(tp)
    accdata = acc.float()
    #print("intersection=",accdata)
    #print(output.size()[0])
    #rint(target.size()[1])
    a = output.size()[0]+target.size()[0]-accdata
    print("area =",a)
    auc = float(acc.data/a.data)
    #print("auc in auc is ",auc)
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


if __name__ == '__main__':
    import torch
    output = torch.tensor([0,0,0,1,1],dtype=torch.float)
    target = torch.tensor([1,1,1,1,1],dtype=torch.float)
    auc = testAUC(output,target)
    print("auc is ",auc)