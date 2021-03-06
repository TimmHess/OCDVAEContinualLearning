import torch
import numpy as np


def to_one_hot(targets, num_classes):
    labels = targets.unsqueeze_(1).long()
    #print("labels", labels.shape)
    one_hot = torch.zeros(labels.shape[0], num_classes, labels.shape[2], labels.shape[3]).to(targets.device)
    #print("one_hot", one_hot.shape)
    one_hot.scatter_(1, labels, 1)
    return one_hot

def from_tensor(tensor):
    #out_img = torch.argmax(tensor.squeeze(), dim=1)
    out_img = torch.argmax(tensor, dim=1)
    return out_img

def iou(pred, target):
    SMOOTH = 1e-6
    
    # Encode 
    num_classes = pred.shape[1]
    pred = from_tensor(pred).unsqueeze_(0)
    pred = to_one_hot(targets=pred, num_classes=num_classes).long()
    target_one_hot = to_one_hot(targets=target, num_classes=num_classes).long()
    
    # Calculate for each class
    ious = []
    for i in range(num_classes):
        curr_pred = pred[:,i,:,:]
        curr_target_one_hot = target_one_hot[:,i,:,:]

        intersection = (curr_pred & curr_target_one_hot).float().sum((1, 2))
        union = (curr_pred | curr_target_one_hot).float().sum((1, 2))

        iou = (intersection + SMOOTH) / (union + SMOOTH)

        ious.append(iou.item())
    return ious

def iou_class_condtitional(pred, target):
    SMOOTH = 1e-6

    #print("pred", pred.shape)
    #print("target", target.shape)

    # Encode
    num_classes = pred.shape[1]
    #print("num_classes", num_classes)
    #pred = from_tensor(pred).unsqueeze_(0)
    pred = from_tensor(pred)
    #print("pred2:", pred.shape)
    pred = to_one_hot(targets=pred, num_classes=num_classes).long()
    target_one_hot = to_one_hot(targets=target, num_classes=num_classes).long()

    ious = torch.zeros(num_classes)
    for i in range(num_classes):
        # Get the current target
        curr_target_one_hot = target_one_hot[:,i,:,:]
        # Get class masked prediction
        curr_pred = pred[:,i,:,:] * curr_target_one_hot

        intersection = (curr_pred & curr_target_one_hot).float().sum((1, 2))
        union = (curr_pred | curr_target_one_hot).float().sum((1, 2))

        iou = (intersection + SMOOTH) / (union + SMOOTH)
        #print("iou", i, iou.mean())

        #ious.append(iou.mean().item())
        ious[i] = iou.mean()
    return ious

def iou_to_accuracy(ious):
    return ious.mean()

def get_seg_confusion(pred, target):
    # get num classes
    num_classes = pred.shape[1]

    # get argmax of prediction
    pred = from_tensor(pred)
    #print("pred unique", np.unique(pred.cpu().numpy()))

    # conert pred and target to one hot
    pred = to_one_hot(targets=pred, num_classes=num_classes).long()
    target_one_hot = to_one_hot(targets=target, num_classes=num_classes).long()

    # get number of pixels classifier for each class
    # for each class map -> mask with each class from target -> sum to get number of pixels
    pixel_counts = torch.zeros(num_classes, num_classes)
    for i in range(num_classes):
        for j in range(num_classes):
            curr_pred = pred[:,i,:,:] * target_one_hot[:,j,:,:]
            pixel_counts[j,i] = curr_pred.sum()
    
    # normalize the rows
    #for i in range(pixel_counts.shape[0]):
    #    pixel_counts[i,:] *= (1/pixel_counts[i,:].sum())
    return pixel_counts


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


class ConfusionMeter:
    """
    Maintains a confusion matrix for a given calssification problem.
    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.

    Parameters:
        k (int): number of classes in the classification problem
        normalized (boolean): Determines whether or not the confusion matrix
            is normalized or not

    Copied from https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    to avoid installation of the entire torchnet package!

    BSD 3-Clause License

    Copyright (c) 2017- Sergey Zagoruyko,
    Copyright (c) 2017- Sasank Chilamkurthy,
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    def __init__(self, k, normalized=False):
        super(ConfusionMeter, self).__init__()
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """
        Computes the confusion matrix of K x K size where K is no of classes

        Paramaters:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors
        """

        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bin-counting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


def accuracy(output, target, topk=(1,)):
    """
    Evaluates a model's top k accuracy

    Parameters:
        output (torch.autograd.Variable): model output
        target (torch.autograd.Variable): ground-truths/labels
        topk (list): list of integers specifying top-k precisions
            to be computed

    Returns:
        float: percentage of correct predictions
    """

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SegConfusionMeter():
    def __init__(self, num_classes):
        self.conf = np.ndarray((num_classes, num_classes), dtype=np.float32)
        self.normalized = True
        self.reset()
        return

    def reset(self):
        self.conf.fill(0)
        return

    def add(self, summed_pixel_predictions):
        self.conf += summed_pixel_predictions
        return

    def value(self):
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf