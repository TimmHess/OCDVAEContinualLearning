import torch
import torch.nn as nn
import torch.nn.functional as F

def get_kl(m, v, m0, v0):
    # adapted from: https://github.com/bunkerj/mlmi4-vcl/blob/master/src/KL.py

    # numerical value for stability of log computation
    eps = 1e-8

    constTerm = -0.5 * m.numel()
    logStdDiff = 0.5 * torch.sum(torch.log(eps+v0**2)-torch.log(eps+v**2))
    muDiffTerm = 0.5 * torch.sum((v**2 + (m0-m)**2) / v0**2)
    return (constTerm + logStdDiff + muDiffTerm) / torch.numel(m)


# def loss_fn_kd(scores, target_scores, T=2.):
#     """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].
#     Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
#     'Hyperparameter': temperature"""
    
#     #print("score", scores.shape)
#     #print("target", target_scores.shape)

#     log_scores_norm = F.log_softmax(scores / T, dim=1)
#     log_scores_norm = log_scores_norm[:,:target_scores.shape[1]] # log_scores_norm[:,:target_scores.shape[1],:,:]
#     targets_norm = F.softmax(target_scores / T, dim=1)
#     #targets_norm = targets_norm[:,:,:,:]

#     # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
#     KD_loss_unnorm = -(targets_norm * log_scores_norm)
#     #print(KD_loss_unnorm.shape)
#     KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                      #--> sum over classes
#     #print(KD_loss_unnorm.shape)
#     KD_loss_unnorm = KD_loss_unnorm.mean()                          #--> average over batch

#     # normalize
#     KD_loss = KD_loss_unnorm * T**2
#     return KD_loss


def loss_fn_kd(scores, target_scores, T=2.):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].
    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""
    
    #print("score", scores.shape)
    #print("target", target_scores.shape)

    device = scores.device

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    n = scores.size(1)
    if n > target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)


    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    #print(KD_loss_unnorm.shape)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                      #--> sum over classes
    #print(KD_loss_unnorm.shape)
    KD_loss_unnorm = KD_loss_unnorm.mean()                          #--> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T**2
    return KD_loss

def loss_fn_kd_multihead(scores, target_scores, task_sizes, T=2.):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].
    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""
    
    #print("score", scores.shape)
    #print("target", target_scores.shape)

    device = scores.device

    # caution! this works only if all tasks are same size and score is dividable (luckily this is provided by this framework) 
    KD_losses = torch.zeros(scores.size(1) // task_sizes).to(device)
    for i in range(scores.size(1) // task_sizes): 
        log_scores_norm = F.log_softmax(scores[:,i*task_sizes:(i+1)*task_sizes] / T, dim=1)
        targets_norm = F.softmax(target_scores[:,i*task_sizes:(i+1)*task_sizes] / T, dim=1)

        # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
        KD_loss_unnorm = -(targets_norm * log_scores_norm)
        #print(KD_loss_unnorm.shape)
        KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                      #--> sum over classes
        #print(KD_loss_unnorm.shape)
        KD_loss_unnorm = KD_loss_unnorm.mean()                          #--> average over batch

        # normalize
        KD_loss = KD_loss_unnorm * T**2

        KD_losses[i] = KD_loss

    KD_losses = KD_losses.sum()
    return KD_losses


def loss_fn_kd_2d(scores, target_scores, T=2.):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].
    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    #log_scores_norm = log_scores_norm[:,:target_scores.shape[1],:,:]
    targets_norm = F.softmax(target_scores / T, dim=1)
    #targets_norm = targets_norm[:,:,:,:]

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    n = scores.size(1)
    if n>target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1), target_scores.size(2), target_scores.size(3))
        zeros_to_add = zeros_to_add.to(device)
        #print("targets_norm", target_norm.size())
        #print("zeros_add", zeros_to_add.size())
        targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    #print(KD_loss_unnorm.shape)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                      #--> sum over classes
    #print(KD_loss_unnorm.shape)
    KD_loss_unnorm = KD_loss_unnorm.mean()                          #--> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T**2
    return KD_loss


def unified_loss_function(output_samples_classification, target, output_samples_recon, inp, mu, std, device, args):
    """
    Computes the unified model's joint loss function consisting of a term for reconstruction, a KL term between
    approximate posterior and prior and the loss for the generative classifier. The number of variational samples
    is one per default, as specified in the command line parser and typically is how VAE models and also our unified
    model is trained. We have added the option to flexibly work with an arbitrary amount of samples.

    Parameters:
        output_samples_classification (torch.Tensor): Mini-batch of var_sample many classification prediction values.
        target (torch.Tensor): Classification targets for each element in the mini-batch.
        output_samples_recon (torch.Tensor): Mini-batch of var_sample many reconstructions.
        inp (torch.Tensor): The input mini-batch (before noise), aka the reconstruction loss' target.
        mu (torch.Tensor): Encoder (recognition model's) mini-batch of mean vectors.
        std (torch.Tensor): Encoder (recognition model's) mini-batch of standard deviation vectors.
        device (str): Device for computation.
        args (dict): Command line parameters. Needs to contain autoregression (bool).

    Returns:
        float: normalized classification loss
        float: normalized reconstruction loss
        float: normalized KL divergence
    """

    # for autoregressive models the decoder loss term corresponds to a classification based on 256 classes (for each
    # pixel value), i.e. a 256-way Softmax and thus a cross-entropy loss.
    # For regular decoders the loss is the reconstruction negative-log likelihood.
    if args.autoregression:
        recon_loss = nn.CrossEntropyLoss(reduction='sum')
    else:
        recon_loss = nn.BCEWithLogitsLoss(reduction='sum')

    class_loss = nn.CrossEntropyLoss(reduction='sum')

    # Place-holders for the final loss values over all latent space samples
    recon_losses = torch.zeros(output_samples_recon.size(0)).to(device)
    cl_losses = torch.zeros(output_samples_classification.size(0)).to(device)

    # numerical value for stability of log computation
    eps = 1e-8

    # loop through each sample for each input and calculate the correspond loss. Normalize the losses.
    for i in range(output_samples_classification.size(0)):
        cl_losses[i] = class_loss(output_samples_classification[i], target) / torch.numel(target)
        recon_losses[i] = recon_loss(output_samples_recon[i], inp) / torch.numel(inp)

    # average the loss over all samples per input
    cl = torch.mean(cl_losses, dim=0)
    rl = torch.mean(recon_losses, dim=0)

    # Compute the KL divergence, normalized by latent dimensionality
    kld = -0.5 * torch.sum(1 + torch.log(eps + std ** 2) - (mu ** 2) - (std ** 2)) / torch.numel(mu)

    # DEBUG
    #mu0 = torch.zeros_like(mu)
    #print(mu0.shape)
    #std0 = torch.ones_like(std)
    #kld_test = get_kl(mu, std, mu0, std0)
    #print(kld.cpu().item(), kld_test.cpu().item())

    return cl, rl, kld


def unified_loss_function_multihead(output_samples_classification, target, output_samples_recon, inp, mu, std, device, args):
    """
    Computes the unified model's joint loss function consisting of a term for reconstruction, a KL term between
    approximate posterior and prior and the loss for the generative classifier. The number of variational samples
    is one per default, as specified in the command line parser and typically is how VAE models and also our unified
    model is trained. We have added the option to flexibly work with an arbitrary amount of samples.

    Parameters:
        output_samples_classification (torch.Tensor): Mini-batch of var_sample many classification prediction values.
        target (torch.Tensor): Classification targets for each element in the mini-batch.
        output_samples_recon (torch.Tensor): Mini-batch of var_sample many reconstructions.
        inp (torch.Tensor): The input mini-batch (before noise), aka the reconstruction loss' target.
        mu (torch.Tensor): Encoder (recognition model's) mini-batch of mean vectors.
        std (torch.Tensor): Encoder (recognition model's) mini-batch of standard deviation vectors.
        device (str): Device for computation.
        args (dict): Command line parameters. Needs to contain autoregression (bool).

    Returns:
        float: normalized classification loss
        float: normalized reconstruction loss
        float: normalized KL divergence
    """

    # for autoregressive models the decoder loss term corresponds to a classification based on 256 classes (for each
    # pixel value), i.e. a 256-way Softmax and thus a cross-entropy loss.
    # For regular decoders the loss is the reconstruction negative-log likelihood.
    if args.autoregression:
        recon_loss = nn.CrossEntropyLoss(reduction='sum')
    else:
        recon_loss = nn.BCEWithLogitsLoss(reduction='sum')

    class_loss = nn.CrossEntropyLoss(reduction='sum')

    # Place-holders for the final loss values over all latent space samples
    recon_losses = torch.zeros(output_samples_recon.size(0)).to(device)
    cl_losses = torch.zeros(output_samples_classification.size(0)).to(device)

    # numerical value for stability of log computation
    eps = 1e-8

    # loop through each sample for each input and calculate the correspond loss. Normalize the losses.
    for i in range(output_samples_classification.size(0)):
        # calculate class loss only for current head (most recently added neurons)
        cl_losses[i] = class_loss(output_samples_classification[i], target) / torch.numel(target)
        recon_losses[i] = recon_loss(output_samples_recon[i], inp) / torch.numel(inp)

    # average the loss over all samples per input
    cl = torch.mean(cl_losses, dim=0)
    rl = torch.mean(recon_losses, dim=0)

    # Compute the KL divergence, normalized by latent dimensionality
    kld = -0.5 * torch.sum(1 + torch.log(eps + std ** 2) - (mu ** 2) - (std ** 2)) / torch.numel(mu)
    return cl, rl, kld



def unified_loss_function_no_vae(output_samples_classification, target, output_samples_recon, inp, mu, std, device, args):
    """
    Computes the unified model's joint loss function consisting of a term for reconstruction, a KL term between
    approximate posterior and prior and the loss for the generative classifier. The number of variational samples
    is one per default, as specified in the command line parser and typically is how VAE models and also our unified
    model is trained. We have added the option to flexibly work with an arbitrary amount of samples.

    Parameters:
        output_samples_classification (torch.Tensor): Mini-batch of var_sample many classification prediction values.
        target (torch.Tensor): Classification targets for each element in the mini-batch.
        output_samples_recon (torch.Tensor): Mini-batch of var_sample many reconstructions.
        inp (torch.Tensor): The input mini-batch (before noise), aka the reconstruction loss' target.
        mu (torch.Tensor): Encoder (recognition model's) mini-batch of mean vectors.
        std (torch.Tensor): Encoder (recognition model's) mini-batch of standard deviation vectors.
        device (str): Device for computation.
        args (dict): Command line parameters. Needs to contain autoregression (bool).

    Returns:
        float: normalized classification loss
        float: normalized reconstruction loss
        float: normalized KL divergence
    """

    # for autoregressive models the decoder loss term corresponds to a classification based on 256 classes (for each
    # pixel value), i.e. a 256-way Softmax and thus a cross-entropy loss.
    # For regular decoders the loss is the reconstruction negative-log likelihood.
    if not args.is_segmentation:
        class_loss = nn.CrossEntropyLoss(reduction='sum')
    else:
        weight = torch.ones(output_samples_classification.size(2))
        #print(output_samples_classification.shape)
        #print(weight.shape)
        weight[0] = 0.05
        #print(weight)
        weight = weight.float().to(device)
        class_loss = nn.CrossEntropyLoss(reduction='sum', weight=weight)

    # Place-holders for the final loss values over all latent space samples
    cl_losses = torch.zeros(output_samples_classification.size(0)).to(device)

    # numerical value for stability of log computation
    eps = 1e-8

    # loop through each sample for each input and calculate the correspond loss. Normalize the losses.
    for i in range(output_samples_classification.size(0)):
        cl_losses[i] = class_loss(output_samples_classification[i], target) / torch.numel(target)

    # average the loss over all samples per input
    cl = torch.mean(cl_losses, dim=0)

    return cl, None, None
