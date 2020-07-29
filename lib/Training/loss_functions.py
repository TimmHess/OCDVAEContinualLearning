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


def loss_fn_kd(scores, target_scores, T=2.):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].
    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    log_scores_norm = log_scores_norm[:,:target_scores.shape[1],:,:]
    targets_norm = F.softmax(target_scores / T, dim=1)
    targets_norm = targets_norm[:,:,:,:]

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


def unified_loss_function_kl_regularized(output_samples_classification, target, output_samples_recon, inp, mu, std, prev_mu, prev_std, device, args):
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
        mu_prev (torch.Tensor): Encoder (recognition model's) mean vector (not including a batch size).
        std_std (torch.Tensor): Encoder (recognition model's) standard deviation vector (not including a batch size).
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
    prev_mu = prev_mu.repeat(mu.size(0),1)
    prev_std = prev_std.repeat(std.size(0),1)
    kld = get_kl(mu, std, prev_mu, prev_std)
    
    #kld_test = -0.5 * torch.sum(1 + torch.log(eps + std ** 2) - (mu ** 2) - (std ** 2)) / torch.numel(mu)
    #print(kld.cpu().item(), kld_test.cpu().item())
    #sys.exit()

    return cl, rl, kld
