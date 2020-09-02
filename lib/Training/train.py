import time
import torch
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import accuracy
from lib.Training.loss_functions import loss_fn_kd
from lib.Utility.visualization import visualize_image_grid
import lib.Models.si as SI


def train(Dataset, model, criterion, epoch, iteration, optimizer, writer, device, args, save_path):
    """
    Trains/updates the model for one epoch on the training dataset.

    Parameters:
        Dataset (torch.utils.data.Dataset): The dataset
        model (torch.nn.module): Model to be trained
        criterion (torch.nn.criterion): Loss function
        epoch (int): Continuous epoch counter
        optimizer (torch.optim.optimizer): optimizer instance like SGD or Adam
        writer (tensorboard.SummaryWriter): TensorBoard writer instance
        device (str): device name where data is transferred to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain print_freq (int) and log_weights (bool).
    """

    # Create instances to accumulate losses etc.
    class_losses = AverageMeter()
    recon_losses = AverageMeter()
    kld_losses = AverageMeter()
    losses = AverageMeter()
    lwf_losses = AverageMeter()
    si_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    # train
    for i, (inp, target) in enumerate(Dataset.train_loader):
        inp = inp.to(device)
        target = target.to(device)

        if epoch % args.epochs == 0 and i == 0:
            visualize_image_grid(inp, writer, epoch + 1, 'train_inp_snapshot', save_path)

        recon_target = inp
        class_target = target

        # this needs to be below the line where the reconstruction target is set
        # sample and add noise to the input (but not to the target!).
        if args.denoising_noise_value > 0.0:
            noise = torch.randn(inp.size()).to(device) * args.denoising_noise_value
            inp = inp + noise

        # measure data loading time
        data_time.update(time.time() - end)

        # compute model forward
        class_samples, recon_samples, mu, std = model(inp)

        # if we have an autoregressive model variant, further calculate the corresponding layers.
        if args.autoregression:
            recon_samples_autoregression = torch.zeros(recon_samples.size(0), inp.size(0), 256, inp.size(1),
                                                       inp.size(2), inp.size(3)).to(device)
            for j in range(model.module.num_samples):
                recon_samples_autoregression[j] = model.module.pixelcnn(recon_target,
                                                                        torch.sigmoid(recon_samples[j])).contiguous()
            recon_samples = recon_samples_autoregression
            # set the target to work with the 256-way Softmax
            recon_target = (recon_target * 255).long()

        # calculate loss
        #if args.use_kl_regularization:
        #    class_loss, recon_loss, kld_loss = criterion(class_samples, class_target, recon_samples, recon_target, mu, std,
        #                                                model.module.prev_mu, model.module.prev_std, device, args)
        #else:
        class_loss, recon_loss, kld_loss = criterion(class_samples, class_target, recon_samples, recon_target, mu, std,
                                                    device, args)

        # add the individual loss components together and weight the KL term.
        loss = class_loss + recon_loss + args.var_beta * kld_loss

        # calculate lwf loss (if there is a previous model stored)
        if args.use_lwf and model.module.prev_model:
            # get prediction from previous model
            prev_pred_class_samples, _, _, _ = model.module.prev_model(inp)
            prev_cl_losses = torch.zeros(prev_pred_class_samples.size(0)).to(device)
            # loop through each sample for each input and calculate the correspond loss. Normalize the losses.
            for s in range(prev_pred_class_samples.size(0)):
                prev_cl_losses[s] = loss_fn_kd(class_samples[s], prev_pred_class_samples[s]) #/ torch.numel(target)
            
            # average the loss over all samples per input
            cl_lwf = torch.mean(prev_cl_losses, dim=0)

            # add lwf loss to overall loss
            loss += args.lmda * cl_lwf

            # record lwf losses
            lwf_losses.update(cl_lwf.item(), inp.size(0))

        # calculate SI loss (if SI is initialized)
        if args.use_si and model.module.si_storage.is_initialized:
            loss_si = args.lmda * SI.surrogate_loss(model.module, model.module.si_storage)
            print(loss_si.item())
            loss += loss_si

            si_losses.update(loss_si.item(), inp.size(0))

        # take mean to compute accuracy. Note if variational samples are 1 this only gets rid of a dummy dimension.
        output = torch.mean(class_samples, dim=0)

        # record precision/accuracy and losses
        prec1 = accuracy(output, target)[0]
        top1.update(prec1.item(), inp.size(0))
        losses.update((class_loss + recon_loss + kld_loss).item(), inp.size(0))
        class_losses.update(class_loss.item(), inp.size(0))
        recon_losses.update(recon_loss.item(), inp.size(0))
        kld_losses.update(kld_loss.item(), inp.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # SI: update running si paramters
        if args.use_si:
            SI.update_si_parameters(model.module, model.module.si_storage)
            #print("SI: Updated running parameters")

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress
        if i % args.print_freq == 0:
            if args.use_lwf and model.module.prev_model:
                print('Training: [{0}][{1}/{2}]\t' 
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Class Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'
                    'LwF Loss {lwf_loss:.4f}\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'
                    'KL {KLD_loss.val:.4f} ({KLD_loss.avg:.4f})'.format(
                    epoch, i, len(Dataset.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, cl_loss=class_losses, top1=top1,
                    recon_loss=recon_losses, KLD_loss=kld_losses, lwf_loss=cl_lwf.item()))
            if args.use_si and model.module.si_storage.is_initialized:
                print('Training: [{0}][{1}/{2}]\t' 
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Class Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'
                    'SI Loss {si_loss:.4f}\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'
                    'KL {KLD_loss.val:.4f} ({KLD_loss.avg:.4f})'.format(
                    epoch, i, len(Dataset.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, cl_loss=class_losses, top1=top1,
                    recon_loss=recon_losses, KLD_loss=kld_losses, si_loss=loss_si.item()))
            
            else:
                print('Training: [{0}][{1}/{2}]\t' 
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Class Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'
                    'KL {KLD_loss.val:.4f} ({KLD_loss.avg:.4f})'.format(
                    epoch, i, len(Dataset.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, cl_loss=class_losses, top1=top1,
                    recon_loss=recon_losses, KLD_loss=kld_losses))

        # increase iteration
        iteration[0] += 1

    # TensorBoard summary logging
    writer.add_scalar('training/train_precision@1', top1.avg, epoch)
    writer.add_scalar('training/train_average_loss', losses.avg, epoch)
    writer.add_scalar('training/train_KLD', kld_losses.avg, epoch)
    writer.add_scalar('training/train_class_loss', class_losses.avg, epoch)
    writer.add_scalar('training/train_recon_loss', recon_losses.avg, epoch)

    writer.add_scalar('training/train_precision_itr@1', top1.avg, iteration[0])
    writer.add_scalar('training/train_average_loss_itr', losses.avg, iteration[0])
    writer.add_scalar('training/train_KLD_itr', kld_losses.avg, iteration[0])
    writer.add_scalar('training/train_class_loss_itr', class_losses.avg, iteration[0])
    writer.add_scalar('training/train_recon_loss_itr', recon_losses.avg, iteration[0])
    
    if args.use_lwf:
        writer.add_scalar('training/train_lwf_loss', lwf_losses.avg, iteration[0])
    if args.use_si:
        writer.add_scalar('training/train_si_loss', si_losses.avg, iteration[0])

    # If the log weights argument is specified also add parameter and gradient histograms to TensorBoard.
    if args.log_weights:
        # Histograms and distributions of network parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram(tag, value.data.cpu().numpy(), epoch, bins="auto")
            # second check required for buffers that appear in the parameters dict but don't receive gradients
            if value.requires_grad and value.grad is not None:
                writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch, bins="auto")

    print(' * Train: Loss {loss.avg:.5f} Prec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))
