import time
import math
import torch
import torch.nn.functional as F
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import ConfusionMeter
from lib.Utility.metrics import accuracy, iou_class_condtitional, iou_to_accuracy
from lib.Utility.visualization import visualize_confusion
from lib.Utility.visualization import visualize_image_grid
from lib.Models.architectures import consolidate_classifier


def validate(Dataset, model, criterion, epoch, iteration, writer, device, save_path, args):
    """
    Evaluates/validates the model

    Parameters:
        Dataset (torch.utils.data.Dataset): The dataset
        model (torch.nn.module): Model to be evaluated/validated
        criterion (torch.nn.criterion): Loss function
        epoch (int): Epoch counter
        writer (tensorboard.SummaryWriter): TensorBoard writer instance
        device (str): device name where data is transferred to
        save_path (str): path to save data to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain print_freq (int), epochs (int), incremental_data (bool), autoregression (bool),
            visualization_epoch (int), cross_dataset (bool), num_base_tasks (int), num_increment_tasks (int) and
            patch_size (int).

    Returns:
        float: top1 precision/accuracy
        float: average loss
    """

    # initialize average meters to accumulate values
    class_losses = AverageMeter()
    recon_losses_nat = AverageMeter()
    kld_losses = AverageMeter()
    losses = AverageMeter()

    # for autoregressive models add an additional instance for reconstruction loss in bits per dimension
    if args.autoregression:
        recon_losses_bits_per_dim = AverageMeter()

    # for continual learning settings also add instances for base and new reconstruction metrics
    # corresponding accuracy values are calculated directly from the confusion matrix below
    if args.incremental_data and ((epoch + 1) % args.epochs == 0 and epoch > 0):
        recon_losses_new_nat = AverageMeter()
        recon_losses_base_nat = AverageMeter()
        if args.autoregression:
            recon_losses_new_bits_per_dim = AverageMeter()
            recon_losses_base_bits_per_dim = AverageMeter()

    batch_time = AverageMeter()
    top1 = AverageMeter()
    
    # confusion matrix
    if not args.incremental_instance:
        confusion = ConfusionMeter(model.module.num_classes, normalized=True)
    else:
        if args.full_conf_mat:
            confusion = ConfusionMeter(model.module.num_classes, normalized=True)
        else:
            confusion = ConfusionMeter(Dataset.num_classes, normalized=True)

    # switch to evaluate mode
    model.eval()

    if args.use_si and not args.is_multiheaded:
        # load consolidated weights for classifier
        consolidate_classifier(model.module)
        print("SI: Consolidated classifier weights for validation")

    end = time.time()

    # evaluate the entire validation dataset
    with torch.no_grad():
        
        if args.is_multiheaded: #and not args.incremental_instance:
            # created dataset loader for each validation set in mh_valsets(ordered by task/head)
            for valset_index in range(len(Dataset.mh_valsets)):
                val_loader = torch.utils.data.DataLoader(Dataset.mh_valsets[valset_index], batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.workers, pin_memory=torch.cuda.is_available(), drop_last=True)
                # iterate validation set
                for i, (inp, target) in enumerate(val_loader):
                    if not args.incremental_instance:
                        # convert targets to respective heads space (indicated by val_index)
                        target_head = target.clone()
                        for i in range(target.size(0)):
                            target_head[i] = Dataset.maps_target_head[valset_index][target.numpy()[i]]
                        target_head = target_head.to(device) # head space
                    
                    inp = inp.to(device)
                    target = target.to(device) # (global) space
                    
                    recon_target = inp
                    if not args.incremental_instance:
                        class_target = target_head
                    else:
                        class_target = target

                    # visualize inp
                    if epoch % args.epochs == 0 and i == 0:
                        visualize_image_grid(inp, writer, epoch + 1, 'val_inp_snapshot', save_path)

                    # compute output
                    class_samples, recon_samples, mu, std = model(inp) # class_samples are (global) target space

                    # computer loss for respective head
                    if not args.incremental_instance:
                        head_start = (valset_index) * args.num_increment_tasks 
                        head_end = (valset_index+1) * args.num_increment_tasks
                    else:
                        head_start = (valset_index) * Dataset.num_classes 
                        head_end = (valset_index+1) * Dataset.num_classes
                    #print("head start, end:", head_start, head_end)
                    class_loss, recon_loss, kld_loss = criterion(class_samples[:,:,head_start:head_end], class_target, recon_samples, recon_target, mu, std,
                                                            device, args)

                    # take mean to compute accuracy
                    # (does nothing if there isn't more than 1 sample per input other than removing dummy dimension)
                    class_output = torch.mean(class_samples, dim=0)
                    #if not args.incremental_instance:
                    #    class_output = torch.mean(class_samples, dim=0)
                    #else:
                    #    class_output = torch.mean(class_samples[:,:,head_start:head_end], dim=0)
                    
                    if not args.no_vae:
                        recon_output = torch.mean(recon_samples, dim=0)

                    # convert model outputs back to (global) target space to map back to known evaluation
                    # copy only current head output back into an zero-ed tensor
                    if not args.incremental_instance or args.full_conf_mat:
                        task_class_output = torch.zeros_like(class_output).fill_(-100000)
                        #print(task_class_output.shape)
                        task_class_output[:,head_start:head_end] = class_output[:,head_start:head_end]
                        #print("task_class_output", task_class_output)

                        # measure accuracy, record loss, fill confusion matrix
                        if args.full_conf_mat:
                            for target_index in range(target.size(0)):
                                target[target_index] += (valset_index*Dataset.num_classes)
                        prec1 = accuracy(task_class_output, target)[0]
                        top1.update(prec1.item(), inp.size(0))
                        #confusion.add(class_output.data, target)
                        confusion.add(task_class_output.data, target)
                    
                    else:
                        # cut only active head from class_output
                        class_output = class_output[:,head_start:head_end]
                        prec1 = accuracy(class_output, target)[0]
                        top1.update(prec1.item(), inp.size(0))
                        confusion.add(class_output.data, target)

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    # skipped code for autoregressive model

                    # If not autoregressive simply apply the Sigmoid and visualize
                    if not args.no_vae:
                        recon = torch.sigmoid(recon_output)
                        if (i == (len(Dataset.val_loader) - 2)) and (epoch % args.visualization_epoch == 0):
                            visualize_image_grid(recon, writer, epoch + 1, 'reconstruction_snapshot', save_path)
                    
                    # update the respective loss values. To be consistent with values reported in the literature we scale
                    # our normalized losses back to un-normalized values.
                    # For the KLD this also means the reported loss is not scaled by beta, to allow for a fair comparison
                    # across potential weighting terms.
                    class_losses.update(class_loss.item() * model.module.num_classes, inp.size(0))
                    if args.no_vae:
                        losses.update(class_loss.item(), inp.size(0))
                    else:
                        kld_losses.update(kld_loss.item() * model.module.latent_dim, inp.size(0))
                        recon_losses_nat.update(recon_loss.item() * inp.size()[1:].numel(), inp.size(0))
                        losses.update((class_loss + recon_loss + kld_loss).item(), inp.size(0))

                    # if we are learning continually, we need to calculate the base and new reconstruction losses at the end
                    # of each task increment.
                    if args.incremental_data and not args.incremental_instance and ((epoch + 1) % args.epochs == 0 and epoch > 0):
                        for j in range(inp.size(0)):
                            # get the number of classes for cross-dataset or class incremental scenarios.
                            if args.cross_dataset:
                                base_classes = range(sum(Dataset.num_classes_per_task[:args.num_base_tasks]))
                                new_classes = range(sum(Dataset.num_classes_per_task[:len(Dataset.seen_tasks) -
                                                                                    args.num_increment_tasks]),
                                                    sum(Dataset.num_classes_per_task[:len(Dataset.seen_tasks)]))
                            else:
                                base_classes = model.module.seen_tasks[:args.num_base_tasks + 1]
                                new_classes = model.module.seen_tasks[-args.num_increment_tasks:]

                            # skipped code for autoregressive model

                            # If the input belongs to one of the base classes also update base metrics
                            if not args.no_vae:
                                if class_target[j].item() in base_classes:
                                    recon_losses_base_nat.update(F.binary_cross_entropy(recon[j], recon_target[j]), 1)
                                # if the input belongs to one of the new classes also update new metrics
                                elif class_target[j].item() in new_classes:
                                    recon_losses_new_nat.update(F.binary_cross_entropy(recon[j], recon_target[j]), 1)

                    # If we are at the end of validation, create one mini-batch of example generations. Only do this every
                    # other epoch specified by visualization_epoch to avoid generation of lots of images and computationally
                    # expensive calculations of the autoregressive model's generation.
                    if not args.no_vae:
                        if i == (len(Dataset.val_loader) - 2) and epoch % args.visualization_epoch == 0:
                            # generation
                            gen = model.module.generate()

                            if args.autoregression:
                                gen = model.module.pixelcnn.generate(gen)
                            visualize_image_grid(gen, writer, epoch + 1, 'generation_snapshot', save_path)

                    # Print progress
                    if i % args.print_freq == 0:
                        print('Validate: [{0}][{1}/{2}]\t' 
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Class Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'
                            'KL {KLD_loss.val:.4f} ({KLD_loss.avg:.4f})'.format(
                            epoch+1, i, len(Dataset.val_loader), batch_time=batch_time, loss=losses, cl_loss=class_losses,
                            top1=top1, recon_loss=recon_losses_nat, KLD_loss=kld_losses))


        else:
            for i, (inp, target) in enumerate(Dataset.val_loader):
                inp = inp.to(device)
                target = target.to(device)

                recon_target = inp
                class_target = target

                # visualize inp
                if epoch % args.epochs == 0 and i == 0:
                    visualize_image_grid(inp, writer, epoch + 1, 'val_inp_snapshot', save_path)

                # compute output
                class_samples, recon_samples, mu, std = model(inp)

                # for autoregressive models convert the target to 0-255 integers and compute the autoregressive decoder
                # for each sample
                if args.autoregression:
                    recon_target = (recon_target * 255).long()
                    recon_samples_autoregression = torch.zeros(recon_samples.size(0), inp.size(0), 256, inp.size(1),
                                                            inp.size(2), inp.size(3)).to(device)
                    for j in range(model.module.num_samples):
                        recon_samples_autoregression[j] = model.module.pixelcnn(
                            inp, torch.sigmoid(recon_samples[j])).contiguous()
                    recon_samples = recon_samples_autoregression

                # compute loss
                class_loss, recon_loss, kld_loss = criterion(class_samples, class_target, recon_samples, recon_target, mu, std,
                                                            device, args)

                # For autoregressive models also update the bits per dimension value, converted from the obtained nats
                if args.autoregression:
                    recon_losses_bits_per_dim.update(recon_loss.item() * math.log2(math.e), inp.size(0))

                # take mean to compute accuracy
                # (does nothing if there isn't more than 1 sample per input other than removing dummy dimension)
                class_output = torch.mean(class_samples, dim=0)
                if not args.no_vae:
                    recon_output = torch.mean(recon_samples, dim=0)

                # measure accuracy, record loss, fill confusion matrix
                if not args.is_segmentation:
                    prec1 = accuracy(class_output, target)[0]
                    top1.update(prec1.item(), inp.size(0))
                    confusion.add(class_output.data, target)
                else:
                    ious_cc = iou_class_condtitional(pred=class_output.clone(), target=target.clone())
                    prec1 = iou_to_accuracy(ious_cc)
                    top1.update(prec1.item(), inp.size(0))
                    # no confusion to add because of missing targets?

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # for autoregressive models generate reconstructions by sequential sampling from the
                # multinomial distribution (Reminder: the original output is a 255 way Softmax as PixelVAEs are posed as a
                # classification problem). This serves two purposes: visualization of reconstructions and computation of
                # a reconstruction loss in nats using a BCE loss, comparable to that of a regular VAE.
                if not args.no_vae:
                    recon_target = inp
                    if args.autoregression:
                        recon = torch.zeros((inp.size(0), inp.size(1), inp.size(2), inp.size(3))).to(device)
                        for h in range(inp.size(2)):
                            for w in range(inp.size(3)):
                                for c in range(inp.size(1)):
                                    probs = torch.softmax(recon_output[:, :, c, h, w], dim=1).data
                                    pixel_sample = torch.multinomial(probs, 1).float() / 255.
                                    recon[:, c, h, w] = pixel_sample.squeeze()

                        if (epoch % args.visualization_epoch == 0) and (i == (len(Dataset.val_loader) - 2)):
                            visualize_image_grid(recon, writer, epoch + 1, 'reconstruction_snapshot', save_path)

                        recon_loss = F.binary_cross_entropy(recon, recon_target)
                    else:
                        # If not autoregressive simply apply the Sigmoid and visualize
                        recon = torch.sigmoid(recon_output)
                        if (i == (len(Dataset.val_loader) - 2)) and (epoch % args.visualization_epoch == 0):
                            visualize_image_grid(recon, writer, epoch + 1, 'reconstruction_snapshot', save_path)

                # update the respective loss values. To be consistent with values reported in the literature we scale
                # our normalized losses back to un-normalized values.
                # For the KLD this also means the reported loss is not scaled by beta, to allow for a fair comparison
                # across potential weighting terms.
                class_losses.update(class_loss.item() * model.module.num_classes, inp.size(0))
                if not args.no_vae:
                    kld_losses.update(kld_loss.item() * model.module.latent_dim, inp.size(0))
                    recon_losses_nat.update(recon_loss.item() * inp.size()[1:].numel(), inp.size(0))
                    losses.update((class_loss + recon_loss + kld_loss).item(), inp.size(0))

                # if we are learning continually, we need to calculate the base and new reconstruction losses at the end
                # of each task increment.
                if args.incremental_data and not args.incremental_instance and ((epoch + 1) % args.epochs == 0 and epoch > 0):
                    for j in range(inp.size(0)):
                        # get the number of classes for cross-dataset or class incremental scenarios.
                        if args.cross_dataset:
                            base_classes = range(sum(Dataset.num_classes_per_task[:args.num_base_tasks]))
                            new_classes = range(sum(Dataset.num_classes_per_task[:len(Dataset.seen_tasks) -
                                                                                args.num_increment_tasks]),
                                                sum(Dataset.num_classes_per_task[:len(Dataset.seen_tasks)]))
                        else:
                            base_classes = model.module.seen_tasks[:args.num_base_tasks + 1]
                            new_classes = model.module.seen_tasks[-args.num_increment_tasks:]

                        if args.autoregression:
                            rec = recon_output[j].view(1, recon_output.size(1), recon_output.size(2),
                                                    recon_output.size(3), recon_output.size(4))
                            rec_tar = recon_target[j].view(1, recon_target.size(1), recon_target.size(2),
                                                        recon_target.size(3))

                        # If the input belongs to one of the base classes also update base metrics
                        if not args.no_vae:
                            if class_target[j].item() in base_classes:
                                if args.autoregression:
                                    recon_losses_base_bits_per_dim.update(F.cross_entropy(rec, (rec_tar * 255).long()) *
                                                                        math.log2(math.e), 1)
                                recon_losses_base_nat.update(F.binary_cross_entropy(recon[j], recon_target[j]), 1)
                            # if the input belongs to one of the new classes also update new metrics
                            elif class_target[j].item() in new_classes:
                                if args.autoregression:
                                    recon_losses_new_bits_per_dim.update(F.cross_entropy(rec, (rec_tar * 255).long()) *
                                                                        math.log2(math.e), 1)
                                recon_losses_new_nat.update(F.binary_cross_entropy(recon[j], recon_target[j]), 1)

                # If we are at the end of validation, create one mini-batch of example generations. Only do this every
                # other epoch specified by visualization_epoch to avoid generation of lots of images and computationally
                # expensive calculations of the autoregressive model's generation.
                if not args.no_vae:
                    if i == (len(Dataset.val_loader) - 2) and epoch % args.visualization_epoch == 0:
                        # generation
                        gen = model.module.generate()

                        if args.autoregression:
                            gen = model.module.pixelcnn.generate(gen)
                        visualize_image_grid(gen, writer, epoch + 1, 'generation_snapshot', save_path)

                # Print progress
                if i % args.print_freq == 0:
                    print('Validate: [{0}][{1}/{2}]\t' 
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Class Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'
                        'KL {KLD_loss.val:.4f} ({KLD_loss.avg:.4f})'.format(
                        epoch+1, i, len(Dataset.val_loader), batch_time=batch_time, loss=losses, cl_loss=class_losses,
                        top1=top1, recon_loss=recon_losses_nat, KLD_loss=kld_losses))

    # TensorBoard summary logging
    writer.add_scalar('validation/val_precision@1', top1.avg, epoch)
    writer.add_scalar('validation/val_average_loss', losses.avg, epoch)
    writer.add_scalar('validation/val_class_loss', class_losses.avg, epoch)
    writer.add_scalar('validation/val_recon_loss_nat', recon_losses_nat.avg, epoch)
    writer.add_scalar('validation/val_KLD', kld_losses.avg, epoch)

    writer.add_scalar('validation/val_precision_itr@1', top1.avg, iteration[0])
    writer.add_scalar('validation/val_average_loss_itr', losses.avg, iteration[0])
    writer.add_scalar('validation/val_class_loss_itr', class_losses.avg, iteration[0])
    writer.add_scalar('validation/val_recon_loss_nat_itr', recon_losses_nat.avg, iteration[0])
    writer.add_scalar('validation/val_KLD_itr', kld_losses.avg, iteration[0])

    if args.autoregression:
        #writer.add_scalar('validation/val_recon_loss_bits_per_dim', recon_losses_bits_per_dim.avg, epoch)
        writer.add_scalar('validation/val_recon_loss_bits_per_dim', recon_losses_bits_per_dim.avg, iteration[0])

    print(' * Validation: Loss {loss.avg:.5f} Prec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))

    # At the end of training isolated, or at the end of every task visualize the confusion matrix
    if (epoch + 1) % args.epochs == 0 and epoch > 0:
        # visualize the confusion matrix
        if args.cross_dataset:
            visualize_confusion(writer, epoch + 1, confusion.value(), Dataset.task_to_idx, save_path)
        else:
            visualize_confusion(writer, epoch + 1, confusion.value(), Dataset.class_to_idx, save_path)

        # If we are in a continual learning scenario (which is not incremental instance), also use the confusion matrix to extract base and new precision.
        if args.incremental_data and not args.incremental_instance:
            prec1_base = 0.0
            prec1_new = 0.0
            if args.cross_dataset:
                for c in range(sum(Dataset.num_classes_per_task[:args.num_base_tasks])):
                    prec1_base += confusion.value()[c][c]
                prec1_base = prec1_base / sum(Dataset.num_classes_per_task[:args.num_base_tasks])
            else:
                # this has to be + 1 because the number of initial tasks is always one less than the amount of classes
                # i.e. 1 task is 2 classes etc.
                for c in range(args.num_base_tasks + 1):
                    prec1_base += confusion.value()[c][c]
                prec1_base = prec1_base / (args.num_base_tasks + 1)

            # For the first task "new" metrics are equivalent to "base"
            if (epoch + 1) / args.epochs == 1:
                prec1_new = prec1_base
                recon_losses_new_nat.avg = recon_losses_base_nat.avg
                if args.autoregression:
                    recon_losses_new_bits_per_dim.avg = recon_losses_base_bits_per_dim.avg
            else:
                if args.cross_dataset:
                    for c in range(sum(Dataset.num_classes_per_task[:len(Dataset.seen_tasks) -
                                                                    args.num_increment_tasks]),
                                   sum(Dataset.num_classes_per_task[:len(Dataset.seen_tasks)])):
                        prec1_new += confusion.value()[c][c]
                    prec1_new = prec1_new / (sum(Dataset.num_classes_per_task[:len(Dataset.seen_tasks)])
                                             - sum(Dataset.num_classes_per_task[:len(Dataset.seen_tasks) -
                                                                                args.num_increment_tasks]))
                else:
                    for c in range(args.num_increment_tasks):
                        prec1_new += confusion.value()[-c-1][-c-1]
                    prec1_new = prec1_new / args.num_increment_tasks

            # At the continual learning metrics to TensorBoard
            writer.add_scalar('validation/base_precision@1', prec1_base, len(model.module.seen_tasks)-1)
            writer.add_scalar('validation/new_precision@1', prec1_new, len(model.module.seen_tasks)-1)
            writer.add_scalar('validation/base_rec_loss_nats', recon_losses_base_nat.avg * args.patch_size *
                              args.patch_size * model.module.num_colors, len(model.module.seen_tasks) - 1)
            writer.add_scalar('validation/new_rec_loss_nats', recon_losses_new_nat.avg * args.patch_size *
                              args.patch_size * model.module.num_colors, len(model.module.seen_tasks) - 1)

            if args.autoregression:
                writer.add_scalar('validation/base_rec_loss_bits_per_dim',
                                  recon_losses_base_bits_per_dim.avg, len(model.module.seen_tasks) - 1)
                writer.add_scalar('validation/new_rec_loss_bits_per_dim',
                                  recon_losses_new_bits_per_dim.avg, len(model.module.seen_tasks) - 1)

            print(' * Incremental validation: Base Prec@1 {prec1_base:.3f} New Prec@1 {prec1_new:.3f}\t'
                  'Base Recon Loss {recon_losses_base_nat.avg:.3f} New Recon Loss {recon_losses_new_nat.avg:.3f}'
                  .format(prec1_base=100*prec1_base, prec1_new=100*prec1_new,
                          recon_losses_base_nat=recon_losses_base_nat, recon_losses_new_nat=recon_losses_new_nat))
        

        # For incremental instances we need to evaluate base_validation set and new_validation set
        if args.incremental_instance:
            prec1_base = 0.0
            prec1_new = 0.0
            top1_base = AverageMeter()
            top1_new = AverageMeter()

            # Base
            print("\n Computing Base Task Validation \n")
            with torch.no_grad():
                for i, (inp, target) in enumerate(Dataset.base_valset_loader):
                    inp = inp.to(device)
                    target = target.to(device)

                    recon_target = inp
                    class_target = target

                    # compute output
                    class_samples, recon_samples, mu, std = model(inp)

                    #for autoregressive models convert the target to 0-255 integers and compute the autoregressive decoder
                    #for each sample
                    if args.autoregression:
                        recon_target = (recon_target * 255).long()
                        recon_samples_autoregression = torch.zeros(recon_samples.size(0), inp.size(0), 256, inp.size(1),
                                                                inp.size(2), inp.size(3)).to(device)
                        for j in range(model.module.num_samples):
                            recon_samples_autoregression[j] = model.module.pixelcnn(
                                inp, torch.sigmoid(recon_samples[j])).contiguous()
                        recon_samples = recon_samples_autoregression

                    if args.is_multiheaded:
                            class_loss, recon_loss, kld_loss = criterion(class_samples[:,:,0:Dataset.num_classes], 
                                                                    class_target, recon_samples, recon_target, mu, std, device, args)
                    else:
                        class_loss, recon_loss, kld_loss = criterion(class_samples, class_target, recon_samples, recon_target, mu, std,
                                                                    device, args)

                    # For autoregressive models also update the bits per dimension value, converted from the obtained nats
                    if args.autoregression:
                        recon_losses_bits_per_dim.update(recon_loss.item() * math.log2(math.e), inp.size(0))

                    # take mean to compute accuracy
                    # (does nothing if there isn't more than 1 sample per input other than removing dummy dimension)
                    class_output = torch.mean(class_samples[:,:,0:Dataset.num_classes], dim=0)
                    if not args.no_vae:
                        recon_output = torch.mean(recon_samples, dim=0)

                    # measure accuracy, record loss, fill confusion matrix
                    prec1_base = accuracy(class_output, target)[0]
                    top1_base.update(prec1_base.item(), inp.size(0))

                    # for autoregressive models generate reconstructions by sequential sampling from the
                    # multinomial distribution (Reminder: the original output is a 255 way Softmax as PixelVAEs are posed as a
                    # classification problem).
                    if not args.no_vae:
                        recon_target = inp
                        if args.autoregression:
                            recon = torch.zeros((inp.size(0), inp.size(1), inp.size(2), inp.size(3))).to(device)
                            for h in range(inp.size(2)):
                                for w in range(inp.size(3)):
                                    for c in range(inp.size(1)):
                                        probs = torch.softmax(recon_output[:, :, c, h, w], dim=1).data
                                        pixel_sample = torch.multinomial(probs, 1).float() / 255.
                                        recon[:, c, h, w] = pixel_sample.squeeze()
                        else:
                            # If not autoregressive simply apply the Sigmoid
                            recon = torch.sigmoid(recon_output)

                        for j in range(inp.size(0)):
                            if args.autoregression:
                                rec = recon_output[j].view(1, recon_output.size(1), recon_output.size(2),
                                                            recon_output.size(3), recon_output.size(4))
                                rec_tar = recon_target[j].view(1, recon_target.size(1), recon_target.size(2),
                                                                recon_target.size(3))

                            # If the input belongs to one of the base classes also update base metrics
                            #if class_target[j].item() in base_classes:
                            if args.autoregression:
                                recon_losses_base_bits_per_dim.update(F.cross_entropy(rec, (rec_tar * 255).long()) *
                                                                        math.log2(math.e), 1)
                            recon_losses_base_nat.update(F.binary_cross_entropy(recon[j], recon_target[j]), 1)

            # New
            print("\n Computing New Task Validation \n")
            if (epoch + 1) / args.epochs == 1:
                #prec1_new = prec1_base
                top1_new = top1_base
                recon_losses_new_nat.avg = recon_losses_base_nat.avg
                if args.autoregression:
                    recon_losses_new_bits_per_dim.avg = recon_losses_base_bits_per_dim.avg
            else:
                with torch.no_grad():
                    for i, (inp, target) in enumerate(Dataset.new_valset_loader):
                        inp = inp.to(device)
                        target = target.to(device)

                        recon_target = inp
                        class_target = target

                        # compute output
                        class_samples, recon_samples, mu, std = model(inp)

                        #for autoregressive models convert the target to 0-255 integers and compute the autoregressive decoder
                        #for each sample
                        if args.autoregression:
                            recon_target = (recon_target * 255).long()
                            recon_samples_autoregression = torch.zeros(recon_samples.size(0), inp.size(0), 256, inp.size(1),
                                                                    inp.size(2), inp.size(3)).to(device)
                            for j in range(model.module.num_samples):
                                recon_samples_autoregression[j] = model.module.pixelcnn(
                                    inp, torch.sigmoid(recon_samples[j])).contiguous()
                            recon_samples = recon_samples_autoregression

                        # compute loss
                        if args.is_multiheaded:
                            class_loss, recon_loss, kld_loss = criterion(class_samples[:,:,-Dataset.num_classes:], 
                                                                        class_target, recon_samples, recon_target, mu, std, device, args)
                        else:
                            class_loss, recon_loss, kld_loss = criterion(class_samples, class_target, recon_samples, recon_target, mu, std,
                                                                        device, args)

                        # For autoregressive models also update the bits per dimension value, converted from the obtained nats
                        if args.autoregression:
                            recon_losses_bits_per_dim.update(recon_loss.item() * math.log2(math.e), inp.size(0))

                        # take mean to compute accuracy
                        # (does nothing if there isn't more than 1 sample per input other than removing dummy dimension)
                        class_output = torch.mean(class_samples[:,:,-Dataset.num_classes:], dim=0)
                        if not args.no_vae:
                            recon_output = torch.mean(recon_samples, dim=0)

                        # measure accuracy, record loss, fill confusion matrix
                        prec1_new = accuracy(class_output, target)[0]
                        top1_new.update(prec1_new.item(), inp.size(0))

                        # for autoregressive models generate reconstructions by sequential sampling from the
                        # multinomial distribution (Reminder: the original output is a 255 way Softmax as PixelVAEs are posed as a
                        # classification problem).
                        if not args.no_vae:
                            recon_target = inp
                            if args.autoregression:
                                recon = torch.zeros((inp.size(0), inp.size(1), inp.size(2), inp.size(3))).to(device)
                                for h in range(inp.size(2)):
                                    for w in range(inp.size(3)):
                                        for c in range(inp.size(1)):
                                            probs = torch.softmax(recon_output[:, :, c, h, w], dim=1).data
                                            pixel_sample = torch.multinomial(probs, 1).float() / 255.
                                            recon[:, c, h, w] = pixel_sample.squeeze()
                            else:
                                # If not autoregressive simply apply the Sigmoid
                                recon = torch.sigmoid(recon_output)

                            for j in range(inp.size(0)):
                                if args.autoregression:
                                    rec = recon_output[j].view(1, recon_output.size(1), recon_output.size(2),
                                                                recon_output.size(3), recon_output.size(4))
                                    rec_tar = recon_target[j].view(1, recon_target.size(1), recon_target.size(2),
                                                                    recon_target.size(3))

                                # If the input belongs to one of the base classes also update base metrics
                                #if class_target[j].item() in base_classes:
                                if args.autoregression:
                                    recon_losses_new_bits_per_dim.update(F.cross_entropy(rec, (rec_tar * 255).long()) *
                                                                            math.log2(math.e), 1)                                            
                                recon_losses_new_nat.update(F.binary_cross_entropy(recon[j], recon_target[j]), 1)
        
            # At the continual learning metrics to TensorBoard
            writer.add_scalar('validation/base_precision@1', top1_base.avg, len(model.module.seen_tasks)-1)
            writer.add_scalar('validation/new_precision@1', top1_new.avg, len(model.module.seen_tasks)-1)
            writer.add_scalar('validation/base_rec_loss_nats', recon_losses_base_nat.avg * args.patch_size *
                              args.patch_size * model.module.num_colors, len(model.module.seen_tasks) - 1)
            writer.add_scalar('validation/new_rec_loss_nats', recon_losses_new_nat.avg * args.patch_size *
                              args.patch_size * model.module.num_colors, len(model.module.seen_tasks) - 1)

            if args.autoregression:
                writer.add_scalar('validation/base_rec_loss_bits_per_dim',
                                  recon_losses_base_bits_per_dim.avg, len(model.module.seen_tasks) - 1)
                writer.add_scalar('validation/new_rec_loss_bits_per_dim',
                                  recon_losses_new_bits_per_dim.avg, len(model.module.seen_tasks) - 1)

            print(' * Incremental validation: Base Prec@1 {prec1_base:.3f} New Prec@1 {prec1_new:.3f}\t'
                  'Base Recon Loss {recon_losses_base_nat.avg:.3f} New Recon Loss {recon_losses_new_nat.avg:.3f}'
                  .format(prec1_base=prec1_base, prec1_new=prec1_new,
                          recon_losses_base_nat=recon_losses_base_nat, recon_losses_new_nat=recon_losses_new_nat))
    
    return top1.avg, losses.avg
