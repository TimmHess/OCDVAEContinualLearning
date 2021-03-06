########################
# Importing libraries
########################
# System libraries
import os
import random
from time import gmtime, strftime
import numpy as np
import pickle
import copy

# Tensorboard for PyTorch logging and visualization
from torch.utils.tensorboard import SummaryWriter

# Torch libraries
import torch
import torch.backends.cudnn as cudnn

# Custom library
import lib.Models.architectures as architectures
from lib.Models.pixelcnn import PixelCNN
import lib.Datasets.datasets as datasets
from lib.Models.initialization import WeightInit, ZeroWeightInit
from lib.Models.architectures import grow_classifier, grow_classifier_seg
from lib.cmdparser import parser
from lib.Training.train import train
from lib.Training.validate import validate
from lib.Utility.utils import save_checkpoint, save_task_checkpoint
from lib.Training.evaluate import get_latent_embedding
from lib.Training.evaluate import get_mu_and_std
from lib.Utility.visualization import args_to_tensorboard
from lib.Utility.visualization import visualize_dataset_in_2d_embedding
from lib.Models.architectures import set_previous_mu_and_std
import lib.Models.si as SI
from lib.Models.architectures import consolidate_classifier

# Comment this if CUDNN benchmarking is not desired
cudnn.benchmark = True


def main():
    # Command line options
    args = parser.parse_args()
    print("Command line options:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    if args.cross_dataset and not args.incremental_data:
        raise ValueError('cross-dataset training possible only if incremental-data flag set')

    # Check whether GPU is available and can be used
    # if CUDA is found then device is set accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Launch a writer for the tensorboard summary writer instance
    save_path = args.save_path_root
    save_path += 'runs/' + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + '_' + args.dataset + '_' + args.architecture +\
                '_variational_samples_' + str(args.var_samples) + '_latent_dim_' + str(args.var_latent_dim)

    # add option specific naming to separate tensorboard log files later
    if args.autoregression:
        save_path += '_pixelcnn'

    if args.incremental_data:
        save_path += '_incremental'
        if args.is_multiheaded:
            save_path += '_multihead'
        if args.train_incremental_upper_bound:
            save_path += '_upper_bound'
        if args.generative_replay:
            save_path += '_genreplay'
        if args.openset_generative_replay:
            save_path += '_opensetreplay'
        #if args.use_kl_regularization:
        #    save_path += '_kl-reg'
        if args.use_si:
            save_path += '_si' + '_' + str(args.lmda)
        if args.use_lwf:
            save_path += '_lwf' + '_' + str(args.lmda)
    if args.cross_dataset:
        save_path += '_cross_dataset_' + args.dataset_order

    # if we are resuming a previous training, note it in the name
    if args.resume:
        save_path = save_path + '_resumed'
    writer = SummaryWriter(save_path)

    # saving the parsed args to file
    log_file = os.path.join(save_path, "stdout")
    log = open(log_file, "a")
    for arg in vars(args):
        log.write(arg + ':' + str(getattr(args, arg)) + '\n')

    # Initialize criterion
    if args.no_vae:
        from lib.Training.loss_functions import unified_loss_function_no_vae as criterion
    elif args.is_multiheaded:
        from lib.Training.loss_functions import unified_loss_function_multihead as criterion
    else:
        from lib.Training.loss_functions import unified_loss_function as criterion


    # Dataset loading
    data_init_method = getattr(datasets, args.dataset)
    dataset = data_init_method(torch.cuda.is_available(), args)
    print("dataset:", dataset)
    
    # get the number of classes from the class dictionary
    num_classes = dataset.num_classes

    # we set an epoch multiplier to 1 for isolated training and increase it proportional to amount of tasks in CL
    epoch_multiplier = 1
    if args.incremental_data:
        from lib.Datasets.incremental_dataset import get_incremental_dataset

        # get the method to create the incremental dataste (inherits from the chosen data loader)
        inc_dataset_init_method = get_incremental_dataset(data_init_method, args)
        print("inc data init method:", inc_dataset_init_method)
        
        # different options for class incremental vs. cross-dataset experiments
        if args.cross_dataset:
            # if a task order file is specified, load the task order from it
            if args.load_task_order:
                # check if file exists and if file ends with extension '.txt'
                if os.path.isfile(args.load_task_order) and len(args.load_task_order) >= 4\
                        and args.load_task_order[-4:] == '.txt':
                    print("=> loading task order from '{}'".format(args.load_task_order))
                    with open(args.load_task_order, 'rb') as fp:
                        task_order = pickle.load(fp)
                # if no file is found default to cmd line task order
                else:
                    # parse and split string at commas
                    task_order = args.dataset_order.split(',')
                    for i in range(len(task_order)):
                        # remove blank spaces in dataset names
                        task_order[i] = task_order[i].replace(" ", "")
            # use task order as specified in command line
            else:
                # parse and split string at commas
                task_order = args.dataset_order.split(',')
                for i in range(len(task_order)):
                    # remove blank spaces in dataset names
                    task_order[i] = task_order[i].replace(" ", "")

            # just for getting the number of classes in the first dataset
            num_classes = 0
            for i in range(args.num_base_tasks):
                temp_dataset_init_method = getattr(datasets, task_order[i])
                temp_dataset = temp_dataset_init_method(torch.cuda.is_available(), args)
                num_classes += temp_dataset.num_classes
                del temp_dataset

            # multiply epochs by number of tasks
            if args.num_increment_tasks:
                epoch_multiplier = ((len(task_order) - args.num_base_tasks) / args.num_increment_tasks) + 1
            else:
                # this branch will get active if num_increment_tasks is set to zero. This is useful when training
                # any isolated upper bound with all datasets present from the start.
                epoch_multiplier = 1.0

        elif args.incremental_instance:
            # instance incremental
            # if specified load task oder from file
            if args.load_task_order:
                if os.path.isfile(args.load_task_order):
                    # TODO
                    pass
                else:
                    # load task order from cmd
                    task_order = args.load_task_order.split(",")
                    task_order = [int(x) for x in task_order]
            else:
                task_order = [] # sub-sequence order
                for i in range(dataset.num_sequences):
                    task_order.append(i)
                if args.randomize_task_order:
                    task_order = np.random.permutation(dataset.num_sequences).tolist()

            # multiply epochs by number of tasks
            if args.num_increment_tasks:
                print("len task_order", len(task_order))
                print("num_base_tasks", args.num_base_tasks)
                print("num_increm_tasks", args.num_increment_tasks)
                epoch_multiplier = ((len(task_order) - args.num_base_tasks) / args.num_increment_tasks) #+ 1
            else:
                # this branch will get active if num_increment_tasks is set to zero. This is useful when training
                # any isolated upper bound with all datasets present from the start.
                epoch_multiplier = 1.0
            print("epoch multiplier:", epoch_multiplier)
        
        else:
            # class incremental
            # if specified load task order from file
            if args.load_task_order:
                if os.path.isfile(args.load_task_order):
                    print("=> loading task order from '{}'".format(args.load_task_order))
                    task_order = np.load(args.load_task_order).tolist()
                else:
                    # if no file is found a random task order is created
                    #print("=> no task order found. Creating randomized task order")
                    #task_order = np.random.permutation(num_classes).tolist()
                    # load task order from cmd
                    task_order = args.load_task_order.split(",")
                    task_order = [int(x) for x in task_order]
            else:
                # if randomize task order is specified create a random task order, else task order is sequential
                task_order = []
                for i in range(dataset.num_classes):
                    task_order.append(i)

                if args.randomize_task_order:
                    task_order = np.random.permutation(num_classes).tolist()

            # save the task order
            np.save(os.path.join(save_path, 'task_order.npy'), task_order)
            # set the number of classes to base tasks + 1 because base tasks is always one less.
            # E.g. if you have 2 classes it's one task. This is a little inconsistent from the naming point of view
            # but we wanted a single variable to work for both class incremental as well as cross-dataset experiments
            if(not args.is_segmentation):
                num_classes = args.num_base_tasks + 1
            else:
                num_classes = args.num_initial_classes
            # multiply epochs by number of tasks
            epoch_multiplier = ((len(task_order) - (args.num_base_tasks + 1)) / args.num_increment_tasks) + 1

        print("Task order: ", task_order)
        # log the task order into the text file
        log.write('task_order:' + str(task_order) + '\n')
        args.task_order = task_order

        # this is a little weird, but it needs to be here because the below method pops items from task_order
        args_to_tensorboard(writer, args)

        assert epoch_multiplier.is_integer(), print("uneven task division, make sure number of tasks are integers.")

        # Get the incremental dataset
        dataset = inc_dataset_init_method(torch.cuda.is_available(), device, task_order, args)
        print("dataset is loaded..")
    else:
        # add command line options to TensorBoard
        args_to_tensorboard(writer, args)

    log.close()

    # Get a sample input from the data loader to infer color channels/size
    net_input, _ = next(iter(dataset.train_loader))
    # get the amount of color channels in the input images
    num_colors = net_input.size(1)

    # import model from architectures class
    net_init_method = getattr(architectures, args.architecture)

    # if we are not building an autoregressive model the number of output channels of the model is equivalent to
    # the amount of input channels. For an autoregressive models we set the number of output channels of the
    # non-autoregressive decoder portion according to the command line option below
    if not args.autoregression:
        args.out_channels = num_colors

    # build the model
    model = net_init_method(device, num_classes, num_colors, args)

    # optionally add the autoregressive decoder
    if args.autoregression:
        model.pixelcnn = PixelCNN(device, num_colors, args.out_channels, args.pixel_cnn_channels,
                                  num_layers=args.pixel_cnn_layers, k=args.pixel_cnn_kernel_size,
                                  padding=args.pixel_cnn_kernel_size//2)

    # Parallel container for multi GPU use and cast to available device
    model = torch.nn.DataParallel(model).to(device)
    print(model)

    # Initialize the weights of the model, by default according to He et al.
    print("Initializing network with: " + args.weight_init)
    WeightInitializer = WeightInit(args.weight_init)
    WeightInitializer.init_model(model)

    # SI: Get ZeroWeightInit instance
    if args.use_si:
        ZeroWeightInitializer = ZeroWeightInit()

    # SI: Register all model paramters
    if args.use_si:
        if not args.is_segmentation:
            SI.register_si_params(model.module.encoder, model.module.si_storage)
            SI.register_si_params(model.module.latent_mu, model.module.si_storage_mu)
            SI.register_si_params(model.module.latent_std, model.module.si_storage_std)
        else:
            SI.register_si_params(model.module.encoder, model.module.si_storage)
            SI.register_si_params(model.module.bottleneck, model.module.si_storage_btn)
            SI.register_si_params(model.module.decoder, model.module.si_storage_dec)
        print("SI: Initial paramters got registered")

    # Define optimizer and loss function (criterion)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    epoch = 0
    iteration = [0] # needs to be type list to be passed as reference
    best_prec = 0
    best_loss = random.getrandbits(128)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # SI: Prepare storing of running parameter updates for this sequence (reset dicts W and p_old)
    if args.use_si:
        if not args.is_segmentation:
            SI.init_si_params(model.module.encoder, model.module.si_storage)
            SI.init_si_params(model.module.latent_mu, model.module.si_storage_mu)
            SI.init_si_params(model.module.latent_std, model.module.si_storage_std)
        else:
            SI.init_si_params(model.module.encoder, model.module.si_storage)
            SI.init_si_params(model.module.bottleneck, model.module.si_storage_btn)
            SI.init_si_params(model.module.decoder, model.module.si_storage_dec)
        print("SI: Reset running paramters for next task")
    
    # optimize until final amount of epochs is reached. Final amount of epochs is determined through the
    while epoch < (args.epochs * epoch_multiplier):
        print("")
        print(epoch, args.epochs*epoch_multiplier)
        # visualize the latent space before each task increment and at the end of training if it is 2-D
        if epoch % args.epochs == 0 and epoch > 0 or (epoch + 1) % (args.epochs * epoch_multiplier) == 0:
            if model.module.latent_dim == 2:
                print("Calculating and visualizing dataset embedding")
                # infer the number of current tasks to plot the different classes in the embedding
                if args.incremental_data:
                    if args.cross_dataset:
                        num_tasks = sum(dataset.num_classes_per_task[:len(dataset.seen_tasks)])
                    else:
                        num_tasks = len(dataset.seen_tasks)
                else:
                    num_tasks = num_classes

                zs = get_latent_embedding(model, dataset.train_loader, num_tasks, device)
                visualize_dataset_in_2d_embedding(writer, zs, args.dataset, save_path, task=num_tasks)

        # continual learning specific part
        if args.incremental_data:
            # at the end of each task increment
            if epoch % args.epochs == 0 and epoch > 0:
                if not args.no_model_store:
                    print('Saving the last checkpoint from the previous task ...')
                    save_task_checkpoint(save_path, epoch // args.epochs)
                
                # save previous model if lwf
                if args.use_lwf:
                    print("Storing previous model")
                    model.module.prev_model = copy.deepcopy(model)
                    model.module.prev_model.eval()
                    print("Storing complete...")

                # perform SI calculations
                if args.use_si:
                    # SI: Update internal paramters
                    if not args.is_segmentation:
                        SI.update_si_integral(model.module.encoder, model.module.si_storage)
                        SI.update_si_integral(model.module.latent_mu, model.module.si_storage_mu)
                        SI.update_si_integral(model.module.latent_std, model.module.si_storage_std)
                    else:
                        SI.update_si_integral(model.module.encoder, model.module.si_storage)
                        SI.update_si_integral(model.module.bottleneck, model.module.si_storage_btn)
                        SI.update_si_integral(model.module.decoder, model.module.si_storage_dec)
                    print("SI: Updated Omega")
                    # If there are previous classifier weighs -> consolidate before saving
                    if not model.module.prev_classifier_weights is None:
                        # consolidate classifier weights -> not needed because weights are still consolidated from validation
                        #consolidate_classifier(model.module)
                        pass
                    # SI: Store (consolidated) classifier weights as prev_classifier_weights
                    model.module.prev_classifier_weights = model.module.classifier[-1].weight.data.clone()
                    print("SI: Stored previous classifier weights before classifier has grown")

                print("Incrementing dataset...")
                dataset.increment_tasks(model, args.batch_size, args.workers, writer, save_path,
                                        is_gpu=torch.cuda.is_available(),
                                        upper_bound_baseline=args.train_incremental_upper_bound,
                                        generative_replay=args.generative_replay,
                                        openset_generative_replay=args.openset_generative_replay,
                                        openset_threshold=args.openset_generative_replay_threshold,
                                        openset_tailsize=args.openset_weibull_tailsize,
                                        autoregression=args.autoregression)
                print("Dataset incremente complete")
                
                # grow the classifier and increment the variable for number of overall classes so we can use it later
                if args.cross_dataset:
                    grow_classifier(device, model.module.classifier,
                                    sum(dataset.num_classes_per_task[:len(dataset.seen_tasks)])
                                    - model.module.num_classes, WeightInitializer)
                    model.module.num_classes = sum(dataset.num_classes_per_task[:len(dataset.seen_tasks)])

                elif args.incremental_instance:
                    print(len(dataset.train_loader), len(dataset.val_loader))
                    # Do not grow the classifier
                    # TODO: check if new dataset contains more classes and grow accordingly
                    pass
                    if args.is_multiheaded:
                        model.module.num_classes += dataset.num_classes
                        grow_classifier(device, model.module.classifier, dataset.num_classes, WeightInitializer)
                        print("Incresed classes by " + str(dataset.num_classes) + ", for multi-headed incremental instance")

                else:
                    if not args.is_segmentation:
                        model.module.num_classes += args.num_increment_tasks
                        grow_classifier(device, model.module.classifier, args.num_increment_tasks, WeightInitializer)
                    else:
                        # TODO: num increment tasks may be not the best way to increment -> cannot add multiple new classes in one increment
                        model.module.num_classes += args.num_increment_tasks
                        model.module.out_channels += args.num_increment_tasks
                        grow_classifier_seg(device, model.module.classifier, args.num_increment_tasks, WeightInitializer)
                        if args.generative_replay:
                            print("Growing reconstructor...")
                            grow_classifier_seg(device, model.module.reconstructor, args.num_increment_tasks, WeightInitializer)
                            print("done...")
                    print("Classifier grown..")
                    
                # Register new paramters to SI
                if args.use_si:
                    #SI.update_registered_si_params(model.module.encoder, model.module.si_storage)
                    #print("SI: Updated registration of SI params for grown model")
                    # SI: Reset running parameters (after growing)
                    if not args.is_segmentation:
                        SI.init_si_params(model.module.encoder, model.module.si_storage)
                        SI.init_si_params(model.module.latent_mu, model.module.si_storage_mu)
                        SI.init_si_params(model.module.latent_std, model.module.si_storage_std)
                    else:
                        SI.init_si_params(model.module.encoder, model.module.si_storage)
                        SI.init_si_params(model.module.bottleneck, model.module.si_storage_btn)
                        SI.init_si_params(model.module.decoder, model.module.si_storage_dec)
                    print("SI: Reset running paramters for next task")
                    # SI: Re-Initialize ALL classifier weights
                    #WeightInitializer.layer_init(model.module.classifier[-1])
                    if not args.is_multiheaded:
                        ZeroWeightInitializer.layer_init(model.module.classifier[-1])
                        print("SI: Re-Initialized classification layer")
                        # SI: Store temp_classifier_weights
                        model.module.temp_classifier_weights = model.module.classifier[-1].weight.data.clone()
                        print("SI: Stored classifier weights to temp_classifier_weights")

                # reset moving averages etc. of the optimizer
                optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

            # change the number of seen classes
            if epoch % args.epochs == 0:
                model.module.seen_tasks = dataset.seen_tasks

        # train
        train(dataset, model, criterion, epoch, iteration, optimizer, writer, device, args, save_path)

        # evaluate on validation set
        prec, loss = validate(dataset, model, criterion, epoch, iteration, writer, device, save_path, args)

        # remember best prec@1 and save checkpoint
        if not args.no_model_store:
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            best_prec = max(prec, best_prec)
            save_checkpoint({'epoch': epoch,
                            'arch': args.architecture,
                            'state_dict': model.state_dict(),
                            'best_prec': best_prec,
                            'best_loss': best_loss,
                            'optimizer': optimizer.state_dict()},
                            is_best, save_path)

        # increment epoch counters
        epoch += 1

        # if a new task begins reset the best prec so that new best model can be stored.
        if args.incremental_data and epoch % args.epochs == 0:
            best_prec = 0
            best_loss = random.getrandbits(128)

    writer.close()


if __name__ == '__main__':     
    main()
