from torch.cuda import current_device
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix,\
    disk_usage_metrics, gpu_usage_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive, SynapticIntelligence, LwF, AR1, EWC, GEM, GDumb

import torch
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss

from lib.Datasets.Custom.incremnetal_instance_set import ClassificationSubSequence
from lib.Models.architectures_avalanche import DCNNNoVAE, DCNNNoVAEIncremental

from datetime import datetime
import argparse


parser = argparse.ArgumentParser(description='Avalanche Continual Training')
# Image-Patches
parser.add_argument('--train_path_to_root', type=str, default=None, 
                    help='Path to root of training set when using incremental sequence')
parser.add_argument('--val_path_to_root', type=str, default=None,
                    help='Path to root of validation set when using incremental sequnece')
parser.add_argument('--labelmap_file', type=str, default=None,
                    help='Path to labelmap file for incremental sequence')

parser.add_argument('--color_transform', type=str, default=None)
parser.add_argument('--sequence_order', nargs="+", type=int, required=True)
parser.add_argument('--task_order', nargs="+", type=int, required=True)
parser.add_argument('--cl_strategy', type=str, required=True)
parser.add_argument('--grow_classifier', action='store_true', default=False)
parser.add_argument('--test_on_all', action='store_true', default=False)

parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--tb_log_dir', type=str, default="./tb_data")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create Task Dataset
training_datasets = []
test_datasets = []

assert(len(args.sequence_order) == len(args.task_order))

for i in range(len(args.sequence_order)):
    training_datasets.append(
        AvalancheDataset(
            ClassificationSubSequence(
            path_to_root=args.train_path_to_root, labelmap_file=args.labelmap_file,
            patch_size=64,
            subsequence_index=args.sequence_order[i],
            is_load_to_ram=True,
            color_transform=args.color_transform,
            is_gdumb = args.cl_strategy=="GDumb"),
        task_labels=args.task_order[i])
    )

    test_datasets.append(
        AvalancheDataset(
            ClassificationSubSequence(
            path_to_root=args.val_path_to_root, labelmap_file=args.labelmap_file,
            patch_size=64,
            subsequence_index=args.sequence_order[i],
            is_load_to_ram=True,
            color_transform=args.color_transform), 
        task_labels=args.task_order[i])
    )

scenario_custom_task_labels = dataset_benchmark(training_datasets, test_datasets)


scenario = scenario_custom_task_labels

if(args.grow_classifier):
    print("using incremental classifier")
    model = DCNNNoVAEIncremental(num_classes=2)
else:      
    model = DCNNNoVAE(num_classes=5) # 5 because I happen to know this

# log to Tensorboard
path = args.tb_log_dir + "/" + ("".join(str(datetime.now()).split(" "))) + "_" + args.cl_strategy
tb_logger = TensorboardLogger(tb_log_dir=path)
# log to text file
text_logger = TextLogger(open('log.txt', 'a'))
# print to stdout
interactive_logger = InteractiveLogger()
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=False, epoch=False, experience=True, stream=True),
    #loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    #timing_metrics(epoch=True),
    #cpu_usage_metrics(experience=True),
    #forgetting_metrics(experience=True, stream=True),
    #StreamConfusionMatrix(num_classes=5, save_image=False),
    StreamConfusionMatrix(save_image=False),
    #disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True)
    loggers=[interactive_logger, text_logger, tb_logger]
)

# CREATE THE STRATEGY INSTANCE (NAIVE)
if(args.cl_strategy == "Naive"):
    cl_strategy = Naive(
        model, Adam(model.parameters(), lr=0.001),
        CrossEntropyLoss(), train_mb_size=args.batch_size, train_epochs=args.num_epochs, eval_mb_size=args.batch_size*2,
        evaluator=eval_plugin, device=device)
elif(args.cl_strategy == "SI"):
    cl_strategy = SynapticIntelligence(model, Adam(model.parameters(), lr=0.001),
        CrossEntropyLoss(), si_lambda=1.0, train_mb_size=args.batch_size, train_epochs=args.num_epochs, eval_mb_size=args.batch_size*2,
        evaluator=eval_plugin, device=device)
elif(args.cl_strategy == "LwF"):
    cl_strategy = LwF(
        model, Adam(model.parameters(), lr=0.001),
        CrossEntropyLoss(), alpha=0.5, temperature=2.0, train_mb_size=args.batch_size, train_epochs=args.num_epochs, eval_mb_size=args.batch_size*2,
        evaluator=eval_plugin, device=device)
elif(args.cl_strategy == "GEM"):
    cl_strategy = GEM(
        model, Adam(model.parameters(), lr=0.001),
        CrossEntropyLoss(), patterns_per_exp=150, memory_strength=0.5, train_mb_size=args.batch_size, train_epochs=args.num_epochs, eval_mb_size=args.batch_size*2,
        evaluator=eval_plugin, device=device)
elif(args.cl_strategy == "EWC"):
    cl_strategy = EWC(
        model, Adam(model.parameters(), lr=0.001),
        CrossEntropyLoss(), ewc_lambda=0.5, mode="separate", train_mb_size=args.batch_size, train_epochs=args.num_epochs, eval_mb_size=args.batch_size*2,
        evaluator=eval_plugin, device=device)
elif(args.cl_strategy == "AR1"):
    cl_strategy = AR1(
        model, Adam(model.parameters(), lr=0.001),
        CrossEntropyLoss(), ewc_lambda=0.5, train_mb_size=args.batch_size, train_epochs=args.num_epochs, eval_mb_size=args.batch_size*2,
        evaluator=eval_plugin, device=device)
elif(args.cl_strategy == "GDumb"):
    cl_strategy = GDumb(
        model, Adam(model.parameters(), lr=0.001),
        CrossEntropyLoss(), mem_size=200, train_mb_size=args.batch_size, train_epochs=args.num_epochs, eval_mb_size=args.batch_size*2,
        evaluator=eval_plugin, device=device)
else:
    print("Strategy is not implemented!")
    raise NotImplementedError


# TRAINING LOOP
print('Starting experiment...')
results = []
for experience in scenario.train_stream:
    curr_experience = experience.current_experience
    print("Start of experience: ", curr_experience)
    print("Current Classes: ", experience.classes_in_this_experience)
    
    if(args.grow_classifier):
        model.classifier.adaptation(experience.dataset)
        print("adpated classifier")
        print("Classiifer:", model.classifier)

    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience, num_workers=4)
    print('Training completed')

    print('Computing accuracy on test set')
    if(not args.test_on_all):
        scenario.test_stream.slice_ids = list(range(curr_experience+1))
        print(scenario.test_stream.slice_ids)
    eval_dict = cl_strategy.eval(scenario.test_stream, num_workers=4)
    
    results.append(eval_dict)
