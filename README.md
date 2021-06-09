# Code for Reproduction of Experimentation in A Procedural World Generation Framework for Systematic Evaluation of Continual Learning

This repository contains the code for reproduction of the experiments in our [paper](https://arxiv.org/abs/2106.02585):



> **Timm Hess, Martin Mundt, Iuliia Pliushch, Visvanathan Ramesh:
> *"A Procedural World Generation Framework for Systematic Evaluation of Continual Learning"*
> https://arxiv.org/abs/2106.02585**

This project builds on the code-basis for generative open-set classifying denoising variational auto-encoder ([OCDVAE](https://github.com/MrtnMndt/OCDVAEContinualLearning)) and the [Avalanche Continual Learning Libraray](https://avalanche.continualai.org/).

#
## Requirements

* Python 3 (3.8.5)
* PyTorch 1.8.1 & torchvision 0.9.1 
* Cython >= 0.17 (for libmr) & libmr 0.1.9 (for open set recognition)
* avalanche 0.0.1
* tqdm 4.61.0 (for progress bars)
* scipy 1.6.3 & librosa 0.6.3 (for creation of AudioMNIST spectrograms)
	
and for visualization:

* Matplotlib 3.4.1
* Seaborn 0.11.1
* Tensorboard 2.5.0

# 
## Specific Experiments
In the following the specific command lines for the reproduction of experimentation as conducted in the paper are provided.

## Datasets
The image-patch datasets used for the experimentation are available [here](https://doi.org/10.5281/zenodo.4899267).

#
## __Naive Continual Leanring__
### Incremental Classes
	python train_OCDVAE.py -a DCNNNoVAE --no-vae --dataset IncrementalClassificationSet --incremental-data --train_path_to_root <Train_ImagePatch_Dataset> --val_path_to_root <Val_ImagePatch_Dataset> --patch-size 64 --epochs 60 --batch-size 64  --num-base-tasks 1 --num-increment-tasks 1 --save_path_root <path>
### Incremental Lighting / Weather
	python train_OCDVAE.py -a DCNNNoVAE --no-vae --dataset IncrementalInstanceSet --incremental-instance --incremental-data --train_path_to_root <Train_ImagePatch_Dataset> --val_path_to_root <Val_ImagePatch_Dataset> --patch-size 64 --epochs 60 --batch-size 64  --num-base-tasks 1 --num-increment-tasks 1 --save_path_root <path>
### Avalanche 
	python train_avalanche.py --train_path_to_root <Train_ImagePatch_Dataset> --val_path_to_root <Val_ImagePatch_Dataset> --num_epochs 60 --batch_size 64 --sequence_order 0 1 2 3 --task_order 0 1 2 3 --cl_strategy Naive --tb_log_dir <path>

#
## __Upper Bound__
### Incremental Classes
	python3 train_OCDVAE.py -a DCNNNoVAE --no-vae --dataset IncrementalClassificationSet --incremental-data --train_path_to_root <Train_ImagePatch_Dataset> --val_path_to_root <Val_ImagePatch_Dataset> --patch-size 64 --epochs 60 --batch-size 64  --num-base-tasks 1 --num-increment-tasks 1 --train-incremental-upper-bound --load-task-order 0,1,2,3,4
### Incremental Lighting / Weather
	python train_OCDVAE.py -a DCNNNoVAE --no-vae --dataset IncrementalInstanceSet --incremental-instance --incremental-data --train_path_to_root <Train_ImagePatch_Dataset> --val_path_to_root <Val_ImagePatch_Dataset> --patch-size 64 --epochs 60 --batch-size 64 --num-base-tasks 0 --num-increment-tasks --train-incremental-upper-bound

#
## __OCDVAE__
### Incremental Classes
	python3 train_OCDVAE.py -a DCNN --dataset IncrementalClassificationSet --train_path_to_root <Train_ImagePatch_Dataset> --val_path_to_root <Val_ImagePatch_Dataset> --patch-size 64 --epochs 120 --batch-size 64 --incremental-data --num-base-tasks 1 --num-increment-tasks 1 --openset-generative-replay --save_path_root <path>

### Incremental Lighting / Weather
	python3 train_OCDVAE.py -a DCNN --dataset IncrementalInstanceSet --incremental-instance --incremental-data --train_path_to_root <Train_ImagePatch_Dataset> --val_path_to_root <Val_ImagePatch_Dataset> --patch-size 64 --epochs 120 --batch-size 64 --num-base-tasks 0 --num-increment-tasks 1 --openset-generative-replay --load-task-order 4,3,2,1,0

#
## __LwF__
### Incremental Classes
	python3 train_OCDVAE.py -a DCNNNoVAE --no-vae --dataset IncrementalClassificationSet --incremental-data --train_path_to_root <Train_ImagePatch_Dataset> --val_path_to_root <Val_ImagePatch_Dataset> --patch-size 64 --epochs 60 --batch-size 64 --num-base-tasks 1 --num-increment-tasks 1 --load-task-order 0,1,2,3 --use-lwf --lmda 0.5

### Incremental Lighting / Weather
	python3 train_OCDVAE.py -a DCNNNoVAE --no-vae --dataset IncrementalInstanceSet --incremental-data --incremental-instance --train_path_to_root <Train_ImagePatch_Dataset> --val_path_to_root <Val_ImagePatch_Dataset> --patch-size 64 --epochs 60 --batch-size 64 --num-base-tasks 0 --num-increment-tasks 1 --load-task-order 0,1,2,3,4 --full-conf-mat --use-lwf --lmda 0.5

### Avalanche 
	python train_avalanche.py --train_path_to_root <Train_ImagePatch_Dataset> --val_path_to_root <Val_ImagePatch_Dataset> --num_epochs 60 --batch_size 64 --sequence_order 0 1 2 3 --task_order 0 1 2 3 --cl_strategy LwF --tb_log_dir <path>

#
## __SI__
### Incremental Classes
	python3 train_OCDVAE.py -a DCNNNoVAE --no-vae --dataset IncrementalClassificationSet --incremental-data --train_path_to_root <Train_ImagePatch_Dataset> --val_path_to_root <Val_ImagePatch_Dataset> --patch-size 64 --epochs 60 --batch-size 64 --num-base-tasks 1 --num-increment-tasks 1 --load-task-order 0,1,2,3 --use-si --lmda 1.0	

### Incremental Lighting / Weather
	python3 train_OCDVAE.py -a DCNNNoVAE --no-vae --dataset IncrementalInstanceSet --incremental-data --incremental-instance --train_path_to_root <Train_ImagePatch_Dataset> --val_path_to_root <Val_ImagePatch_Dataset> --patch-size 64 --epochs 60 --batch-size 64 --num-base-tasks 0 --num-increment-tasks 1 --load-task-order 0,1,2,3,4 --full-conf-mat --use-si --lmda 1.0

### Avalanche
	python train_avalanche.py --train_path_to_root <Train_ImagePatch_Dataset> --val_path_to_root <Val_ImagePatch_Dataset> --num_epochs 60 --batch_size 64 --sequence_order 0 1 2 3 --task_order 0 1 2 3 --cl_strategy SI --tb_log_dir <path>

#
## __EWC__ (Avalanche Only)
### Incremental Classes / Lighting / Weather
	python train_avalanche.py --train_path_to_root <Train_ImagePatch_Dataset> --val_path_to_root <Val_ImagePatch_Dataset> --num_epochs 60 --batch_size 64 --sequence_order 0 1 2 3 --task_order 0 1 2 3 --cl_strategy EWC --tb_log_dir <path>

#
## __GEM__ (Avalanche Only)
### Incremental Classes / Lighting / Weather
	python train_avalanche.py --train_path_to_root <Train_ImagePatch_Dataset> --val_path_to_root <Val_ImagePatch_Dataset> --num_epochs 60 --batch_size 64 --sequence_order 0 1 2 3 --task_order 0 1 2 3 --cl_strategy GEM --tb_log_dir <path>





