import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import csv
import json
import os
import os.path
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image



class ClassificationSequence(data.Dataset):
    def __init__(self, path_to_root, labelmap_file, patch_size):
        self.path_to_root = path_to_root
        self.patch_size = patch_size

        # Dict mapping string to label
        self.label_dict = self.__load_label_dict(labelmap_file)
        self.class_to_idx = self.__get_class_to_idx(self.label_dict)
        self.num_classes = self.__get_num_classes(self.label_dict)

        # List holding list for each sequence containing (img_path, target)
        self.sequence_data = []
        # Load data of all sequences
        self.__load_data()
        
        # Number of sequences in this set
        self.num_sequences = self.__get_num_sequences()
        # Index of the currently active sequence
        self.active_sequence_index = 0

        # Transforms
        self.transforms = transforms.Compose([
            transforms.Resize(size=(self.patch_size, self.patch_size)),
            transforms.ToTensor()
        ])

        return
    
    def __load_label_dict(self, labelmap_file):
        label_dict = {}
        with open(labelmap_file) as file:
            json_array = json.load(file)
            label_dict = json_array["SegmentationClasses"]
        return label_dict

    def __load_data(self):
        # Get all sub-sequence dirs from root_path
        sub_sequences_dirs = [f.name for f in os.scandir(self.path_to_root) if f.is_dir()]
        #print("subsequence_dirs", sub_sequences_dirs)
        
        # For each sub-sequence dir
        for sub_sequence_dir in sub_sequences_dirs:
            data = []
            #print("curr sub_sequenc_dir",sub_sequence_dir)
            # Get all sub dirs (a.k.a. classes)
            class_name_dirs = [f.name for f in os.scandir(self.path_to_root+sub_sequence_dir+"/") if f.is_dir()]
            #print("class_name_dirs", class_name_dirs)
    
            # For each class get the respective label from labelmap 
            for class_name in class_name_dirs:
                label = self.label_dict[class_name]
                #print("class_name",class_name,label)
                # Read all paths to list (path_to_img, label)
                path = self.path_to_root + sub_sequence_dir + "/" + class_name + "/"
                #print("path", path)    
                for file in os.listdir(path):
                    #print(path+file)
                    data.append((path + file, label))
            
            self.sequence_data.append(data)
        return

    def __get_class_to_idx(self, label_dict):
        class_to_idx = {}
        for key in label_dict:
            class_to_idx[label_dict[key]] = key
        return class_to_idx

    def __get_num_classes(self, label_dict):
        num_classes = len(np.unique(list(label_dict.values())))
        return num_classes

    def __get_num_sequences(self):
        num_sequences = len([f.name for f in os.scandir(self.path_to_root) if f.is_dir()])
        return num_sequences

    def pil_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        return img

    def set_sequence_index(self, index):
        self.active_sequence_index = index
        return

    def __getitem__(self, index):
        data_pair = self.sequence_data[self.active_sequence_index][index]
        img_path, target = data_pair

        img = self.pil_loader(img_path)

        img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.sequence_data[self.active_sequence_index])


if(__name__ == "__main__"):
    path_to_root = "/home/shared/hess/UE4_EndlessRunner/test_light/"
    labelmap_file = "/home/shared/hess/UE4_EndlessRunner/test_light/labelmap.json"
    
    dataset = ClassificationSequence(path_to_root, labelmap_file, 64)
    print("num_sequences", dataset.num_sequences)
    print("len", len(dataset))
    img, _ = dataset[0]

    img = np.asarray(img.cpu().numpy().transpose(1,2,0)*255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    
    #cv2.imwrite("img.png", img)