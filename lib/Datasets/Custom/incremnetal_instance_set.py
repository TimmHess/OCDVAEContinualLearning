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
    def __init__(self, path_to_root, labelmap_file, patch_size, use_single_container=False):
        self.path_to_root = path_to_root
        self.patch_size = patch_size

        self.use_single_container = use_single_container

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
        sub_sequences_dirs = sorted([f.name for f in os.scandir(self.path_to_root) if f.is_dir()])

        data = []
        # For each sub-sequence dir
        for sub_sequence_dir in sub_sequences_dirs:
            # Get all sub dirs (a.k.a. classes)
            class_name_dirs = [f.name for f in os.scandir(self.path_to_root+sub_sequence_dir+"/") if f.is_dir()]

            # For each class get the respective label from labelmap 
            for class_name in class_name_dirs:
                label = self.label_dict[class_name]
                # Read all paths to list (path_to_img, label)
                path = self.path_to_root + sub_sequence_dir + "/" + class_name + "/"  
                for file in os.listdir(path):
                    data.append((path + file, label))
            
            if(not self.use_single_container):
                # Store data of this sequence
                self.sequence_data.append(data)
                # reset data 
                data = []

        # Store data of all sequences
        if(self.use_single_container):
            self.sequence_data.append(data)
        return

    def __get_class_to_idx(self, label_dict):
        class_to_idx = {}
        for key in label_dict:
            #print(key)
            #class_to_idx[label_dict[key]] = key
            class_to_idx[key] = label_dict[key]
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



class SegmentationSequence(data.Dataset):
    def __init__(self, path_to_color, path_to_seg, sequence_file, segmentation_file, 
            classmap_file, patch_size, use_single_container=False):
        self.patch_size = patch_size
        self.use_single_container = use_single_container
        
        # Dict mapping a range of labels to a single class label
        self.classmap = self.__load_classmap(classmap_file)
        #print("classmap:")
        #print(self.classmap)
        #print("")
        self.label_dict = self.__load_label_dict(segmentation_file)
        #print("label_dict:")
        #print(self.label_dict)
        self.sequence_dict = self.__load_sequence_indices(sequence_file)
        #print("sequence_dict:")
        #print(self.sequence_dict)

        self.num_classes = self.__get_num_classes(self.classmap)
        print("Num Classes: ", self.num_classes)
        self.class_to_idx = self.__get_class_to_idx(self.classmap)

        # List holding list for each sequence containing (img_path, target)
        self.sequence_data = []
        # Load data of all sequences
        self.__load_data(path_to_color=path_to_color, path_to_seg=path_to_seg)

        # Number of sequences in this set
        self.num_sequences = self.__get_num_sequences()
        # Index of the currently active sequence
        self.active_sequence_index = 0

        # Transforms
        self.transforms = transforms.Compose([
            transforms.Resize(size=(self.patch_size[1], self.patch_size[0])),
            transforms.ToTensor()
        ])

        return
    
    def __load_label_dict(self, labelmap_file):
        label_dict = {}
        with open(labelmap_file) as file:
            json_array = json.load(file)

            segMin = json_array[0]["ObjectClassMapping"]
            segMax = json_array[1]["ObjectClassMapping"]

            for key in segMin:
                label_dict[key] = [segMin[key], segMax[key]]
        return label_dict

    def __load_classmap(self, classmap_file):
        classmap = {}
        with open(classmap_file) as file:
            json_array = json.load(file)
            classmap = json_array["ClassMapping"]
        return classmap

    def __load_sequence_indices(self, sequence_file):
        sequence_dict = {}
        with open(sequence_file) as file:
            json_array = json.load(file)

            for i in range(len(json_array)):
                sequence_dict[i] = json_array[i]["Sequence"]["ImageCounter"]
        return sequence_dict

    def __get_name_for_label(self, label):
        for key in self.label_dict:
            #print("key", key)
            min_val, max_val = self.label_dict[key]
            #print(min_val, max_val)
            if(min_val == max_val):
                if(label == min_val):
                    return key
            else:
                if(label >= min_val and label < max_val):
                    return key
        print("Label could not be converted! This is bad..")
        sys.exit()
        return None

    def __convert_target(self, target):
        """ 
        Converts segmentation target (instance-segmented) according to classmap
        """ 
        # get all unique labels
        unique_labels = np.unique(target)
        # for each label get respective class name from label_dict (in numerical order to prevent overwriting of wrong labels)
        for unique_label in unique_labels:
            #print("unique_label", unique_label)
            # get class_name for label
            label_name = self.__get_name_for_label(unique_label)
            #print("label_name", label_name)
            # get class_label for name
            class_label = self.classmap[label_name]
            #print("class_label", class_label)
            # convert all labels to class labels
            target[target == unique_label] = class_label # should access the original target no return required
        return

    def __load_data(self, path_to_color, path_to_seg):
        # Get list of all file names in color dir
        color_files = []
        for f in sorted(os.listdir(path_to_color)):
            color_files.append(path_to_color + f)
        # Get list of all file names in seg dir
        target_files = []
        for f in sorted(os.listdir(path_to_seg)):
            target_files.append(path_to_seg + f)
        assert(len(color_files) == len(target_files))

        # split by sequence
        if(not self.use_single_container):
            for i in range(len(self.sequence_dict)):
                last_index = self.sequence_dict[i]
                if((i+1) == len(self.sequence_dict)):
                    next_index = len(color_files)
                else:
                    next_index = self.sequence_dict[i+1]

                color_sub_paths = color_files[last_index:next_index]
                target_sub_paths = target_files[last_index:next_index]

                print(len(color_sub_paths), len(target_sub_paths))

                # append to sequence_data 
                data = (color_sub_paths, target_sub_paths)
                self.sequence_data.append(data)
        else:
            data = (color_files, target_files)
            self.sequence_data.append(data)

        return

    def __get_class_to_idx(self, label_dict):
        class_to_idx = {}
        for key in label_dict:
            class_to_idx[key] = label_dict[key]
        return class_to_idx

    def __get_num_classes(self, label_dict):
        num_classes = len(np.unique(list(label_dict.values())))
        return num_classes

    def __get_num_sequences(self):
        return len(self.sequence_data)

    def pil_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        return img

    def target_loader(self, path):
        with open(path, 'rb') as f:
            target = Image.open(f).convert("L").resize((self.patch_size[0],
            self.patch_size[1]), Image.NEAREST)
        return target

    def set_sequence_index(self, index):
        self.active_sequence_index = index
        return

    def __getitem__(self, index):
        image_path = self.sequence_data[self.active_sequence_index][0][index]
        target_path = self.sequence_data[self.active_sequence_index][1][index]
    
        # load image
        img = self.pil_loader(image_path)
        img = self.transforms(img)

        # load target
        target = np.array(self.target_loader(target_path))
        self.__convert_target(target)

        return img, target

    def __len__(self):
        return len(self.sequence_data[self.active_sequence_index])




if(__name__ == "__main__"):
    # path_to_root = "/home/shared/hess/UE4_EndlessRunner/test_light/"
    # labelmap_file = "/home/shared/hess/UE4_EndlessRunner/test_light/labelmap.json"
    
    # dataset = ClassificationSequence(path_to_root, labelmap_file, 64)
    # print("num_sequences", dataset.num_sequences)
    # print("len", len(dataset))
    # img, _ = dataset[0]

    # img = np.asarray(img.cpu().numpy().transpose(1,2,0)*255, dtype=np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #cv2.imwrite("img.png", img)


    train_path_to_color = "/home/shared/hess/UE4_EndlessRunner/Train_IncrementalClasses/Color/0/"
    train_path_to_seg = "/home/shared/hess/UE4_EndlessRunner/Train_IncrementalClasses/Seg/0/"
    path_to_sequence = "/home/shared/hess/UE4_EndlessRunner/Train_IncrementalClasses/Sequence.json"
    path_to_segmentation = "/home/shared/hess/UE4_EndlessRunner/Train_IncrementalClasses/Segmentation.json"
    path_to_classmap = "/home/shared/hess/UE4_EndlessRunner/Train_IncrementalClasses/Classmap.json"

    dataset = SegmentationSequence(path_to_color=train_path_to_color, 
                            path_to_seg=train_path_to_seg, 
                            sequence_file=path_to_sequence, 
                            segmentation_file=path_to_segmentation, 
                            classmap_file=path_to_classmap, 
                            patch_size=[320, 180], 
                            use_single_container=False)
    
    img, target = dataset[0]
    print(target.shape)
    print(np.unique(target))