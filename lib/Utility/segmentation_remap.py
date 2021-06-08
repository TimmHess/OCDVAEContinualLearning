import numpy as np
import json

class ColorMapper():
    def __init__(self, path_to_colormap_file, path_to_classmap_file):
        self.color_dict = self.__load_color_dict(path_to_colormap_file, path_to_classmap_file)

    def __load_color_dict(self, path_to_colormap_file, path_to_classmap_file):
        # Get name -> class relation
        colormap = None
        with open(path_to_colormap_file) as file:
            json_array = json.load(file)
            colormap = json_array["ColorMapping"]
        # Get name -> color relation
        classmap = None
        with open(path_to_classmap_file) as file:
            json_array = json.load(file)
            classmap = json_array["ClassMapping"]
        # Generate class -> color
        color_dict = {}
        for key in colormap:
            color_dict[classmap[key]] = colormap[key] 
        return color_dict

    def segmentation_to_color(self, segmentation_prediction):
        for key in self.color_dict:
            segmentation_prediction[segmentation_prediction==key] = self.color_dict[key]
        return segmentation_prediction