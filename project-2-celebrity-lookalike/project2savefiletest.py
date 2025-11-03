import os,time
import sys
import dlib
import cv2
import numpy as np

data = np.load('directory_embeddings_map.npz', allow_pickle=True)
folder_name_data_map = data['map'].item()
for key, value  in folder_name_data_map.items():
    a = folder_name_data_map[key]
    b = folder_name_data_map[key]['embeddings']
    c = folder_name_data_map[key]['file_paths']