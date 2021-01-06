import json
from pathlib import Path

import torch
import torch.utils.data as data

from .loader import VideoLoader
import os


def get_database(root_path, video_path_formatter):
    video_paths = []
    video_list = os.listdir(root_path)
    for video in video_list:
        video_paths.append(video_path_formatter(root_path, video))
    return video_paths


class VideoDataset(data.Dataset):
    def __init__(self,
                 root_path,
                 spatial_transform=None,
                 temporal_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, video_name: root_path / video_name),
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg'):
        self.data = get_database(root_path, video_path_formatter)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        return clip

    def __getitem__(self, index):
        path = self.data[index]
        frame_list = os.listdir(path)
        frame_indices = [indice for indice in range(1, len(frame_list)+1)]
#         print(frame_indices)
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
#         print(frame_indices)
        clip = self.__loading(path, frame_indices)
       
        return clip

    def __len__(self):
        return len(self.data)