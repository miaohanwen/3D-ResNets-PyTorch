import json
import copy
import functools

import torch
from torch.utils.data.dataloader import default_collate

from .videodataset import VideoDataset
import os


def collate_fn(batch):
    batch_clips = zip(*batch)
    batch_clips = [clip for multi_clips in batch_clips for clip in multi_clips]
    return default_collate(batch_clips)


class VideoDatasetMultiClips(VideoDataset):

    def __loading(self, path, video_frame_indices):
        clips = []
        segments = []
        for clip_frame_indices in video_frame_indices:
            clip = self.loader(path, clip_frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            clips.append(torch.stack(clip, 0).permute(1, 0, 2, 3))
            segments.append(
                [min(clip_frame_indices),
                 max(clip_frame_indices) + 1])

        return clips, segments

    def __getitem__(self, index):
        path = self.data[index]
        frame_list = os.listdir(path)
        video_frame_indices = [indice for indice in range(1, len(frame_list)+1)]
        if self.temporal_transform is not None:
            video_frame_indices = self.temporal_transform(video_frame_indices)
        clips, segments = self.__loading(path, video_frame_indices)
        return clips
    
    