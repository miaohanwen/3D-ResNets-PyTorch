from torchvision import get_image_backend

from datasets.videodataset import VideoDataset
from datasets.videodataset_multiclips import (VideoDatasetMultiClips,
                                              collate_fn)
from torch.utils.data.dataloader import default_collate
from datasets.activitynet import ActivityNet
from datasets.loader import VideoLoader, VideoLoaderHDF5, VideoLoaderFlowHDF5


def image_name_formatter(x):
    return f'image_{x:05d}.jpg'


def get_training_data(video_path,
                      annotation_path,
                      dataset_name,
                      input_type,
                      file_type,
                      spatial_transform=None,
                      temporal_transform=None,
                      target_transform=None):
    assert dataset_name in [
        'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit'
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id)
    else:
        if input_type == 'rgb':
            loader = VideoLoaderHDF5()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = (lambda root_path, label, video_id: root_path /
                                label / f'{video_id}.hdf5')

    if dataset_name == 'activitynet':
        training_data = ActivityNet(video_path,
                                    annotation_path,
                                    'training',
                                    spatial_transform=spatial_transform,
                                    temporal_transform=temporal_transform,
                                    target_transform=target_transform,
                                    video_loader=loader,
                                    video_path_formatter=video_path_formatter)
    else:
        training_data = VideoDataset(video_path,
                                     annotation_path,
                                     'training',
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform,
                                     video_loader=loader,
                                     video_path_formatter=video_path_formatter)

    return training_data


def get_validation_data(video_path,
                        annotation_path,
                        dataset_name,
                        input_type,
                        file_type,
                        spatial_transform=None,
                        temporal_transform=None,
                        target_transform=None):
    assert dataset_name in [
        'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit'
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id)
    else:
        if input_type == 'rgb':
            loader = VideoLoaderHDF5()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = (lambda root_path, label, video_id: root_path /
                                label / f'{video_id}.hdf5')

    if dataset_name == 'activitynet':
        validation_data = VideoDataset(video_path,
                                      annotation_path,
                                      'validation',
                                      spatial_transform=spatial_transform,
                                      temporal_transform=temporal_transform,
                                      target_transform=target_transform,
                                      video_loader=loader,
                                      video_path_formatter=video_path_formatter)
    else:
        validation_data = VideoDatasetMultiClips(
            video_path,
            annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter)

    return validation_data, collate_fn


def get_inference_data(video_path,
                       input_type,
                       file_type,
                       spatial_transform=None,
                       temporal_transform=None):
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, video_name: root_path / video_name)
    else:
        if input_type == 'rgb':
            loader = VideoLoaderHDF5()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = (lambda root_path, label, video_id: root_path /
                                label / f'{video_id}.hdf5')

#     inference_data = VideoDataset(video_path,
#                                   spatial_transform=spatial_transform,
#                                   video_loader=loader,
#                                   video_path_formatter=video_path_formatter)
    inference_data = VideoDatasetMultiClips(video_path,
                                            spatial_transform=spatial_transform,
                                            temporal_transform=temporal_transform,
                                            video_loader=loader,
                                            video_path_formatter=video_path_formatter)


    return inference_data, collate_fn