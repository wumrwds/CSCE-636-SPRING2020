#!/usr/bin/env python
# coding: utf-8
import cv2
import os
import numpy as np


def video_to_frames(video, resize=(320, 240)):
    """
    Convert video to frame arrays
    """
    # extract frames from a video and return a frame array
    vidcap = cv2.VideoCapture(video)
    frames = []
    while vidcap.isOpened():
        success, image = vidcap.read()
        
        if success:
            frames.append(image)
        else:
            break
    
    cv2.destroyAllWindows()
    vidcap.release()
    
    # resize each frame
    resized_frames = []
    for frame in frames:
        resized_frames.append(cv2.resize(frame, dsize=resize, interpolation=cv2.INTER_CUBIC))
    
    return np.array(resized_frames)


def parse_videos_in_folder(folder_path, label):
    """
    Parse all videos in a folfer and mark all video with the given tag
    """
    train_data = []
    train_label = []
    
    for file_path in os.listdir(folder_path):
        if os.path.splitext(file_path)[1] in (".avi", ".mp4"):
            file_path = folder_path + file_path if folder_path[-1] == '/' else folder_path + '/' + file_path
            print(file_path, label)

            # parse videos into frame arrays
            frames = video_to_frames(file_path)

            train_data.append(frames)
            train_label.append(label)
    
    return train_data, train_label


def merge_dataset(data_1, label_1, data_2, label_2):
    """
    Merge 2 datasets and label sets
    """
    merged_data = data_1 + data_2
    merged_label = label_1 + label_2

    merged_data = np.array(merged_data)
    merged_label = np.array(merged_label)

    return merged_data, merged_label


# Parse train set, validation set and test set
train_data_other, train_label_other = parse_videos_in_folder('dataset/train/other', 0)
train_data_slip, train_label_slip = parse_videos_in_folder('dataset/train/slip', 1)
train_data, train_label = merge_dataset(train_data_other, train_label_other, train_data_slip, train_label_slip)
del train_data_other, train_label_other, train_data_slip, train_label_slip

valid_data_other, valid_label_other = parse_videos_in_folder('dataset/valid/other', 0)
valid_data_slip, valid_label_slip = parse_videos_in_folder('dataset/valid/slip', 1)
valid_data, valid_label = merge_dataset(valid_data_other, valid_label_other, valid_data_slip, valid_label_slip)
del valid_data_other, valid_label_other, valid_data_slip, valid_label_slip

test_data_other, test_label_other = parse_videos_in_folder('dataset/test/other', 0)
test_data_slip, test_label_slip = parse_videos_in_folder('dataset/test/slip', 1)
test_data, test_label = merge_dataset(test_data_other, test_label_other, test_data_slip, test_label_slip)
del test_data_other, test_label_other, test_data_slip, test_label_slip


# get the median length of video frame array of the training set
frame_lens = []
for frames in train_data:
    frame_lens.append(len(frames))

median_len = int(np.median(np.array(frame_lens)))
    
print("The median of the frame length is %d." % median_len)



def normalize_frames(frames, length):
    """
    Truncate the excess frames and pad the missing frames using the head & tail frame
    """
    mid = len(frames) // 2
    half_len = length // 2
    if len(frames) >= length:
        norm_frames = frames[mid - half_len : mid + half_len]
    else:
        left = half_len - mid
        norm_frames = []
        for i in range(length):
            if i < left:
                norm_frames.append(frames[0])
            elif i - left < len(frames):
                norm_frames.append(frames[i-left])
            else:
                norm_frames.append(frames[-1])

    return np.array(norm_frames)
    
    
def normalize_frame_array(frame_array, length):
    """
    Normalize all frame arrays and convert it to a numpy array
    """
    norm_frame_list = []
    for frames in frame_array:
        norm_frame_list.append(normalize_frames(frames, length))

    return np.array(norm_frame_list)


# normalize train set, validation set and test set
train_data = normalize_frame_array(train_data, median_len // 2 * 2)
test_data = normalize_frame_array(test_data, median_len // 2 * 2)
valid_data = normalize_frame_array(valid_data, median_len // 2 * 2)

print(train_data.shape, train_label.shape)
print(valid_data.shape, valid_label.shape)
print(test_data.shape, test_label.shape)

# save train set, validation set and test set as a .npy file
np.save('train_data', train_data)
np.save('train_label', train_label)

np.save('valid_data', valid_data)
np.save('valid_label', valid_label)

np.save('test_data', test_data)
np.save('test_label', test_label)
