# %%
import sys
sys.path.append('..')

import os
import torch
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from src.paths import DATA

# %%
class BirdClefData(Dataset):
    ''' Similar to FirstStageDataSnippet, but loads .wav data, augments it, and extracts features on the fly. '''

    def __init__(self, 
                 train_or_val                   = 'train',
                 val_frac                       = 0.1,
                 data_item_length               = 300):

        self.train_or_val                       = train_or_val
        self.val_frac                           = val_frac
        self.data_item_length                   = data_item_length

        self.df = self.df[self.df['augment_index'].isin(self.use_augment_index)]
        self.df.reset_index(drop=True, inplace=True) # Drop rows based on augment_index here

        self.df_augindx_by_id = self.df.groupby('id')['augment_index'].apply(list)
        self.df_ids = self.df[['dataset', 'id', 'speaker', 'case_no']].drop_duplicates().reset_index(drop=True)
        self.df_ids = self.df_ids.join(self.df_augindx_by_id, on='id')

    def __len__(self):
        if self.train_or_val == 'val':
            return len(self.df)
        else:
            return len(self.df_ids)

    def __getitem__(self, index, max_out_phoneme_label_vals=True, limit_displacement_values=False):
        if self.train_or_val == 'val':
            case = self.df.iloc[index]
        else:
            case = self.df_ids.iloc[index]
            aug_indx = random.choice(case['augment_index'])
            case = self.df[(self.df['id']==case['id']) & (self.df['augment_index']==aug_indx)].iloc[0]

        audio_features = np.load(case['audio_npy'], allow_pickle=True)

        if case['dataset'] == 'BIWI':
            landmark_mean, landmark_std = self.LANDMARK_FEATURES_MEAN_BIWI, self.LANDMARK_FEATURES_STD_BIWI
        elif case['dataset'] == 'GRID':
            landmark_mean, landmark_std = self.LANDMARK_FEATURES_MEAN_GRID, self.LANDMARK_FEATURES_STD_GRID
        elif case['dataset'] == 'SAVEE':
            landmark_mean, landmark_std = self.LANDMARK_FEATURES_MEAN_SAVEE, self.LANDMARK_FEATURES_STD_SAVEE
        

        phoneme_labels = np.load(case['phoneme_npy'])
        landmark_features = np.load(case['landmark_npy'])

        padded_indx = self.get_padded_indices(audio_features, window_size=self.data_item_length)
        random_padded_slice_indx = padded_indx[random.choice(range(padded_indx.shape[0]))] # Take a random slice from the generated padded indices

        audio_features      = audio_features[random_padded_slice_indx, :]         # [300, 65] -> 300 is the specified self.data_item_length; could change. 
        phoneme_labels      = phoneme_labels[random_padded_slice_indx, :]         # [300, 15]
        landmark_features   = landmark_features[random_padded_slice_indx, :, :]   # [300, 68, 2]
        landmark_features   = landmark_features[:, self.landmark_subset, :]       # [300, 68, 2] -> Needs to be [300, 32, 2] (subset of 68 landmarks)

        if max_out_phoneme_label_vals: # Represent phoneme groups' multi-labels this way [0, 1, 1, 0] (as opposed to [0, 0.5, 0.5, 0]) for thresholding in post-processing
            phoneme_labels[phoneme_labels > 0] = 1.0
        
        # Centralize landmark displacement values
        if self.centralize_displacement_values:
            landmark_features = landmark_features - landmark_features.mean(axis=0)

        # Standardize landmark displacement values
        if self.standardize_landmark_displacement:
            landmark_features = (landmark_features - landmark_mean) / landmark_std

        # Return only y displacement values if specified
        if self.only_y_displacement:
            landmark_features = landmark_features[:, :, 1]

        if limit_displacement_values: # Clip any values that are outside of the range [-1, 1]
            landmark_features[landmark_features > 1.0] = 1.0
            landmark_features[landmark_features < -1.0] = -1.0

        audio_features = torch.from_numpy(audio_features)
        phoneme_labels = torch.from_numpy(phoneme_labels)
        landmark_features = torch.from_numpy(landmark_features)

        return audio_features, phoneme_labels, landmark_features

    def get_padded_indices(self, normalized_feat: torch.Tensor, window_size=24):
        ''' See temp_get_padded_indices.py for examples '''
        num_frames = normalized_feat.shape[0]
        wav_idxs = [i for i in range(0, num_frames)]

        half_win_size = window_size // 2
        pad_head = [0 for _ in range(half_win_size)]
        pad_tail = [wav_idxs[-1] for _ in range(half_win_size)]
        padded_idxs = np.array(pad_head + wav_idxs + pad_tail)

        target_wav_idxs = np.zeros(shape=(num_frames, window_size)).astype(int)
        for i in range(0, num_frames):
            target_wav_idxs[i] = padded_idxs[i:i+window_size].reshape(-1, window_size)

        if num_frames > window_size: # Remove ones with padding if features are longer than window.
            target_wav_idxs = target_wav_idxs[half_win_size:-half_win_size]

        return target_wav_idxs


# %% Test for FirstStageData Dataset Class
# if __name__=='__main__':
#     first_stage_data = FirstStageDataPreAugmented()
#     dataloader = DataLoader(dataset     = first_stage_data, 
#                             shuffle     = False, 
#                             batch_size  = 2, 
#                             num_workers = 1)

#     for audio_feature, phoneme_label, landmark_displacement in tqdm(dataloader):
#         print('Audio Feature Shape:', audio_feature.shape)
#         print('Phoneme Label Shape:', phoneme_label.shape)
#         print('Landmark Displ. Shape:', landmark_displacement.shape)

    # # For cases where the multiple workers are throwing an error:
    # first_stage_data = FirstStageDataPreAugmented()
    # first_stage_data[0]

# %% Test for FirstStagetData Dataset Class
if __name__=='__main__':
    from preprocess_codes.util_visualize_data import visualize_features

    first_stage_data = FirstStageDataPreAugmented(train_or_val='train',
                                                  data_item_length=300,
                                                  centralize_displacement_values=False,
                                                  use_augment_index=[0, 2, 3])
    dataloader = DataLoader(dataset     = first_stage_data, 
                            shuffle     = True, 
                            batch_size  = 256, 
                            num_workers = 1)
    for audio_feature, phoneme_label, landmark_displacement in tqdm(dataloader):
        print('='*90)
        print('Audio Feature Shape:', audio_feature.shape)
        print('Phoneme Label Shape:', phoneme_label.shape)
        print('Landmark Displ. Shape:', landmark_displacement.shape)

        visualize_features(audio_feature[0, :, :], type='audio')
        visualize_features(phoneme_label[0, :, :], type='phoneme')
        visualize_features(landmark_displacement[0, :, :], type='landmark')
        break


    # # For cases where the multiple workers are throwing an error:
    # first_stage_data = FirstStageDataPreAugmented()
    # first_stage_data[0]
