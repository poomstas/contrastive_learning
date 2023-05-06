# %%
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

# %%
class TemporalConvNetFirst(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=4, tcn_dropout=0.2, use_only_y_landmark_displacements=False):
        super(TemporalConvNetFirst, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=tcn_dropout)]

        self.network = nn.Sequential(*layers)

        self.n_phoneme = 15
        self.n_landmark = 32 if use_only_y_landmark_displacements else 64
        self.n_features_final = num_channels[-1]

        self.bn_phoneme_1 = nn.BatchNorm1d(self.n_features_final, affine=True)
        self.fc_phoneme_1 = nn.Linear(self.n_features_final, self.n_features_final)
        self.fc_phoneme_2 = nn.Linear(self.n_features_final, self.n_phoneme)

        self.bn_landmark_1 = nn.BatchNorm1d(self.n_features_final, affine=True)
        self.fc_landmark_1 = nn.Linear(self.n_features_final, self.n_features_final)
        self.fc_landmark_2 = nn.Linear(self.n_features_final, self.n_landmark)
        
        self.initialize_layer(self.fc_landmark_1, bias_constant_val=0.1)
        self.initialize_layer(self.fc_landmark_2, bias_constant_val=0.0)
        self.initialize_layer(self.fc_phoneme_1, bias_constant_val=0.1)
        self.initialize_layer(self.fc_phoneme_2, bias_constant_val=0.1)

        self.relu = nn.ReLU()

    def initialize_layer(self, layer, bias_constant_val=0.0):
        ''' Zero values for bias and xavier_normal for weights '''
        for name, param in layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, bias_constant_val)
            elif 'weight' in name:
                nn.init.xavier_normal_(param, gain=1.0)

    def forward(self, x):
        network_out = self.network(x).permute(0, 2, 1)                              # [BS, L, C]

        # Visemenet standard
        landmark_out = self.fc_landmark_1(network_out)                              # [BS, L, C]
        landmark_out = self.bn_landmark_1(network_out.permute(0, 2, 1))             # [BS, C, L]
        landmark_out = self.relu(landmark_out)                                      # [BS, C, L]
        landmark_out = self.fc_landmark_2(landmark_out.permute(0, 2, 1))            # [BS, L, C]

        phoneme_out = self.fc_phoneme_1(network_out)
        phoneme_out = self.bn_phoneme_1(phoneme_out.permute(0, 2, 1))
        phoneme_out = self.relu(phoneme_out)
        phoneme_out = self.fc_phoneme_2(phoneme_out.permute(0, 2, 1))

        return phoneme_out, landmark_out

# %%
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# %%
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# %%
if __name__=='__main__':
    from data_first_model import FirstStageDataPreAugmented

    batch_size = 512
    n_features = 65
    n_timesteps = 200

    model = TemporalConvNetFirst(num_inputs=n_features, num_channels=[512, 256, 128])
    summary(model = model,
            input_size = (batch_size, n_features, n_timesteps),
            device='cpu')

    first_stage_data = FirstStageDataPreAugmented(train_or_val='train')
    dataloader = DataLoader(dataset     = first_stage_data,
                            shuffle     = False, 
                            batch_size  = batch_size, 
                            num_workers = 1)

    for audio_features, phoneme_label, landmark_displacement in tqdm(dataloader):
        print('='*90)
        print('Audio Feature Shape:', audio_features.shape)
        print('='*90)

        phoneme_out, landmark_out = model(audio_features.permute(0, 2, 1).float()) # [512, 300, 39] -> [512, 39, 300]

        print('Phoneme Pred. Shape:', phoneme_out.shape)
        print('Phoneme Label Shape:', phoneme_label.shape)
        print('='*90)
        print('Landmark Pred. Shape:', landmark_out.shape)
        print('Landmark Displ. Shape:', landmark_displacement.shape)
        print('='*90)

        break
