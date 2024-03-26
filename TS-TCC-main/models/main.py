from torch import nn
import torch
import numpy as np
from seed_utils import left_lower_channel, left_upper_channel, right_lower_channel, right_upper_channel, channelID2str
from SEED_Configs import Config as Configs
class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 1024, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(1024, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        # print(model_output_dim * configs.final_out_channels)
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x

    
class Conv2d_model(nn.Module):
    def __init__(self, configs):
        super(Conv2d_model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm2d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(6656, configs.num_classes)

    def forward(self, x_in):
        x = x_in.unsqueeze(1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x
    


class Conv1d_single_Model(nn.Module):
    def __init__(self, configs):
        super(Conv1d_single_Model, self).__init__()

        # Define a list to hold individual convolutional layers
        self.conv_blocks = nn.ModuleList()

        for _ in range(configs.input_channels):
            conv_block = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=configs.kernel_size,
                          stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(configs.dropout)
            )
            self.conv_blocks.append(conv_block)

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32 * 62, 256, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(256, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        # Split the input tensor along the channel dimension
        x_splits = torch.split(x_in, 1, dim=1)

        # Apply each convolutional layer independently
        x_blocks = [conv_block(x_split) for conv_block, x_split in zip(self.conv_blocks, x_splits)]

        # Concatenate the results along the channel dimension
        x = torch.cat(x_blocks, dim=1)

        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x
    

class Conv1d_single3_Model(nn.Module):
    def __init__(self, configs):
        super(Conv1d_single3_Model, self).__init__()

        # Define a list to hold individual convolutional layers
        self.conv_blocks = nn.ModuleList()

        for _ in range(configs.input_channels):
            conv_block = nn.Sequential(
                # the first conv
                nn.Conv1d(1, 32, kernel_size=configs.kernel_size,
                          stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(configs.dropout),

                # the second conv
                nn.Conv1d(32, 4, kernel_size=configs.kernel_size,
                          stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
                nn.BatchNorm1d(4),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
                
                nn.Conv1d(4, 1, kernel_size=configs.kernel_size,
                          stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
                nn.BatchNorm1d(1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
            )

            self.conv_blocks.append(conv_block)

        model_output_dim = 62
        final_out_channels = 105
        self.logits = nn.Linear(model_output_dim * final_out_channels, configs.num_classes)

    def forward(self, x_in):
        # Split the input tensor along the channel dimension
        x_splits = torch.split(x_in, 1, dim=1)

        # Apply each convolutional layer independently
        # 62 * 1 * 27 
        x_blocks = [conv_block(x_split) for conv_block, x_split in zip(self.conv_blocks, x_splits)]

        # Concatenate the results along the channel dimension
        # 62 * 27
        x = torch.cat(x_blocks, dim=1)
        print(x.shape)
        exit(0)

        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x
    
    
class Conv_seperate_model(nn.Module):
    def __init__(self, configs):
        super(Conv_seperate_model, self).__init__()

        def map_channel_to_id(channel):
            id = next((id for id, name in channelID2str.items() if name == channel), None)
            if id is not None:
                return id - 1
            return None
        
        self.channel_mapped_ids = [
            list(map(map_channel_to_id, left_upper_channel)),
            list(map(map_channel_to_id, right_upper_channel)),
            list(map(map_channel_to_id, left_lower_channel)),
            list(map(map_channel_to_id, right_lower_channel))
        ]
        
        
        self.encoder = nn.ModuleList()
        
        for i in range(0,4):
            layer = nn.Sequential( 
                    nn.Conv2d(1, 128, kernel_size=8,
                            stride=2, bias=False, padding = 4),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=4, stride=4),
                    nn.Dropout(0.35),

                    nn.Conv2d(128, 64, kernel_size=8, stride=1, bias=False, padding = 4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),

                    nn.Conv2d(64, 16, kernel_size=8, stride=1, bias=False, padding = 4),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=1)
            )
            
            self.encoder.append(layer)
        
        self.linear = nn.Sequential(
            nn.Linear(4160, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.logits = nn.Linear(4160, 3)
        
    def seperate_x(self,x):
        x_seperate = []
        for i in range(0,4):
            x_seperate.append(x[:,self.channel_mapped_ids[i],:])

        x_feature = []
        for i in range (0,4):
            x_seperate[i] = x_seperate[i].reshape(x_seperate[i].shape[0], -1, 100)
            x_seperate[i] = x_seperate[i].unsqueeze(1) 
            # [128, 16, 5, 13]
            feature = self.encoder[i](x_seperate[i])
            # [128, 16, 65]  
            feature = feature.reshape(feature.shape[0], feature.shape[1], -1)
            # feature = feature.reshape(feature.shape[0], -1)
            x_feature.append(feature)
        return x_feature
        
    def forward(self, x_in):
        x_feature = self.seperate_x(x_in)
        x_cat = torch.cat(x_feature, dim = 1)
        # print(x_cat.shape)
        x_flat = x_cat.reshape(-1, 64 * 65)
        # print(x_flat.shape)
        # feature = self.linear(x_flat)
        logits = self.logits(x_flat)
        return logits, x_cat



if __name__ == "__main__":
    configs = Configs()
    model = Conv1d_single3_Model(configs)
    # print(model)
    x = torch.randn(32, 62, 200)
    logits, x = model(x)
    print(logits.shape)
    print(x.shape)
    print("Done")
