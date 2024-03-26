from torch import nn
import torch
from .seed_utils import left_lower_channel, left_upper_channel, right_lower_channel, right_upper_channel, channelID2str
# from .SEED_Configs import Config as Configs
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
        self.linear = nn.Sequential(
            nn.Linear(52 * configs.final_out_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256)
        )

        self.logits = nn.Linear(52 * configs.final_out_channels, configs.num_classes)

        
    def forward(self, x_in):
        x = x_in.unsqueeze(1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)

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
        # 62 * 1 * 105
        x_blocks = [conv_block(x_split) for conv_block, x_split in zip(self.conv_blocks, x_splits)]

        # Concatenate the results along the channel dimension
        # 62 * 105
        x = torch.cat(x_blocks, dim=1)
        # print(x.shape)
        return x
    
    def set_first_grad_false(self):
        for conv_block in self.conv_blocks:
            for idx, layer in enumerate(conv_block):
                if idx < 5:
                    # print(layer)
                    for param in layer.parameters():
                        param.requires_grad = False
        
            
class Channel_Conv1d_single3_Model(nn.Module):
    def __init__(self, model, configs, device, logger, if_scale):
        super(Channel_Conv1d_single3_Model, self).__init__()
        self.conv1d_single3_model = model
        self.scale = nn.Parameter(torch.ones(configs.final_out_channels).unsqueeze(0).unsqueeze(-1).to(device), requires_grad=True)
        self.logits = nn.Linear(configs.features_len * configs.final_out_channels, configs.num_classes)
        self.logger = logger
        self.if_scale = if_scale
        
        
    def forward(self, x_in):
        # x [62, 105]
        x = self.conv1d_single3_model(x_in)

        if self.if_scale:
            x = x * self.scale
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x
    
    def load_state_dict_inner(self, model_dict):
        self.conv1d_single3_model.load_state_dict(model_dict)
        
    def set_grad_false(self, pretrained_dict):
        def set_requires_grad(model, dict_, requires_grad=True):
            for param in model.named_parameters():
                if param[0] in dict_:
                    param[1].requires_grad = requires_grad
        set_requires_grad(self.conv1d_single3_model, pretrained_dict, requires_grad=False)
        
    def get_weight(self):
        scale = self.scale.squeeze(0)
        scale = scale.squeeze(1)
        self.logger.debug(scale)
        return scale.detach().cpu().numpy()
        
    def set_first_layer_grad_false(self):
        self.conv1d_single3_model.set_first_grad_false()
        


# if __name__ == "__main__":
#     configs = Configs()
#     model = Conv1d_single2_Model(configs)
#     # print(model)
#     x = torch.randn(32, 62, 200)
#     logits, x = model(x)
#     print(logits.shape)
#     print(x.shape)
#     print("Done")
