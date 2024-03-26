from torch import nn
import torch
from .seed_utils import left_lower_channel, left_upper_channel, right_lower_channel, right_upper_channel, channelID2str
# from .SEED_Configs import Config as Configs
    
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
        # 62 * 1 * 105
        x_blocks = [conv_block(x_split) for conv_block, x_split in zip(self.conv_blocks, x_splits)]

        # Concatenate the results along the channel dimension
        # 62 * 105
        x = torch.cat(x_blocks, dim=1)
        # print(x.shape)

        # x_flat = x.reshape(x.shape[0], -1)
        # logits = self.logits(x_flat)
        # return logits, x
        return x
    
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
    
    def get_inner(self, model_dict):
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
        
        


# if __name__ == "__main__":
#     configs = Configs()
#     model = Conv1d_single2_Model(configs)
#     # print(model)
#     x = torch.randn(32, 62, 200)
#     logits, x = model(x)
#     print(logits.shape)
#     print(x.shape)
#     print("Done")
