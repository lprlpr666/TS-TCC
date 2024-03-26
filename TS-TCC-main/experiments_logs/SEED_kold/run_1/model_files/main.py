import torch

import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
from dataloader.dataloader import data_generator,data_generator_all,data_generator_all_seed_34, data_generator_partial, data_generator_free, data_generator_all_seed4
from dataloader.dataloader_kfold import data_generator_seed4_kfolder, data_generator_seed_kfolder, data_generator_sample
from trainer.trainer import Trainer, model_evaluate
from models.TC import TC
from utils import _calc_metrics, copy_Files
from models.model import Conv1d_single3_Model, Channel_Conv1d_single3_Model, Conv2d_model
from models.SEED_Configs import Config3, Config4
import warnings


warnings.filterwarnings('ignore')
# Args selections
start_time = datetime.now()


parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--selected_dataset', default='Epilepsy', type=str,
                    help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
parser.add_argument('--id', default='1', type=str,
                    help='SEED id')
parser.add_argument('--ratio', default=3, type=int)

parser.add_argument('--froze_first', default='F', type=str)
parser.add_argument('--scale_channel', type=str, default='T')
parser.add_argument('--encoder', type=str, default='conv1d', help='conv1d or conv2d')
parser.add_argument('--dataset', type = str, default= 'SEED', help= 'SEED or SEED4')
parser.add_argument('--file_seed',type=int, default=1)

args = parser.parse_args()



device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'TS-TCC'
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

print("------")
if args.dataset == 'SEED':
    configs = Config3()
elif args.dataset == 'SEED4':
    configs = Config4()
# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0

load_seed = 123

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
data_path = f"./data/{data_type}"
trial = int(args.id)
if data_type == 'SEED':
    data_path = f"../autodl-tmp/SEED-Dataset/SEED/{trial}/"

if training_mode == "self_supervised" and data_type == 'SEED':
    # train_dl, valid_dl, test_dl = data_generator_all(configs, training_mode)
    train_dl, valid_dl, test_dl = data_generator_seed_kfolder(args.file_seed, configs, training_mode)
    print("finish loading seed")
elif training_mode == "self_supervised" and data_type == 'SEED4':
    # train_dl, valid_dl, test_dl = data_generator_all_seed4(configs, training_mode)
    train_dl, valid_dl, test_dl = data_generator_seed4_kfolder(args.file_seed, configs, training_mode)
    print("finish loading seed4")
else:
    root_path = "../autodl-tmp/SEED-Dataset/"
    # train_dl, valid_dl, test_dl = data_generator_free(root_path, configs, training_mode, trial, args.ratio, file_name)
    train_dl, valid_dl, test_dl = data_generator_sample(training_mode, configs, args.id, args.ratio, args.file_seed, args.dataset)
    
logger.debug("Data loaded ...")
logger.debug(f"Subjet {trial}")


def train(lr,weight_decay):
    # Load Model

    if args.scale_channel == 'T' and training_mode != 'self_supervised':
        if_scale = True
    else:
        if_scale = False
    logger.debug(f"if scale {if_scale}")
    
    if args.encoder == 'conv1d':
        model_inner = Conv1d_single3_Model(configs).to(device)
        model = Channel_Conv1d_single3_Model(model=model_inner, configs = configs, device = device, logger=logger, if_scale=if_scale).to(device)
    elif args.encoder == 'conv2d':
        model = Conv2d_model(configs=configs).to(device)
        print("conv2d!")

    temporal_contr_model = TC(configs, device).to(device)

    if training_mode == "fine_tune":
        # load saved model of this experiment
        load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{load_seed}", "saved_models"))
        chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
        # print(chkpoint.keys())
        pretrained_dict = chkpoint["model_state_dict"]
        # pretrained_dict = chkpoint
        model_dict = model.state_dict()
        # model_dict = model_inner.state_dict()
        del_list = ['logits','scale', 'projector']
        pretrained_dict_copy = pretrained_dict.copy()
        for i in pretrained_dict_copy.keys():
            for j in del_list:
                if j in i:
                    del pretrained_dict[i]
        model_dict.update(pretrained_dict)
        # model.load_state_dict_inner(model_dict)
        model.load_state_dict(model_dict)
        if args.froze_first == 'T':
            model.set_first_layer_grad_false()

    if training_mode == "train_linear" or "tl" in training_mode:
        load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{load_seed}", "saved_models"))
        chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
        pretrained_dict = chkpoint["model_state_dict"]
        model_dict = model.state_dict()
        # model_dict = model_inner.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # delete these parameters (Ex: the linear layer at the end)
        del_list = ['logits','scale']
        pretrained_dict_copy = pretrained_dict.copy()
        for i in pretrained_dict_copy.keys():
            for j in del_list:
                if j in i:
                    del pretrained_dict[i]

        model_dict.update(pretrained_dict)
        # model.load_state_dict_inner(model_dict)
        # model.set_grad_false(pretrained_dict=pretrained_dict)
        model.load_state_dict(model_dict)
        set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

    if training_mode == "random_init":
        model_dict = model.state_dict()
        # model_dict = model_inner.state_dict()
        # delete all the parameters except for logits
        del_list = ['logits', 'scale']
        pretrained_dict_copy = model_dict.copy()
        for i in pretrained_dict_copy.keys():
            for j in del_list:
                if j in i:
                    del model_dict[i]
        # model.set_grad_false(pretrained_dict=pretrained_dict)
        set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.



    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(configs.beta1, configs.beta2), weight_decay=weight_decay)
    temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=lr, betas=(configs.beta1, configs.beta2), weight_decay=weight_decay)

    if training_mode == "self_supervised":  # to do it only once
        copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)
        file_save_path = os.path.join(experiment_log_dir, str(args.file_seed))
        os.makedirs(file_save_path, exist_ok=True)

    # Trainer
    best_model_loss, best_model_acc, _ = Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, file_save_path, training_mode)

    acc = 0
    loss_acc = 0
    acc_acc = 0
    if training_mode != "self_supervised":
        # Testing
        outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        outs_best_loss = model_evaluate(best_model_loss, temporal_contr_model, test_dl, device, training_mode)
        outs_best_acc = model_evaluate(best_model_acc, temporal_contr_model, test_dl, device, training_mode)
        total_loss, total_acc, _, pred_labels, true_labels = outs
        _, loss_acc, _, _, _ = outs_best_loss
        _, acc_acc, _, _, _ = outs_best_acc
        # _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)
        acc = total_acc

    logger.debug(f"Training time is : {datetime.now()-start_time}")
    return acc, loss_acc, acc_acc

if training_mode != "self_supervised":
    acc, loss_acc, acc_acc = train(0.001, 0.01)
    # acc = train(configs.lr, 3e-4)
    
else:
    _ = train(configs.lr, 3e-4)