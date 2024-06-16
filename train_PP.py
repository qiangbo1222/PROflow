import argparse
import logging
import os
from os.path import join

import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import yaml
from easydict import EasyDict as edict
from yaml import Dumper, Loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data_utils.data_module import get_dataloader
from utils.diffusion_utils import get_t_schedule
from train_module.PP_diffusion import PP_diffusion
from train_module.ema_callback import EMACallback

#os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--data", type=str, default="default")
parser.add_argument("--run_name", type=str, default="full_run")
parser.add_argument("--sample_num", type=int, default=3)
parser.add_argument("--run_dir", type=str, default="/storage/boqiang/proflow/lightning_logs")
ter_args = parser.parse_args()

config_path = "config"
args_dict = edict({
    "dataset": yaml.load(open(join(config_path, f"dataset/{ter_args.data}.yml")), Loader=Loader),
    "model": yaml.load(open(join(config_path, "model/default.yml")), Loader=Loader),
    "t_to_sigma": yaml.load(open(join(config_path, "model/t_to_sigma.yml")), Loader=Loader),
    "optim": yaml.load(open(join(config_path, "optim/default.yml")), Loader=Loader),
    "scheduler": yaml.load(open(join(config_path, "optim/scheduler.yml")), Loader=Loader),
    "trainer": yaml.load(open(join(config_path, "trainer/default.yml")), Loader=Loader),
    "checkpoint": yaml.load(open(join(config_path, "checkpoint/default.yml")), Loader=Loader),
    "inference": yaml.load(open(join(config_path, "inference/default.yml")), Loader=Loader),
})

if __name__ == "__main__":
    tb_logger = TensorBoardLogger(ter_args.run_dir, name=f"protac_diffusion({ter_args.run_name})")
    lr_logger = LearningRateMonitor(logging_interval='step')
    #ema_callback = EMACallback(decay=0.99)
    checkpoint_callback = ModelCheckpoint(**args_dict.checkpoint)
    earlystop_callback = EarlyStopping(monitor='val_loss', patience=20, mode='min')

    
    data_module = get_dataloader(args_dict)
    model = PP_diffusion(args_dict)
    trainer = Trainer(**args_dict.trainer, logger=tb_logger,
                      callbacks=[lr_logger, checkpoint_callback],#, ema_callback
                      default_root_dir=ter_args.run_dir)
    
    if ter_args.checkpoint is not None:
        model = model.load_from_checkpoint(ter_args.checkpoint,
            args_dict=args_dict)
    
    if ter_args.mode == "train":
        trainer.fit(model, data_module)
    
    elif ter_args.mode == "test":
        args_dict.trainer.max_epochs = 0
        args_dict.trainer.num_sanity_val_steps = -1
        model.full_inference = os.path.join(ter_args.run_dir, f"protac_diffusion({ter_args.run_name})/inference")
        if not os.path.exists(model.full_inference):
            os.makedirs(model.full_inference)
        model.data_root = args_dict.dataset.data_dir
        model.visualize_results = True
        model.test_sample_num = ter_args.sample_num
        model.args.inference = args_dict.inference
        model.t_schedule = get_t_schedule(inference_steps=args_dict.inference.inference_steps)
        trainer = Trainer(**args_dict.trainer, logger=tb_logger, callbacks=[lr_logger, checkpoint_callback], 
                          flush_logs_every_n_steps=1)
        trainer.fit(model, data_module)
        
    elif ter_args.mode == "preprocess":
        pass#use this to preprocess the data into cached lmdb
    