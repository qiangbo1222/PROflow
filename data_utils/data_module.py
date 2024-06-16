from functools import partial
import pickle

import torch
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.transforms import BaseTransform

from utils.diffusion_utils import t_to_sigma
from data_utils.dips import NoiseTransform_forPP, DIPS, ListDataset


def get_dataloader(args):
    t_to_sigma_ = partial(t_to_sigma, args=args.t_to_sigma)
    transform = NoiseTransform_forPP(t_to_sigma=t_to_sigma_, linker_dict=pickle.load(open(args.dataset.linker_dict, 'rb')))
    args = args.dataset

    common_args = {'root': args.data_dir, 'esm_lm_path': args.esm_embeddings_path, 'transform': transform, 'data_source': args.data_source,
                   'cache_path': args.cache_path, 'limit_complexes': args.limit_complexes, 'msms_bin': args.msms_bin, 'server_cache': args.server_cache,
                }
    

    if args.data_source == 'dips':
        #train_dataset = DIPS(split='train', **common_args)
        #val_dataset = DIPS(split='val', **common_args)
        #test_dataset = DIPS(split='test', **common_args)
        full_dataset = DIPS(split='full', **common_args)
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [int(len(full_dataset) * 0.9), len(full_dataset) - int(len(full_dataset) * 0.9)])
        data_module = LightningDataset(train_dataset, val_dataset, **args.loader)

    elif args.data_source == 'protac':
        #for debug
        train_dataset = ListDataset(DIPS(split='train', **common_args), 3)
        val_dataset = DIPS(split='val', **common_args)
        test_dataset = DIPS(split='test', **common_args)
        data_module = LightningDataset(train_dataset, val_dataset, test_dataset=test_dataset, **args.loader)

    elif args.data_source == 'e3':
        dataset = DIPS(split=args.split, **common_args)
        torch.manual_seed(2023)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
        data_module = LightningDataset(train_dataset, val_dataset, **args.loader)
    
    elif args.data_source == 'confidence':
        common_args['transform'] = None
        train_dataset = DIPS(split='train', **common_args)
        val_dataset = DIPS(split='val', **common_args)
        test_dataset = DIPS(split='test', **common_args)
        data_module = LightningDataset(train_dataset, val_dataset, test_dataset=test_dataset, **args.loader)
        
    
    return data_module