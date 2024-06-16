import os
import pandas as pd
import pickle
from collections import defaultdict
import torch.multiprocessing as mp
import random
import copy
import glob
import dill

from torch_cluster import knn_graph
import torch_scatter
import lmdb

import numpy as np
import torch
import networkx as nx
from torch.utils.data.distributed import DistributedSampler
from kmeans_pytorch import kmeans
import rdkit.Chem as Chem
from rdkit.Chem import MolToSmiles, MolFromSmiles, AddHs
from rdkit.Geometry import Point3D
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

from dataset.process_mols import read_molecule, get_rec_graph, generate_conformer, \
    get_lig_graph_with_matching, extract_receptor_structure, parse_receptor, parse_pdb_from_path, get_protein_surface_graph, extract_protein_structure, extract_c_alpha_coords, mol_to_graph
from dataset.linker_distribution import generate_linker_length, norm_dist_4_points, linker_interpolate
from utils.diffusion_utils import modify_conformer, modify_conformer_torsion, set_time
from utils.geometry import rigid_transform_Kabsch_3D_torch, matrix_to_axis_angle, axis_angle_to_matrix, transform_norm_vector, angle_exp, rot_axis_to_matrix, angle_sum, change_R_T_ref
from utils import so3, torus
from utils.preprocess import ret_anchor_idx
from dataset.linker_matching import match_complex

class NoiseTransform_forPP(BaseTransform):
    def __init__(self, t_to_sigma, linker_dict=None):
        super().__init__()
        self.bandwidth = 0.3 #TODO: move to config
        self.t_to_sigma = t_to_sigma
        self.linker_dict = linker_dict
        if self.linker_dict is not None:
            self.link_key_points = torch.concat([self.linker_dict[k]['key_points'] for k in self.linker_dict.keys()], dim=0)
            self.linker_mol = {k: self.linker_dict[k]['mol'] for k in self.linker_dict.keys()}
            self.idx_to_key = []
            self.idx_to_subkey = []
            for k in self.linker_dict.keys():
                self.idx_to_key.extend([k] * len(self.linker_dict[k]['key_points']))
                self.idx_to_subkey.extend(list(range(len(self.linker_dict[k]['key_points']))))


    def __call__(self, data, from_start=False):
        """
        Args:
            data (HeteroData): The data object.
            from_start (bool): Whether to sample from the start of the diffusion process.
            energy_train (bool): Whether to sample from the EBM training.
            linker (torch.tensor: (4, 3)): the normal vector of the linker
        """
        if from_start:
            t = 1
            #restore the original conformer
            data['ligand'].pos = copy.deepcopy(data['ligand'].org_pos)
            data['ligand'].c_alpha_coords = copy.deepcopy(data['ligand'].org_c_alpha_coords)
            data['ligand'].x = copy.deepcopy(data['ligand'].org_x)
            #data['ligand'].fake_pocket_pos = copy.deepcopy(data['ligand'].org_fake_pocket_pos)
            #data['ligand'].fake_pocket_norm = copy.deepcopy(data['ligand'].org_fake_pocket_norm)
            #data.linker_pos = data.org_linker_pos
            t_linker, t_tor = t, t
            train = False
        else:
            t = np.random.uniform()
            t_linker, t_tor = t, t
            train = True
        return self.apply_noise(data, t_linker, t_tor, train=train)

    def apply_noise(self, data, t_linker, t_tor, rand=None, rand_idx=None, dt=0.1, train=False):
        '''
        dt = min(1 - t_linker, dt)
        dt = np.random.uniform() * dt
        #clip too small dt for numerical stability
        if dt < 0.01:
            dt = 0.01
        '''
        if not torch.is_tensor(data['ligand'].pos):
            data['ligand'].pos = random.choice(data['ligand'].pos)
        if isinstance(data['ligand'].c_alpha_coords, np.ndarray):
            data['ligand'].c_alpha_coords = torch.from_numpy(data['ligand'].c_alpha_coords).to(data['ligand'].pos.device).float()
        data['ligand'].org_pos = copy.deepcopy(data['ligand'].pos)
        data['ligand'].org_c_alpha_coords = copy.deepcopy(data['ligand'].c_alpha_coords)
        data['ligand'].org_x = copy.deepcopy(data['ligand'].x)
        if hasattr(data, 'linker'):
            data.org_linker = copy.deepcopy(data.linker)
        else:
            data.org_linker = None

        data = set_time(data, t_tor, t_linker, 1)
        if data.linker_key != "linker":
            if data.linker_subkey == rand_idx:
                rand_idx = (rand_idx + 1) % len(self.linker_dict[data['linker_key']]['key_points'])
        data.linker_sigma = self.t_to_sigma(t_linker)
        #sample_dic = self.sample_linker(data, random_idx=rand_idx, sigma = data.linker_sigma)
        #sample_dic_dt = self.sample_linker(data, random_idx=rand_idx, sigma = self.t_to_sigma(t_linker + dt))
        #data.tr_score = (- sample_dic['noised_tr_linker'] / data.linker_sigma).float()
        #data.rot_score = angle_exp(sample_dic['noised_rot'], - 1/data.linker_sigma).unsqueeze(0).float()
        #data.rot_score = data.rot_score % (np.pi * 2)
        
        sample_dic = self.sample_linker(data, random_idx=rand_idx, sigma = 1)
        #data.tr_score = (sample_dic['noised_tr_linker'] - sample_dic_dt['noised_tr_linker']) / dt
        #data.rot_score = angle_sum(sample_dic['noised_rot'], angle_exp(sample_dic_dt['noised_rot'], -1))
        #data.rot_score = angle_exp(data.rot_score, 1/dt).unsqueeze(0).float()

        #data.tr_update = sample_dic['noised_tr_linker'].float()
        #data.rot_update = sample_dic['noised_rot'].float()
        data.tr_update = sample_dic['noised_tr_linker'].float() * data.linker_sigma
        data.rot_update = (angle_exp(sample_dic['noised_rot'].float(), data.linker_sigma)) % (np.pi * 2)
        
        #data_cache = modify_conformer(copy.deepcopy(data), rot_update.float(), tr_update_backbone=tr_update, has_norm=True, 
        #                             tr_update_surface=change_R_T_ref(tr_update, axis_angle_to_matrix(rot_update), 
        #                                                               data['ligand'].c_alpha_coords,
        #                                                               torch.cat([data['ligand'].pos, data['ligand'].fake_pocket_pos])).squeeze(-1).to(data.linker_pos.device).float(),
        ##                             tr_update_linker=change_R_T_ref(tr_update, axis_angle_to_matrix(rot_update),
        #                                                            data['ligand'].c_alpha_coords,
        #                                                               data.linker_pos).squeeze(-1).to(data.linker_pos.device).float())
        #search_dic = self.search_linker_conf(data_cache, data['ligand'].c_alpha_coords, 
        #                                     torch.cat([data['ligand'].pos, data['ligand'].fake_pocket_pos]), 
        #                                     data.linker_pos)
        #print("search dic:", search_dic, "tr_update:", tr_update, "rot_update:", rot_update)
        data = modify_conformer(data, data.rot_update, 
                                tr_update_backbone=change_R_T_ref(data.tr_update, axis_angle_to_matrix(data.rot_update),
                                                                  data.linker_pos,
                                                                  data['ligand'].c_alpha_coords), 
                                tr_update_linker=data.tr_update,
                                tr_update_surface=change_R_T_ref(data.tr_update, axis_angle_to_matrix(data.rot_update),
                                                                data.linker_pos,
                                                                torch.cat([data['ligand'].pos, data['ligand'].fake_pocket_pos])), has_norm=True)
        data.linker_mol = sample_dic['noised_linker_mol']
        #data = data_cache
        tor_sigma = self.t_to_sigma(t_tor)
        data.tor_sigma = tor_sigma
        if rand is None:
            rand = torch.rand(1) * np.pi * 2
        else:
            rand = torch.tensor(rand) * np.pi * 2
        tor_l = rand[0] * tor_sigma
        data = modify_conformer_torsion(data, tor_l.float(), tor_l.float() * 0, has_norm=True)
        data.tor_l_update = tor_l.float()
        
        R, t = rigid_transform_Kabsch_3D_torch(data['ligand'].pos.T, data['ligand'].org_pos.T)
        rot_angles = matrix_to_axis_angle(R)
        #data.tr_score = t.T.unsqueeze(0).float() / data.linker_sigma
        R_, t_ = rigid_transform_Kabsch_3D_torch(data['ligand'].c_alpha_coords.T, data['ligand'].org_c_alpha_coords.T)
        data.tr_score = t_.T.unsqueeze(0).float() / data.linker_sigma
        data.rot_score = (angle_exp(rot_angles, 1/ data.linker_sigma).unsqueeze(0) % (2*np.pi)).float()
        data.tr_update = - t.T.float()
        data.rot_update = (angle_exp(rot_angles, -1) % (2*np.pi)).float()
        
        if train:
            band_sigma = data.linker_sigma * (1 - data.linker_sigma)
            band_tr = torch.randn([1, 3]) * np.sqrt(10 * self.bandwidth * band_sigma)
            band_rot = so3.sample_vec(self.bandwidth * band_sigma)
            data.band_tr_update = band_tr.float()
            data.band_rot_update = torch.from_numpy(band_rot).float()
            
            data = modify_conformer(data, data.band_rot_update,
                                tr_update_backbone=change_R_T_ref(data.band_tr_update, axis_angle_to_matrix(data.band_rot_update),
                                                                data['ligand'].pos,
                                                                data['ligand'].c_alpha_coords),
                                tr_update_linker=change_R_T_ref(data.band_tr_update, axis_angle_to_matrix(data.band_rot_update),
                                                                  data['ligand'].pos,
                                                                  data.linker_pos), 
                                tr_update_surface=change_R_T_ref(data.band_tr_update, axis_angle_to_matrix(data.band_rot_update),
                                                                  data['ligand'].pos,
                                                                  torch.cat([data['ligand'].pos, data['ligand'].fake_pocket_pos])), has_norm=True)
            data.tr_update += data.band_tr_update
            data.rot_update = angle_sum(data.rot_update, data.band_rot_update)
            data.rot_update = data.rot_update % (np.pi * 2)
        
         
        #data, tr_mod, rot_mod = match_complex(data)
        #data.tr_update += tr_mod
        #data.rot_update = angle_sum(data.rot_update, rot_mod)
        return data
    

    def search_for_linker(self, data):
        if hasattr(data, 'linker'):
            linker = data.linker
        else:
            linker = None
        if linker is None:
            if not hasattr(data['ligand'], 'fake_pocket_idx'):
                fake_pocket_ligand_pos_idx, fake_pocket_receptor_pos_idx= sample_pocket(data['ligand'].pos, data['receptor'].pos, 
                                                                                        data['ligand'].curv, data['receptor'].curv)

            else:
                fake_pocket_ligand_pos_idx, fake_pocket_receptor_pos_idx = data['ligand'].fake_pocket_idx, data['receptor'].fake_pocket_idx

            fake_pocket_ligand_pos = torch.mean(data['ligand'].pos[fake_pocket_ligand_pos_idx], dim=0)
            fake_pocket_ligand_norm = torch.mean(data['ligand'].x[:, 0, -4:-1][fake_pocket_ligand_pos_idx], dim=0) / \
                                        torch.norm(torch.mean(data['ligand'].x[:, 0, -4:-1][fake_pocket_ligand_pos_idx], dim=0, keepdim=True))
            #fake_pocket_ligand_pos += fake_pocket_ligand_norm * torch.rand(1) * 3 

            fake_pocket_receptor_pos = torch.mean(data['receptor'].pos[fake_pocket_receptor_pos_idx], dim=0)
            fake_pocket_receptor_norm = torch.mean(data['receptor'].x[:, 0, -4:-1][fake_pocket_receptor_pos_idx], dim=0) / \
                                        torch.norm(torch.mean(data['receptor'].x[:, 0, -4:-1][fake_pocket_receptor_pos_idx], dim=0, keepdim=True))
            #fake_pocket_receptor_pos += fake_pocket_receptor_norm * torch.rand(1) * 3
            if torch.isnan(fake_pocket_ligand_pos).any() or torch.isnan(fake_pocket_ligand_norm).any() or torch.isnan(fake_pocket_receptor_pos).any() or torch.isnan(fake_pocket_receptor_norm).any():
                print("fake pocket nan")
            '''
            #clip protein surface
            clean_idx_lig = torch.where(torch.sum((data['ligand'].pos - fake_pocket_ligand_pos)**2, dim=-1) < 400)[0]#clip 20 from pocket
            clean_idx_rec = torch.where(torch.sum((data['receptor'].pos - fake_pocket_receptor_pos)**2, dim=-1) < 400)[0]

            data['ligand'].pos = data['ligand'].pos[clean_idx_lig]
            data['ligand'].x = data['ligand'].x[clean_idx_lig]
            data['receptor'].pos = data['receptor'].pos[clean_idx_rec]
            data['receptor'].x = data['receptor'].x[clean_idx_rec]
            '''
        
        else:
            fake_pocket_receptor_pos, fake_pocket_ligand_pos = linker[0], linker[2]
            fake_pocket_receptor_norm, fake_pocket_ligand_norm = linker[1] - linker[0] / torch.norm(linker[1] - linker[0]), linker[3] - linker[2] / torch.norm(linker[3] - linker[2])

        PP_key_points = torch.stack([fake_pocket_receptor_pos, fake_pocket_receptor_pos + fake_pocket_receptor_norm / (torch.norm(fake_pocket_receptor_norm) + 1e-6),
                                        fake_pocket_ligand_pos, fake_pocket_ligand_pos + fake_pocket_ligand_norm / (torch.norm(fake_pocket_ligand_norm) + 1e-6),])
        
        if linker is None:
            PP_key_points_norm = norm_dist_4_points(PP_key_points.float())

            #search for the closest linker according to the cross distance
            linker_idx = torch.argmin(torch.sum(torch.sum((self.link_key_points[:, 2:] - PP_key_points_norm[2:].unsqueeze(0))**2, dim=-1), dim=-1))
            linker_key = self.idx_to_key[linker_idx]
            linker_subkey = self.idx_to_subkey[linker_idx]

            data['linker_key'] = linker_key
            data['linker_subkey'] = linker_subkey
            new_mol = copy.deepcopy(self.linker_mol[linker_key])
            new_mol.RemoveAllConformers()
            new_mol.AddConformer(self.linker_mol[linker_key].GetConformer(linker_subkey))
            #TODO fix the bug for get positions here
            #This bug only happens on the rbgquanta2 server, rosetta server is fine #still don't know why here is come with a segmenation fault
            mol_conf = torch.tensor(np.array(new_mol.GetConformer().GetPositions()), device=data['ligand'].pos.device).float()
            #kabsch fit back to the original position
            linker_mol_key_points = torch.tensor(np.array(new_mol.GetConformer().GetPositions()), 
                                                 device=data['ligand'].pos.device).float()[[self.linker_dict[linker_key]['anchor_idx']]]
            linker_mol_key_points[1] = linker_mol_key_points[0] + linker_mol_key_points[1] - linker_mol_key_points[0] / torch.norm(linker_mol_key_points[1] - linker_mol_key_points[0])
            linker_mol_key_points[3] = linker_mol_key_points[2] + linker_mol_key_points[3] - linker_mol_key_points[2] / torch.norm(linker_mol_key_points[3] - linker_mol_key_points[2])
            linker_mol_key_points = kabsch_fit(linker_mol_key_points, PP_key_points,
                                            torch.cat([linker_mol_key_points, mol_conf], dim=0))
            mol_conf = linker_mol_key_points[4:]
            
            linker_mol_key_points = linker_mol_key_points[:4]
            #set conf to mol
            for i in range(mol_conf.shape[0]):
                new_mol.GetConformer().SetAtomPosition(i, Point3D(float(mol_conf[i, 0]), float(mol_conf[i, 1]), float(mol_conf[i, 2])))
            data.linker = linker_mol_key_points
            
            data.linker_mol = new_mol
            data.linker_cache = self.linker_dict[linker_key]
            
            fake_pocket_receptor_pos = linker_mol_key_points[0]
            fake_pocket_receptor_norm = linker_mol_key_points[1] - linker_mol_key_points[0] / torch.norm(linker_mol_key_points[1] - linker_mol_key_points[0])
            fake_pocket_ligand_pos = linker_mol_key_points[2]
            fake_pocket_ligand_norm = linker_mol_key_points[3] - linker_mol_key_points[2] / torch.norm(linker_mol_key_points[3] - linker_mol_key_points[2])
            
        
        else:
            data['linker_key'] = "linker"
        data["ligand"].fake_pocket_pos = fake_pocket_ligand_pos.unsqueeze(0).float()
        data["ligand"].fake_pocket_norm = fake_pocket_ligand_norm.unsqueeze(0).float()
        data["ligand"].org_fake_pocket_pos = fake_pocket_ligand_pos.unsqueeze(0).float()
        data["ligand"].org_fake_pocket_norm = fake_pocket_ligand_norm.unsqueeze(0).float()
        data["receptor"].fake_pocket_pos = fake_pocket_receptor_pos.unsqueeze(0).float()
        data["receptor"].fake_pocket_norm = fake_pocket_receptor_norm.unsqueeze(0).float()
        data.linker_pos =  PP_key_points[[2, 3]].float()
        data.ref_linker_pos = PP_key_points[[0, 1]].float()
        data.org_linker_pos = copy.deepcopy(data.linker_pos)

    
    def search_linker_conf(self, data, ref_backbone, ref_surface, ref_linker):

        modified_fake_pocket_ligand_pos = data['ligand'].fake_pocket_pos
        modified_fake_pocket_ligand_norm = data['ligand'].fake_pocket_norm
        modified_fake_pocket_receptor_pos = data['receptor'].fake_pocket_pos
        modified_fake_pocket_receptor_norm = data['receptor'].fake_pocket_norm

        PP_key_points = torch.stack([modified_fake_pocket_receptor_pos, 
                                                            modified_fake_pocket_receptor_pos + modified_fake_pocket_receptor_norm / (torch.norm(modified_fake_pocket_receptor_norm) + 1e-6),
                                                            modified_fake_pocket_ligand_pos,
                                                            modified_fake_pocket_ligand_pos + modified_fake_pocket_ligand_norm / (torch.norm(modified_fake_pocket_ligand_norm) + 1e-6),])

        if len(PP_key_points.shape) == 3:
            PP_key_points = PP_key_points.squeeze(1)
        PP_key_points_norm = norm_dist_4_points(PP_key_points)

        if data.linker_key != "linker":
            linker_key_points_search = self.linker_dict[data['linker_key']]['key_points'].to(data['ligand'].pos.device)
        else:
            linker_key_points_search = data.linker_cache['key_points']

        linker_subidx = torch.argmin(torch.sum(torch.sum((linker_key_points_search[:, 2:] - PP_key_points_norm[2:].unsqueeze(0))**2, dim=-1), dim=-1))
        
        linker_key_points = linker_key_points_search[linker_subidx].to(data['ligand'].pos.device).float()
        linker_key_points = kabsch_fit(linker_key_points[:2], data.ref_linker_pos, linker_key_points)

        

        noised_R, noised_T = transform_norm_vector(ref_linker, linker_key_points[2:].float())
        if data.linker_key != "linker":
            data.noised_subkey = linker_subidx

        noised_translation = noised_T.to(data.linker_pos.device)
        #return noised_translation.squeeze(-1), noised_rotation, PP_key_points

        #translate the R, t to different ref
        noised_tr_surface = change_R_T_ref(noised_translation, noised_R, ref_linker[[0]], ref_surface)
        noised_tr_backbone = change_R_T_ref(noised_translation, noised_R, ref_linker[[0]], ref_backbone)
        noised_translation = change_R_T_ref(noised_translation, noised_R, ref_linker[[0]], ref_linker)
        

        output =  {
            'noised_tr_linker': noised_translation.squeeze(-1).to(data.linker_pos.device).float(),
            'noised_rot': matrix_to_axis_angle(noised_R).to(data.linker_pos.device).float(),
            'noised_tr_backbone': noised_tr_backbone.squeeze(-1).to(data.linker_pos.device).float(),
            'noised_tr_surface': noised_tr_surface.squeeze(-1).to(data.linker_pos.device).float(),
        }
        return output


    def sample_linker(self, data, sampling=False, random_idx=None, sigma=1):
        #fit the searched linker to the fake pocket
        linker_key = data['linker_key']
        fake_pocket_receptor_pos = data['receptor'].fake_pocket_pos
        fake_pocket_receptor_norm = data['receptor'].fake_pocket_norm
        if not sampling:
            if data.__contains__('linker_subkey'):
                linker_subkey = data['linker_subkey']
            else:
                linker_subkey = -1
            fake_pocket_ligand_pos = data['ligand'].fake_pocket_pos
            fake_pocket_ligand_norm = data['ligand'].fake_pocket_norm
            device = fake_pocket_ligand_pos.device
        else:
            device = data['ligand'].pos.device
            fake_pocket_receptor_pos = fake_pocket_receptor_pos.squeeze(0)
            fake_pocket_receptor_norm = fake_pocket_receptor_norm.squeeze(0)
            fake_pocket_ligand_pos = data['ligand'].fake_pocket_pos.squeeze(0)
            fake_pocket_ligand_norm = data['ligand'].fake_pocket_norm.squeeze(0)
            
        
        
        PP_key_points = torch.stack([fake_pocket_receptor_pos, fake_pocket_receptor_pos + fake_pocket_receptor_norm / (torch.norm(fake_pocket_receptor_norm) + 1e-6),
                                     fake_pocket_ligand_pos, fake_pocket_ligand_pos + fake_pocket_ligand_norm/ (torch.norm(fake_pocket_ligand_norm) + 1e-6),]).squeeze(1)

        #random sample another conformation
        if data.linker_key != "linker":
            sample_len = len(self.linker_dict[linker_key]['key_points'])
        else:
            sample_len = len(data.linker_cache['key_points'])
        if random_idx is None:
            if not sampling:
                random_idx = random.choice([i for i in range(sample_len) if i != linker_subkey])
            else:
                random_idx = random.choice([i for i in range(sample_len)])
        
        if data.linker_key != "linker":
            noised_linker_mol = self.linker_dict[linker_key]['mol']
        else:
            noised_linker_mol = data.linker_cache['mol']
        
        #keep only the random_idx conformer and remove the others
        new_mol = copy.deepcopy(noised_linker_mol)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(noised_linker_mol.GetConformer(random_idx))
        noised_linker_mol = new_mol
        #align to PP_key_points
        noised_coords = torch.tensor(np.array(noised_linker_mol.GetConformer().GetPositions()), device=device).float()
        #add random rotation to noised linker
        R_random = axis_angle_to_matrix(torch.rand(3) * np.pi * 2).to(noised_coords.device).float()
        noised_coords = noised_coords @ R_random
        noised_linker_key_points = noised_coords[[data.linker_cache['anchor_idx']]]
        noised_coords = kabsch_fit(noised_linker_key_points[:2], PP_key_points[:2], noised_coords)
        #assign the new conformer to the mol
        for i in range(noised_coords.shape[0]):
            noised_linker_mol.GetConformer().SetAtomPosition(i, Point3D(float(noised_coords[i, 0]), float(noised_coords[i, 1]), float(noised_coords[i, 2])))

        org_noised_linker_mol = copy.deepcopy(noised_linker_mol)
        #noised_linker_mol = linker_interpolate(data.linker_mol, noised_linker_mol, sigma, data.linker_cache['anchor_idx'][:2])
        noised_linker_key_points = torch.tensor(np.array(noised_linker_mol.GetConformer().GetPositions()), device=device).float()[[data.linker_cache['anchor_idx']]]
        
        linker_conform = torch.tensor(np.array(noised_linker_mol.GetConformer().GetPositions()), device=device).float()
        #aligned_noised_linker_key_points = kabsch_fit(noised_linker_key_points[:2], PP_key_points[:2], torch.cat([noised_linker_key_points, linker_conform], dim=0))
        #linker_conform = aligned_noised_linker_key_points[4:]
        #aligned_noised_linker_key_points = aligned_noised_linker_key_points[:4]
        
        #rot_angle = align_axis(aligned_noised_linker_key_points[2:].float(), PP_key_points[2:].float(), PP_key_points[:2])
        #rot_mat = rot_axis_to_matrix(PP_key_points[1] - PP_key_points[0], rot_angle).to(aligned_noised_linker_key_points.device).float()
        #aligned_noised_linker_key_points[2: ] = (aligned_noised_linker_key_points[2: ] - PP_key_points[:1]) @ (rot_mat) + PP_key_points[:1]
        #linker_conform = (linker_conform - PP_key_points[:1]) @ (rot_mat) + PP_key_points[:1]
        noised_R, noised_t = transform_norm_vector(PP_key_points[2:].float(), noised_linker_key_points[2:].float())
        if not sampling:
            data.noised_subkey = random_idx
        #translate the R, t to different ref
        noised_t_surface = change_R_T_ref(noised_t, noised_R, PP_key_points[[2]].float(), torch.cat([data['ligand'].pos, data['ligand'].fake_pocket_pos], dim=0))
        noised_t_backbone = change_R_T_ref(noised_t, noised_R, PP_key_points[[2]].float(), data['ligand'].c_alpha_coords)
        noised_t_c = change_R_T_ref(noised_t, noised_R, PP_key_points[[2]].float(), PP_key_points[[2, 3]].float())

        #set conf to mol
        linker_conf_array = linker_conform.cpu().numpy()
        for i in range(linker_conform.shape[0]):
            noised_linker_mol.GetConformer().SetAtomPosition(i, Point3D(float(linker_conf_array[i, 0]), float(linker_conf_array[i, 1]), float(linker_conf_array[i, 2])))
        return {
            'noised_tr_linker': noised_t_c.squeeze(-1).to(device).float(),
            'noised_rot': matrix_to_axis_angle(noised_R).to(device).float(),
            'noised_tr_backbone': noised_t_backbone.squeeze(-1).to(device).float(),
            'noised_tr_surface': noised_t_surface.squeeze(-1).to(device).float(),
            'noised_linker_mol': noised_linker_mol,
            'noised_linker_conform': linker_conform,
            'org_noised_linker_mol': org_noised_linker_mol,
            'noised_tr_linker_test': noised_t_c.squeeze(-1).to(device).float(),
        }
    

class ListDataset(Dataset):
    def __init__(self, dataset, multi_time=1):
        super().__init__()
        self.data_list = dataset
        self.multi_time = multi_time
        self.cache_num = 0
        self.cache = None

    def len(self) -> int:
        return len(self.data_list) * self.multi_time

    def get(self, idx: int):            
        if self.cache_num % self.multi_time == 0:
            self.cache = self.data_list.get(idx % (len(self.data_list)// 8))
            self.cache_num += 1
            return self.cache
        else:
            self.cache_num += 1
            return self.cache
        
        
    

class DIPS(Dataset):
    def __init__(self, root, esm_lm_path, transform=None, cache_path='data/cache', split='train', limit_complexes=0,
                 receptor_radius=30, num_workers=64, c_alpha_max_neighbors=None,  max_lig_size=0, server_cache=None,
                msms_bin='msms', data_source='dips'):

        super(DIPS, self).__init__()
        self.transform = transform
        self.esm_lm_path = esm_lm_path
        self.max_lig_size = max_lig_size
        self.split = split
        self.limit_complexes = limit_complexes
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.msms_bin = msms_bin
        self.no_torsion = True
        self.ds = data_source
        self.dips_dir = root
        
        if self.ds == 'dips':
            self.full_cache_path = os.path.join(cache_path, f'DIPS_limit{self.limit_complexes}'
                                                        f'_INDEX{os.path.splitext(os.path.basename(self.split))[0]}'
                                                        f'_maxLigSize{self.max_lig_size}'
                                                        f'_recRad{self.receptor_radius}_recMax{self.c_alpha_max_neighbors}')
            if server_cache is not None:
                self.full_cache_path_server = os.path.join(server_cache, f'DIPS_limit{self.limit_complexes}'
                                                        f'_INDEX{os.path.splitext(os.path.basename(self.split))[0]}'
                                                        f'_maxLigSize{self.max_lig_size}'
                                                        f'_recRad{self.receptor_radius}_recMax{self.c_alpha_max_neighbors}')
        elif self.ds == 'protac':
            self.full_cache_path = os.path.join(cache_path, f'PROTAC___')
            if server_cache is not None:
                self.full_cache_path_server = os.path.join(server_cache, f'PROTAC___')
        
        elif self.ds == 'e3':
            self.full_cache_path = os.path.join(cache_path, f'E3___')
            if server_cache is not None:
                self.full_cache_path_server = os.path.join(server_cache, f'E3___')
                
        elif self.ds == 'confidence':
            self.full_cache_path = os.path.join(cache_path, f'CONFIDENCE___{self.split}')
            if server_cache is not None:
                self.full_cache_path_server = os.path.join(server_cache, f'CONFIDENCE___{self.split}')
            
        else:
            raise NotImplementedError
        

        if not os.path.exists(os.path.join(self.full_cache_path, 'data.lmdb')):
            os.makedirs(self.full_cache_path, exist_ok=True)
            self.preprocessing()
        
        self.env_path = os.path.join(self.full_cache_path, 'data.lmdb')
        if server_cache is not None:
            if not os.path.exists(os.path.join(self.full_cache_path_server, 'data.lmdb')):
                os.makedirs(self.full_cache_path_server, exist_ok=True)
                os.system(f"cp -r {self.full_cache_path}/* {self.full_cache_path_server}/")
                self.env_path = os.path.join(self.full_cache_path_server, 'data.lmdb')
            else:
                self.env_path = os.path.join(self.full_cache_path_server, 'data.lmdb')

        print('loading data from memory: ', self.env_path)
        self.graph_cache_env = lmdb.open(self.env_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=32, map_size=int(10e9))


    def len(self):
        #read the len from the lmdb
        with self.graph_cache_env.begin(write=False) as txn:
            self.data_len = pickle.loads(txn.get('len'.encode()))
            keys = list(txn.cursor().iternext(values=False))
            self.keys = []
            for k in keys:
                try:
                    self.keys.append(int(k))
                except:
                    continue
        if self.split == 'train' and self.ds == 'protac':
            return self.data_len * 100
        return self.data_len
    
    def get(self, idx):
        #if self.ds == 'protac':
            #idx = 0
            #idx = idx % 14
        idx = self.keys[idx]
        with self.graph_cache_env.begin(write=False) as txn:
            complex_graph = pickle.loads(txn.get(str(idx).encode()))
            # normalize the norm vector
            complex_graph['ligand'].x[:, :, -4:-1] = complex_graph['ligand'].x[:, :, -4:-1] / torch.norm(complex_graph['ligand'].x[:, :, -4:-1], dim=-1, keepdim=True)
            complex_graph['receptor'].x[:, :, -4:-1] = complex_graph['receptor'].x[:, :, -4:-1] / torch.norm(complex_graph['receptor'].x[:, :, -4:-1], dim=-1, keepdim=True)
        
        if self.ds == 'protac':
            complex_graph.linker_cache['anchor_idx'] = torch.stack(complex_graph.linker_cache['anchor_idx'])
        if self.transform is not None:
            return self.transform(complex_graph)
        else:
            set_time(complex_graph, 1., 1., 1)
            return complex_graph
    
    def __getitem__(self, idx):
        return self.get(idx)



    def preprocessing(self):
        print(f'Preprocessing {self.split}---{self.root} and saving to {self.full_cache_path}')
        print(f'running with {self.num_workers} workers')

        if self.ds == 'dips':#files description for DIPS dataset
            #get all the paired ligand/receptor files
            if self.dips_dir.endswith('/'):
                self.dips_dir = self.dips_dir[:-1]
            files_dills_all = glob.glob(self.dips_dir+ '/*/*.dill', recursive=True)
            pdb_dir = os.path.join(self.full_cache_path, 'pdbs')
            
            files_all = [[f.split('/')[-1].replace('.dill', '_0.pdb'), f.split('/')[-1].replace('.dill', '_1.pdb')] for f in files_dills_all]
            files_all = [[os.path.join(pdb_dir, f[0]), os.path.join(pdb_dir, f[1])] for f in files_all]
            if not os.path.exists(pdb_dir):
                os.makedirs(pdb_dir) 
                with mp.Pool(self.num_workers) as p:
                    with tqdm(total=len(files_all)) as pbar:
                        for i, _ in tqdm(enumerate(p.imap_unordered(dill_to_pdb, zip(files_dills_all, files_all)))):
                            pbar.update()
            process_args = [(p[0], p[1], os.path.join(self.full_cache_path, "pieces"), i, None) for i, p in enumerate(files_all)]
            

        elif self.ds == 'protac':#files description for PROTAC dataset
            files_all = [f for f in os.listdir(self.dips_dir) if f.endswith('.pdb')]
            #naming = set([p[:4] for p in files_all])
            naming = set([p.split('_')[0] for p in files_all])
            naming = [os.path.join(self.dips_dir, n) for n in naming]
            files_paired = [[n + '_poi.pdb', n + '_e3.pdb'] for n in naming]
            ligand_files_paired = [{'e3': n + '_e3_lig.sdf', 
                                    'poi': n + '_poi_lig.sdf', 
                                    'linker': n + '_linker.sdf'} for n in naming]

            process_args = [(p[0], p[1], os.path.join(self.full_cache_path, "pieces"), i, ligand_files_paired[i]) for i, p in enumerate(files_paired)]
            print(f"processing {len(process_args)} PROTAC complexes")
            
        
        elif self.ds == 'e3':#files description for E3 dataset
            self.split_csv = pd.read_csv(self.split)
            files_paired = [[os.path.join(self.dips_dir, f'{self.split_csv.iloc[i, 0]}'),
                            os.path.join(self.dips_dir, f'{self.split_csv.iloc[i, 1]}')] for i in range(len(self.split_csv))]
            process_args = [(p[0], p[1], os.path.join(self.full_cache_path, "pieces"), i, None) for i, p in enumerate(files_paired)]
            print(f"processing {len(process_args)} E3 complexes")
        
        elif self.ds == 'confidence':
            files_all = [f for f in os.listdir(self.dips_dir) if f.endswith('.pdb')]
            #hard write here: the pdb split
            naming = list(set([p.split('_')[0] for p in files_all]))
            if self.split == 'test' or self.split == 'val':
                naming = [n for n in naming if n[:4] in ['6W8I', '6W7O']]
            else:
                naming = [n for n in naming if n[:4] not in ['6W8I', '6W7O']]
            files_paired = [[os.path.join(self.dips_dir, f) for f in os.listdir(self.dips_dir) if f.startswith(n) and 'e3' in f and f.endswith('.pdb')][0] for n in naming]
            files_paired = [[f.replace('e3', 'poi'), f] for f in files_paired]
            process_args = [(p[0], p[1], os.path.join(self.full_cache_path, "pieces"), i, None) for i, p in enumerate(files_paired)]
            
        
        os.makedirs(os.path.join(self.full_cache_path, 'pieces'), exist_ok=True)
        if self.num_workers > 1:
            with mp.Pool(self.num_workers) as p:
                with tqdm(total=len(process_args)) as pbar:
                    for i, _ in tqdm(enumerate(p.imap_unordered(self.get_pp, process_args))):
                        pbar.update()
    
        else:
            for i, _ in tqdm(enumerate(map(self.get_pp, process_args))):
                pass
        #save all to an lmdb file
        print('saving all cached data to lmdb')
        lmdb_path = os.path.join(self.full_cache_path, 'data.lmdb')
        env = lmdb.open(lmdb_path, subdir=False, readonly=False, lock=False, readahead=False, meminit=False, map_size=1099511627776)
        num = 0
        with env.begin(write=True) as txn:
            for d in tqdm(os.listdir(os.path.join(self.full_cache_path, 'pieces'))):
                with open(os.path.join(self.full_cache_path, 'pieces', d), 'rb') as f:
                    data = pickle.load(f)
                    txn.put(str(num).encode('ascii'), pickle.dumps(data))
                    num += 1
            txn.put('len'.encode('ascii'), pickle.dumps(num))

        
    def get_pp(self, par):
        try:
            if self.ds != 'protac':
                pdb_1, pdb_2, output_path, output_num, _ = par
                graph_path = os.path.join(output_path, f'heterograph_{output_num}.pkl')
                if os.path.isfile(graph_path):
                    print("exits!")
                    return
            else:
                pdb_1, pdb_2, output_path, output_num, linker = par
                graph_path = os.path.join(output_path, f'heterograph_{output_num}.pkl')
                if os.path.isfile(graph_path):
                    print("exits!")
                    return
                linker_mol, e3_lig_mol, poi_lig_mol = read_molecule(linker["linker"]), read_molecule(linker["e3"]), read_molecule(linker["poi"])
                linker_conf = torch.tensor(np.array(linker_mol.GetConformer().GetPositions()))
                e3_lig_conf = torch.tensor(np.array(e3_lig_mol.GetConformer().GetPositions()))
                poi_lig_conf = torch.tensor(np.array(poi_lig_mol.GetConformer().GetPositions()))

                anchor_results = ret_anchor_idx(linker_mol, poi_lig_mol, e3_lig_mol)
                poi_anchor_i, poi_anchor_j, poi_anchor = anchor_results['poi']
                e3_anchor_i, e3_anchor_j, e3_anchor = anchor_results['e3']
                
                bonds_direction = torch.cat([poi_anchor, e3_anchor], dim=0)
                bonds_direction[1] = bonds_direction[0] + (bonds_direction[1] - bonds_direction[0]) / torch.norm(bonds_direction[1] - bonds_direction[0])
                bonds_direction[3] = bonds_direction[2] + (bonds_direction[3] - bonds_direction[2]) / torch.norm(bonds_direction[3] - bonds_direction[2])
                
                linker_conf_cache = generate_linker_length(linker_mol, anchor_idx_set=[poi_anchor_i[0], poi_anchor_i[1], e3_anchor_i[0], e3_anchor_i[1]])
                linker_conf_cache = {'size': torch.from_numpy(linker_conf_cache[0]),
                                    'mol': linker_conf_cache[2], 
                                    'key_points': linker_conf_cache[3],
                                    'anchor_idx': linker_conf_cache[4]}

            
            if os.path.exists(graph_path):
                return
            
            if not os.path.exists(pdb_1) or not os.path.exists(pdb_2):
                print(f'file {pdb_1} or {pdb_2} does not exist')
                return [], []

            name = pdb_1.split('/')[-1].split('_')[0]
            rec_model_1 = parse_pdb_from_path(pdb_1)
            rec_model_2 = parse_pdb_from_path(pdb_2)
        
        except Exception as e:
            print(f'Skipping {name} because of the error:')
            print(e)
            return [], []
        
        complex_graph = HeteroData()
        complex_graph['name'] = name
        rec_1, rec_coords_1, atom_to_res_1, c_alpha_coords_1, n_coords_1, c_coords_1, _ = extract_receptor_structure(copy.deepcopy(rec_model_1))
        rec_2, rec_coords_2, atom_to_res_2, c_alpha_coords_2, n_coords_2, c_coords_2, _ = extract_receptor_structure(copy.deepcopy(rec_model_2))
        
        if self.ds == 'protac':
            complex_graph.linker = bonds_direction
            complex_graph.linker_mol = linker_mol
            complex_graph.linker_cache = linker_conf_cache
        
        else:
            protein_graph = get_protein_surface_graph(pdb_1, self.msms_bin, rec_1, rec_coords_1, atom_to_res_1, copy.deepcopy(complex_graph), lig_coord=rec_coords_2)
            if protein_graph is None:
                return [], []
            complex_graph['receptor'].x = protein_graph['protein_surface'].x
            complex_graph['receptor'].pos = protein_graph['protein_surface'].pos

            complex_graph['receptor'].curv = curve_feat(complex_graph['receptor'].pos, complex_graph['receptor'].x[:, 0, -4:-1])
            complex_graph['receptor'].c_alpha_coords = c_alpha_coords_1
        
            protein_graph = get_protein_surface_graph(pdb_2, self.msms_bin, rec_2, rec_coords_2, atom_to_res_2, copy.deepcopy(complex_graph), lig_coord=rec_coords_1)
            if protein_graph is None:
                return [], []
            complex_graph['ligand'].x = protein_graph['protein_surface'].x
            complex_graph['ligand'].pos = protein_graph['protein_surface'].pos

            complex_graph['ligand'].curv = curve_feat(complex_graph['ligand'].pos, complex_graph['ligand'].x[:, 0, -4:-1])
            complex_graph['ligand'].c_alpha_coords = c_alpha_coords_2

        if self.ds == 'confidence':
            rmsd_int = float(pdb_1.split('/')[-1].split('_')[1])
            complex_graph['rmsd_int'] = torch.tensor(rmsd_int)
        
        self.transform.search_for_linker(complex_graph)

        
        protein_graph = get_protein_surface_graph(pdb_1, self.msms_bin, rec_1, rec_coords_1, atom_to_res_1, copy.deepcopy(complex_graph), lig_coord=np.array(complex_graph['receptor'].fake_pocket_pos))
        if protein_graph is None:
            return [], []
        complex_graph['receptor'].x = protein_graph['protein_surface'].x.to(torch.float32)
        complex_graph['receptor'].pos = protein_graph['protein_surface'].pos.to(torch.float32)

        complex_graph['receptor'].curv = curve_feat(complex_graph['receptor'].pos, complex_graph['receptor'].x[:, 0, -4:-1])
        complex_graph['receptor'].c_alpha_coords = c_alpha_coords_1.astype(np.float32)
    
        protein_graph = get_protein_surface_graph(pdb_2, self.msms_bin, rec_2, rec_coords_2, atom_to_res_2, copy.deepcopy(complex_graph), lig_coord=np.array(complex_graph['ligand'].fake_pocket_pos))
        if protein_graph is None:
            return [], []
        complex_graph['ligand'].x = protein_graph['protein_surface'].x.to(torch.float32)
        complex_graph['ligand'].pos = protein_graph['protein_surface'].pos.to(torch.float32)

        complex_graph['ligand'].curv = curve_feat(complex_graph['ligand'].pos, complex_graph['ligand'].x[:, 0, -4:-1])
        complex_graph['ligand'].c_alpha_coords = c_alpha_coords_2.astype(np.float32)
            
        graph_path = os.path.join(output_path, f'heterograph_{output_num}.pkl')
        with open(graph_path, 'wb') as f:
            pickle.dump(complex_graph, f)
        #except Exception as e:
        #    print(e, pdb_1, pdb_2)
    

def curve_feat(x, norm):#x: [N, 3]
    #knn graph
    edge_index = knn_graph(x, k=6, batch=None, loop=False, flow= 'source_to_target')
    edge_vec = x[edge_index[1]] - x[edge_index[0]]
    edge_norm = norm[edge_index[1]] - norm[edge_index[0]]

    edge_length = torch.norm(edge_vec, dim=1, keepdim=True)
    edge_norm = torch.norm(edge_norm, dim=1, keepdim=True)

    #Gaussian average
    edge_length = torch.exp(-edge_length**2/1.414**2)
    #average pooling from neighbors
    curvature = torch_scatter.scatter(edge_length * edge_norm, edge_index[1], dim=0, reduce='sum') / (torch_scatter.scatter(edge_length, edge_index[1], dim=0, reduce='sum') + 1e-6)
    return curvature

def kabsch_fit(points_A, points_B, new_points, random=True):

    if points_A.shape[0] > 2 or random:
        R, t = rigid_transform_Kabsch_3D_torch(points_A.T.float(), points_B.T.float())
        new_points = (new_points - torch.mean(points_A, dim=0, keepdim=True))@R.T + t.T + torch.mean(points_A, dim=0, keepdim=True)
        return new_points

    else:
        R, t = transform_norm_vector(points_A, points_B)
        new_points = (new_points - points_A[0].unsqueeze(0))@R.T + points_A[0].unsqueeze(0) + t.T
        
        return new_points


def get_cross_distance(rec_pos, rec_norm, lig_pos, lig_norm):
    PP_key_points = torch.stack([rec_pos, rec_pos + rec_norm / torch.norm(rec_norm), 
                                lig_pos, lig_pos + lig_norm / torch.norm(lig_norm)])
    cross_distances = []
    for i in [0, 1]:
        for j in [2, 3]:
            cross_distances.append(torch.norm(PP_key_points[i] - PP_key_points[j]))
    return torch.stack(cross_distances, dim=0)



def sample_pocket(pos_lig, pos_rec, curve_lig, curve_rec):

    filter_PPI_dist = 10
    sample_point_far_PPI = 500
    high_curvature_sample = 200
    pocket_num = 4

    cross_distance = torch.cdist(pos_lig, pos_rec)
    #find the PPI dist < 10
    PPI_ligand_idx = torch.where(torch.min(cross_distance, dim=1)[0] < filter_PPI_dist)[0]
    PPI_receptor_idx = torch.where(torch.min(cross_distance, dim=0)[0] < filter_PPI_dist)[0]

    if PPI_ligand_idx.shape[0] == 0 or PPI_receptor_idx.shape[0] == 0:#if no PPI, sample the closest 500 points
        PPI_ligand_idx = torch.argsort(torch.min(cross_distance, dim=1)[0])[:sample_point_far_PPI]
        PPI_receptor_idx = torch.argsort(torch.min(cross_distance, dim=0)[0])[:sample_point_far_PPI]

    #sample 100 points with high curvature

    PPI_ligand_curve, PPI_receptor_curve = curve_lig[PPI_ligand_idx], curve_rec[PPI_receptor_idx]
    lig_c, rec_c = PPI_ligand_curve.shape[0] < high_curvature_sample, PPI_receptor_curve.shape[0] < high_curvature_sample
    try:#if there is nan curvature points, sample randomly
        pocket_ligand_idx = torch.multinomial(PPI_ligand_curve.squeeze(-1), num_samples=high_curvature_sample, replacement=lig_c)
    except:
        pocket_ligand_idx = PPI_ligand_curve.squeeze(-1).argsort()[:high_curvature_sample]
    try:
        pocket_receptor_idx = torch.multinomial(PPI_receptor_curve.squeeze(-1), num_samples=high_curvature_sample, replacement=rec_c)
    except:
        pocket_receptor_idx = PPI_receptor_curve.squeeze(-1).argsort()[:high_curvature_sample]

    pocket_ligand_patch_idx = kmeans(pos_lig[PPI_ligand_idx][pocket_ligand_idx], pocket_num, device=pos_lig.device, tqdm_flag=False, tol=0.1)[0]
    pocket_receptor_patch_idx = kmeans(pos_rec[PPI_receptor_idx][pocket_receptor_idx], pocket_num, device=pos_rec.device, tqdm_flag=False, tol=0.1)[0]
    #FIND alternative solution, this is incredible slow on CPU
    
    #get mean patch pos
    patch_ligand_pos, patch_receptor_pos = [], []
    for i in range(pocket_num):
        patch_ligand_pos.append(torch.mean(pos_lig[PPI_ligand_idx][pocket_ligand_idx][torch.where(pocket_ligand_patch_idx == i)[0]], dim=0, keepdim=True))
        patch_receptor_pos.append(torch.mean(pos_rec[PPI_receptor_idx][pocket_receptor_idx][torch.where(pocket_receptor_patch_idx == i)[0]], dim=0, keepdim=True))
    
    #cross distance between patches
    patched_cross_distance = torch.cdist(torch.cat(patch_ligand_pos, dim=0), torch.cat(patch_receptor_pos, dim=0))
    #random select 1 patch pairs with distance > 6
    pocket_ligand_patch_id, pocket_receptor_patch_id = torch.where(patched_cross_distance < 15)
    if pocket_ligand_patch_id.shape[0] == 0:
        pocket_ligand_patch_id, pocket_receptor_patch_id = torch.argsort(patched_cross_distance.view(-1))[0] // pocket_num, \
                                                        torch.argsort(patched_cross_distance.view(-1))[0] % pocket_num
    else:
        pocket_ligand_patch_id, pocket_receptor_patch_id = pocket_ligand_patch_id[0], pocket_receptor_patch_id[0]
    
    pocket_ligand_patch_idx = torch.where(pocket_ligand_patch_idx == pocket_ligand_patch_id)[0]
    pocket_ligand_idx = PPI_ligand_idx[pocket_ligand_idx][pocket_ligand_patch_idx]

    pocket_receptor_patch_idx = torch.where(pocket_receptor_patch_idx == pocket_receptor_patch_id)[0]
    pocket_receptor_idx = PPI_receptor_idx[pocket_receptor_idx][pocket_receptor_patch_idx]
    
    return pocket_ligand_idx, pocket_receptor_idx

def reverse_angles(angle):
    rot_mat = axis_angle_to_matrix(angle)
    return matrix_to_axis_angle(rot_mat.T)

def align_axis(center_A, center_B, axis):
    if len(center_A.shape) == 1:
        center_A = center_A.unsqueeze(0)
    if len(center_B.shape) == 1:
        center_B = center_B.unsqueeze(0)
    point_A = torch.cat([axis, center_A], dim=0)
    point_B = torch.cat([axis, center_B], dim=0)

    norm_axis = torch.tensor([[0., 0., 0.], [0., 0., 1.]]).to(point_A.device)
    point_A_norm = kabsch_fit(point_A[:2], norm_axis, point_A)
    point_B_norm = kabsch_fit(point_B[:2], norm_axis, point_B)


    angle_A = torch.atan(point_A_norm[2, 1] / (point_A_norm[2, 0] + 1e-5))
    angle_B = torch.atan(point_B_norm[2, 1] / (point_B_norm[2, 0] + 1e-5))

    return angle_B - angle_A

def dill_to_pdb(args):
    dill_file, pdb_files = args
    #dill_file: input dill file
    #pdb_files: output pdb files [receptor, ligand]
    with open(dill_file, 'rb') as f:
        data = dill.load(f)
    pdb1, pdb2 = data.df0, data.df1
    for i, p in enumerate([pdb1, pdb2]):
        with open(pdb_files[i], 'w') as f:
            for _, row in p.iterrows():
                try:
                    res_num = int(row['residue'])
                except:
                    row['residue'] = row['residue'][:-1]
                line = line = f"ATOM  {row['aid']:>5} {row['atom_name']:<4} {row['resname']:<3} {row['chain']}{row['residue']:>4}    {row['x']:>8.3f}{row['y']:>8.3f}{row['z']:>8.3f}  1.00  0.00          {row['element']:>2}"
                f.write(line + '\n')
    
    
    