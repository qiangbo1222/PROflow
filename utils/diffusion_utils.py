import math
import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import tqdm
from scipy.stats import beta
from functools import partial

from rdkit.Geometry import Point3D
import rdkit.Chem as Chem

from utils.geometry import axis_angle_to_matrix, rigid_transform_Kabsch_3D_torch
from utils.torsion import modify_conformer_torsion_angles
from utils.geometry import matrix_to_axis_angle, rot_axis_to_matrix
from dataset.conformer_matching import optimize_rotatable_bonds, get_torsion_angles


def t_to_sigma(t, args):
    """old exponential decay"""
    #tr_sigma = args.tr_sigma_min ** (1-t) * args.tr_sigma_max ** t
    #rot_sigma = args.rot_sigma_min ** (1-t) * args.rot_sigma_max ** t
    #tor_sigma = args.tor_sigma_min ** (1-t) * args.tor_sigma_max ** t
    #return {'tr': tr_sigma, 'rot': rot_sigma, 'tor': tor_sigma}

    """simple Linear decay"""
    sigma = args.sigma_min + (args.sigma_max - args.sigma_min) * t
    return sigma

def modify_conformer(data, rot_update, tr_update_surface, tr_update_backbone, tr_update_linker=None, has_norm=True):
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    #rigid_new_pos = (data['ligand'].pos - lig_center) @ rot_mat.T.to(lig_center.device) + tr_update.to(lig_center.device) + lig_center
    pos_pact = torch.mean(torch.cat([data['ligand'].pos, data['ligand'].fake_pocket_pos]), dim=0).detach()
    data['ligand'].pos = (data['ligand'].pos - pos_pact) @ rot_mat.T \
                        + tr_update_surface + pos_pact
    data['ligand'].fake_pocket_pos = (data['ligand'].fake_pocket_pos - pos_pact) @ rot_mat.T \
                        + tr_update_surface + pos_pact
    
    #if np array convert to tensor
    if isinstance(data['ligand'].c_alpha_coords, np.ndarray):
        data['ligand'].c_alpha_coords = torch.from_numpy(data['ligand'].c_alpha_coords).to(data['ligand'].pos.device).float()
    if isinstance(data['receptor'].c_alpha_coords, np.ndarray):
        data['receptor'].c_alpha_coords = torch.from_numpy(data['receptor'].c_alpha_coords).to(data['receptor'].pos.device).float()

   
    if tr_update_backbone is not None:
        data['ligand'].c_alpha_coords = (data['ligand'].c_alpha_coords - torch.mean(data['ligand'].c_alpha_coords, dim=0, keepdim=True)) @ rot_mat.T \
                        + tr_update_backbone + torch.mean(data['ligand'].c_alpha_coords, dim=0, keepdim=True)
    

    if has_norm:
        data['ligand'].x = data['ligand'].x.float()
        #rewrite to avoid inplace operation(fail for autograd)
        #data['ligand'].x[:, :, -4:-1] = F.normalize(data['ligand'].x[:, :, -4:-1], dim=-1)
        data['ligand'].x = torch.cat([data['ligand'].x[:, :, :-7],
                                      data['ligand'].x[:, :, -7:-4] @ rot_mat.T,
                                      data['ligand'].x[:, :, -4:-1] @ rot_mat.T,
                                      data['ligand'].x[:, :, -1:]], dim=-1)
        data['ligand'].fake_pocket_norm = (data['ligand'].fake_pocket_norm / torch.norm(data['ligand'].fake_pocket_norm)) @ rot_mat.T
    
    if tr_update_linker is not None:

        linker_pos_mod = (data.linker_pos[[0, 1]] - torch.mean(data.linker_pos[[0, 1]], dim=0)) @ rot_mat.T\
                            + torch.mean(data.linker_pos[[0, 1]], dim=0) + tr_update_linker


        data.linker_pos = linker_pos_mod.float()
    
        #res_vec = data['ligand'].res_x[:, :, -3].float()
        #res_vec = (res_vec @ rot_mat.T.to(res_vec.device)).to(data['ligand'].pos.device)
        #data['ligand'].res_x[:, :, -3:] = res_vec

    #update axis for torsion
    data['ligand'].fake_pocket_pos = linker_pos_mod[0].float().unsqueeze(0)
    data['ligand'].fake_pocket_norm = (linker_pos_mod[1] - linker_pos_mod[0]).float().unsqueeze(0)
    data.linker = torch.cat([data.ref_linker_pos, data.linker_pos], dim=0).float()
    return data

def modify_conformer_torsion(data, tor_l_update, tor_r_update, has_norm=True):
    edge_1 = [data['ligand'].fake_pocket_pos, data['ligand'].fake_pocket_pos + data['ligand'].fake_pocket_norm / torch.norm(data['ligand'].fake_pocket_norm)]
    edge_2 = [data['receptor'].fake_pocket_pos, data['receptor'].fake_pocket_pos + data['receptor'].fake_pocket_norm / torch.norm(data['receptor'].fake_pocket_norm)]

    edge_1 = [edge_1[0].float(), edge_1[1].float()]
    edge_2 = [edge_2[0].float(), edge_2[1].float()]

    #update the torsion angles
    rot_vec_1 =  (edge_1[1] - edge_1[0]).squeeze(0)
    rot_vec_2 =  (edge_2[1] - edge_2[0]).squeeze(0)

    rot_mat_1 = rot_axis_to_matrix(rot_vec_1, tor_l_update)
    rot_mat_2 = rot_axis_to_matrix(rot_vec_2, tor_r_update)

    data['ligand'].pos = (data['ligand'].pos - edge_1[0]) @ rot_mat_1 + edge_1[0]
    data['ligand'].fake_pocket_pos = (data['ligand'].fake_pocket_pos - edge_1[0]) @ rot_mat_1 + edge_1[0]
    data['ligand'].c_alpha_coords = (data['ligand'].c_alpha_coords - edge_1[0]) @ rot_mat_1 + edge_1[0]


    if has_norm:
        data['ligand'].x = data['ligand'].x.float()
        data['ligand'].x[:, :, -4:-1] = data['ligand'].x[:, :, -4:-1] / (torch.norm(data['ligand'].x[:, :, -4:-1].float(), dim=-1, keepdim=True))
        data['ligand'].x[:, :, -4:-1] = (data['ligand'].x[:, :, -4:-1] @ rot_mat_1).to(data['ligand'].pos.device)

        data['ligand'].x[:, :, -7:-4] = (data['ligand'].x[:, :, -7:-4] @ rot_mat_1).to(data['ligand'].pos.device)
        data['ligand'].fake_pocket_norm = (data['ligand'].fake_pocket_norm / torch.norm(data['ligand'].fake_pocket_norm)) @ rot_mat_1

    
    #data['ligand'].pos = (data['ligand'].pos - edge_2[0]) @ rot_mat_2.T + edge_2[0]
    
    data['ligand'].pos = (data['ligand'].pos - edge_2[0]) @ rot_mat_2 + edge_2[0]
    data['ligand'].fake_pocket_pos = (data['ligand'].fake_pocket_pos - edge_2[0]) @ rot_mat_2 + edge_2[0]
    data['ligand'].c_alpha_coords = (data['ligand'].c_alpha_coords - edge_2[0]) @ rot_mat_2 + edge_2[0]
    data.linker_pos = (data.linker_pos - edge_2[0]) @ rot_mat_2 + edge_2[0]
    
    if has_norm:
        data['ligand'].x[:, :, -4:-1] = data['ligand'].x[:, :, -4:-1] / (torch.norm(data['ligand'].x[:, :, -4:-1].float(), dim=-1, keepdim=True))
        data['ligand'].x[:, :, -4:-1] = (data['ligand'].x[:, :, -4:-1] @ rot_mat_2).to(data['ligand'].pos.device)
        

        data['ligand'].x[:, :, -7:-4] = (data['ligand'].x[:, :, -7:-4] @ rot_mat_2).to(data['ligand'].pos.device)
        data['ligand'].fake_pocket_norm = (data['ligand'].fake_pocket_norm / torch.norm(data['ligand'].fake_pocket_norm)) @ rot_mat_2
    
    return data
    







def compare_conformer(data_ref, mol):
    gen_mol = copy.deepcopy(mol)
    ref_pos = data_ref['ligand'].pos
    gen_pos = torch.from_numpy(gen_mol.GetConformer().GetPositions()).to(ref_pos.device).float()
    

    data_ref.mol = Chem.RemoveHs(data_ref.mol)
    #set new atom positions on mol
    #for i in range(data_ref.mol.GetNumAtoms()):
    #    data_ref.mol.GetConformer(0).SetAtomPosition(i, Point3D(gen_pos[i, 0].cpu().item(), gen_pos[i, 1].cpu().item(), gen_pos[i, 2].cpu().item()))

    rotable_bonds = get_torsion_angles(gen_mol)
    #get torsion angles
    if rotable_bonds:
        opt_mol, angles, match_dif = optimize_rotatable_bonds(gen_mol, data_ref.mol, rotable_bonds=rotable_bonds, return_angle=True)
    else:
        opt_mol = gen_mol
        angles = np.zeros(0)
        match_dif = 0
                          
    opt_pos = torch.from_numpy(opt_mol.GetConformer().GetPositions()).to(ref_pos.device).float()
    minus_R, minus_t = rigid_transform_Kabsch_3D_torch(opt_pos.T, ref_pos.T)

    opt_pos = (opt_pos - torch.mean(opt_pos, dim=0, keepdim=True)) @ minus_R.T + minus_t.T + torch.mean(opt_pos, dim=0, keepdim=True)

    flexible_new_pos = modify_conformer_torsion_angles(gen_pos,
                                                        data_ref['ligand', 'lig_bond', 'ligand'].edge_index.T[data_ref['ligand'].edge_mask],
                                                        data_ref['ligand'].mask_rotate if isinstance(data_ref['ligand'].mask_rotate, np.ndarray) else data_ref['ligand'].mask_rotate[0],
                                                        - angles).to(gen_pos.device)
    R_align, t_align = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, gen_pos.T)
    flexible_new_pos = (flexible_new_pos - torch.mean(flexible_new_pos, dim=0, keepdim=True)) @ R_align.T + t_align.T + torch.mean(flexible_new_pos, dim=0, keepdim=True)
    R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, opt_pos.T)
    #compare error
    #error = (flexible_new_pos - torch.mean(flexible_new_pos, dim=0, keepdim=True)) @ R.T - (opt_pos - torch.mean(opt_pos, dim=0, keepdim=True))
    #error = torch.mean(torch.sqrt(torch.sum(error**2, dim=-1)))
    #print("compare error:", error)
    

    return - t, matrix_to_axis_angle(R), angles, match_dif, opt_pos


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size//2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb



def get_t_schedule(inference_steps):
    return np.linspace(1, 0, inference_steps + 1)[:-1]


def set_time(complex_graphs, t_tor, t_linker, batchsize):
    device = complex_graphs['ligand'].pos.device
    complex_graphs.complex_t = {'tor': t_tor * torch.ones(batchsize).to(device),
                                'linker': t_linker * torch.ones(batchsize).to(device)}
    #expand t to node level
    if not isinstance(t_tor, torch.Tensor):
        t_tor = torch.tensor(t_tor)
    if not isinstance(t_linker, torch.Tensor):
        t_linker = torch.tensor(t_linker)

    t_tor_l = t_tor.repeat_interleave(complex_graphs['ligand'].num_nodes).to(device)
    t_linker_l = t_linker.repeat_interleave(complex_graphs['ligand'].num_nodes).to(device)
    complex_graphs['ligand'].node_t = {'tor': t_tor_l, 'linker': t_linker_l}
    t_tor_r = t_tor.repeat_interleave(complex_graphs['receptor'].num_nodes).to(device)
    t_linker_r = t_linker.repeat_interleave(complex_graphs['receptor'].num_nodes).to(device)
    complex_graphs['receptor'].node_t = {'tor': t_tor_r, 'linker': t_linker_r}
    return complex_graphs