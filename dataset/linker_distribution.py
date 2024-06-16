import sys
sys.path.append('..')

import rdkit
from rdkit import Chem
import rdkit.Chem.AllChem as AllChem
import rdkit.Chem.Descriptors as Descriptors

import copy
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import networkx as nx
import multiprocessing as mp
import tqdm
import pickle
from rdkit.Geometry import Point3D

from utils.geometry import rot_axis_to_matrix, transform_norm_vector, rigid_transform_Kabsch_3D_torch, get_optimal_R
from dataset.conformer_matching import get_torsions

def generate_conformer(mol, n_conformers=800, max_iters=500):
    """
    Generate conformer for a molecule
    """
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, maxAttempts=max_iters, numThreads=1, useRandomCoords=True)
    #AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=1000)
    mol = Chem.RemoveHs(mol)
    return mol

def kabsch_fit(points_A, points_B, new_points, random=True):

    if points_A.shape[0] > 2 or random:
        R, t = rigid_transform_Kabsch_3D_torch(points_A.T.float(), points_B.T.float())
        new_points = (new_points - torch.mean(points_A, dim=0, keepdim=True))@R.T + t.T + torch.mean(points_A, dim=0, keepdim=True)
        return new_points

    else:
        R, t = transform_norm_vector(points_A, points_B)
        new_points = (new_points - points_A[0].unsqueeze(0))@R.T + points_A[0].unsqueeze(0) + t.T
        
        return new_points

def generate_linker_length(mol, sample_num=1024, anchor_idx_set=None):


    if anchor_idx_set is None:
        mol = Chem.AddHs(mol, addCoords=True)
    AllChem.EmbedMultipleConfs(mol, numConfs=sample_num, maxAttempts=500)
    mol = Chem.RemoveHs(mol)

    mol_graph = nx.Graph()
    edges = [(e.GetBeginAtomIdx(), e.GetEndAtomIdx()) for e in mol.GetBonds()]
    mol_graph.add_edges_from(edges)
    mol_graph = mol_graph.to_undirected()

    linker_size = []
    key_points = []
    #try:
    for conf in mol.GetConformers():
        coords = np.array(conf.GetPositions())
        inner_distance = np.zeros((len(coords), len(coords)))
        for i in range(len(coords)):
            for j in range(len(coords)):
                #topological distance measured by bonds between atoms
                if i == j:
                    inner_distance[i, j] = 0
                else:
                    inner_distance[i, j] = nx.shortest_path_length(mol_graph, i, j)
        
        #get the bond direction of the farthest bond pair
        #get the index of the farthest bond pair
        
        if anchor_idx_set is not None:
            bond_idx_start = [anchor_idx_set[0], anchor_idx_set[2]]
            bond_idx_end = [anchor_idx_set[1], anchor_idx_set[3]]
            atoms_start = [mol.GetAtomWithIdx(int(x)) for x in bond_idx_start]
            atoms_end = [mol.GetAtomWithIdx(int(x)) for x in bond_idx_end]
        else:
            bond_idx_start = np.unravel_index(np.argmax(inner_distance), inner_distance.shape)
            #get these bonds from mol
            atoms_start = [mol.GetAtomWithIdx(int(x)) for x in bond_idx_start]
            bonds = [x.GetBonds()[0] for x in atoms_start]
            atoms_end = [x.GetOtherAtom(y) for x, y in zip(bonds, atoms_start)]
            bond_idx_end = [x.GetIdx() for x in atoms_end]
        anchor_idx = [bond_idx_start[0], bond_idx_end[0], bond_idx_start[1], bond_idx_end[1]]
        bonds_direction = np.array([conf.GetAtomPosition(atoms_start[0].GetIdx()) , conf.GetAtomPosition(atoms_end[0].GetIdx()),
                                    conf.GetAtomPosition(atoms_start[1].GetIdx()) , conf.GetAtomPosition(atoms_end[1].GetIdx())])
        #normalize the bond direction
        bonds_direction[1] = bonds_direction[0] + (bonds_direction[1] - bonds_direction[0]) / np.linalg.norm(bonds_direction[1] - bonds_direction[0])
        bonds_direction[3] = bonds_direction[2] + (bonds_direction[3] - bonds_direction[2]) / np.linalg.norm(bonds_direction[3] - bonds_direction[2])
        
        bonds_direction = norm_dist_4_points(torch.from_numpy(bonds_direction).float())
        key_points.append(copy.deepcopy(bonds_direction))
        linker_size.append(np.linalg.norm(coords[bond_idx_start[0]] - coords[bond_idx_start[1]]))
        
            

    #except:
    #    print("Error: ", Chem.MolToSmiles(mol))
    #    return None
    return np.array(linker_size), Chem.MolToSmiles(mol), mol, torch.stack(key_points), anchor_idx


def norm_dist_4_points(P):
    """
    P contains two normal vectors of two bonds
    align P[0] to [0, 0, 0]
    align P[1] to [0, 0, 1]
    align P[2] to [0, x, y]
    """

    P[1] = P[0] + (P[1] - P[0]) / torch.norm(P[1] - P[0])
    P[3] = P[2] + (P[3] - P[2]) / torch.norm(P[3] - P[2])
    
    P = P - P[0]
    #rotate P[1] to [0, 0, 1]
    angle = torch.acos(P[1] @ torch.tensor([0, 0, 1.0]) / torch.norm(P[1]))
    axis = torch.cross(P[1], torch.tensor([0, 0, 1.0]))
    axis = axis / torch.norm(axis)
    rot_mat = rot_axis_to_matrix(axis, angle)
    P = P @ rot_mat.T
    #rotate P[2] to [0, x, y] on axis [0, 0, 1]
    axis = torch.tensor([0, 0, 1.0])
    angle = torch.atan(P[2, 0] / P[2, 1])
    rot_mat = rot_axis_to_matrix(axis, angle)
    P = P @ rot_mat.T
    if P[2, 1] < 0:
        P = P * torch.tensor([1, -1, 1])

    return P

def get_torsion_angles(mol, idx_filter=None):
    #find all rotatable bonds and return the torsion angles as a np.array
    tor_angles = []
    tor_angle_idx = get_torsions([mol])
    if idx_filter is not None:
        tor_angle_idx = [idx for idx in tor_angle_idx if idx[0] in idx_filter and idx[1] in idx_filter and idx[2] in idx_filter and idx[3] in idx_filter]
    for tor_idx in tor_angle_idx:
        tor_angles.append(Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(), 
                                                                tor_idx[0], 
                                                                tor_idx[1],
                                                                tor_idx[2],
                                                                tor_idx[3]))
    return np.array(tor_angles)

def set_torsion_angles(mol, angles, idx_filter=None):
    #set the torsion angles of a molecule
    tor_angles = []
    tor_angle_idx = get_torsions([mol])
    if idx_filter is not None:
        tor_angle_idx = [idx for idx in tor_angle_idx if idx[0] in idx_filter and idx[1] in idx_filter and idx[2] in idx_filter and idx[3] in idx_filter]
    for i, tor_idx in enumerate(tor_angle_idx):
        Chem.rdMolTransforms.SetDihedralDeg(mol.GetConformer(), 
                                                tor_idx[0], 
                                                tor_idx[1],
                                                tor_idx[2],
                                                tor_idx[3],
                                                angles[i])
    return mol


def linker_interpolate(mol_ref, mol_noise, sigma, anchor_idx):
    tor_ref = get_torsion_angles(mol_ref)
    noise_ref = get_torsion_angles(mol_noise)
    tor_noise = tor_ref + sigma * (noise_ref - tor_ref)
    mol_noise_full = set_torsion_angles(copy.deepcopy(mol_noise), noise_ref)
    mol_noise = set_torsion_angles(mol_noise, tor_noise)
    #align 2 molecules
    ref_conf = torch.from_numpy(np.array(mol_ref.GetConformer().GetPositions())).float()
    noise_conf = torch.from_numpy(np.array(mol_noise.GetConformer().GetPositions())).float()
    full_noise_conf = torch.from_numpy(np.array(mol_noise_full.GetConformer().GetPositions())).float()
    noise_conf = kabsch_fit(noise_conf[anchor_idx], ref_conf[anchor_idx], noise_conf)
    mol_noise_full = kabsch_fit(full_noise_conf[anchor_idx], ref_conf[anchor_idx], full_noise_conf)
    noise_conf = get_optimal_R(noise_conf, [ref_conf, mol_noise_full], ref_conf[anchor_idx], sigma)
    #set the new coordinates
    mol_inter = copy.deepcopy(mol_ref)
    for i, conf in enumerate(noise_conf):
        mol_inter.GetConformer().SetAtomPosition(i, Point3D(conf[0].item(), conf[1].item(), conf[2].item()))
    return mol_inter


if __name__ == '__main__':

    suppl = Chem.SDMolSupplier('linker.sdf')
    linker_mols = [x for x in suppl if x is not None]
    linker_mols = [m for m in linker_mols if '.' not in Chem.MolToSmiles(m) and Descriptors.NumRotatableBonds(m) > 3]
    linker_size = {}
    
    p = mp.Pool(64)
    for r in tqdm.tqdm(p.imap_unordered(generate_linker_length, linker_mols), total=len(linker_mols)):
        linker_size[r[1]] = {'size': torch.from_numpy(r[0]), 'mol': r[2], 'key_points': r[3], 'anchor_idx': r[4]}
    p.close()
    #for mol in tqdm.tqdm(linker_mols):
    #    try:
    #        generate_linker_length(mol)
    #    except:
    #        print("Error: ", Chem.MolToSmiles(mol))

    with open('linker_conf.pkl', 'wb') as f:
        pickle.dump(linker_size, f)
    