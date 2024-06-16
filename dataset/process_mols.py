import copy
import os
import warnings

import numpy as np
import scipy.spatial as spa
import torch
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs
from rdkit.Geometry import Point3D
from scipy import spatial
from scipy.special import softmax
from torch_cluster import radius_graph, knn_graph
from tqdm import tqdm

import networkx as nx


import torch.nn.functional as F

from dataset.conformer_matching import get_torsion_angles, optimize_rotatable_bonds
from utils.torsion import get_transformation_mask

from utils.geometry import axis_angle_to_matrix, rigid_transform_Kabsch_3D_torch
from utils.torsion import modify_conformer_torsion_angles
from utils.geometry import matrix_to_axis_angle

biopython_parser = PDBParser()
periodic_table = GetPeriodicTable()
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

lig_feature_dims = (list(map(len, [
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_chirality_list'],
    allowable_features['possible_degree_list'],
    allowable_features['possible_formal_charge_list'],
    allowable_features['possible_implicit_valence_list'],
    allowable_features['possible_numH_list'],
    allowable_features['possible_number_radical_e_list'],
    allowable_features['possible_hybridization_list'],
    allowable_features['possible_is_aromatic_list'],
    allowable_features['possible_numring_list'],
    allowable_features['possible_is_in_ring3_list'],
    allowable_features['possible_is_in_ring4_list'],
    allowable_features['possible_is_in_ring5_list'],
    allowable_features['possible_is_in_ring6_list'],
    allowable_features['possible_is_in_ring7_list'],
    allowable_features['possible_is_in_ring8_list'],
])), 0)  # number of scalar features

rec_atom_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids'],
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_atom_type_2'],
    allowable_features['possible_atom_type_3'],
])), 0)

rec_residue_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids']
])), 0)


def lig_atom_featurizer(mol):
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append([
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
        ])

    return torch.tensor(atom_features_list)


def rec_residue_featurizer(rec, index=None):
    feature_list = []
    for i, residue in enumerate(rec.get_residues()):
        if index is not None and i in index:
            feature_list.append([safe_index(allowable_features['possible_amino_acids'], residue.get_resname())])
        elif index is None:
            feature_list.append([safe_index(allowable_features['possible_amino_acids'], residue.get_resname())])
    return torch.tensor(feature_list, dtype=torch.float32)  # (N_res, 1)

def get_rec_from_atom(rec):
    rec_residue_id = []
    for atom in rec.get_atoms():
        rec_residue_id.append(atom.get_parent().get_id()[1])
    return rec_residue_id


def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index """
    try:
        return l.index(e)
    except:
        return len(l) - 1



def parse_receptor(pdbid, pdbbind_dir):
    rec = parsePDB(pdbid, pdbbind_dir)
    return rec


def parsePDB(pdbid, pdbbind_dir):
    rec_path = os.path.join(pdbbind_dir, pdbid, f'{pdbid}_protein_processed.pdb')
    return parse_pdb_from_path(rec_path)

def parse_pdb_from_path(path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', path)
        rec = structure[0]
    return rec


def extract_receptor_structure(rec, lm_embedding_chains=None):
    coords = []
    atom_to_res = []
    c_alpha_coords = []
    n_coords = []
    c_coords = []
    valid_chain_ids = []
    lengths = []
    for i, chain in enumerate(rec):
        chain_coords = []  # num_residues, num_atoms, 3
        chain_c_alpha_coords = []
        chain_n_coords = []
        chain_c_coords = []
        chain_atom_to_res = []
        count = 0
        invalid_res_ids = []
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                invalid_res_ids.append(residue.get_id())
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
                residue_coords.append(list(atom.get_vector()))

            if c_alpha != None and n != None and c != None:
                # only append residue if it is an amino acid and not some weird molecule that is part of the complex
                chain_c_alpha_coords.append(c_alpha)
                chain_n_coords.append(n)
                chain_c_coords.append(c)
                chain_coords.append(np.array(residue_coords))
                chain_atom_to_res.append(np.zeros((len(residue_coords)), dtype=np.int32) + count)
                count += 1
            else:
                invalid_res_ids.append(residue.get_id())
        for res_id in invalid_res_ids:
            chain.detach_child(res_id)
        if len(chain_coords) > 0:
            all_chain_coords = np.concatenate(chain_coords, axis=0)

        lengths.append(count)
        coords.append(chain_coords)
        atom_to_res.append(chain_atom_to_res)
        c_alpha_coords.append(np.array(chain_c_alpha_coords))
        n_coords.append(np.array(chain_n_coords))
        c_coords.append(np.array(chain_c_coords))
        if not count == 0: valid_chain_ids.append(chain.get_id())

    valid_coords = []
    valid_atom_to_res = []
    valid_c_alpha_coords = []
    valid_n_coords = []
    valid_c_coords = []
    valid_lengths = []
    invalid_chain_ids = []
    valid_lm_embeddings = []
    for i, chain in enumerate(rec):
        if chain.get_id() in valid_chain_ids:
            valid_coords.append(coords[i])
            valid_atom_to_res.append(atom_to_res[i])
            valid_c_alpha_coords.append(c_alpha_coords[i])
            if lm_embedding_chains is not None:
                if i >= len(lm_embedding_chains):
                    raise ValueError('Encountered valid chain id that was not present in the LM embeddings')
                valid_lm_embeddings.append(lm_embedding_chains[i])
            valid_n_coords.append(n_coords[i])
            valid_c_coords.append(c_coords[i])
            valid_lengths.append(lengths[i])
        else:
            invalid_chain_ids.append(chain.get_id())
    coords = [item for sublist in valid_coords for item in sublist]  # list with n_residues arrays: [n_atoms, 3]
    atom_to_res = [item for sublist in valid_atom_to_res for item in sublist]  # list with n_residues arrays: [n_atoms]

    c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3]
    n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
    c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]
    lm_embeddings = np.concatenate(valid_lm_embeddings, axis=0) if lm_embedding_chains is not None else None
    for invalid_id in invalid_chain_ids:
        rec.detach_child(invalid_id)

    assert len(c_alpha_coords) == len(n_coords)
    assert len(c_alpha_coords) == len(c_coords)
    assert sum(valid_lengths) == len(c_alpha_coords)
    assert len(atom_to_res) == len(coords)
    return rec, coords, atom_to_res, c_alpha_coords, n_coords, c_coords, lm_embeddings

def extract_c_alpha_coords(rec):
    c_alpha_coords = []
    valid_chain_ids = []
    for i, chain in enumerate(rec):
        chain_c_alpha_coords = []
        count = 0
        invalid_res_ids = []
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                invalid_res_ids.append(residue.get_id())
                continue
            c_alpha = None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())

            if c_alpha != None:
                # only append residue if it is an amino acid and not some weird molecule that is part of the complex
                chain_c_alpha_coords.append(c_alpha)
                count += 1
            else:
                invalid_res_ids.append(residue.get_id())

        c_alpha_coords.append(np.array(chain_c_alpha_coords))
        if not count == 0: valid_chain_ids.append(chain.get_id())


    valid_c_alpha_coords = []
    for i, chain in enumerate(rec):
        if chain.get_id() in valid_chain_ids:
            valid_c_alpha_coords.append(c_alpha_coords[i])
        else:
            invalid_chain_ids.append(chain.get_id())

    c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3]
    return c_alpha_coords

def extract_protein_structure(rec):#need single chain as input
    coords = []
    c_alpha_coords = []
    n_coords = []
    c_coords = []
    valid_chain_ids = []
    lengths = []
    for i, chain in enumerate(rec):
        chain_coords = []  # num_residues, num_atoms, 3
        chain_c_alpha_coords = []
        chain_n_coords = []
        chain_c_coords = []
        count = 0
        invalid_res_ids = []
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                invalid_res_ids.append(residue.get_id())
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C': 
                    c = list(atom.get_vector())
                residue_coords.append(list(atom.get_vector()))

            if c_alpha != None and n != None and c != None:
                # only append residue if it is an amino acid and not some weird molecule that is part of the complex
                chain_c_alpha_coords.append(c_alpha)
                chain_n_coords.append(n)
                chain_c_coords.append(c)
                chain_coords.append(np.array(residue_coords))
                count += 1
            else:
                invalid_res_ids.append(residue.get_id())
        for res_id in invalid_res_ids:
            chain.detach_child(res_id)

        lengths.append(count)
        coords.append(chain_coords)
        c_alpha_coords.append(np.array(chain_c_alpha_coords))
        n_coords.append(np.array(chain_n_coords))
        c_coords.append(np.array(chain_c_coords))
        if not count == 0: valid_chain_ids.append(chain.get_id())

    valid_coords = []
    valid_c_alpha_coords = []
    valid_n_coords = []
    valid_c_coords = []
    valid_lengths = []
    invalid_chain_ids = []
    for i, chain in enumerate(rec):
        if chain.get_id() in valid_chain_ids:
            valid_coords.append(coords[i])
            valid_c_alpha_coords.append(c_alpha_coords[i])
            valid_n_coords.append(n_coords[i])
            valid_c_coords.append(c_coords[i])
            valid_lengths.append(lengths[i])
        else:
            invalid_chain_ids.append(chain.get_id())
    coords = [item for sublist in valid_coords for item in sublist]  # list with n_residues arrays: [n_atoms, 3]

    c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3]
    n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
    c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]

    assert len(c_alpha_coords) == len(n_coords)
    assert len(c_alpha_coords) == len(c_coords)
    assert sum(valid_lengths) == len(c_alpha_coords)
    return rec, coords, c_alpha_coords, n_coords, c_coords

def get_lig_graph(mol, complex_graph):
    lig_coords = torch.from_numpy(mol.GetConformer().GetPositions()).float()
    atom_feats = lig_atom_featurizer(mol)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]] if bond.GetBondType() != BT.UNSPECIFIED else [0, 0]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    complex_graph['ligand'].x = atom_feats
    complex_graph['ligand'].pos = lig_coords
    complex_graph['ligand', 'lig_bond', 'ligand'].edge_index = edge_index
    complex_graph['ligand', 'lig_bond', 'ligand'].edge_attr = edge_attr
    return

def generate_conformer(mol):
    mol = Chem.RemoveHs(mol)
    mol_cache = copy.deepcopy(mol)
    smiles_org = Chem.MolToSmiles(mol)
    org_conf = torch.from_numpy(mol.GetConformer().GetPositions()).float()
    #remove conformers
    old_conformer = Chem.AddHs(mol).GetConformer()
    mol.RemoveAllConformers()
    try:
        ps = AllChem.ETKDGv2()
        mol = Chem.AddHs(mol)
        id = AllChem.EmbedMolecule(mol, ps)
    except:
        id = -1
    if id == -1:
        #print('rdkit coords could not be generated without using random coords. using random coords now.')
        try:
            ps.useRandomCoords = True
            AllChem.EmbedMolecule(mol, ps)
        except:
            pass#raise error for some invalid bonds
        #for those cannot be embedded, use the original coords
        if mol.GetNumConformers() == 0:
            mol.AddConformer(old_conformer)
            #add some random noise
            for i, atom in enumerate(mol.GetAtoms()):
                atom_pos = np.array(mol.GetConformer().GetAtomPosition(i))
                atom_pos += np.random.normal(0, 0.5, 3)
                mol.GetConformer().SetAtomPosition(i, Point3D(float(atom_pos[0]), float(atom_pos[1]), float(atom_pos[2])))
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    # else:
    #    AllChem.MMFFOptimizeMolecule(mol_rdkit, confId=0)
    assert mol.GetNumConformers() == 1
    mol = Chem.RemoveHs(mol)

    #reassign to mol cache
    for i in range(mol.GetNumAtoms()):
        mol_cache.GetConformer().SetAtomPosition(i, mol.GetConformer().GetAtomPosition(i))
    mol = mol_cache

    #not too much noise on rotation
    new_conf = torch.from_numpy(mol.GetConformer().GetPositions()).float()
    R, t = rigid_transform_Kabsch_3D_torch(new_conf.T, org_conf.T)
    reduce_R = axis_angle_to_matrix(matrix_to_axis_angle(R) * 0.75)
    new_conf = (new_conf - torch.mean(new_conf, dim=0)) @ reduce_R + torch.mean(new_conf, dim=0)
    assert Chem.MolToSmiles(mol) == smiles_org, 'smiles: ' + Chem.MolToSmiles(mol) + ' smiles_org: ' + smiles_org
    #set new conf
    for i in range(mol.GetNumAtoms()):
        mol.GetConformer().SetAtomPosition(i, Point3D(new_conf[i][0].item(), new_conf[i][1].item(), new_conf[i][2].item()))
    assert Chem.MolToSmiles(mol) == smiles_org
    return mol

def get_lig_graph_with_matching(mol_, complex_graph, popsize, maxiter, matching, keep_original, num_conformers, remove_hs):
    if matching:
        mol_maybe_noh = copy.deepcopy(mol_)
        if remove_hs:
            mol_maybe_noh = RemoveHs(mol_maybe_noh, sanitize=True)
        if keep_original:
            complex_graph['ligand'].orig_pos = mol_maybe_noh.GetConformer().GetPositions()

        rotable_bonds = get_torsion_angles(mol_maybe_noh)
        if not rotable_bonds: print("no_rotable_bonds but still using it")

        for i in range(num_conformers):
            mol_rdkit = copy.deepcopy(mol_)

            mol_rdkit.RemoveAllConformers()
            mol_rdkit = AllChem.AddHs(mol_rdkit)
            generate_conformer(mol_rdkit)
            if remove_hs:
                mol_rdkit = RemoveHs(mol_rdkit, sanitize=True)
            mol = copy.deepcopy(mol_maybe_noh)
            if rotable_bonds:
                optimize_rotatable_bonds(mol_rdkit, mol, rotable_bonds, popsize=popsize, maxiter=maxiter)
            mol.AddConformer(mol_rdkit.GetConformer())
            rms_list = []
            AllChem.AlignMolConformers(mol, RMSlist=rms_list)
            mol_rdkit.RemoveAllConformers()
            mol_rdkit.AddConformer(mol.GetConformers()[1])

            if i == 0:
                complex_graph.rmsd_matching = rms_list[0]
                get_lig_graph(mol_rdkit, complex_graph)
            else:
                if torch.is_tensor(complex_graph['ligand'].pos):
                    complex_graph['ligand'].pos = [complex_graph['ligand'].pos]
                complex_graph['ligand'].pos.append(torch.from_numpy(mol_rdkit.GetConformer().GetPositions()).float())

    else:  # no matching
        complex_graph.rmsd_matching = 0
        if remove_hs: mol_ = RemoveHs(mol_)
        get_lig_graph(mol_, complex_graph)

    used_mol = mol_rdkit if matching else mol_
    edge_mask, mask_rotate = get_transformation_mask(complex_graph, used_mol)
    
    
    complex_graph['ligand'].edge_mask = torch.tensor(edge_mask)
    complex_graph['ligand'].mask_rotate = mask_rotate

    return


def get_calpha_graph(rec, c_alpha_coords, n_coords, c_coords, complex_graph, cutoff=20, max_neighbor=None, lm_embeddings=None, pocket_size=20, lig_center=None):
    if lig_center is not None:
        box_distance = np.linalg.norm(c_alpha_coords - lig_center, axis=1)
        box_indx = np.where(box_distance < pocket_size)[0]
        c_alpha_coords = c_alpha_coords[box_indx]
        n_coords = n_coords[box_indx]
        c_coords = c_coords[box_indx]

        box_distance_full = np.linalg.norm(rec_coords - lig_center, axis=1)
        box_indx_full = np.where(box_distance_full < pocket_size)[0]
        rec_coords = rec_coords[box_indx_full]
        if lm_embeddings is not None:
            lm_embeddings = [lm_embeddings[b] for b in box_indx]
        
    else:
        box_indx = np.arange(len(c_alpha_coords))
        box_indx_full = np.arange(len(rec_coords))
    
    
    n_rel_pos = n_coords - c_alpha_coords
    c_rel_pos = c_coords - c_alpha_coords
    num_residues = len(c_alpha_coords)
    if num_residues <= 1:
        raise ValueError(f"rec contains only 1 residue!")

    # Build the k-NN graph
    distances = spa.distance.cdist(c_alpha_coords, c_alpha_coords)
    src_list = []
    dst_list = []
    mean_norm_list = []
    for i in range(num_residues):
        dst = list(np.where(distances[i, :] < cutoff)[0])
        dst.remove(i)
        if max_neighbor != None and len(dst) > max_neighbor:
            dst = list(np.argsort(distances[i, :]))[1: max_neighbor + 1]
        if len(dst) == 0:
            dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
            print(f'The c_alpha_cutoff {cutoff} was too small for one c_alpha such that it had no neighbors. '
                  f'So we connected it to the closest other c_alpha')
        assert i not in dst
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distances[i, dst])
        valid_dist_np = distances[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = c_alpha_coords[src, :] - c_alpha_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)

    node_feat = rec_residue_featurizer(rec)
    mu_r_norm = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    side_chain_vecs = torch.from_numpy(
        np.concatenate([np.expand_dims(n_rel_pos, axis=1), np.expand_dims(c_rel_pos, axis=1)], axis=1))

    complex_graph['receptor'].x = torch.cat([node_feat, torch.tensor(lm_embeddings)], axis=1) if lm_embeddings is not None else node_feat
    complex_graph['receptor'].pos = torch.from_numpy(c_alpha_coords).float()
    complex_graph['receptor'].mu_r_norm = mu_r_norm
    complex_graph['receptor'].side_chain_vecs = side_chain_vecs.float()
    complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = torch.from_numpy(np.asarray([src_list, dst_list]))

    return


def rec_atom_featurizer(rec, index=None):
    atom_feats = []
    for i, atom in enumerate(rec.get_atoms()):
        if index is not None and i not in index:
            continue
        atom_name, element = atom.name, atom.element
        if element == 'CD':
            element = 'C'
        assert not element == ''
        try:
            atomic_num = periodic_table.GetAtomicNumber(element)
        except:
            atomic_num = -1
        atom_feat = [safe_index(allowable_features['possible_amino_acids'], atom.get_parent().get_resname()),
                     safe_index(allowable_features['possible_atomic_num_list'], atomic_num),
                     safe_index(allowable_features['possible_atom_type_2'], (atom_name + '*')[:2]),
                     safe_index(allowable_features['possible_atom_type_3'], atom_name)]
        atom_feats.append(atom_feat)

    return atom_feats


def get_rec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords, complex_graph, rec_radius, c_alpha_max_neighbors=None, all_atoms=False,
                  atom_radius=5, atom_max_neighbors=None, remove_hs=False, lm_embeddings=None, pocket_size=20, lig_center=(0, 0, 0)):
    if all_atoms:
        return get_fullrec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords, complex_graph,
                                 c_alpha_cutoff=rec_radius, c_alpha_max_neighbors=c_alpha_max_neighbors,
                                 atom_cutoff=atom_radius, atom_max_neighbors=atom_max_neighbors, remove_hs=remove_hs,lm_embeddings=lm_embeddings, pocket_size=pocket_size, lig_center=lig_center)
    else:
        return get_calpha_graph(rec, c_alpha_coords, n_coords, c_coords, complex_graph, rec_radius, c_alpha_max_neighbors,lm_embeddings=lm_embeddings, pocket_size=pocket_size, lig_center=lig_center)


def get_fullrec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords, complex_graph, c_alpha_cutoff=20,
                      c_alpha_max_neighbors=None, atom_cutoff=5, atom_max_neighbors=None, remove_hs=False, lm_embeddings=None, pocket_size=20, lig_center=None):
    # builds the receptor graph with both residues and atoms
    
    if lig_center is not None:
        box_distance = np.linalg.norm(c_alpha_coords - lig_center, axis=1)
        box_indx = np.where(box_distance < pocket_size)[0]
        c_alpha_coords = c_alpha_coords[box_indx]
        n_coords = n_coords[box_indx]
        c_coords = c_coords[box_indx]
        if lm_embeddings is not None:
            lm_embeddings = [lm_embeddings[b] for b in box_indx]

        box_indx_full = []
        cumsum = 0
        for i in range(len(rec_coords)):
            if i in box_indx:
                box_indx_full.extend(list(np.arange(len(rec_coords[i])) + cumsum))
            cumsum += len(rec_coords[i])
        
        rec_coords = [rec_coords[b] for b in box_indx]
        
        
        
    else:
        box_indx = np.arange(len(c_alpha_coords))
        box_indx_full = np.arange(len(rec_coords))


    n_rel_pos = n_coords - c_alpha_coords
    c_rel_pos = c_coords - c_alpha_coords
    num_residues = len(c_alpha_coords)
    if num_residues <= 1:
        raise ValueError(f"rec contains only 1 residue!")

    # Build the k-NN graph of residues
    distances = spa.distance.cdist(c_alpha_coords, c_alpha_coords)
    src_list = []
    dst_list = []
    mean_norm_list = []
    for i in range(num_residues):
        dst = list(np.where(distances[i, :] < c_alpha_cutoff)[0])
        dst.remove(i)
        if c_alpha_max_neighbors != None and len(dst) > c_alpha_max_neighbors:
            dst = list(np.argsort(distances[i, :]))[1: c_alpha_max_neighbors + 1]
        if len(dst) == 0:
            dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
            print(f'The c_alpha_cutoff {c_alpha_cutoff} was too small for one c_alpha such that it had no neighbors. '
                  f'So we connected it to the closest other c_alpha')
        assert i not in dst
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distances[i, dst])
        valid_dist_np = distances[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert 1 - 1e-2 < weights[0].sum() < 1.01
        diff_vecs = c_alpha_coords[src, :] - c_alpha_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)

    node_feat = rec_residue_featurizer(rec, index=box_indx)
    mu_r_norm = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    side_chain_vecs = torch.from_numpy(
        np.concatenate([np.expand_dims(n_rel_pos, axis=1), np.expand_dims(c_rel_pos, axis=1)], axis=1))

    complex_graph['receptor'].x = torch.cat([node_feat, torch.tensor(lm_embeddings)], axis=1) if lm_embeddings is not None else node_feat
    complex_graph['receptor'].pos = torch.from_numpy(c_alpha_coords).float()
    complex_graph['receptor'].mu_r_norm = mu_r_norm
    complex_graph['receptor'].side_chain_vecs = side_chain_vecs.float()
    complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = torch.from_numpy(np.asarray([src_list, dst_list]))

    src_c_alpha_idx = np.concatenate([np.asarray([i]*len(l)) for i, l in enumerate(rec_coords)])
    atom_feat = torch.from_numpy(np.asarray(rec_atom_featurizer(rec, index=box_indx_full)))
    atom_coords = torch.from_numpy(np.concatenate(rec_coords, axis=0)).float()

    if remove_hs:
        not_hs = (atom_feat[:, 1] != 0)
        src_c_alpha_idx = src_c_alpha_idx[not_hs]
        atom_feat = atom_feat[not_hs]
        atom_coords = atom_coords[not_hs]

    atoms_edge_index = radius_graph(atom_coords, atom_cutoff, max_num_neighbors=atom_max_neighbors if atom_max_neighbors else 1000)
    atom_res_edge_index = torch.from_numpy(np.asarray([np.arange(len(atom_feat)), src_c_alpha_idx])).long()

    complex_graph['atom'].x = atom_feat
    complex_graph['atom'].pos = atom_coords
    complex_graph['atom', 'atom_contact', 'atom'].edge_index = atoms_edge_index
    complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index = atom_res_edge_index

    return

def write_mol_with_coords(mol, new_coords, path):
    w = Chem.SDWriter(path)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x,y,z = new_coords.astype(np.double)[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))
    w.write(mol)
    w.close()

def read_molecule(molecule_file, sanitize=True, calc_charges=False, remove_hs=True):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception as e:
        print(e)
        print("RDKit was unable to read the molecule.")
        return None

    return mol


def read_sdf_or_mol2(sdf_fileName, mol2_fileName):

    mol = Chem.MolFromMolFile(sdf_fileName, sanitize=False)
    problem = False
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
    except Exception as e:
        problem = True
    if problem:
        mol = Chem.MolFromMol2File(mol2_fileName, sanitize=False)
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            problem = False
        except Exception as e:
            problem = True

    return mol, problem

def get_protein_surface_graph(fileName, msms_bin_dir, rec_model, atom_coords, atom_to_res, complex_graph, agg_num=6, lig_coord=None, lm_embedding=None):
    try:
        if not os.path.exists(fileName):
            raise ValueError('File {} does not exist.'.format(fileName))

        # convert to xyzr
        if os.path.exists(fileName[:-4] + '.xyzr'):
            os.remove(fileName[:-4] + '.xyzr')

        xyzr_file = fileName[:-4] + '.xyzr'
        os.system(f"cd {msms_bin_dir}; {os.path.join(msms_bin_dir, 'pdb_to_xyzr')} {fileName} > {xyzr_file}")
        
        # run msms
        if os.path.exists(fileName[:-4] + '.vert'):
            os.remove(fileName[:-4] + '.vert')
        if os.path.exists(fileName[:-4] + '.face'):
            os.remove(fileName[:-4] + '.face')
        vert_file = fileName[:-4]
        os.system(f"{os.path.join(msms_bin_dir, 'msms.x86_64Linux2.2.6.1')} -if {fileName[:-4]}.xyzr -of {vert_file} -d 0.5  > null.out  2>&1")

        # read vertices
        if not os.path.isfile(fileName[:-4] + '.vert') or not os.path.isfile(fileName[:-4] + '.face'):
            return None
        vertices, vnormals, faces, faces_type = read_msms(fileName[:-4])
        #filter out vertices distance to ligand center > cut_off
        if isinstance(lig_coord, list):
            lig_coord = np.concatenate(lig_coord, axis=0)
        if len(lig_coord.shape) == 1:
            lig_coord = np.expand_dims(lig_coord, axis=0)
            cut_off = 40
        elif lig_coord.shape[0] == 1:
            cut_off = 40
        else:
            cut_off = 20
        if lig_coord is not None:
            #get the cross distance (shape rec * lig) with in 20A
            cross_distance = cdist_np(vertices, lig_coord)
            cross_distance = cross_distance.min(axis=1)
            vertives_idx = np.where(cross_distance < cut_off)[0]
            vertices = vertices[vertives_idx]
            vnormals = vnormals[vertives_idx]
        surface_features = []
        atom_feature = torch.from_numpy(np.asarray(rec_atom_featurizer(rec_model)))
        atom_coords = torch.from_numpy(np.concatenate(atom_coords, axis=0)).float()

        #filter H atom in protein
        no_H_idx = torch.where(atom_feature[:, 1] != 0)[0]
        atom_feature = atom_feature[no_H_idx]
        atom_coords = atom_coords[no_H_idx]
        atom_to_res = np.concatenate(atom_to_res)[no_H_idx]
        if lm_embedding is not None:
            assert len(lm_embedding) == np.max(atom_to_res) + 1

        new_faces = []
        for i in range(faces.shape[0]):
            if lig_coord is not None:
                if faces[i][0] not in vertives_idx or faces[i][1] not in vertives_idx or faces[i][2] not in vertives_idx:
                    continue
                faces[i] = [np.where(vertives_idx == faces[i][0])[0][0], np.where(vertives_idx == faces[i][1])[0][0], np.where(vertives_idx == faces[i][2])[0][0]]
            vert_coords = vertices[faces[i]]# [3, 3]
            vert_normals = vnormals[faces[i]]# [3, 3]
            new_faces.append(faces[i])
            center_coord = np.mean(vert_coords, axis=0)
            center_normal = np.mean(vert_normals, axis=0)

            # get the nearest atoms
            distances = np.linalg.norm(atom_coords - center_coord, axis=1)
            nearest_atom_idx = np.argsort(distances)[:agg_num]
            nearest_atom_feat = atom_feature[nearest_atom_idx]
            nearest_atom_coord = atom_coords[nearest_atom_idx]
            nearest_atom_vect = nearest_atom_coord - center_coord

            #get the res ID for the nearest atom
            nearest_atom_res_id = atom_to_res[nearest_atom_idx[0]]
            if lm_embedding is not None:
                res_embedding = lm_embedding[nearest_atom_res_id]
                res_vect = atom_coords[nearest_atom_idx[0]] - center_coord
            else:
                res_embedding = None
                res_vect = None

            surface_features.append({'center_coord': center_coord,
                                'center_normal': center_normal,
                                'nearest_atom_feat': nearest_atom_feat,
                                'nearest_atom_coord': nearest_atom_coord,
                                'nearest_atom_vect': nearest_atom_vect,
                                'face_type': faces_type[i],
                                'res_embedding': res_embedding,
                                'res_vect': res_vect,})
            
        if len(faces) == 0:
            return None
        faces = np.asarray(new_faces)
        #convert to graph

        graph_x = np.concatenate([np.stack([s['nearest_atom_feat'] for s in surface_features], axis=0),
                    np.stack([s['nearest_atom_vect'] for s in surface_features], axis=0)], axis=-1)
        norm_feature = np.stack([s['center_normal'] for s in surface_features], axis=0)

        feat = torch.from_numpy(graph_x).view(-1, agg_num, graph_x.shape[-1])
        face_center = torch.from_numpy(np.stack([s['center_coord'] for s in surface_features], axis=0)).float()
        norm_f = torch.from_numpy(norm_feature).unsqueeze(1).repeat(1, agg_num, 1)
        faces_type = torch.from_numpy(np.asarray([s['face_type'] for s in surface_features])).long().unsqueeze(-1).unsqueeze(-1).repeat(1, agg_num, 1)
        
        #del unused variables
        #del graph_x, norm_feature
        complex_graph['protein_surface'].x = torch.cat([feat, norm_f, faces_type], axis=-1)
        if lm_embedding is not None:
            res_feat = torch.from_numpy(np.stack([s['res_embedding'] for s in surface_features], axis=0)).float()
            res_vect = torch.from_numpy(np.stack([s['res_vect'] for s in surface_features], axis=0)).float()
            complex_graph['protein_surface'].res_x = torch.cat([res_feat, res_vect], axis=-1)
        complex_graph['protein_surface'].pos = face_center
        
        '''
        #knn graph
        edge_index = knn_graph(face_center, k=3, loop=False)
        distance = torch.norm(face_center[edge_index[0]] - face_center[edge_index[1]], dim=-1).unsqueeze(-1)
        #filter distance larger than 1.5
        edge_index = edge_index[:, distance.squeeze() < 1.0]
        print("edge_index shape:", edge_index.shape)
        '''
        
        #add edges
        #complex_graph['protein_surface', 'share_vertic', 'protein_surface'].edge_index = edge_index
        #complex_graph['protein_surface', 'share_vertic', 'protein_surface'].edge_attr = torch.norm(face_center[edge_index[0]] - face_center[edge_index[1]], dim=-1).unsqueeze(-1)
        return complex_graph
    except:
        return None


    
        

def read_msms(mesh_prefix):
    """Read and parse MSMS output
    Args:
        mesh_prefix (path): path prefix for MSMS output mesh. 
            The directory should contain .vert and .face files from MSMS
    
    Returns:
        vertices (np.ndarray): (N, 3) vertex coordinates
        vnormals (np.ndarray): (N, 3) vertex normals
        faces (np.ndarray): (F, 3) vertex ids of faces
    """
    assert os.path.isfile(mesh_prefix + '.vert')
    assert os.path.isfile(mesh_prefix + '.face')
    # vertices
    with open(mesh_prefix + '.vert') as f:
        vert_data = f.read().rstrip().split('\n')
    num_verts = int(vert_data[2].split()[0])
    assert num_verts == len(vert_data) - 3
    vertices = []
    vnormals = []
    for idx in range(3, len(vert_data)):
        ifields = vert_data[idx].split()
        assert len(ifields) == 9
        vertices.append(ifields[:3])
        vnormals.append(ifields[3:6])
    assert len(vertices) == num_verts

    # faces
    with open(mesh_prefix + '.face') as f:
        face_data = f.read().rstrip().split('\n')
    num_faces = int(face_data[2].split()[0])
    assert num_faces == len(face_data) - 3
    faces = []
    faces_type = []
    for idx in range(3, len(face_data)):
        ifields = face_data[idx].split()
        assert len(ifields) == 5
        faces.append(ifields[:3]) # one-based, to be converted
        faces_type.append(ifields[3])
    assert len(faces) == num_faces

    # solvent excluded surface info
    vertices = np.array(vertices, dtype=float)
    vnormals = np.array(vnormals, dtype=float)
    faces = np.array(faces, dtype=int) - 1 # convert to zero-based indexing
    faces_type = np.array(faces_type, dtype=int) - 1
    assert np.amin(faces) == 0
    assert np.amax(faces) < num_verts
    
    return vertices, vnormals, faces, faces_type



def cdist_np(coords_A, coords_B):
    #coords_A: (N, 3)
    #coords_B: (M, 3)

    #compute distance matrix
    coords_A = coords_A.reshape(-1, 1, 3)
    coords_B = coords_B.reshape(1, -1, 3)

    dist_mat = np.sqrt(np.sum((coords_A - coords_B)**2, axis=-1))
    return dist_mat

def mol_to_graph(mol):
    mol_graph = nx.Graph()
    edges = [(e.GetBeginAtomIdx(), e.GetEndAtomIdx()) for e in mol.GetBonds()]
    mol_graph.add_edges_from(edges)
    mol_graph = mol_graph.to_undirected()

    atom_num = mol.GetNumAtoms()
    inner_distance = np.zeros((atom_num, atom_num))
    for i in range(atom_num):
        for j in range(atom_num):
            #topological distance measured by bonds between atoms
            if i == j:
                inner_distance[i, j] = 0
            else:
                inner_distance[i, j] = nx.shortest_path_length(mol_graph, i, j)
    return inner_distance