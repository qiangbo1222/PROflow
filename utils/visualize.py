from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.vectors import Vector
import prody
#prody.confProDy(verbosity='none')

import rdkit
import rdkit.Chem as Chem
from rdkit.Geometry import Point3D

import torch
import numpy as np

from utils.geometry import rigid_transform_Kabsch_3D_torch
from dataset.process_mols import extract_receptor_structure, parse_pdb_from_path

def extract_io_coords(input_pdb):
    protein_structure = prody.parsePDB(input_pdb)
    coords = protein_structure.getCoords()
    coords = torch.tensor(coords).float()
    return coords

def extract_CA_coords(input_pdb):
    protein_structure = prody.parsePDB(input_pdb)
    coords = protein_structure.select('name CA').getCoords()
    coords = torch.tensor(coords).float()
    return coords

def transform_pdb(input_pdb, output_pdb, new_pos=None, rot_mat=None, trans=None):
    parser = PDBParser()
    structure = parser.get_structure('X', input_pdb)

    #get all atom coordinates
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.get_coord())
    
    _, _, _, c_alpha_coords, _, _, _ = extract_receptor_structure(parse_pdb_from_path(input_pdb))
    
    #transform all atom coordinates
    coords = torch.tensor(coords)
    c_alpha_coords = torch.tensor(c_alpha_coords).float()
    
    if rot_mat is not None and trans is not None and new_pos is None:
        coords = (coords - torch.mean(coords, dim=0)) @ rot_mat.T + torch.mean(coords, dim=0)
        coords += trans.T
        
    if new_pos is not None:
        rot_mat, trans = rigid_transform_Kabsch_3D_torch(c_alpha_coords.T, new_pos.T.float())
        coords = (coords - torch.mean(coords, dim=0)) @ rot_mat.T + torch.mean(coords, dim=0)
        coords += trans.T
    
    coords = coords.numpy()
    #write to new pdb file
    coord_id = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom.set_coord(coords[coord_id])
                    anum = atom.get_serial_number()
                    atom.set_serial_number(anum)
                    coord_id += 1
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, preserve_atom_numbering = True)

def transform_sdf(mol, trans, rot, write=None):
    if isinstance(mol, str):
        mol = Chem.SDMolSupplier(mol)[0]
    mol_coords = torch.tensor(np.array(mol.GetConformer().GetPositions())).float()
    mol_coords = (mol_coords - torch.mean(mol_coords, dim=0)) @ rot.T + torch.mean(mol_coords, dim=0)
    mol_coords += trans
    
    mol_coords = mol_coords.numpy()
    for i in range(mol.GetNumAtoms()):
        mol.GetConformer().SetAtomPosition(i, Point3D(float(mol_coords[i][0]),
                                                      float(mol_coords[i][1]),
                                                      float(mol_coords[i][2])))
    
    if write is not None:
        w = Chem.SDWriter(write)
        w.write(mol)
        w.close()
        
    return mol