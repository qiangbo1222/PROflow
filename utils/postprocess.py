#post process have 3 steps:
#align the docked or ground truth warheads back to the new protein positions
#use linker matching to get the best matched linker
#ensemble best matched linker and the warheads into a sdf; combine POI and E3 to one pdb file
import sys
sys.path.append('..')

import os
import tqdm
import copy
from prody import *
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFMCS import FindMCS

from dataset.linker_matching import *
from utils.visualize import transform_pdb, transform_sdf, extract_io_coords, extract_CA_coords

from dataset.process_mols import mol_to_graph, extract_receptor_structure, parse_pdb_from_path

def align_warheads(poi_org, poi_new, poi_lig_org_, e3_org, e3_new, e3_lig_org_):
    poi_lig_org, e3_lig_org = copy.deepcopy(poi_lig_org_), copy.deepcopy(e3_lig_org_)
    new_ligs = []
    for protein_org, protein_new, warhead_org in [(poi_org, poi_new, poi_lig_org),
                                                  (e3_org, e3_new, e3_lig_org)]:
        R, t = rigid_transform_Kabsch_3D_torch(protein_org.T, protein_new.T)
        t = change_R_T_ref(t.T, R, 
                              torch.tensor(protein_org).float(),
                              torch.tensor(warhead_org.GetConformer().GetPositions()).float())
        warhead_new = transform_sdf(warhead_org, t, R)
        new_ligs.append(warhead_new)
    
    return new_ligs

def get_anchor(mol1, mol2, check_distance=0.4):
    #mol1: the warhead with anchor bonds connect to linker
    #mol2: the linker 
    
    #get the 2 closest atoms idx in both mol
    coords1 = torch.tensor(np.array(mol1.GetConformer().GetPositions()))
    coords2 = torch.tensor(np.array(mol2.GetConformer().GetPositions()))
    
    cross_dist = torch.cdist(coords1, coords2)
    min_dist, anchor_i = torch.topk(torch.min(cross_dist, dim=1)[0], 2, largest=False)
    anchor_j = [torch.argmin(cross_dist[i]) for i in anchor_i]
    assert min_dist.max() < check_distance
    
    inner_dist_mol1 = np.sum(mol_to_graph(mol1), axis=-1)
    
    #assign order to the anchors
    if inner_dist_mol1[anchor_i[1]] < inner_dist_mol1[anchor_i[0]]:
        anchor_i = [anchor_i[1], anchor_i[0]]
        anchor_j = [anchor_j[1], anchor_j[0]]
    
    return list(anchor_i), list(anchor_j)

def clear_atoms_and_bonds_with_type(molecule, atom_type):
    """
    Clear all atoms and bonds of a specific atom type in an RDKit molecule.

    Args:
        molecule (Chem.Mol): The RDKit molecule.
        atom_type (str): The atom type to be cleared, e.g., '*'.

    Returns:
        Chem.Mol: The modified RDKit molecule with specified atoms and bonds removed.
    """
    # Create a copy of the input molecule
    modified_molecule = Chem.Mol(molecule)

    # Find and remove atoms of the specified type
    atoms_to_remove = [atom.GetIdx() for atom in modified_molecule.GetAtoms() if atom.GetSymbol() == atom_type]
    modified_molecule = Chem.EditableMol(modified_molecule)
    for atom_idx in reversed(atoms_to_remove):
        modified_molecule.RemoveAtom(atom_idx)
    modified_molecule = modified_molecule.GetMol()

    # Remove bonds associated with the removed atoms
    modified_molecule = Chem.DeleteSubstructs(modified_molecule, Chem.MolFromSmiles(atom_type))

    return modified_molecule

def split_pdb(pdb_file, chain1_file, chain2_file):
    """
    Split a PDB file containing two chains into two separate files.

    Parameters:
    pdb_file (str): Path to the input PDB file.
    chain1_file (str): Path to the output file for the first chain.
    chain2_file (str): Path to the output file for the second chain.
    """
    with open(pdb_file, 'r') as infile:
        chain1_lines = []
        chain2_lines = []
        current_chain = None

        for line in infile:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                chain_id = line[21]
                if current_chain is None:
                    current_chain = chain_id

                if chain_id == current_chain:
                    chain1_lines.append(line)
                else:
                    chain2_lines.append(line)

    with open(chain1_file, 'w') as outfile1:
        for line in chain1_lines:
            outfile1.write(line)

    with open(chain2_file, 'w') as outfile2:
        for line in chain2_lines:
            outfile2.write(line)


def assign_3d_conformation_from_fragments(fragments, complete_smiles):
    """
    Assigns a 3D conformation to a complete molecule using 3D coordinates from its fragments
    via common substructure matching.

    Parameters:
    fragments (list of tuple): Each tuple contains a fragment as an RDKit Mol object and its 3D coordinates.
    complete_smiles (str): SMILES string of the complete molecule.

    Returns:
    Chem.Mol: RDKit Mol object of the complete molecule with assigned 3D conformation.
    """
    # Convert the complete molecule to a RDKit Mol object
    complete_mol = Chem.MolFromSmiles(complete_smiles)
    complete_mol = Chem.AddHs(complete_mol)

    # Embed the molecule to get an initial 3D structure
    AllChem.EmbedMolecule(complete_mol)
    
    # Initialize a new conformation object for the complete molecule
    conf = complete_mol.GetConformer()

    for frag in fragments:
        # Ensure the fragment is in 3D and has Hs
        frag_conf = frag.GetConformer()
        frag_coords = np.array(frag_conf.GetPositions())

        # Find the maximum common substructure between the fragment and the complete molecule
        mcs = FindMCS([complete_mol, frag])
        mcs_smarts = mcs.smartsString

        mcs_mol = Chem.MolFromSmarts(mcs_smarts)

        # Get the atom mappings
        complete_match = complete_mol.GetSubstructMatch(mcs_mol)
        frag_match = frag.GetSubstructMatch(mcs_mol)

        # Assign coordinates from the fragment to the complete molecule
        for c_atom_idx, f_atom_idx in zip(complete_match, frag_match):
            f_pos = frag_conf.GetAtomPosition(f_atom_idx)
            conf.SetAtomPosition(c_atom_idx, f_pos)

    # Set the updated conformation to the complete molecule
    complete_mol.AddConformer(conf, assignId=True)
    complete_mol = Chem.RemoveHs(complete_mol)
    complete_mol = Chem.AddHs(complete_mol, addCoords=True)
    return complete_mol

def merge_molecules(mol1, mol2, merge_distance=0.2):
    # Create a copy of the input molecules to avoid modifying them
    merged_mol1 = Chem.Mol(mol1)
    merged_mol2 = Chem.Mol(mol2)
    
    # Get the positions of atoms in the merged molecules
    mol1_positions = merged_mol1.GetConformer().GetPositions()
    mol2_positions = merged_mol2.GetConformer().GetPositions()
                
    merged_mol2 = clear_atoms_and_bonds_with_type(merged_mol2, "*")
    
    mol2_positions = merged_mol2.GetConformer().GetPositions()
    
    # Create a new molecule to store the merged result
    merged_molecule = Chem.EditableMol(Chem.CombineMols(merged_mol1, merged_mol2))
    # Loop through the atoms in the first molecule
    merge_index = [None, None]
    merge_index_dist = [1e8, 1e8]
    for idx, atom1 in enumerate(merged_mol1.GetAtoms()):
        pos1 = mol1_positions[atom1.GetIdx()]
        for idy, atom2 in enumerate(merged_mol2.GetAtoms()):
            pos2 = mol2_positions[atom2.GetIdx()]
            
            # Calculate the distance between the two atoms
            distance = np.sqrt(np.sum((pos1 - pos2)**2))
            #if idy == 0:
            #    print("chekc distance: ", distance)
            if merge_index_dist[0] > distance and merge_index_dist[1] > distance:
                merge_index[0] = [(idx, atom1), (idy, atom2)]
                merge_index_dist[0] = distance
            elif merge_index_dist[1] > distance:
                merge_index[1] = [(idx, atom1), (idy, atom2)]
                merge_index_dist[1] = distance
    
    #print("check merge index: ", merge_index)
    #assert len(merge_index) == 2
    if len(merge_index[0][1][1].GetBonds()) == 1:
        merge_index = [merge_index[1], merge_index[0]]
    
    cached = copy.deepcopy(merged_molecule.GetMol())
    
    merged_molecule.AddBond(merge_index[0][1][0] + idx + 1, merge_index[1][0][0],order=Chem.rdchem.BondType.SINGLE)
    merged_molecule.RemoveAtom(merge_index[0][0][0])
    merged_molecule.RemoveAtom(merge_index[1][1][0] + idx)
    
    merged_molecule = merged_molecule.GetMol()
    # Remove any disconnected fragments from the merged molecule
    merged_molecule = remove_hydrogen_on_tetrahedral_nitrogen(merged_molecule)
    merged_molecule = Chem.RemoveHs(merged_molecule)
    #merged_molecule = Chem.RemoveHs(Chem.GetMolFrags(merged_molecule, asMols=True)[0])
    

    return merged_molecule

def check_idx(merged_molecule, mols, merge_distance=0.4):
    #collect index of the 2 mols in the merged mol
    merged_position = merged_molecule.GetConformer().GetPositions()
    mols_idx = [[] for _ in range(len(mols))]
    for idx, pos in enumerate(merged_position):
        for i, mol in enumerate(mols):
            mol_position = mol.GetConformer().GetPositions()
            for atom in mol.GetAtoms():
                if np.linalg.norm(pos - mol_position[atom.GetIdx()]) < merge_distance:
                    mols_idx[i].append(idx)
    return mols_idx

def remove_hydrogen_on_tetrahedral_nitrogen(molecule):
    """
    Remove a hydrogen atom from a tetrahedral nitrogen atom (N) in an RDKit molecule if present.

    Args:
        molecule (Chem.Mol): The RDKit molecule.

    Returns:
        Chem.Mol: The modified RDKit molecule with the hydrogen removed.
    """
    modified_molecule = Chem.Mol(molecule)
    
    # Iterate through the atoms in the molecule
    for atom in modified_molecule.GetAtoms():
        if atom.GetAtomicNum() == 7:  # Check if the atom is nitrogen (N)
            if len(atom.GetBonds()) == 4:  # Check if the nitrogen has 4 bonds (tetrahedral)
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1:  # Check if the neighbor is a hydrogen atom
                        # Remove the hydrogen atom
                        modified_molecule.RemoveAtom(neighbor.GetIdx())
                        break  # Remove only one hydrogen atom if there are multiple
    
    return modified_molecule

def read_sdf(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file)
    return next(suppl)

def write_sdf(mol, output_path):
    writer = Chem.SDWriter(output_path)
    writer.write(mol)
    writer.close()


def constrained_FF_min(mol, fix_idx):
    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s'))
    ff.Initialize()
    for idx in range(mol.GetNumAtoms()):
        if idx in fix_idx:
            ff.AddFixedPoint(idx)
    ff.Minimize()
    return mol

def ensemble_all(linker_file, e3_lig_file, poi_lig_file, poi_file, e3_file,
                 poi_org_file, e3_org_file, output_path=None, check_distance=0.1, ref_smiles=None):
    linker_mol, e3_lig, poi_lig = read_sdf(linker_file), read_sdf(e3_lig_file), read_sdf(poi_lig_file)
    #print("checky sep smiles: ", Chem.MolToSmiles(linker_mol), Chem.MolToSmiles(e3_lig), Chem.MolToSmiles(poi_lig))
    poi_org_pos, poi_pos = extract_CA_coords(poi_org_file), extract_CA_coords(poi_file)
    e3_org_pos, e3_pos = extract_CA_coords(e3_org_file), extract_CA_coords(e3_file)
    
    poi_anchor, poi_linker_anchor = get_anchor(poi_lig, linker_mol, check_distance)
    e3_anchor, e3_linker_anchor = get_anchor(e3_lig, linker_mol, check_distance)
    
    
    poi_lig, e3_lig = align_warheads(poi_org_pos, poi_pos, poi_lig,
                                     e3_org_pos, e3_pos, e3_lig)
    
    key_points = np.concatenate([poi_lig.GetConformer().GetPositions()[poi_anchor],
                                 e3_lig.GetConformer().GetPositions()[e3_anchor]], axis=0)
    
    linker_opt, fun, opt_key_points = optimize_linker(linker_mol, torch.tensor(key_points).float(),
                                                      torch.tensor(poi_linker_anchor + e3_linker_anchor), 
                                                      return_mol=True, backbone=None)#torch.tensor(e3_ca_pos).float()
    
    #merged_mol = merge_molecules(linker_opt, poi_lig)
    #merged_mol = merge_molecules(merged_mol, e3_lig)
    if ref_smiles is not None:
        merged_mol = assign_3d_conformation_from_fragments([poi_lig, e3_lig, linker_opt], ref_smiles)
    
    #poi_idx, linker_idx, e3_idx = check_idx(merged_mol, [poi_lig, linker_opt, e3_lig])
    
    #merged_mol = constrained_FF_min(merged_mol, poi_idx + e3_idx)
    if output_path is not None:
        write_sdf(linker_opt, output_path + '_linker.sdf')
        write_sdf(poi_lig, output_path + '_poi_lig.sdf')
        write_sdf(e3_lig, output_path + '_e3_lig.sdf')
        write_sdf(merged_mol, output_path + '_merged.sdf')
    
    return {'linker': linker_opt, 'poi_lig': poi_lig, 'e3_lig': e3_lig, 'rmsd': fun.item()}

if __name__ == '__main__':
    '''
    OUTPUT_file = '/storage/boqiang/proflow/lightning_logs/protac_diffusion_test/'
    ROOT_file = '/data/rsg/chemistry/boqiang/protac_test_data/proflow_temp/'
    ensemble_all(ROOT_file + '7Q2J_linker.sdf',
                 ROOT_file + '7Q2J_e3_lig.sdf',
                ROOT_file + '7Q2J_poi_lig.sdf',
                OUTPUT_file + '7Q2J_poi_vis_113.pdb',
                OUTPUT_file + '7Q2J_e3_vis_113.pdb',
                ROOT_file + '7Q2J_poi.pdb',
                ROOT_file + '7Q2J_e3.pdb', 'matched_7Q2J_113',
                ref_smiles='Cc1c(scn1)c2ccc(cc2)CNC(=O)[C@@H]3C[C@H](CN3C(=O)[C@H](C(C)(C)C)NC(=O)CCCCNC(=O)c4ccc(cc4)c5ccc(c(c5)NC(=O)C6=CNC(=O)C=C6C(F)(F)F)N7CCN(CC7)C)O'
            )
    '''
    
    '''
    
    OUTPUT_file = '/storage/boqiang/case_study_AFM/NC_paper/data/'
    ROOT_file = '/storage/boqiang/case_study_AFM/7z76_e3/'
    afm_files = [os.path.join(ROOT_file, f) for f in os.listdir(ROOT_file) if f.startswith('rank') and f.endswith('.pdb')]
    afm_files = [f for f in afm_files if 'e3' not in f.split('/')[-1]]
    afm_files = [f for f in afm_files if 'poi' not in f.split('/')[-1]]
    rmsd_data = np.zeros((len(afm_files), 15))
    for f in tqdm.tqdm(afm_files):
        split_pdb(f, f + '_e3.pdb', f + '_poi.pdb')
        for i in range(1, 16):
            rmsd = ensemble_all(OUTPUT_file + f'COMPOUND{i}_linker.sdf',
                         OUTPUT_file + f'COMPOUND{i}_e3_lig.sdf',
                         OUTPUT_file + f'COMPOUND{i}_poi_lig.sdf',
                         f + '_poi.pdb',
                         f + '_e3.pdb',
                         OUTPUT_file + f'COMPOUND{i}_poi.pdb',
                         OUTPUT_file + f'COMPOUND{i}_e3.pdb', f + f'_matched_COMPOUND{i}',
                         1000)['rmsd']
        
            rmsd_data[afm_files.index(f), i-1] = rmsd
    
    np.save(ROOT_file + '_rmsd.npy', rmsd_data)
    '''
    
    ROOT_file = '/storage/boqiang/MD_5t35/randomize_input'
    GT_file = '/data/rsg/chemistry/boqiang/protac_test_data/proflow_temp/'
    '''
    for i in tqdm.tqdm(range(100)):
        ensemble_all(os.path.join(GT_file, f'5T35_linker.sdf'),
                     os.path.join(GT_file, f'5T35_e3_lig.sdf'),
                     os.path.join(GT_file, f'5T35_poi_lig.sdf'),
                     os.path.join(ROOT_file, f'5T35_poi_vis_{i}.pdb'),
                     os.path.join(ROOT_file, f'5T35_e3_vis_{i}.pdb'), 
                     os.path.join(GT_file, f'5T35_poi.pdb'),
                     os.path.join(GT_file, f'5T35_e3.pdb'),
                     os.path.join(ROOT_file, f'5T35_{i}_matched'),
                     ref_smiles='Cc1c(sc-2c1C(=N[C@H](c3n2c(nn3)C)CC(=O)NCCOCCOCCOCC(=O)N[C@H](C(=O)N4C[C@@H](C[C@H]4C(=O)NCc5ccc(cc5)C6=C(NCS6)C)O)C(C)(C)C)c7ccc(cc7)Cl)C'
                    )
    '''
    ensemble_all(os.path.join(GT_file, f'5T35_linker.sdf'),
                     os.path.join(GT_file, f'5T35_e3_lig.sdf'),
                     os.path.join(GT_file, f'5T35_poi_lig.sdf'),
                     os.path.join(GT_file, f'5T35_poi.pdb'),
                     os.path.join(GT_file, f'5T35_e3.pdb'),
                     os.path.join(GT_file, f'5T35_poi.pdb'),
                     os.path.join(GT_file, f'5T35_e3.pdb'),
                     os.path.join('.', f'5T35_00_matched'),
                     ref_smiles='Cc1c(sc-2c1C(=N[C@H](c3n2c(nn3)C)CC(=O)NCCOCCOCCOCC(=O)N[C@H](C(=O)N4C[C@@H](C[C@H]4C(=O)NCc5ccc(cc5)C6=C(NCS6)C)O)C(C)(C)C)c7ccc(cc7)Cl)C'
                    )
                         