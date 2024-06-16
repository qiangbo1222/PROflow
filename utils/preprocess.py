import os
import tqdm
import sys
sys.path.append('..')
import warnings

# Disable PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import multiprocessing as mp

import prody
prody.confProDy(verbosity='none')

from dataset.process_mols import mol_to_graph, read_molecule
from dataset.linker_matching import *
from utils.visualize import transform_pdb, transform_sdf, extract_io_coords
from utils.postprocess import align_warheads
from utils.geometry import rigid_transform_Kabsch_3D_torch, matrix_to_axis_angle, axis_angle_to_matrix, transform_norm_vector

FFTW_PATH = '/data/rsg/chemistry/boqiang/bin/fftw-3.3.9/install_bin/lib'
HDock_PATH = '/data/rsg/chemistry/boqiang/bin/HDOCKlite-v1.1/hdock'
Createpl_PATH = '/data/rsg/chemistry/boqiang/bin/HDOCKlite-v1.1/createpl'

DB_csv_file = '/Mounts/rbg-storage1/users/urop/qiangbo/data/warheads/protac_db'
OUTPUT_PATH = '/data/rsg/chemistry/boqiang/protac_test_data/proflow_temp_hdock'


#input protac smiles; docked warheads-protein

def distance(p1, p2):
    if p1.endswith('.sdf'):
        p1 = read_molecule(p1)
        p1 = np.array(p1.GetConformer().GetPositions())
    elif p1.endswith('.pdb'):
        p1 = prody.parsePDB(p1)
        p1 = np.array(p1.getCoords())
    
    if p2.endswith('.sdf'):
        p2 = read_molecule(p2)
        p2 = np.array(p2.GetConformer().GetPositions())
    elif p2.endswith('.pdb'):
        p2 = prody.parsePDB(p2)
        p2 = np.array(p2.getCoords())
    
    #min cross distance
    return np.min(np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=-1), axis=1)


def Hdock(pdb_poi, pdb_e3, sdf_poi, sdf_e3, tmp_path):
    
    #some case pdb and sdf are reversed or too far away, filter out these cases
    poi_lig_e3_pdb_dist = distance(sdf_poi, pdb_e3).sum()
    poi_lig_poi_pdb_dist = distance(sdf_poi, pdb_poi).sum()
    e3_lig_poi_pdb_dist = distance(sdf_e3, pdb_poi).sum()
    e3_lig_e3_pdb_dist = distance(sdf_e3, pdb_e3).sum()
    
    #if poi_lig_e3_pdb_dist < poi_lig_poi_pdb_dist and e3_lig_poi_pdb_dist < e3_lig_e3_pdb_dist:
    #    #reverse the order
    #    sdf_poi, sdf_e3 = sdf_e3, sdf_poi
    
    poi_lig_poi_pdb_dist = distance(sdf_poi, pdb_poi)
    e3_lig_e3_pdb_dist = distance(sdf_e3, pdb_e3)
    
    
    if poi_lig_poi_pdb_dist.min() > 6.0 or e3_lig_e3_pdb_dist.min() > 6.0:
        print("too far: ", poi_lig_poi_pdb_dist.min(), e3_lig_e3_pdb_dist.min())
        return None
    
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    results_len = len([os.path.join(tmp_path, p) for p in os.listdir(tmp_path) if p.endswith('.pdb') and p.startswith('model')])
    if results_len == 0:
        os.system(f'cp {pdb_poi} {os.path.join(tmp_path, "receptor.pdb")}')
        os.system(f'cp {pdb_e3} {os.path.join(tmp_path, "ligand.pdb")}')
        os.system(f'cp {sdf_poi} {os.path.join(tmp_path, "receptor.sdf")}')
        os.system(f'cp {sdf_e3} {os.path.join(tmp_path, "ligand.sdf")}')
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        identify_binding_pocket('receptor.pdb', 'receptor.sdf', 'rec_param.txt')
        identify_binding_pocket('ligand.pdb', 'ligand.sdf', 'lig_param.txt')
        os.system(f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{FFTW_PATH}; {HDock_PATH} receptor.pdb ligand.pdb -out Hdock.out -rsite rec_param.txt -lsite lig_param.txt")
        os.system(f"{Createpl_PATH} Hdock.out top100.pdb -nmax 100 -models")
        os.chdir(original_dir)
    
    return [os.path.join(tmp_path, p) for p in os.listdir(tmp_path) if p.endswith('.pdb') and p.startswith('model')]

def identify_binding_pocket(protein_pdb, ligand_sdf, param_file, distance_cutoff=6.0):
    # Load protein structure
    protein_structure = prody.parsePDB(protein_pdb)

    # Load ligand molecule from SDF file
    ligand_molecule_supplier = Chem.SDMolSupplier(ligand_sdf)
    ligand_molecule = next(ligand_molecule_supplier)
    
    # Get ligand coordinates
    ligand_coords = ligand_molecule.GetConformer().GetPositions()

    distances = np.linalg.norm(protein_structure.getCoords()[:, None, :] - ligand_coords[None, :, :], axis=-1)

    # Find the minimum distance for each protein atom
    min_distances = np.min(distances, axis=1)

   # Select residues within a specified distance from the ligand
    binding_residues_indices = np.where(min_distances < distance_cutoff)[0]
    binding_residues = protein_structure[binding_residues_indices]

    # Extract information about binding residues
    binding_residue_info = []
    for residue in binding_residues.getResnums():
        chain_id = binding_residues.select(f'resnum {residue}').getChids()[0]
        binding_residue_info.append(f'{residue}:{chain_id}')
    
    binding_residue_info = list(set(binding_residue_info))

    # Create a formatted string
    binding_pocket_str = ', '.join(binding_residue_info)

    param_file = open(param_file, 'w')
    param_file.write(binding_pocket_str)
    

def ret_anchor_idx(linker_mol, poi_lig_mol, e3_lig_mol):
    linker_conf = torch.tensor(np.array(linker_mol.GetConformer().GetPositions()))
    e3_lig_conf = torch.tensor(np.array(e3_lig_mol.GetConformer().GetPositions()))
    poi_lig_conf = torch.tensor(np.array(poi_lig_mol.GetConformer().GetPositions()))

    #find the 2 closest atom to the linker
    e3_linker_dist = torch.cdist(linker_conf, e3_lig_conf)
    e3_min_dist, e3_anchor_i = torch.topk(torch.min(e3_linker_dist, dim=1)[0], k=2, largest=False)
    e3_anchor_j = [torch.argmin(e3_linker_dist[e3_anchor_i[0]]), torch.argmin(e3_linker_dist[e3_anchor_i[1]])]
    if e3_min_dist.max() > 1.:
        print("got error fitting e3: ", e3_min_dist.max())
        raise ValueError
    e3_anchor = linker_conf[e3_anchor_i]

    poi_linker_dist = torch.cdist(linker_conf, poi_lig_conf)
    poi_min_dist, poi_anchor_i = torch.topk(torch.min(poi_linker_dist, dim=1)[0], k=2, largest=False)
    poi_anchor_j = [torch.argmin(poi_linker_dist[poi_anchor_i[0]]), torch.argmin(poi_linker_dist[poi_anchor_i[1]])]
    if poi_min_dist.max() > 1.:
        print("got error fitting poi: ", poi_min_dist.max())
        raise ValueError
    poi_anchor = linker_conf[poi_anchor_i]

    inner_dist_e3 = mol_to_graph(e3_lig_mol)
    inner_dist_poi = mol_to_graph(poi_lig_mol)
    inner_dist_e3 = np.sum(inner_dist_e3, axis=-1)
    inner_dist_poi = np.sum(inner_dist_poi, axis=-1)

    #assign order to the anchors
    if inner_dist_e3[e3_anchor_j[0]] < inner_dist_e3[e3_anchor_j[1]]:
        e3_anchor = e3_anchor[[1, 0]]
        e3_anchor_i = e3_anchor_i[[1, 0]]
        e3_anchor_j = [e3_anchor_j[1], e3_anchor_j[0]]
    if inner_dist_poi[poi_anchor_j[0]] < inner_dist_poi[poi_anchor_j[1]]:
        poi_anchor = poi_anchor[[1, 0]]
        poi_anchor_i = poi_anchor_i[[1, 0]]
        poi_anchor_j = [poi_anchor_j[1], poi_anchor_j[0]]
    return {"poi": [poi_anchor_i, poi_anchor_j, poi_anchor], "e3": [e3_anchor_i, e3_anchor_j, e3_anchor]}


def Hdock_screen(docked_pdbs, ref_pdb_poi, ref_pdb_e3, ref_sdf_poi, ref_sdf_e3, ref_sdf_linker, output_path, prefix):
    #screen docked pdbs
    poi_lig_mol, e3_lig_mol, linker_mol = read_molecule(ref_sdf_poi), read_molecule(ref_sdf_e3), read_molecule(ref_sdf_linker)
    try:
        anchor_result = ret_anchor_idx(linker_mol, poi_lig_mol, e3_lig_mol)
    except:
        print("anchor fitting error")
        return
    anchor_linker = torch.cat([anchor_result['poi'][0], anchor_result['e3'][0]], dim=0)
   
    poi_org, poi_new = extract_io_coords(ref_pdb_poi), extract_io_coords(ref_pdb_poi)
    e3_org, e3_docked = extract_io_coords(ref_pdb_e3), [extract_io_coords(pdb) for pdb in docked_pdbs]
    
    rmsd_results = [1e5, None, None, 1e5]
    closest_idx = None
    for i, e3_new in tqdm.tqdm(enumerate(e3_docked), desc='screening HDock Results'):
        _, e3_ligs_docked = align_warheads(poi_org, poi_new, poi_lig_mol, e3_org, e3_new, e3_lig_mol)
        e3_anchor_docked = torch.tensor(np.array(e3_ligs_docked.GetConformer().GetPositions()))[[anchor_result['e3'][1]]]
        key_points = torch.cat([anchor_result['poi'][2], e3_anchor_docked], dim=0).float()
        linker_opt, fun, opt_key_points = optimize_linker(linker_mol, key_points, anchor_linker, return_mol=True, backbone=e3_new)
        #print("fun: ", fun)
        #TODO also screen conflict with poi
        R, t = transform_norm_vector(anchor_result['e3'][2], opt_key_points[2:])
        t_prot = change_R_T_ref(t, R, torch.tensor(anchor_result['e3'][2][:1]).float(), torch.tensor(e3_org).float())
        e3_prot_coord = (e3_org - torch.mean(e3_org, dim=0)) @ R.T + t_prot + torch.mean(e3_org, dim=0)
        R, t = transform_norm_vector(anchor_result['poi'][2], opt_key_points[:2])
        t_prot = change_R_T_ref(t, R, torch.tensor(anchor_result['poi'][2][:1]).float(), torch.tensor(poi_org).float())
        poi_prot_coord = (poi_org - torch.mean(poi_org, dim=0)) @ R.T + t_prot + torch.mean(poi_org, dim=0)
        conflict_num = torch.sum(torch.cdist(poi_prot_coord, e3_prot_coord) < 2.0)
        conflict_num = max(conflict_num.item(), 20)
        if conflict_num < rmsd_results[3]:
            if fun < rmsd_results[0]:
                rmsd_results = [fun, linker_opt, opt_key_points, conflict_num]
                closest_idx = i
    #print("best rmsd: ", rmsd_results[0], "  best idx: ", docked_pdbs[closest_idx])
    
    rmsd_results[2] = torch.tensor(rmsd_results[1].GetConformer().GetPositions())[[anchor_linker]]
    R, t = transform_norm_vector(anchor_result['e3'][2], rmsd_results[2][2:])
    t_prot = change_R_T_ref(t, R, torch.tensor(anchor_result['e3'][2][:1]).float(), torch.tensor(e3_org).float())
    transform_pdb(ref_pdb_e3, os.path.join(output_path, f'{prefix}_e3.pdb'), rot_mat=R, trans=t_prot.T)
    t_lig = change_R_T_ref(t, R, torch.tensor(anchor_result['e3'][2][:1]).float(), torch.tensor(np.array(read_molecule(ref_sdf_e3).GetConformer().GetPositions())).float())
    transform_sdf(ref_sdf_e3, t_lig, R, os.path.join(output_path, f'{prefix}_e3_lig.sdf'))
   
    writer = Chem.SDWriter(os.path.join(output_path, f'{prefix}_linker.sdf'))
    writer.write(rmsd_results[1])
    
    R, t = transform_norm_vector(anchor_result['poi'][2], rmsd_results[2][:2])
    t_prot = change_R_T_ref(t.T, R, torch.tensor(anchor_result['poi'][2][:1]).float(), torch.tensor(poi_org).float())
    transform_pdb(ref_pdb_poi, os.path.join(output_path, f'{prefix}_poi.pdb'), rot_mat=R, trans=t_prot.T)
    t_lig = change_R_T_ref(t, R, anchor_result['poi'][2][:1].float(), torch.tensor(np.array(read_molecule(ref_sdf_poi).GetConformer().GetPositions())).float())
    transform_sdf(ref_sdf_poi, t_lig, R, os.path.join(output_path, f'{prefix}_poi_lig.sdf'))
    
def hdock_wrap(args):
    poi_pdb, e3_pdb, poi_lig, e3_lig, linker, tmp_path, prefix = args
    #if os.path.exists(os.path.join(OUTPUT_PATH, f'{prefix}_poi.pdb')):
    #    return
    docked_pdbs = Hdock(poi_pdb, e3_pdb, poi_lig, e3_lig, tmp_path)
    if docked_pdbs is not None:
        Hdock_screen(docked_pdbs, poi_pdb, e3_pdb, poi_lig, e3_lig, linker, OUTPUT_PATH, prefix)
        
if __name__ == '__main__':
    #db_id = [p.split('_')[2] for p in os.listdir(DB_csv_file)]
    #db_id = list(set(db_id))
    #db_id = sorted([int(i) for i in db_id])
    
    
    
    
    # preprocess data for PROTAC DB data
    p = mp.Pool(32)
    '''
    for _ in tqdm.tqdm(p.imap_unordered(hdock_wrap, zip([os.path.join(DB_csv_file, f'protac_db_{i}__poi_prot.pdb') for i in db_id],
                                                        [os.path.join(DB_csv_file, f'protac_db_{i}__e3_prot.pdb') for i in db_id],
                                                        [os.path.join(DB_csv_file, f'protac_db_{i}__poi_lig.sdf') for i in db_id],
                                                        [os.path.join(DB_csv_file, f'protac_db_{i}__e3_lig.sdf') for i in db_id],
                                                        [os.path.join(DB_csv_file, f'protac_db_{i}__linker.sdf') for i in db_id],
                                                        [os.path.join(OUTPUT_PATH, f'protac_{i}_tmp') for i in db_id],
                                                        [f'protacDB-{i}' for i in db_id])), total=len(db_id)):
        pass
    p.close()
    
    '''
    #preprocess PDB test set
    
    DATA_Path = '/data/rsg/chemistry/boqiang/protac_test_data/proflow_temp'
    pdb_id = [p[:4] for p in os.listdir(DATA_Path)]
    pdb_id = list(set(pdb_id))
    
    
    for _ in tqdm.tqdm(p.imap_unordered(hdock_wrap, zip([os.path.join(DATA_Path, f'{i}_poi.pdb') for i in pdb_id],
                                                        [os.path.join(DATA_Path, f'{i}_e3.pdb') for i in pdb_id],
                                                        [os.path.join(DATA_Path, f'{i}_poi_lig.sdf') for i in pdb_id],
                                                        [os.path.join(DATA_Path, f'{i}_e3_lig.sdf') for i in pdb_id],
                                                        [os.path.join(DATA_Path, f'{i}_linker.sdf') for i in pdb_id],
                                                        [os.path.join(OUTPUT_PATH, f'{i}_tmp') for i in pdb_id],
                                                        pdb_id)), total=len(pdb_id), desc='preprocess full test set'):
        pass
    
    p.close()
   