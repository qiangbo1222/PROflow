import os
import tqdm
import sys
sys.path.append('..')
import warnings
import pandas as pd

# Disable PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import multiprocessing as mp

import prody
prody.confProDy(verbosity='none')

from dataset.process_mols import mol_to_graph, read_molecule, generate_conformer
from dataset.linker_matching import *
from utils.visualize import transform_pdb, transform_sdf, extract_io_coords
from utils.postprocess import align_warheads
from utils.geometry import rigid_transform_Kabsch_3D_torch, matrix_to_axis_angle, axis_angle_to_matrix, transform_norm_vector

def generate_protac_three_body(linker_smi, poi_lig_sdf, e3_lig_sdf, 
                               poi_pdb, e3_pdb,
                               anchor_dic, output_dir, prefix, num=1):
    
    poi_lig_mol = read_molecule(poi_lig_sdf)
    e3_lig_mol = read_molecule(e3_lig_sdf)
    
    # generate linker conformation
    linker_mol = Chem.MolFromSmiles(linker_smi)
    linker_mol_ = generate_conformer(linker_mol, n_conformers=num)
    
    success = 0
    for i, conf in enumerate(linker_mol_.GetConformers()):
        linker_mol = copy.deepcopy(linker_mol_)
        linker_mol.RemoveAllConformers()
        linker_mol.AddConformer(conf)
    
        linker_conf = np.array(linker_mol.GetConformer().GetPositions())
        e3_conf = np.array(e3_lig_mol.GetConformer().GetPositions())
        poi_conf = np.array(poi_lig_mol.GetConformer().GetPositions())
        
        linker_e3_anchor, linker_poi_anchor = anchor_dic['linker_e3'], anchor_dic['linker_poi']
        linker_e3_anchor_pos = linker_conf[linker_e3_anchor]
        linker_poi_anchor_pos = linker_conf[linker_poi_anchor]
        
        e3_anchor, poi_anchor = anchor_dic['e3'], anchor_dic['poi']
        e3_anchor_pos = e3_conf[e3_anchor][::-1].copy()
        poi_anchor_pos = poi_conf[poi_anchor][::-1].copy()
        
        #transform sdf to fit

        R, t = rigid_transform_Kabsch_3D_torch(torch.tensor(poi_anchor_pos).T.float(), 
                                            torch.tensor(linker_poi_anchor_pos).T.float())

        t = change_R_T_ref(t.T, R, poi_anchor_pos, poi_conf)
        transform_sdf(poi_lig_sdf, t, R, os.path.join(output_dir, f'{prefix}_{i}_poi_lig.sdf'))
        
        
        R, t = rigid_transform_Kabsch_3D_torch(torch.tensor(e3_anchor_pos).T.float(), 
                                            torch.tensor(linker_e3_anchor_pos).T.float())
        t = change_R_T_ref(t.T, R, e3_anchor_pos, e3_conf)
        transform_sdf(e3_lig_sdf, t, R, os.path.join(output_dir, f'{prefix}_{i}_e3_lig.sdf'))
        

        
        #transform pdb to fit
        e3_pdb_coord = extract_io_coords(e3_pdb)
        poi_pdb_coord = extract_io_coords(poi_pdb)
        
        R, t = rigid_transform_Kabsch_3D_torch(torch.tensor(e3_anchor_pos).T.float(), 
                                            torch.tensor(linker_e3_anchor_pos).T.float())
        t = change_R_T_ref(t.T, R, e3_anchor_pos, e3_pdb_coord)
        e3_new_pos = torch.matmul(e3_pdb_coord - torch.mean(e3_pdb_coord, dim=0), R.T) + torch.mean(e3_pdb_coord, dim=0) + t
        transform_pdb(e3_pdb, os.path.join(output_dir, f'{prefix}_{i}_e3.pdb'), rot_mat=R, trans=t.T)
        
        R, t = rigid_transform_Kabsch_3D_torch(torch.tensor(poi_anchor_pos).T.float(), 
                                            torch.tensor(linker_poi_anchor_pos).T.float())
        t = change_R_T_ref(t, R, poi_anchor_pos, poi_pdb_coord)
        poi_new_pos = torch.matmul(poi_pdb_coord - torch.mean(poi_pdb_coord, dim=0), R.T) + torch.mean(poi_pdb_coord, dim=0) + t
        
        #get the cross distance (different size) (B, N, 3), (B, M, 3) --> (B, N, M)
        cross_dist = torch.cdist(poi_new_pos, e3_new_pos)
        transform_pdb(poi_pdb, os.path.join(output_dir, f'{prefix}_{i}_poi.pdb'), rot_mat=R, trans=t.T)
        if cross_dist.max() > 2.0:
            print(f'{prefix}_{i} is too close')
            os.remove(os.path.join(output_dir, f'{prefix}_{i}_poi.pdb'))
            os.remove(os.path.join(output_dir, f'{prefix}_{i}_e3.pdb'))
            os.remove(os.path.join(output_dir, f'{prefix}_{i}_poi_lig.sdf'))
            os.remove(os.path.join(output_dir, f'{prefix}_{i}_e3_lig.sdf'))
            continue
        
        
        
        
        #save linker sdf
        linker_writer = Chem.SDWriter(os.path.join(output_dir, f'{prefix}_{i}_linker.sdf'))
        linker_writer.write(linker_mol)
        linker_writer.close()
        success += 1
    print(f'{success} out of {num} are successful')