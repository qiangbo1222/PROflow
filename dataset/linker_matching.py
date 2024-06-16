import torch
import rdkit.ForceField.rdForceField as rdFF
from scipy.optimize import direct

from dataset.conformer_matching import *
from dataset.linker_distribution import *
from utils.diffusion_utils import modify_conformer
from utils.geometry import rigid_transform_Kabsch_3D_torch, matrix_to_axis_angle, change_R_T_ref, transform_norm_vector
from utils.visualize import transform_sdf

def optimize_linker(mol, keypoints, anchor_idx, popsize=20, maxiter=200, FF_iter=200,
                             mutation=(0.5, 1), recombination=0.8, workers=1, seed=0, return_mol=False, linker_idx=None,  backbone=None):
    org_mol = copy.deepcopy(mol)
    if linker_idx is not None:
        anchor_idx = [linker_idx[i] for i in anchor_idx]
        linker_idx = [i for i in linker_idx if i not in anchor_idx]
    opt = OptimizeLinker(mol, keypoints, anchor_idx, seed=seed, linker_idx=linker_idx, backbone=backbone, FF_iter=FF_iter)
    tor_num = get_torsion_angles(mol, linker_idx)
    if len(tor_num) == 0:#no torsion angles in linker
        opt_keypoints = torch.tensor(mol.GetConformer().GetPositions()).to(keypoints.device).float()[anchor_idx]
        opt_inter = torch.cat([interpolate_points(opt_keypoints[:2]), opt_keypoints[[2]]], dim=0)
        key_inter = torch.cat([interpolate_points(keypoints[:2]), keypoints[[2]]], dim=0)
        R, t = rigid_transform_Kabsch_3D_torch(opt_inter.T, key_inter.T)
        t = change_R_T_ref(t.T, R, opt_inter, opt_keypoints)
        opt_keypoints = (opt_keypoints - torch.mean(opt_keypoints, dim=0)) @ R.T + t + torch.mean(opt_keypoints, dim=0)
        if return_mol:
            return mol, 0, opt_keypoints
        else:
            return opt_keypoints
        
    max_bound = [np.pi] * len(tor_num)
    min_bound = [-np.pi] * len(tor_num)
    bounds = (min_bound, max_bound)
    bounds = list(zip(bounds[0], bounds[1]))
    
    if backbone is not None:
        atol = 1.0
    else:
        atol = 0.5
    
    if backbone is None:
        result = differential_evolution(opt.score_conformation, bounds,
                                        maxiter=maxiter, popsize=popsize,
                                        mutation=mutation, recombination=recombination, disp=False, seed=seed, workers=workers, atol=atol)
    else:
        result = direct(opt.score_conformation, bounds, maxiter=maxiter, f_min=atol)

    mol = opt.best_mol[0]
    mol_conf = torch.tensor(mol.GetConformer().GetPositions()).to(keypoints.device).float()
    opt_keypoints = copy.deepcopy(mol_conf[anchor_idx])
    opt_inter = torch.cat([interpolate_points(opt_keypoints[:2]), opt_keypoints[[2]]], dim=0)
    key_inter = torch.cat([interpolate_points(keypoints[:2]), keypoints[[2]]], dim=0)
    R, t = rigid_transform_Kabsch_3D_torch(opt_inter.T, key_inter.T)
    t = change_R_T_ref(t.T, R, opt_inter, opt_keypoints)
    t_mol = change_R_T_ref(t, R, opt_keypoints, mol_conf)
    opt_keypoints = (opt_keypoints - torch.mean(opt_keypoints, dim=0)) @ R.T + t + torch.mean(opt_keypoints, dim=0)
    #print("key points rmsd: ", kabsch_rmsd(opt_keypoints[:2], keypoints[:2], align=False)) 
    mol = transform_sdf(mol, t_mol, R)
    #print("checky: ", opt_keypoints[:2], torch.tensor(mol.GetConformer().GetPositions()).to(keypoints.device).float()[anchor_idx][:2])
    rmsd_match = kabsch_rmsd(torch.tensor(mol.GetConformer().GetPositions()).to(keypoints.device).float()[anchor_idx][2:], keypoints[2:], align=False)
    #print("match rmsd e3 : ", rmsd_match)
    
    

    if return_mol:
        return mol, rmsd_match, opt_keypoints
    else:
        return opt_keypoints

def interpolate_points(line_p, alpha=20):
    #interpolate points on a line 
    #line_p: [2, 3]
    #output: [alpha, 3]
    line_out = []
    for i in range(alpha):
        line_out.append(line_p[0] * (alpha - i) / alpha + line_p[1] * i / alpha)
    return torch.stack(line_out, dim=0)



class OptimizeLinker:
    def __init__(self, mol, key_points, anchor_idx, seed=None, linker_idx=None, backbone=None, FF_iter=100):
        super(OptimizeLinker, self).__init__()
        if seed:
            np.random.seed(seed)
        self.mol = mol
        self.key_points = key_points
        self.linker_idx = linker_idx
        self.anchor_idx = anchor_idx
        self.backbone = backbone
        self.FF_iter = FF_iter
        self.best_mol = [mol, 1e8]
        self.call_time = 0
    
    def optimize_mff(self, mol):
        org_conf = np.array(mol.GetConformer().GetPositions())
        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s'))
        ff.Initialize()
        if self.linker_idx is not None:
            for idx in range(mol.GetNumAtoms()):
                if idx not in self.linker_idx:
                    ff.AddFixedPoint(idx)
        ff.Minimize()
        #new_conf = np.array(mol.GetConformer().GetPositions())
        #print("conf diff: ", np.sqrt(np.sum((org_conf - new_conf) ** 2, axis=-1)))
        

    def score_conformation(self, values):
        self.mol = set_torsion_angles(self.mol, values, self.linker_idx)
        #MFF optimize
        if self.FF_iter > 0:
            AllChem.MMFFOptimizeMolecule(self.mol, maxIters=self.FF_iter, nonBondedThresh=3.)
        ref_points = torch.tensor(self.mol.GetConformer().GetPositions()).to(self.key_points.device).float()
        ref_points = ref_points[self.anchor_idx]
        ref_inter = torch.cat([interpolate_points(ref_points[:2]), ref_points[[2]]], dim=0)
        key_inter = torch.cat([interpolate_points(self.key_points[:2]), self.key_points[[2]]], dim=0)
        R, t = rigid_transform_Kabsch_3D_torch(ref_inter.T, key_inter.T)
        t = change_R_T_ref(t.T, R, ref_inter, ref_points)
        ref_points_mod = (ref_points - torch.mean(ref_points, dim=0)) @ R.T + t + torch.mean(ref_points, dim=0)
        rmsd_l = kabsch_rmsd(ref_points_mod[2:], self.key_points[2:], align=False)
        
        if self.backbone is not None:
            R, t = transform_norm_vector(self.key_points[2:], ref_points_mod[2:])
            t = change_R_T_ref(t, R, self.key_points[[2]], self.backbone)
            backbone_mod = (self.backbone - torch.mean(self.backbone, dim=0)) @ R.T + t + torch.mean(self.backbone, dim=0)
            rmsd_l = kabsch_rmsd(backbone_mod, self.backbone, align=False)
        
        if rmsd_l < self.best_mol[1]:
            self.best_mol = [copy.deepcopy(self.mol), rmsd_l]
            
        self.call_time += 1
        #print("call time: ", self.call_time, "rmsd: ", self.best_mol[1])
        return rmsd_l


def kabsch_rmsd(points_A, points_B, align=True):
    if align:
        R, t = rigid_transform_Kabsch_3D_torch(points_A.T, points_B.T)
        points_A = (points_A - torch.mean(points_A, dim=0)) @ R.T + t.T + torch.mean(points_A, dim=0)
    return torch.sqrt(torch.mean((points_A - points_B) ** 2))

def match_complex(data, FF_iter=100):
    org_key_points = torch.cat([data.ref_linker_pos, data.linker_pos], dim=0)
    new_key_points = optimize_linker(data.linker_mol,
                                    org_key_points,
                                    data.linker_cache["anchor_idx"],
                                    backbone=data['ligand'].c_alpha_coords,
                                    FF_iter=FF_iter)
                                    
    
    #rmsd_match = torch.sqrt(torch.mean((new_key_points[2:] - org_key_points[2:]) ** 2))
    rot_mod, tr_mod = transform_norm_vector(org_key_points[2:], new_key_points[2:])
    #print("checky mod: ", rot_mod, tr_mod)

    tr_surface = change_R_T_ref(tr_mod, rot_mod, org_key_points[[2]], data['ligand'].pos)
    tr_backbone = change_R_T_ref(tr_mod, rot_mod, org_key_points[[2]], data['ligand'].c_alpha_coords)
    tr_linker = change_R_T_ref(tr_mod, rot_mod, org_key_points[[2]], data.linker_pos)

    rot_mod = matrix_to_axis_angle(rot_mod)
    modify_conformer(data, rot_mod, tr_update_surface=tr_surface, 
                     tr_update_backbone=tr_backbone, tr_update_linker=tr_linker, has_norm=True)
    return data, tr_backbone, rot_mod