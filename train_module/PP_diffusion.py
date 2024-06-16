from functools import partial
import os

import numpy as np
import pandas as pd
import math
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pytorch_lightning as pl
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.loader import DataLoader
import torch.distributed as dist
import random
import pickle


from rdkit.Geometry import Point3D
import rdkit.Chem as Chem
from rdkit.Chem import AllChem

from model.dynamic_surface_model_PP_patch import TensorProductScoreModel
from data_utils.dips import NoiseTransform_forPP
from utils.diffusion_utils import t_to_sigma, get_t_schedule, modify_conformer, set_time, modify_conformer_torsion
from utils.geometry import  axis_angle_to_matrix, matrix_to_axis_angle, angle_exp, angle_sum, change_R_T_ref, rigid_transform_Kabsch_3D_torch
from utils.visualize import transform_pdb
from dataset.linker_matching import match_complex

class PP_diffusion(pl.LightningModule):
    def __init__(self, args):
        super(PP_diffusion, self).__init__()
        self.args = args
        self.save_hyperparameters()
        self.t_to_sigma_ = partial(t_to_sigma, args=args.t_to_sigma)
        self.full_inference = False
        self.data_root = args.dataset.data_dir
        self.visualize_results = False
        self.confidence_mode = args.model.confidence_mode
        

        args = args.model
        
        #args.confidence_mode = False
        #self.confidence_mode = False
        #TODO remove this after retrain our model
        
        self.score_model = TensorProductScoreModel(t_to_sigma=self.t_to_sigma_,
                                                   time_scale=args.embedding_scale,
                                                    num_conv_layers=args.num_conv_layers,
                                                    scale_by_sigma=args.scale_by_sigma,
                                                    sigma_embed_dim=args.sigma_embed_dim,
                                                    ns=args.ns, nv=args.nv,
                                                    distance_embed_dim=args.distance_embed_dim,
                                                    cross_distance_embed_dim=args.cross_distance_embed_dim,
                                                    dropout=args.dropout,
                                                    cross_max_distance=args.cross_max_distance,
                                                    dynamic_max_cross=args.dynamic_max_cross,
                                                    lm_embedding_type="esm",
                                                    batch_norm=args.batch_norm,
                                                    confidence_mode=args.confidence_mode
                                )
        self.t_schedule = get_t_schedule(inference_steps=self.args.inference.inference_steps)
        self.valid_tobe_inference = []
        self.validation_step_outputs =[]
        self.transform = NoiseTransform_forPP(t_to_sigma=self.t_to_sigma_, linker_dict = pickle.load(open(args.linker_dict, 'rb')),
                            )
        self.confidence_mode = args.confidence_mode
        self.epoch_idx = 0
        self.snr = 500 #TODO move to args
        self.test_sample_num = 3  #TODO move to args
    

    def forward(self, batch, train=False):
        if self.confidence_mode:
            if train:
                energy_pred, batch_loss = self.score_model(batch, train=True)
                return energy_pred, batch_loss
            else:
                energy_pred = self.score_model(batch)
                return energy_pred
        else:
            if train:
                #tr_pred, rot_pred, batch_loss = self.score_model(batch, train=True)
                #return tr_pred, rot_pred, batch_loss
                tr_pred, rot_pred, tr_loss, rot_loss, batch_loss, cfm_loss = self.score_model(batch, train=True)
                return tr_pred, rot_pred, tr_loss, rot_loss, batch_loss, cfm_loss
            else:
                tr_pred, rot_pred = self.score_model(batch)
                return tr_pred, rot_pred
        
    def compute_loss(self, tr_pred, rot_pred, data, 
                        tr_weight=1, rot_weight=1, tor_l_weight=1):
        #check double float
        
        tr_score, rot_score = data.tr_score, data.rot_score
        
        #tr_loss = F.mse_loss(tr_pred, tr_score)
        tr_pred_dir = tr_pred / torch.norm(tr_pred, dim=-1, keepdim=True)
        tr_pred_norm = torch.norm(tr_pred, dim=-1, keepdim=True)
        tr_score_dir = tr_score / torch.norm(tr_score, dim=-1, keepdim=True)
        tr_score_norm = torch.norm(tr_score, dim=-1, keepdim=True)
        tr_norm_loss = F.mse_loss(tr_pred_norm, tr_score_norm)
        tr_dir_loss = (- torch.sum(tr_score_norm * tr_pred_dir * tr_score_dir, dim=-1).mean())
        tr_loss = tr_norm_loss * 0.1 + tr_dir_loss
        #print("tr_loss scale:", ((tr_score_norm**2).mean()), F.mse_loss(tr_pred, tr_score))
        #rot_loss = F.mse_loss(rot_pred, rot_score)
        
        rot_mat = torch.stack([axis_angle_to_matrix(rot_pred[i]) for i in range(len(rot_pred))])
        rot_mat_score = torch.stack([axis_angle_to_matrix(rot_score[i]) for i in range(len(rot_score))])
        
        rot_loss = F.mse_loss(rot_mat, rot_mat_score)

        loss = tr_weight * tr_loss + rot_weight * rot_loss
        #print("checky tr:  ", (- torch.sum(tr_score_norm * tr_pred_dir * tr_score_dir, dim=-1).mean()), ((tr_pred_norm - tr_score_norm) ** 2 / tr_score_norm).mean())
        #print("checky rot: ", rot_loss)
        return loss, tr_loss, rot_loss, tr_norm_loss, tr_dir_loss
    
    def distance_uniform(self, data, width=(-2* np.pi)):
        data = data.view(-1)
        data = torch.sort(data, dim=-1)[0]
        sampled_data = torch.rand_like(data) * width
        sampled_data = torch.sort(sampled_data, dim=-1)[0]
        distance = torch.abs(data - sampled_data)
        return distance.mean()

    def visualize(self, batch_data, sample_step=None):
        log_dir = self.full_inference
        pdb_keys = set([data.name for data in batch_data])
        pdb_keys = {k: 0 for k in pdb_keys}
        for data in batch_data:
            rec_data_dir = os.path.join(self.data_root, data.name + "_poi.pdb")
            lig_data_dir = os.path.join(self.data_root, data.name + "_e3.pdb")

            

            output_rec_dir = os.path.join(log_dir, data.name + f"_poi_vis_{pdb_keys[data.name]}.pdb")
            output_lig_dir = os.path.join(log_dir, data.name + f"_e3_vis_{pdb_keys[data.name]}.pdb")
            if sample_step is not None:
                output_rec_dir = os.path.join(log_dir, data.name + f"_poi_vis_{sample_step}_{pdb_keys[data.name]}.pdb")
                output_lig_dir = os.path.join(log_dir, data.name + f"_e3_vis_{sample_step}_{pdb_keys[data.name]}.pdb")
            pdb_keys[data.name] += 1

            transform_pdb(rec_data_dir, output_rec_dir)
            transform_pdb(lig_data_dir, output_lig_dir, new_pos = data['ligand'].c_alpha_coords.cpu())

    def training_step(self, batch, batch_idx):
        if self.confidence_mode:
            energy_pred, pos_loss = self.forward(batch, train=True)
        else:
            #tr_pred, rot_pred, pos_loss = self.forward(batch, train=True)
            tr_pred, rot_pred, tr_loss, rot_loss, pos_loss, cfm_loss = self.forward(batch, train=True)
        '''
        loss, tr_loss, rot_loss, tr_norm_loss, tr_dir_loss = self.compute_loss(tr_pred, rot_pred,
                                                                batch,
                                                                self.args.model.tr_weight, self.args.model.rot_weight,
                                                                self.args.model.tor_weight)
        bsz = len(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bsz)
        self.log('train_tr_loss', tr_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bsz)
        self.log('train_rot_loss', rot_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bsz)
        self.log('train_tr_norm_loss', tr_norm_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bsz)
        self.log('train_tr_dir_loss', tr_dir_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bsz)
       
        return {
            'loss': loss,
            'tr_loss': tr_loss,
            'rot_loss': rot_loss
        }
        '''
        bsz = len(batch)
        self.log('train_loss', pos_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bsz)
        self.log('train_cfm_loss', cfm_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bsz)
        self.log('train_tr_loss', tr_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bsz)
        self.log('train_rot_loss', rot_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bsz)
        return {'loss': pos_loss ** 2 * 0.4 + cfm_loss}
        #return {'loss': cfm_loss}

    #def on_train_epoch_end(self):
    #    sch = self.trainer.lr_schedulers[0]['scheduler']
    #    sch.step(self.trainer.callback_metrics['train_loss'])


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.optim.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
        scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "train_loss"}

        return [optimizer], [scheduler]

    
    def validation_step(self, batch, batch_idx):
        if self.confidence_mode:
            energy_pred, pos_loss = self.forward(batch, train=True)
        else:
            #tr_pred, rot_pred, pos_loss = self.forward(batch, train=True)
            tr_pred, rot_pred, tr_loss, rot_loss, pos_loss, cfm_loss = self.forward(batch, train=True)
        
        '''
        loss, tr_loss, rot_loss, _, _ = self.compute_loss(tr_pred, rot_pred,
                                                                batch,
                                                                self.args.model.tr_weight, self.args.model.rot_weight,
                                                                self.args.model.tor_weight)
        batch_size = len(batch)
        valid_log = {
            'loss': loss,
            'tr_loss': tr_loss,
            'rot_loss': rot_loss,
            'batch_size': torch.tensor(batch_size).to(loss.device).float()
        }
        '''
        
        valid_log = {
            'loss': pos_loss,
            'cfm_loss': cfm_loss,
            'tr_loss': tr_loss,
            'rot_loss': rot_loss,
            'batch_size': torch.tensor(len(batch)).to(pos_loss.device).float()
        }
        self.validation_step_outputs.append(valid_log)
        
        if not self.confidence_mode:
            for i in range(len(batch)):
                batch[i]['ligand'].pos = batch[i]['ligand'].org_pos.to(batch['ligand'].pos.device)
                batch[i]['ligand'].pos = copy.deepcopy(batch[i]['ligand'].org_pos)
                batch[i]['ligand'].c_alpha_coords = copy.deepcopy(batch[i]['ligand'].org_c_alpha_coords)
                batch[i]['ligand'].x = copy.deepcopy(batch[i]['ligand'].org_x)
                batch[i].linker_pos = batch[i].org_linker_pos
            if self.full_inference:
                self.valid_tobe_inference.extend([batch[i].to(torch.device('cpu')) for i in range(len(batch)) ])
                
                
            else:#random choose on each batch to be inference
                self.valid_tobe_inference.append(batch[random.randint(0, len(batch)-1)].to(torch.device('cpu')))
            return valid_log

    
    def compute_rmsd(self, data_list):
        ligand_pos = [complex_graph['ligand'].c_alpha_coords.cpu().numpy() for complex_graph in data_list]
        org_ligand_pos = [complex_graph['ligand'].org_c_alpha_coords for complex_graph in data_list]
        if isinstance(org_ligand_pos[0], torch.Tensor):
            org_ligand_pos = [org_ligand_pos[i].cpu().numpy() for i in range(len(org_ligand_pos))]
        lig_rmsds = np.asarray([np.sqrt(((ligand_pos[i] - org_ligand_pos[i]) ** 2).sum(axis=1).mean(axis=0)) for i in range(len(data_list))])
        
        
        receptor_pos = [complex_graph['receptor'].c_alpha_coords.cpu() for complex_graph in data_list]
        ligand_pos = [complex_graph['ligand'].c_alpha_coords.cpu() for complex_graph in data_list]
        org_ligand_pos = [complex_graph['ligand'].org_c_alpha_coords for complex_graph in data_list]
        complex_pos = [torch.cat([receptor_pos[i], ligand_pos[i]]) for i in range(len(data_list))]
        org_complex_pos = [torch.cat([receptor_pos[i], org_ligand_pos[i]]) for i in range(len(data_list))]
        for i, comp_pos in enumerate(complex_pos):
            R, t = rigid_transform_Kabsch_3D_torch(comp_pos.T, org_complex_pos[i].T)
            complex_pos[i] = torch.matmul(comp_pos - torch.mean(comp_pos, dim=0), R.T) + t.T + torch.mean(complex_pos[i], dim=0)
        complex_rmsd = np.asarray([np.sqrt(((complex_pos[i].numpy() - org_complex_pos[i].numpy()) ** 2).sum(axis=1).mean(axis=0)) for i in range(len(data_list))])
        #check nan in complex_rmsd

        distances = [torch.cdist(org_ligand_pos[i], receptor_pos[i]) for i in range(len(data_list))]
        #find interface < 10
        interface_rec_idx = [torch.where(distances[i] < 10)[1] for i in range(len(data_list))]
        interface_lig_idx = [torch.where(distances[i] < 10)[0] for i in range(len(data_list))]

        interface_pos = [torch.cat([receptor_pos[i][interface_rec_idx[i]], ligand_pos[i][interface_lig_idx[i]]]) for i in range(len(data_list))]
        org_interface_pos = [torch.cat([receptor_pos[i][interface_rec_idx[i]], org_ligand_pos[i][interface_lig_idx[i]]]) for i in range(len(data_list))]
        for i, int_pos in enumerate(interface_pos):
            R, t = rigid_transform_Kabsch_3D_torch(int_pos.T, org_interface_pos[i].T)
            interface_pos[i] = torch.matmul(int_pos - torch.mean(int_pos, dim=0), R.T) + t.T + torch.mean(interface_pos[i], dim=0)
        interface_rmsd = np.asarray([np.sqrt(((interface_pos[i].numpy() - org_interface_pos[i].numpy()) ** 2).sum(axis=1).mean(axis=0)) for i in range(len(data_list))])
        return {'complex_rmsd': complex_rmsd, 'interface_rmsd': interface_rmsd, 'ligand_rmsd': lig_rmsds}
        '''
        return {'complex_rmsd': lig_rmsds}
        '''

    def sampling_PC(self, data_list, inference_steps, time_schedule, 
                    batch_size, device, conflict_grad=5):
        if self.trainer.is_global_zero:
            print("sampling with conflict grad step: ", conflict_grad)
        if self.test_sample_num > 1:#multiply data_list
            data_list = [data_list[i % len(data_list)].clone() for i in range(self.test_sample_num * len(data_list))]
        data_list = [self.transform(d, from_start=1) for d in tqdm(data_list, desc="transforming data to raw")]
        #get initial rmsd
        if self.trainer.is_global_zero:
            rmsds = self.compute_rmsd(data_list)['complex_rmsd']
            print(f"average rmsd {rmsds.mean()} at init step")
        
        for t_idx in tqdm(range(inference_steps), desc="inferencing diffusion times"):
            t_tor = time_schedule[t_idx]
            dt = time_schedule[t_idx] - time_schedule[t_idx + 1] if t_idx < inference_steps - 1 else time_schedule[t_idx]
            
            #update linker
            step_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
            data_list_r = self.forward_sample_loader_linker(step_loader, t_tor, dt, device=device)
            
            
            for i, (complex_graph, tr_update, rot_update) in enumerate(data_list_r):
                data_cache = modify_conformer(copy.deepcopy(complex_graph), rot_update, 
                                        tr_update_surface=change_R_T_ref(tr_update, axis_angle_to_matrix(rot_update), 
                                                                        complex_graph['ligand'].c_alpha_coords, 
                                                                        torch.cat([complex_graph['ligand'].pos, complex_graph['ligand'].fake_pocket_pos])),
                                        tr_update_backbone=change_R_T_ref(tr_update, axis_angle_to_matrix(rot_update), 
                                                                        complex_graph['ligand'].c_alpha_coords, 
                                                                        complex_graph['ligand'].c_alpha_coords),
                                        tr_update_linker=change_R_T_ref(tr_update, axis_angle_to_matrix(rot_update), 
                                                                        complex_graph['ligand'].c_alpha_coords, 
                                                                        torch.cat([complex_graph.linker_pos])),
                                        has_norm=True)
                
                #data_cache, tr_mod, rot_mod = match_complex(data_cache)
                #data_cache.tr_update +=  (tr_update + tr_mod)
                #data_cache.rot_update = angle_sum(data_cache.rot_update, rot_update)
                #data_cache.rot_update = angle_sum(data_cache.rot_update, rot_mod)
                #except:
                #    data_list_r[i] = complex_graph
                #    print("skip one step for svd error")
                data_list[i] = data_cache
            #self.visualize(data_list, sample_step=str(t_idx) + '__before_matching')
            #get rmsd
            rmsds = self.compute_rmsd(data_list)['complex_rmsd']
            if self.trainer.is_global_zero:
                print(f"average rmsd {rmsds.mean()} at step {t_idx}")
            
            if conflict_grad:
                conflict_dt = 1.0 / conflict_grad
                for conflict_t_idx in range(conflict_grad):
                    time_t = 1 - (t_idx + 1) / inference_steps
                    g_t = 0.02 ** time_t * 1 ** (1 - time_t)
                    for i, complex_graph in enumerate(data_list):
                        #print("stuck at conflict grad")
                        complex_graph = complex_graph.to(device)
                        tr_grad, rot_grad = self.conflict_grad(complex_graph)
                        tr_grad = tr_grad * conflict_dt * g_t
                        rot_grad = angle_exp(rot_grad, conflict_dt * g_t)
                        complex_graph.tr_update += tr_grad
                        complex_graph.rot_update = angle_sum(complex_graph.rot_update, rot_grad)
                        
                        
                        data_list[i] = modify_conformer(complex_graph, rot_grad,
                                            tr_update_surface=change_R_T_ref(tr_grad, axis_angle_to_matrix(rot_grad), 
                                                                            complex_graph['ligand'].pos, 
                                                                            torch.cat([complex_graph['ligand'].pos, complex_graph['ligand'].fake_pocket_pos])),
                                            tr_update_backbone=change_R_T_ref(tr_grad, axis_angle_to_matrix(rot_grad), 
                                                                            complex_graph['ligand'].pos, 
                                                                            complex_graph['ligand'].c_alpha_coords),
                                            tr_update_linker=change_R_T_ref(tr_grad, axis_angle_to_matrix(rot_grad), 
                                                                            complex_graph['ligand'].pos, 
                                                                            torch.cat([complex_graph.linker_pos])),
                                            has_norm=True).to(torch.device('cpu'))
                        
                        #print("stuck at matching")
                        if conflict_t_idx == conflict_grad - 1:
                            if t_idx == (inference_steps - 1):# or random.random() < 0.5
                                data_list[i], tr_mod, rot_mod = match_complex(data_list[i], FF_iter=60)
                                data_list[i].tr_update += tr_mod
                                data_list[i].rot_update = angle_sum(data_list[i].rot_update, rot_mod)
                            elif random.random() < 0.5:
                                data_list[i], tr_mod, rot_mod = match_complex(data_list[i], FF_iter=30)
                                data_list[i].tr_update += tr_mod
                                data_list[i].rot_update = angle_sum(data_list[i].rot_update, rot_mod)
                            #print("avoid conflict :", tr_grad, rot_grad)
                #self.visualize(data_list, sample_step=str(t_idx) + '__after_matching')
                rmsds = self.compute_rmsd(data_list)['complex_rmsd']
                if self.trainer.is_global_zero:
                    print(f"average rmsd {rmsds.mean()} after conflict tune")
            
        return data_list

    def conflict_grad(self, data_):
        data = copy.deepcopy(data_)
        with torch.enable_grad():
            device = data['receptor'].pos.device
            tr_zero = torch.zeros(data.tr_update.shape).requires_grad_(True).to(device)
            rot_zero = torch.zeros(data.rot_update.shape).requires_grad_(True).to(device)

            data = modify_conformer(data, rot_zero, tr_update_surface=change_R_T_ref(tr_zero, axis_angle_to_matrix(rot_zero),
                                                                            data['ligand'].pos,
                                                                            torch.cat([data['ligand'].pos, data['ligand'].fake_pocket_pos])),
                                                    tr_update_backbone=change_R_T_ref(tr_zero, axis_angle_to_matrix(rot_zero),
                                                                            data['ligand'].pos,
                                                                            data['ligand'].c_alpha_coords),
                                                    tr_update_linker=change_R_T_ref(tr_zero, axis_angle_to_matrix(rot_zero),
                                                                            data['ligand'].pos,
                                                                            data.linker_pos), has_norm=True)
            sub_sample = 1
            rand_idx_rec = torch.randperm(data['receptor'].pos.shape[0])[:data['receptor'].pos.shape[0] // sub_sample]
            rand_idx_lig = torch.randperm(data['ligand'].pos.shape[0])[:data['ligand'].pos.shape[0] // sub_sample]
            conf_term_alpha = self.conflict(data['receptor'].c_alpha_coords, data['ligand'].c_alpha_coords, 4.)
            conf_term_surface = self.conflict(data['receptor'].pos, data['ligand'].pos, 2.)
            conf_term = conf_term_alpha + conf_term_surface.sum()
            #print("check grad: ", tr_zero.requires_grad, rot_zero.requires_grad, tor_l_zero.requires_grad)
            tr_grad = torch.autograd.grad(conf_term, tr_zero, create_graph=True)[0].detach()
            rot_grad = torch.autograd.grad(conf_term, rot_zero, create_graph=True)[0].detach()

            #clean nan
            if conf_term > -1e-5:
                tr_grad = torch.zeros_like(tr_grad)
                rot_grad = torch.zeros_like(rot_grad)
        return tr_grad, rot_grad
            
            
    
    def conflict(self, rec_pos, lig_pos, threshold=4.):
        #  a loss term that penalize too close and too far
        distances = torch.cdist(lig_pos, rec_pos)
        conflict_term = torch.clamp(distances, max=threshold) - threshold
        return conflict_term.mean() * self.snr
                                                                          

    @torch.no_grad()
    def forward_sample_loader_linker(self, loader, t_linker, dt_linker, device):
        new_data_list = []
        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            if device is not None:
                complex_graph_batch = complex_graph_batch.to(device)
            else:
                device = complex_graph_batch['receptor'].x.device
                complex_graph_batch = complex_graph_batch.to(device)
            set_time(complex_graph_batch, t_linker, t_linker, b)
            

            tr_pred, rot_pred = self.score_model(complex_graph_batch)
            '''
            #GT test
            tr_pred_gt = - complex_graph_batch.tr_update / self.t_to_sigma_(t_linker)
            rot_pred_gt = complex_graph_batch.rot_update
            rot_pred_gt = angle_exp(rot_pred_gt, - 1 / self.t_to_sigma_(t_linker)).unsqueeze(0) % (2 * np.pi)

            R,t = rigid_transform_Kabsch_3D_torch(complex_graph_batch[0]['ligand'].pos.T, complex_graph_batch[0]['ligand'].org_pos.T)
            rot_gap = matrix_to_axis_angle(R.T)
            tr_gap = t.unsqueeze(0)
            '''
            #print("tr pred: ", tr_pred[0], "tr gt: ", tr_pred_gt[0], "tr gap: ", tr_gap)
            #print("rot pred: ", rot_pred[0], "rot gt: ", rot_pred_gt[0], "rot gap: ", rot_gap)
            #print("tr gap: ", tr_gap)
            #print("rot gap: ", rot_gap)
            #tr_pred = tr_pred_gt
            #rot_pred = rot_pred_gt
            

            #Euler Solver
            tr_perturb = tr_pred.cpu() * dt_linker
            #print("gap: ", complex_graph_batch[0].linker_pos.mean(0) - complex_graph_batch[0].org_linker_pos.mean(0), "perturb:  ", tr_perturb[0],  "update:  ", complex_graph_batch[0].tr_update)
            rot_perturb = torch.stack([angle_exp(rot_pred[i].cpu(), dt_linker) for i in range(b)])
            new_data_list.extend([(complex_graph.to(torch.device('cpu')), tr_perturb[i], rot_perturb[i])
                    for i, complex_graph in enumerate(complex_graph_batch.to_data_list())])
        return new_data_list
    

    @torch.no_grad()
    def forward_sample_loader_torsion(self, loader, t_tor, dt_tor, device):
        new_data_list = []
        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            if device is not None:
                complex_graph_batch = complex_graph_batch.to(device)
            else:
                device = complex_graph_batch['receptor'].x.device
                complex_graph_batch = complex_graph_batch.to(device)
            set_time(complex_graph_batch, t_tor, t_tor, b)
            
            _, _, tor_l_pred = self.score_model(complex_graph_batch)
           #GT test
            tor_l_pred_gt = - complex_graph_batch.tor_l_update / self.t_to_sigma_(t_tor)
            #print("tor_l_pred: ", tor_l_pred[0], tor_l_pred_gt[0])
            print("tor_l pred: ", tor_l_pred[0], "tor_l gt: ", tor_l_pred_gt[0])
            #tor_l_pred = tor_l_pred_gt

            tor_perturb_l = dt_tor * tor_l_pred.cpu()
            

            tor_perturb_l = torch.tensor(tor_perturb_l).cpu().float()
            new_data_list.extend([(complex_graph.to(torch.device('cpu')), tor_perturb_l[i]) \
                                  for i, complex_graph in enumerate(complex_graph_batch.to_data_list())])
        return new_data_list

    def _compute_metrics(self, result):
        return {k: l.mean() for k, l in result.items()}

    def on_validation_epoch_end(self):

        result = self.validation_step_outputs
        loss = self._compute_metrics(self._gather_result(result))
        bs = int(loss['batch_size'].cpu().item())
        for k, l in loss.items():
            if k.endswith('loss'):
                self.log('val_' + k, l.mean(), on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        #flush logger to save the log
        self.logger.log_metrics(loss, step=self.global_step)
        self.validation_step_outputs = []
        
        if self.epoch_idx % 10 == 0:#TODO move this to args
            #inference on validation set
            device = loss['loss'].device
            if self.trainer.is_global_zero:
                print("sampling num per GPU: ", len(self.valid_tobe_inference))
            
            if not self.confidence_mode:
                predictions_list = self.sampling_PC(self.valid_tobe_inference, self.args.inference.inference_steps, 
                                            self.t_schedule, self.args.inference.batch_size * 16, device=device)
                
                if self.visualize_results:
                    dist.barrier()
                    self.visualize(predictions_list)
                rmsds = self.compute_rmsd(predictions_list)

                rmsds = {'complex_rmsd': torch.cat(list(self.all_gather(rmsds['complex_rmsd']))),
                        'interface_rmsd': torch.cat(list(self.all_gather(rmsds['interface_rmsd']))),
                        'ligand_rmsd': torch.cat(list(self.all_gather(rmsds['ligand_rmsd'])))}

                rmsds_complex = {'rmsds_lt20': (100 * (rmsds['complex_rmsd'] < 20).sum() / len(rmsds['complex_rmsd'])),
                        'rmsds_lt10': (100 * (rmsds['complex_rmsd'] < 10).sum() / len(rmsds['complex_rmsd'])),
                        'rmsds_mean': rmsds['complex_rmsd'].mean(),
                        'rmsds_median': torch.median(rmsds['complex_rmsd']),
                        'rmsds_min': rmsds['complex_rmsd'].min()}
                rmsds_interface = {'rmsds_lt20': (100 * (rmsds['interface_rmsd'] < 20).sum() / len(rmsds['interface_rmsd'])),
                        'rmsds_lt10': (100 * (rmsds['interface_rmsd'] < 10).sum() / len(rmsds['interface_rmsd'])),
                        'rmsds_mean': rmsds['interface_rmsd'].mean(),
                        'rmsds_median': torch.median(rmsds['interface_rmsd']),
                        'rmsds_min': rmsds['interface_rmsd'].min()}
                
                #print on rank 0
                if self.trainer.is_global_zero:
                    print("RMSD statistics: ", "complex: ", rmsds_complex, "interface: ", rmsds_interface)
                for k, v in rmsds_complex.items():
                    self.log('val_c_' + k, v, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
                for k, v in rmsds_interface.items():
                    self.log('val_i_' + k, v, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)

            elif self.confidence_mode and self.full_inference:
                #save the confidence prediction to a csv file
                log_dir = self.logger.log_dir
                pdb_keys = set([data.name for data in self.valid_tobe_inference])
                pdb_keys = {k: 0 for k in pdb_keys}
                loader = DataLoader(self.valid_tobe_inference, batch_size=self.args.inference.batch_size, shuffle=False)
                results = []
                for data in loader:
                    energy_pred = self.forward(data, train=False)
                    results.extend(list(energy_pred.cpu().numpy()))
                results = pd.DataFrame([pdb_keys, results]).T
                results.columns = ['pdb', 'energy']
                
                pd.to_csv(os.path.join(log_dir, "confidence.csv"))
        
        self.epoch_idx += 1
        self.valid_tobe_inference = []
    
    def test_step(self, batch, batch_idx):
        self.full_inference = True
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        torch.set_grad_enabled(True)
        test_result = self.on_validation_epoch_end()
        return test_result
    
    def _gather_result(self, result):
        # collect steps
        result = {
            key: torch.cat([x[key] for x in result])
            if len(result[0][key].shape) > 0
            else torch.tensor([x[key] for x in result]).to(result[0][key])
            for key in result[0].keys()
        }
        # collect machines
        result = {
            key: torch.cat(list(self.all_gather(result[key])))
            for key in result.keys()
        }
        #debug on single machine
        #result = {
        #    key: result[key]
        #    for key in result.keys()
        #}

        #clean nan
        for key in result.keys():
            result[key] = result[key][~torch.isnan(result[key])]
        return result

    




def check_none(k_list):
    none_num = 0
    for k in k_list:
        if k is None:
            none_num += 1
    return none_num