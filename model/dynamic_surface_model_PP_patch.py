
import copy
import numpy as np
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph, knn_graph
from torch_scatter import scatter_mean, scatter_sum
from e3nn import o3

from kmeans_pytorch import kmeans


from model.score_model import AtomEncoder, TensorProductConvLayer, GaussianSmearing, TensorProductMLPLayer
from utils import so3, torus
from utils.geometry import axis_angle_to_matrix, matrix_to_axis_angle
from dataset.process_mols import lig_feature_dims, rec_residue_feature_dims, rec_atom_feature_dims


class TensorProductScoreModel(torch.nn.Module):
    def __init__(self, t_to_sigma, time_scale=10000, in_lig_edge_features=4, sigma_embed_dim=32, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, num_sample=1500, prot_max_radius=2, cross_max_distance=8,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32,
                 scale_by_sigma=True, batch_norm=False,
                 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=False, confidence_mode=False):
                 #confidence_dropout=0, confidence_no_batchnorm=False, num_confidence_outputs=1):
        super(TensorProductScoreModel, self).__init__()
        
        
        
        self.time_scale = time_scale
        self.t_to_sigma = t_to_sigma
        self.num_sample = num_sample
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.prot_max_radius = prot_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.num_conv_layers = num_conv_layers
        self.confidence_mode = confidence_mode
        self.num_conv_layers = num_conv_layers

        # embedding layers
        self.surface_embedding = nn.Embedding(3, ns)

        self.atom_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_atom_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.atom_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.lr_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.prot_distance_expansion = GaussianSmearing(0.0, prot_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        self.protein_embed = TensorProductMLPLayer(f'{ns}x0e + {self.sh_irreps} + {self.sh_irreps} + 1x0e', f'{ns}x0e + {nv}x1o', dropout=dropout)

        irrep_seq = [
            f'{ns}x0e + {nv}x1o + {self.sh_irreps}',
            f'{ns}x0e + {nv}x1o  + {self.sh_irreps}',
            f'{ns}x0e + {nv}x1o + {nv}x1e + {self.sh_irreps}',
            f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o + {self.sh_irreps}'
            ]

        # convolutional layers
        conv_layers = []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }

            for k in range(4): # 3 intra & 6 inter per each layer
                conv_layers.append(TensorProductConvLayer(**parameters))

        self.conv_layers = nn.ModuleList(conv_layers)

        self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
        self.center_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        
        if self.confidence_mode:
            self.final_conv_linker = TensorProductConvLayer(
                    in_irreps=f'{self.conv_layers[-1].out_irreps}',
                    sh_irreps=self.sh_irreps,
                    out_irreps=f'1x0e',
                    n_edge_features=2 * ns,
                    residual=False,
                    dropout=dropout,
                    batch_norm=batch_norm
                )
            
        else:
            self.final_conv_linker = TensorProductConvLayer(
                in_irreps=f'{self.conv_layers[-1].out_irreps} + 1o + {self.sh_irreps}',
                sh_irreps=self.sh_irreps,
                out_irreps=f'2x1o + 2x1e',
                n_edge_features=2 * ns,
                residual=False,
                dropout=dropout,
                batch_norm=batch_norm
            )

        #self.tor_scale_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns), nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.ligand_key = 'ligand'


    def build_graph_forward_RT(self, data):
        # build ligand graph
        #print("lig shape", lig_node_attr.shape, l_atom_edge_index.shape)
        l_atom_node_attr, l_atom_edge_index, l_atom_edge_attr, l_atom_edge_sh, l_batch, l_pos, l_sigma = self.build_atom_conv_graph(data, self.ligand_key)
        l_atom_edge_attr_emb = self.atom_edge_embedding(l_atom_edge_attr)

        # build receptor surface graph
        r_atom_node_attr, r_atom_edge_index, r_atom_edge_attr, r_atom_edge_sh, r_batch, r_pos, r_sigma = self.build_atom_conv_graph(data, 'receptor')
        r_atom_edge_attr = self.atom_edge_embedding(r_atom_edge_attr)


        # build cross graph
        cross_cutoff = self.cross_max_distance
        lr_edge_index, lr_edge_attr, lr_edge_sh = self.build_cross_conv_graph(l_pos, r_pos, l_batch, r_batch, l_sigma, cross_cutoff)
        lr_edge_attr = self.lr_edge_embedding(lr_edge_attr)


        for l in range(self.num_conv_layers):
            # LIGAND updates
            l_atom_edge_attr_ = torch.cat([l_atom_edge_attr_emb, l_atom_node_attr[l_atom_edge_index[0], :self.ns], l_atom_node_attr[l_atom_edge_index[1], :self.ns]], -1)
            lig_update = self.conv_layers[4*l](l_atom_node_attr, l_atom_edge_index, l_atom_edge_attr_, l_atom_edge_sh)

            lr_edge_attr_ = torch.cat([lr_edge_attr, l_atom_node_attr[lr_edge_index[0], :self.ns], r_atom_node_attr[lr_edge_index[1], :self.ns]], -1)
            lr_update = self.conv_layers[4*l+1](l_atom_node_attr, torch.flip(lr_edge_index, dims=[0]), lr_edge_attr_, lr_edge_sh,
                                                out_nodes=r_atom_node_attr.shape[0])


            #if l != self.num_conv_layers-1:  # last layer optimisation

            # ATOM UPDATES
            r_atom_edge_attr_ = torch.cat([r_atom_edge_attr, r_atom_node_attr[r_atom_edge_index[0], :self.ns], r_atom_node_attr[r_atom_edge_index[1], :self.ns]], -1)
            r_atom_update = self.conv_layers[4*l+2](r_atom_node_attr, r_atom_edge_index, r_atom_edge_attr_, r_atom_edge_sh)

            rl_edge_attr_ = torch.cat([lr_edge_attr, l_atom_node_attr[lr_edge_index[0], :self.ns], r_atom_node_attr[lr_edge_index[1], :self.ns]], -1)
            rl_update = self.conv_layers[4*l+3](r_atom_node_attr, lr_edge_index, rl_edge_attr_,
                                                lr_edge_sh, out_nodes=l_atom_node_attr.shape[0])


            # padding original features and update features with residual updates
            r_atom_node_attr = F.pad(r_atom_node_attr, (0, r_atom_update.shape[-1] - r_atom_node_attr.shape[-1]))
            r_atom_node_attr = r_atom_node_attr + lr_update + r_atom_update

            if l != self.num_conv_layers - 1:  # last layer optimisation
                l_atom_node_attr = F.pad(l_atom_node_attr, (0, lig_update.shape[-1] - l_atom_node_attr.shape[-1]))
                l_atom_node_attr = l_atom_node_attr + rl_update + lig_update
            
        
        return l_atom_node_attr, l_pos, l_batch, l_sigma

    def cal_moment(self, axis, pos, force, batch_idx, norm=2*np.pi):
        moments = []
        for b in range(batch_idx.max() + 1):
            idx = torch.where(batch_idx == b)[0]
            #get the distance from pos to axis(2 points)
            direction_vec = (axis[1][b] - axis[0][b]).float()
            t = (pos[idx, 0] - pos[idx, 1]) * direction_vec[0] \
                + (pos[idx, 1] - pos[idx, 1]) * direction_vec[1] \
                + (pos[idx, 2] - pos[idx, 1]) * direction_vec[2]

            t = t / torch.norm(direction_vec)
            #get the projection point3
            proj = (axis[0][b] + t.unsqueeze(1) * direction_vec.unsqueeze(0).repeat(len(idx), 1)).float()
        
            #get the moment
            moment = torch.mean(torch.cross(pos[idx] - proj, force[idx]), dim=0)
            moment = torch.sum(moment / (direction_vec / torch.norm(direction_vec)))
            #agg each graph
            moments.append(moment.unsqueeze(0))
        moments = torch.stack(moments)
        moments = torch.sigmoid(moments) * norm
        
        return moments

    def forward(self, data, train=False):
        self.ligand_key = 'ligand'
        lig_node_attr, l_pos, l_batch, l_sigma = self.build_graph_forward_RT(data)

        # compute translational and rotational score vectors
        center_edge_index, center_edge_attr, center_edge_sh, linker_attr = self.build_center_conv_graph(data, l_pos, l_batch, l_sigma)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
        if self.confidence_mode:
            energy_pred = self.final_conv_linker(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)
        else:
            energy_pred = self.final_conv_linker(torch.cat([lig_node_attr, linker_attr], dim=-1), center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)

        
        #edge_sigma = self.sinusoidal_embedding(data.complex_t['tor'].float() * self.time_scale, self.sigma_embed_dim)
        #tor_sigma = torch.stack([data[i].tor_sigma for i in range(len(data))], dim=0).float()
        #tor_l_scale = self.tor_scale_layer(torch.cat([edge_sigma, l_moment], dim=1)) * torch.sqrt(tor_sigma)
        
        #print("checky: ", tr_pred[0], rot_pred[0])
        if not self.confidence_mode:
            tr_pred, rot_pred = energy_pred[:, :3] + energy_pred[:, 6:9], energy_pred[:, 3:6] + energy_pred[:, 9:12]
            #rot_pred = torch.sigmoid(rot_pred) * np.pi * 2
            
            if train:
                #all_pos = [data[i][self.ligand_key].pos for i in range(len(data))]
                all_pos = [data[i][self.ligand_key].c_alpha_coords for i in range(len(data))]
                rot_mat_pred = [axis_angle_to_matrix(rot_pred[i]) for i in range(len(rot_pred))]
                pos_pred = [(all_pos[i] - torch.mean(all_pos[i], dim=0))@ rot_mat_pred[i].T + torch.mean(all_pos[i], dim=0) + tr_pred[i] for i in range(len(rot_pred))]
                
                rot_mat_score = [axis_angle_to_matrix(data[i].rot_score[0]) for i in range(len(data))]
                tr_score = [data[i].tr_score[0] for i in range(len(data))]
                
                pos_score = [(all_pos[i] - torch.mean(all_pos[i], dim=0))@ rot_mat_score[i].T + torch.mean(all_pos[i], dim=0) + tr_score[i] for i in range(len(rot_pred))]
                #return tr_pred, rot_pred, torch.mean(torch.stack([torch.norm(pos_pred[i] - pos_score[i], dim=-1).mean() for i in range(len(rot_pred))]))
            
                tr_loss = F.l1_loss(tr_pred, torch.stack(tr_score, dim=0))
                rot_loss = [rot_mat_score[i] @ rot_mat_pred[i].T for i in range(len(rot_pred))]
                rot_loss = torch.stack([(matrix_to_axis_angle(R)**2).sum() for R in rot_loss]).mean()
                cfm_loss = tr_loss + rot_loss
                
                return tr_pred, rot_pred, tr_loss, rot_loss, torch.mean(torch.stack([torch.norm(pos_pred[i] - pos_score[i], dim=-1).mean() for i in range(len(rot_pred))])), cfm_loss
            else:
                return tr_pred, rot_pred
        else:
            if train:
                rmsd_int = torch.stack([data[i].rmsd_int for i in range(len(data))], dim=0).float()
                confidence_loss = F.mse_loss(energy_pred, rmsd_int)
                return energy_pred, confidence_loss
            else:
                return energy_pred
                


    def build_atom_conv_graph(self, data, key, patch_size=256):
        
        data[key].node_sigma_emb = torch.cat([self.sinusoidal_embedding(data[key].node_t['tor'] * self.time_scale, self.sigma_embed_dim //2),
                                                self.sinusoidal_embedding(data[key].node_t['linker'] * self.time_scale, self.sigma_embed_dim //2)], dim=-1)
        node_attr = torch.cat([data[key].x[:, :, :-7], data[key].node_sigma_emb.unsqueeze(1).repeat(1, data[key].x.shape[1], 1), data[key].x[:, :, -7:]], -1)
        
        node_attr_f = self.atom_node_embedding(node_attr[:, :, :-7]) + self.surface_embedding(node_attr[:, :, -1].long())
        nearest_atom_sh = torch.cat([o3.spherical_harmonics(self.sh_irreps, node_attr[:, :, -7:-4], normalize=True, normalization='component'),
                                     o3.spherical_harmonics(self.sh_irreps, node_attr[:, :, -4:-1], normalize=True, normalization='component')], dim=2)
        surface_norm_sh = o3.spherical_harmonics(self.sh_irreps, node_attr[:, :, -4:-1], normalize=True, normalization='component')
        node_attr = torch.cat([node_attr_f, nearest_atom_sh, torch.norm(node_attr[:, :, -7:-4], dim=-1, keepdim=True)], dim=2)
        node_attr_f = self.protein_embed(node_attr)
        node_attr = torch.cat([node_attr_f, surface_norm_sh], dim=2)
        node_attr = torch.sum(node_attr, dim=1)

        patch_idx, patch_batch = self.kmeans_batch(data[key].pos, data[key].batch, patch_size)

        node_attr = scatter_mean(node_attr, patch_idx, dim=0)
        patch_pos = scatter_mean(data[key].pos, patch_idx, dim=0)

        #use knn graph
        edge_index = knn_graph(patch_pos, k=3, batch=patch_batch, loop=False)
        src, dst = edge_index
        

        edge_vec = data[key].pos[dst] - data[key].pos[src]

        edge_length_emb = self.prot_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = scatter_mean(data[key].node_sigma_emb, patch_idx, dim=0)[dst]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        
        return node_attr, edge_index, edge_attr, edge_sh, patch_batch, patch_pos, scatter_mean(data[key].node_sigma_emb, patch_idx, dim=0)

    def build_cross_conv_graph(self, ligand_pos, receptor_pos, ligand_batch, receptor_batch, ligand_sigma_emb, cross_cutoff):
        # build the cross edges between ligan atoms, receptor residues and receptor atoms
        # LIGAND to RECEPTOR
        lr_edge_index = radius(ligand_pos, receptor_pos, cross_cutoff, ligand_batch, receptor_batch, max_num_neighbors=256)
        lr_edge_index = torch.flip(lr_edge_index, dims=[0])
        lr_edge_vec = receptor_pos[lr_edge_index[1].long()] - ligand_pos[lr_edge_index[0].long()]
        lr_edge_length_emb = self.cross_distance_expansion(lr_edge_vec.norm(dim=-1))
        lr_sigma_emb = ligand_sigma_emb[lr_edge_index[0].long()]
        lr_edge_attr = torch.cat([lr_sigma_emb, lr_edge_length_emb], 1)
        lr_edge_sh = o3.spherical_harmonics(self.sh_irreps, lr_edge_vec, normalize=True, normalization='component')

        return lr_edge_index, lr_edge_attr, lr_edge_sh
    

    def build_center_conv_graph(self, data, l_pos, l_batch, l_sigma):
        # build the filter for the convolution of the center with the ligand atoms
        # for translational and rotational score
        
        edge_index = torch.cat([l_batch.unsqueeze(0), torch.arange(len(l_batch)).to(data[self.ligand_key].x.device).unsqueeze(0)], dim=0)

        '''for C alpha as center'''
        
        center_pos = torch.zeros((data.num_graphs, 3)).to(data[self.ligand_key].x.device)
        pos_batch = []
        for i in range(len(data)):
            #pos_batch.append((torch.zeros(data[i][self.ligand_key].pos.shape[0]).to(data[self.ligand_key].x.device) + i).int())
            pos_batch.append((torch.zeros(data[i][self.ligand_key].c_alpha_coords.shape[0]).to(data[self.ligand_key].x.device) + i).long())
        pos_batch = torch.cat(pos_batch, dim=0)

        #center_pos.index_add_(0, index=pos_batch, source=data[self.ligand_key].pos)
        center_pos.index_add_(0, index=pos_batch, source=data[self.ligand_key].c_alpha_coords)
        center_pos = center_pos / torch.bincount(pos_batch).unsqueeze(1)
        

       
        edge_vec = l_pos[edge_index[1]] - center_pos[edge_index[0]]

        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        
        if self.confidence_mode:
            linker_attr = None
        else:
            linker_attr = torch.stack(
                [data[self.ligand_key].fake_pocket_norm[l_b] for l_b in l_batch]
            )
            linker_sh = o3.spherical_harmonics(self.sh_irreps, linker_attr, normalize=True, normalization='component')
            linker_attr = torch.cat([linker_attr, linker_sh], dim=-1)

        edge_sigma_emb = l_sigma[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh, linker_attr
    

    def sinusoidal_embedding(self, timesteps, embedding_dim, max_positions=10000):
        """ from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
        assert len(timesteps.shape) == 1
        half_dim = embedding_dim // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (timesteps.shape[0], embedding_dim)
        return emb

    def kmeans_batch(self, x, batch, k):
        # x: (N, D)
        # batch: (N)
        # k: int
        batch_size = batch.max() + 1
        index_count = 0
        cluster_ids, cluster_batches = [], []
        for i in range(batch_size):
            x_index = torch.where(batch == i)[0]
            x_batch = x[x_index]
            if k > len(x_batch):
                k = len(x_batch)
            cluster_id, _ = kmeans(x_batch, k, distance='euclidean', device=x.device, tqdm_flag=False)
            cluster_id = cluster_id + index_count
            cluster_batches.append(torch.zeros(k) + i)
            index_count += k
            cluster_ids.append(cluster_id)
        
        cluster_ids = torch.cat(cluster_ids, dim=0).to(x.device)
        cluster_batch = torch.cat(cluster_batches, dim=0).to(x.device).long()

        return cluster_ids, cluster_batch
        
        