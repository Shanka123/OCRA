import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import math
from util import log
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         # print(pe.shape,(position*div_term).shape)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return x

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, dim)))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        # hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs,feats,pos, device,num_slots = None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs) 
        pos = pos.expand(b,pos.shape[1],pos.shape[2])

        k, v = self.to_k(inputs), self.to_v(inputs)
        # total_attn = torch.Tensor().to(device).float()
        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            # total_attn = torch.cat((total_attn,attn.unsqueeze(1)),dim=1)
            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots,torch.einsum('bjd,bij->bid', feats, attn),torch.einsum('bjd,bij->bid', pos, attn),attn

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).cuda()

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid, grid

class Encoder(nn.Module):
    def __init__(self, resolution, hid_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hid_dim, 5, padding = 2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        feat = x.permute(0,2,3,1)

        y,pos = self.encoder_pos(feat)

        y = torch.flatten(y, 1, 2)
        
        return y,torch.flatten(feat, 1, 2), torch.flatten(pos, 1, 2)

class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
      
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv6 = nn.ConvTranspose2d(hid_dim, 4, 3, stride=(1, 1), padding=1)

        self.decoder_initial_size = resolution
        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        # print(x.shape)
        x,_ = self.decoder_pos(x)
        # print(x.shape)
        x = x.permute(0,3,1,2)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = F.relu(x)
        x = self.conv2(x)
        # print(x.shape)
        x = F.relu(x)
#         x = F.pad(x, (4,4,4,4)) # no longer needed
        x = self.conv3(x)
        # print(x.shape)
        x = F.relu(x)
        x = self.conv4(x)
    
        x = F.relu(x)
        x = self.conv5(x)
    
        x = F.relu(x)
      
        x = self.conv6(x)




        # print(x.shape)

        # x = x[:,:,:self.resolution[0], :self.resolution[1]]
        x = x.permute(0,2,3,1)
        # print(x.shape)
        return x

"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, num_slots, num_iterations, hid_dim):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.encoder_cnn = Encoder(self.resolution, self.hid_dim)
        self.decoder_cnn = Decoder(self.hid_dim, self.resolution)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=hid_dim,
            iters = self.num_iterations,
            eps = 1e-8, 
            hidden_dim = 128)

    def forward(self, image,device):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x,feat,pos = self.encoder_cnn(image)  # CNN Backbone.
        x = nn.LayerNorm(x.shape[1:]).to(device)(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots,feat_slots,pos_slots,attn = self.slot_attention(x,feat,pos,device)
        # print("attention>>",attn.shape)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        slots_reshaped = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots_reshaped = slots_reshaped.repeat((1, image.shape[2], image.shape[3], 1))
        
        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        
        x = self.decoder_cnn(slots_reshaped)

        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-1)
        # recons, masks = x.reshape(image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([1,1], dim=-1)
        
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0,3,1,2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].

        # return slots 
        return recon_combined, recons, masks, feat_slots,pos_slots,attn.reshape(image.shape[0],-1,image.shape[2],image.shape[3],1)



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.0):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self,opt, dim, depth, heads, mlp_dim,num_slots, pool = 'cls', dim_head = 64, dropout = 0.0, emb_dropout = 0.):
        super().__init__()
        

     
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_slots*9 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim ))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim , depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim , 1)
    

    def forward(self,x,device):
        
        # print(x.shape)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class scoring_model(nn.Module):
    def __init__(self, opt,in_dim,depth,heads,mlp_dim,num_slots):
        super(scoring_model, self).__init__()
        self.in_dim = in_dim
       
        # self.posemb_fc = nn.Linear((num_slots*(num_slots+1))//2, in_dim)

        if opt.apply_context_norm:
            self.contextnorm = True
            self.gamma = nn.Parameter(torch.ones(in_dim))
            self.beta = nn.Parameter(torch.zeros(in_dim))
        else:
            self.contextnorm = False
        self.num_slots = num_slots
        
        self.embed_feat = nn.Linear(in_dim,in_dim)

        self.embed_layer = nn.Linear(1,in_dim)
        self.embed_pos = nn.Linear(in_dim,in_dim)

        self.transformer = ViT(opt,in_dim,depth,heads,mlp_dim,num_slots)


        # self.lstm = LSTM(in_dim)
    
    def apply_context_norm(self, z_seq):
        eps = 1e-8
        z_mu = z_seq.mean(1)
        z_sigma = (z_seq.var(1) + eps).sqrt()
        # print("seq, mean, var shape>>",z_seq.shape,z_mu.shape,z_sigma.shape)
        z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
        z_seq = (z_seq * self.gamma.unsqueeze(0).unsqueeze(0)) + self.beta.unsqueeze(0).unsqueeze(0)
        return z_seq

    def forward(self,feat_panels,pos_panels, device):
        # for i in range(self.in_dim):
        

        #   ABC[:,:,i] = ABC[:,:,i].clone() * torch.sigmoid(self.weights[i])
        #   D_choices[:,:,i] = D_choices[:,:,i].clone() * torch.sigmoid(self.weights[i])
            
        # AB = AB * torch.unsqueeze(torch.unsqueeze(torch.sigmoid(self.weights),0),0)
        # C_choices = C_choices * torch.unsqueeze(torch.unsqueeze(torch.sigmoid(self.weights),0),0)
        
        # given_panels_posencoded=[]
        
        # given_panels_posencoded_seq = torch.stack(given_panels_posencoded,dim=1)
        
        # print("given panels shape after position encoding>>", given_panels_posencoded_seq.shape)
        # ABC = ABC.clone() * torch.sigmoid(self.weights)[None,None,:]
        # D_choices = D_choices.clone() * torch.sigmoid(self.weights)[None,None,:]
            
        scores = []
        # feat_panels = self.embed_feat(feat_panels)
        # pos_panels = self.embed_pos(pos_panels)
        

        # Loop through all choices and compute scores
        for d in range(feat_panels.shape[1]):
           
            # print(AB,C_choices[:,d,:],AB.shape,C_choices[:,d,:].shape)
            # x_seq = torch.cat([given_panels_posencoded_seq,torch.cat((answer_panels[:,d],self.row_fc(third).unsqueeze(1).repeat((1,answer_panels.shape[2],1)), self.column_fc(third).unsqueeze(1).repeat((1,answer_panels.shape[2],1))),dim=2).unsqueeze(1)],dim=1)
            x_seq = feat_panels[:,d]
            pos_seq = pos_panels[:,d]
            
            # print("seq min and max>>",torch.min(x_seq),torch.max(x_seq))
            # x_seq = torch.cat([AB,C_choices[:,d,:].unsqueeze(1)],dim=1)
         #      x_seq = self.embed_feat(x_seq)
            # pos_seq = self.embed_pos(pos_seq)

            if self.contextnorm:
                x_seq = self.apply_context_norm(x_seq)
                pos_seq = self.apply_context_norm(pos_seq)
                # x_seq_all_seg = []
                # for i in range(x_seq.shape[1]):

          
                #     x_seq_all_seg.append(self.apply_context_norm(x_seq[:,i]).unsqueeze(1))
                # x_seq = torch.cat(x_seq_all_seg, dim=1)
          
            # x_seq = torch.cat((x_seq,all_posemb_concat_flatten),dim=2)
            # print(x_seq.shape)
            x_seq = self.embed_feat(x_seq)
            pos_seq = self.embed_pos(pos_seq)

            sim_matrix = torch.matmul(x_seq,x_seq.transpose(2,1))
            sim_matrix = F.softmax(sim_matrix, dim=2)
            # print("matrix>>",sim_matrix.shape)
            # print("pos feats>>",pos_seq.shape)
            sim_matrix_triu_withpos = torch.Tensor().to(device).float()
            for i in range(sim_matrix.shape[1]):
                for j in range(i,sim_matrix.shape[1]):
                    sim_matrix_triu_withpos = torch.cat((sim_matrix_triu_withpos,(self.embed_layer(sim_matrix[:,i,j].unsqueeze(1))+pos_seq[:,i]+pos_seq[:,j]).unsqueeze(1)),dim=1)

                    # sim_matrix_triu_withpos = torch.cat((sim_matrix_triu_withpos,(self.embed_layer(sim_matrix[:,i,j].unsqueeze(1))+self.embed_pos(pos_seq[:,i])+self.embed_pos(pos_seq[:,j])).unsqueeze(1)),dim=1)

            # print("sim matrix triu with pos>>",sim_matrix_triu_withpos.shape)
                # sim_matrix_triu = torch.cat((sim_matrix_triu, sim_matrix[i][torch.triu_indices(self.num_slots,self.num_slots)[0], torch.triu_indices(self.num_slots,self.num_slots)[1]].unsqueeze(0)),dim=0)
            # print("matrix triu>>",sim_matrix_triu.shape)
            # sim_matrix_triu_embed = self.embed_layer(sim_matrix_triu.unsqueeze(2))
            # print("matrix triu embed>>",sim_matrix_triu_embed.shape)
            # posembs = torch.Tensor().to(device).float()
            # for i in range(sim_matrix_triu_embed.shape[1]):
                # posembs = torch.cat((posembs,self.posemb_fc(F.one_hot( torch.Tensor([i]).expand(sim_matrix.shape[0]).to(device).long(),sim_matrix_triu_embed.shape[1]).float()).unsqueeze(1)),dim=1)
            # print("posembs>>",posembs.shape)
            # print(sim_matrix.shape)
            # mask_s = torch.ones_like(sim_matrix[0]).triu(diagonal=1).unsqueeze(0)
            # mask_s = mask_s.repeat(sim_matrix.shape[0], 1, 1).bool()
            # sim_matrix.masked_fill_(mask_s, float('-inf'))
            # sim_matrix = F.softmax(sim_matrix, dim=2)
            
            # sim_matrix = sim_matrix + sim_matrix.tril(diagonal=-1).transpose(2,1)
           
            
            score = self.transformer(sim_matrix_triu_withpos,device)

            

            scores.append(score)
        scores = torch.cat(scores,dim=1)
        return scores 

