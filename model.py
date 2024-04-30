import torch
import torch.nn as nn
import math


class DCPML(nn.Module):
    def __init__(self, opt, n_items):
        super(DCPML, self).__init__()

        self.z_dim = opt['z_dim']
        self.enc_dim = eval(opt['enc_dim'])
        self.dec_dim = eval(opt['dec_dim'])
        self.n_items = n_items
        self.batch_size = opt['batch_size']

        self.lmb = opt['lambda']
        self.temp = opt['temp']

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        # encoder initialization
        enc_layers = []
        for i in range(len(self.enc_dim)):
            if i == 0:
                enc_layers.append(nn.Linear(self.n_items, self.enc_dim[i]))
            else:
                enc_layers.append(nn.Linear(self.enc_dim[i-1], self.enc_dim[i]))
            enc_layers.append(nn.Tanh())
        enc_layers.append(nn.Linear(self.enc_dim[-1], self.z_dim*2))
        self.encoder = nn.Sequential(*enc_layers)
        self.encoder.apply(self.init_weights)

        # decoder initialization
        dec_layers = []
        for i in range(len(self.dec_dim)):
            if i == 0:
                dec_layers.append(nn.Linear(self.z_dim, self.dec_dim[i]))
            else:
                dec_layers.append(nn.Linear(self.dec_dim[i-1], self.dec_dim[i]))
            dec_layers.append(nn.Tanh())
        dec_layers.append(nn.Linear(self.dec_dim[-1], self.n_items))
        self.decoder = nn.Sequential(*dec_layers)
        self.decoder.apply(self.init_weights)

        self.instance_projector = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
            nn.Linear(self.z_dim, self.z_dim),
        )        
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            self.truncated_normal_(m.bias, std=0.001)

    def truncated_normal_(self, tensor, mean=0, std=0.09):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size+(4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor
        
    def forward(self, input_sup, input_que=None):
        if self.training:
            h_context = self.encoder(input_sup)
            h_target = self.encoder(input_que)

            mu_context = h_context[:, :self.z_dim]
            logvar_context = h_context[:, self.z_dim:]
            std_context = torch.exp(0.5 * logvar_context)

            eps = torch.randn_like(logvar_context)
            z_context = mu_context + eps * std_context

            mu_target = h_target[:, :self.z_dim]
            logvar_target = h_target[:, self.z_dim:]
            std_target = torch.exp(0.5 * logvar_target)

            eps = torch.randn_like(logvar_target)
            z_target = mu_target + eps * std_target

            logit = self.decoder(z_target)

            neg_ll, loss_con, loss = self.loss(input_que, logit, self.instance_projector(z_context), self.instance_projector(z_target))
        else:
            h_context = self.encoder(input_sup)

            mu_context = h_context[:, :self.z_dim]
            logvar_context = h_context[:, self.z_dim:]
            std_context = torch.exp(0.5 * logvar_context)

            eps = torch.randn_like(logvar_context)
            z_context = mu_context + eps * std_context

            logit = self.decoder(z_context)

            neg_ll, loss_con, loss = None, None, None
        
        return logit, neg_ll, loss_con, loss

    def loss(self, input, logit, gamma_context, gamma_target):
        log_softmax_logit = nn.functional.log_softmax(logit, dim=1)
        neg_ll = -torch.mean(torch.sum(input * log_softmax_logit, dim=-1))

        loss_contrastive = self.forward_label(gamma_context, gamma_target, temperature_l=self.temp)
    
        loss = neg_ll + self.lmb * loss_contrastive
        return neg_ll, self.lmb * loss_contrastive, loss

    def forward_label(self, q_i, q_j, temperature_l, normalized=False):
        N_batch = q_i.shape[0]
        N = 2 * N_batch
        q = torch.cat((q_i, q_j), dim=0)

        sim = (torch.matmul(q, q.T) / temperature_l).to(q.device)

        sim_i_j = torch.diag(sim, N_batch)
        sim_j_i = torch.diag(sim, N_batch)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)
    
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss
    
    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)  # 自身置为0
        for i in range(N // 2):  # 正样本置为0
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        mask = mask.bool()

        return mask

