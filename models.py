import torch.nn as nn
import math
from ctgan.synthesizers.ctgan import Discriminator, Generator
from torch import functional, optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import os
import time
import warnings
import numpy as np
import pandas as pd
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
import torch
import torch.nn.functional as F
import torch.optim as optim
import utils
import copy
#GCN layer
from tqdm import tqdm


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        #for 3_D batch, need a loop!!!


        if self.bias is not None:
            return output + self.bias
        else:
            return output

#Multihead attention layer
class MultiHead(Module):#currently, allowed for only one sample each time. As no padding mask is required.
    def __init__(
        self,
        input_dim,
        num_heads,
        kdim=None,
        vdim=None,
        embed_dim = 128,#should equal num_heads*head dim
        v_embed_dim = None,
        dropout=0.1,
        bias=True,
    ):
        super(MultiHead, self).__init__()
        self.input_dim = input_dim
        self.kdim = kdim if kdim is not None else input_dim
        self.vdim = vdim if vdim is not None else input_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.v_embed_dim = v_embed_dim if v_embed_dim is not None else embed_dim

        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.bias = bias
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        assert self.v_embed_dim % num_heads ==0, "v_embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5


        self.q_proj = nn.Linear(self.input_dim, self.embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, self.embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, self.v_embed_dim, bias=bias)

        self.out_proj = nn.Linear(self.v_embed_dim, self.v_embed_dim//self.num_heads, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if True:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.normal_(self.k_proj.weight)
            nn.init.normal_(self.v_proj.weight)
            nn.init.normal_(self.q_proj.weight)

        nn.init.normal_(self.out_proj.weight)

        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

        if self.bias:
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.q_proj.bias, 0.)

    def forward(
        self,
        query,
        key,
        value,
        need_weights: bool = False,
        need_head_weights: bool = False,
    ):
        """Input shape: Time x Batch x Channel
        Args:
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        batch_num, node_num, input_dim = query.size()

        assert key is not None and value is not None

        #project input
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q * self.scaling

        #compute attention
        q = q.view(batch_num, node_num, self.num_heads, self.head_dim).transpose(-2,-3).contiguous().view(batch_num*self.num_heads, node_num, self.head_dim)
        k = k.view(batch_num, node_num, self.num_heads, self.head_dim).transpose(-2,-3).contiguous().view(batch_num*self.num_heads, node_num, self.head_dim)
        v = v.view(batch_num, node_num, self.num_heads, self.vdim).transpose(-2,-3).contiguous().view(batch_num*self.num_heads, node_num, self.vdim)
        attn_output_weights = torch.bmm(q, k.transpose(-1,-2))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        #drop out
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        #collect output
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.view(batch_num, self.num_heads, node_num, self.vdim).transpose(-2,-3).contiguous().view(batch_num, node_num, self.v_embed_dim)
        attn_output = self.out_proj(attn_output)


        if need_weights:
            attn_output_weights = attn_output_weights #view: (batch_num, num_heads, node_num, node_num)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output


#Graphsage layer
class SageConv(Module):
    """
    Simple Graphsage layer
    """

    def __init__(self, in_features, out_features, bias=False):
        super(SageConv, self).__init__()

        self.proj = nn.Linear(in_features*2, out_features, bias=bias)

        self.reset_parameters()

        #print("note: for dense graph in graphsage, require it normalized.")

    def reset_parameters(self):

        nn.init.normal_(self.proj.weight)

        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)

    def forward(self, features, adj):
        """
        Args:
            adj: can be sparse or dense matrix.
        """

        #fuse info from neighbors. to be added:
        if adj.layout != torch.sparse_coo:
            if len(adj.shape) == 3:
                neigh_feature = torch.bmm(adj, features) / (adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1],-1))+1)
            else:
                neigh_feature = torch.mm(adj, features) / (adj.sum(dim=1).reshape(adj.shape[0], -1)+1)
        else:
            #print("spmm not implemented for batch training. Note!")
            
            neigh_feature = torch.spmm(adj, features) / (adj.to_dense().sum(dim=1).reshape(adj.shape[0], -1)+1)

        #perform conv
        data = torch.cat([features,neigh_feature], dim=-1)
        combined = self.proj(data)

        return combined

#GraphAT layers

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        if isinstance(adj, torch.sparse.FloatTensor):
            adj = adj.to_dense()

        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




#--------------
### models ###
#--------------

#gcn_encode
class GCN_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(GCN_En, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class GCN_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(GCN_Classifier, self).__init__()

        self.gc1 = GraphConvolution(nembed, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x

#sage model

class Sage_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(Sage_En, self).__init__()

        self.sage1 = SageConv(nfeat, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x

class Sage_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(Sage_Classifier, self).__init__()

        self.sage1 = SageConv(nembed, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x


#GAT model

class GAT_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout, alpha=0.2, nheads=8):
        super(GAT_En, self).__init__()

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_proj = nn.Linear(nhid * nheads, nembed)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight,std=0.05)

    def forward(self, x, adj):

        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_proj(x))

        return x

class GAT_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout, alpha=0.2, nheads=8):
        super(GAT_Classifier, self).__init__()

        
        self.attentions = [GraphAttentionLayer(nembed, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_proj = nn.Linear(nhid * nheads, nhid)

        self.dropout = dropout
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)
        nn.init.normal_(self.out_proj.weight,std=0.05)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_proj(x))
        x = self.mlp(x)

        return x


class Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(Classifier, self).__init__()

        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, adj):
        x = self.mlp(x)

        return x

class Decoder(Module):
    """
    Simple Graphsage layer
    """

    def __init__(self, nembed, dropout=0.1):
        super(Decoder, self).__init__()
        self.dropout = dropout

        self.de_weight = Parameter(torch.FloatTensor(nembed, nembed))

        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)


    def forward(self, node_embed):
        
        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1,-2)))

        return adj_out

class GAT_GAN(BaseSynthesizer):

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=100, discriminator_steps=1,
                 log_frequency=True, verbose=True, epochs=300, pac=10, cuda=True):

        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)


        self._transformer = None
        self._data_sampler = None
        self._generator = None

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):

        for _ in range(10):
            transformed = F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):

        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    @random_state
    def fit(self, encoder, decoder, classifier, optimizer_en, optimizer_cls, optimizer_de, features, labels, adj, idx_train, idx_val, args, epochs=None,
            im_class=None, gan_epoch=10, pretrain=False):

        encoder.train()
        classifier.train()
        decoder.train()

        print('-----------------load training data for GNN-GAN-------------------------')
        ori_num = labels.shape[0]
        labels=labels.detach().cpu().numpy()
        features=features.detach().cpu()
        train_label=[]
        train_features=[]
        for i in im_class:
            train_label.append(labels[labels==i]+(1,))
            train_features.append(features[labels==i,:])
        train_label=np.hstack(train_label)
        train_label=np.array(train_label).squeeze()
        train_features=np.row_stack(train_features)
        train_features=np.array(train_features).squeeze()
        labels_ex = np.expand_dims(train_label, 1)
        real_data = np.hstack((train_features,labels_ex))
        T = np.array(real_data).astype(np.float32)
        train_data = pd.DataFrame(T)
        if train_data.shape[0]<300:
            index=np.random.permutation(train_data.shape[0])
        else:
            index=np.random.permutation(train_data.shape[0])[:300]
        train_data = train_data.iloc[index, :]
        train_label=train_data.iloc[:,-1]
        train_label=np.array(train_label)

        ##-----------sampling colomns
        discrete_columns = []
        for i in range(train_data.shape[1] - 1):
            discrete_columns.append(str(i))
        discrete_columns.append('type')
        train_data.columns = discrete_columns
        c_class,counts = np.unique(labels,return_counts=True)
        c_max=np.max(counts)
        c_min=np.min(counts)
        c_total=np.sum(counts)
        self._validate_discrete_columns(train_data, discrete_columns)
        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        self._transformer = DataTransformer()

        self._transformer.fit(train_data, discrete_columns)
        print('\n')
        print('-------------------start train transformer-----------------------')
        st=time.time()
        train_data = self._transformer.transform(train_data)
        ed=time.time()
        print("total time cost: {}".format(ed-st))
        print('-------------------finish training transformer-----------------------')

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions



        self._generator = Generator(
            self._embedding_dim +  self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)


        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Discriminator Loss'])
        metric_values = pd.DataFrame(columns=['Epoch', 'acc_train', 'loss_train', 'acc_val', 'loss_val','time'])

        epoch_iterator = tqdm(range(gan_epoch), disable=(not self._verbose))
        epoch_iterator_2 = tqdm(range(epochs-gan_epoch), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            description_2 = 'Acc_train. ({acc:.4f}) | Loss_train. ({loss:.4f})'
        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        print('\n')


        print('-------------------start loading pretrained GAN and GNN-----------------------')

        acc_best = None
        if pretrain:
            if os.path.exists('checkpoint/{}/GNN_best.pth'.format(args.dataset)):
                loaded_content=torch.load('checkpoint/{}/GNN_best.pth'.format(args.dataset))
                encoder.load_state_dict(loaded_content['encoder'])
                decoder.load_state_dict(loaded_content['decoder'])
                classifier.load_state_dict(loaded_content['classifier'])
                acc_best=loaded_content['acc_best']
            else:
                pass



        print('\n')
        print('-------------------start training GAN-----------------------')
        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):
                ### training the discriminator
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std).to(self._device)
                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)
                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()
                    optimizerD.zero_grad(set_to_none=False)
                ### training the generator
                fakez = torch.normal(mean=mean, std=std).to(self._device)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy
                loss_g.backward()
                optimizerG.step()
                optimizerG.zero_grad(set_to_none=False)

                generator_loss = loss_g.detach().cpu()
                discriminator_loss = loss_d.detach().cpu()
                epoch_loss_df = pd.DataFrame({
                    'Epoch': [i],
                    'Generator Loss': [generator_loss],
                    'Discriminator Loss': [discriminator_loss]
                })
                if not self.loss_values.empty:
                    self.loss_values = pd.concat(
                        [self.loss_values, epoch_loss_df]
                    ).reset_index(drop=True)
                else:
                    self.loss_values = epoch_loss_df

                if self._verbose:
                    epoch_iterator.set_description(
                        description.format(gen=generator_loss, dis=discriminator_loss)
                    )
                ### saving model
                if (steps_per_epoch * i + id_) > 0 and (steps_per_epoch * i + id_) % 500 == 0:
                    saved_gan = {}
                    saved_gan['generator'] = self._generator.state_dict()
                    saved_gan['discriminator'] = discriminator.state_dict()
                    saved_gan['loss'] = self.loss_values




        print('-------------------sampling nodes-----------------------')
        if args.confidence is None:
            num_sample=200
        else:
            num_sample=int(args.confidence*len(train_data))
        steps = num_sample // self._batch_size + 1
        data = []
        for id_step in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            sample = self._generator(fakez)
            sampleact = self._apply_activate(sample)
            data.append(sampleact.detach().cpu().numpy())
            # y_fake = discriminator(sampleact)
            # loss_ge = -torch.mean(y_fake)

        data = np.concatenate(data, axis=0)
        data = data[:num_sample]

        print('\n')
        print('-------------------start training GNN-----------------------')
        for i_1 in  epoch_iterator_2:
            t = time.time()
            ## merge synthetic nodes into the original datasets
            optimizer_en.zero_grad()
            optimizer_cls.zero_grad()
            optimizer_de.zero_grad()
            new_chosen = self._transformer.inverse_transform(data)
            new_chosen = np.array(new_chosen)
            adj_back = adj.to_dense()
            if not isinstance(labels, np.ndarray):
                labels = labels.detach().cpu().numpy()
            chosen = []
            for k in range(c_class.shape[0]):
                c = c_class[k]
                if c in im_class:
                    c_i = counts[c_class == c][0]
                    c_i_dot = np.sum(new_chosen[:, -1] == c)
                    if args.imbalance_ratio is None:
                        new = int(c_max * c_i_dot / ((c_i) + 0.1))
                    else:
                        new= int(c_i_dot / args.imbalance_ratio )
                    # print(c_max,c_i_dot , c_min,c_i,new)
                    if new > 0 and new < new_chosen.shape[0]:
                        chosen.append(new_chosen[new_chosen[:, -1] == c][:new, :])
                    else:
                        chosen.append(new_chosen[new_chosen[:, -1] == c])
                else:
                    c_i = np.sum(train_label == c)
                    c_i_dot = np.sum(new_chosen[:, -1] == c)
                    if args.imbalance_ratio is None:
                        new = int((c_max - c_min) * c_i_dot / ((c_i - c_min) + 0.1))
                    else:
                        new= int(c_i_dot/args.imbalance_ratio )
                    # print(c_max,c_i_dot , c_min,c_i,new)
                    if new > 0 and new < new_chosen.shape[0]:
                        chosen.append(new_chosen[new_chosen[:, -1] == c][:new, :])
                    else:
                        chosen.append(new_chosen[new_chosen[:, -1] == c])

            if len(chosen) >= 1:
                chosen = np.row_stack(chosen)
                # print(chosen.shape)
                chosen = np.array(chosen).squeeze()
                add_num = chosen.shape[0]
                new_adj = adj_back.new(torch.Size((adj_back.shape[0] + add_num, adj_back.shape[
                    0] + add_num)))  # 创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致，且无内容。
                new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:, :]
                features_append = copy.deepcopy(chosen[:, :-1])
                labels_append = copy.deepcopy(chosen[:, -1])
                idx_new = np.arange(adj_back.shape[0], adj_back.shape[0] + add_num)
                idx_train_append = idx_train.new(idx_new)
                # print(features_append, labels_append)
                features_append = torch.FloatTensor(features_append)
                labels_append = torch.LongTensor(labels_append)
                labels = torch.LongTensor(labels)
                features = features.cuda()
                labels = labels.cuda()
                adj = adj.cuda()
                features_append = features_append.cuda()
                labels_append = labels_append.cuda()
                features_new = torch.cat((features, features_append), 0)
                labels_new = torch.cat((labels, labels_append), 0)
                idx_train_new = torch.cat((idx_train, idx_train_append), 0)
                adj_up = new_adj.detach()

                embed = encoder(features_new, new_adj)
                generated_G = decoder(embed)
                loss_rec = utils.adj_mse_loss(generated_G[:ori_num, :][:, :ori_num], adj.detach().to_dense())

            else:
                features = features.cuda()
                adj = adj.cuda()
                embed = encoder(features, adj)
                generated_G = decoder(embed)
                loss_rec = utils.adj_mse_loss(generated_G[:ori_num, :][:, :ori_num], adj.detach().to_dense())

            if not args.opt_new_G:
                adj_new = copy.deepcopy(generated_G.detach())
                threshold = 0.5
                adj_new[adj_new < threshold] = 0.0
                adj_new[adj_new >= threshold] = 1.0

                # ipdb.set_trace()
                edge_ac = adj_new[:ori_num, :ori_num].eq(adj.to_dense()).double().sum() / (ori_num ** 2)
            else:
                adj_new = generated_G
                edge_ac = F.l1_loss(adj_new[:ori_num, :ori_num], adj.to_dense(), reduction='mean')
                # calculate generation information

            adj_new = torch.mul(adj_up, adj_new)
            adj_new[:ori_num, :][:, :ori_num] = adj.detach().to_dense()
            # adj_new = adj_new.to_sparse()
            # ipdb.set_trace()

            if not args.opt_new_G:
                adj_new = adj_new.detach()

            output = classifier(embed, adj_new)
            loss_train = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new])
            acc_train = utils.accuracy(output[idx_train_new], labels_new[idx_train_new])
            loss_val = F.cross_entropy(output[idx_val], labels_new[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels_new[idx_val])

            # loss=loss_train
            loss_train.backward()
            optimizer_en.step()
            optimizer_cls.step()
            # optimizer_de.step()



            epoch_loss_df = pd.DataFrame({
                'Epoch': [i_1+1],
                'acc_train': [acc_train.item()],
                'loss_train': [loss_train.item()],
                'acc_val': [acc_val.item()],
                'loss_val': [loss_val.item()],
                'time':[time.time()-t]
            })
            if not metric_values.empty:
                metric_values = pd.concat(
                    [metric_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                metric_values = epoch_loss_df

            if self._verbose:
                epoch_iterator_2.set_description(
                    description_2.format(acc=acc_train.item(), loss=loss_train.item())
                )


            if acc_best is None or acc_train > acc_best:
                acc_best = acc_train
                saved_content = {}
                saved_content['encoder'] = encoder.state_dict()
                saved_content['decoder'] = decoder.state_dict()
                saved_content['classifier'] = classifier.state_dict()
                saved_content['acc_best'] = acc_best
                saved_content['metric_value'] = metric_values
                if args.confidence is None:
                    if args.imbalance_ratio is None:
                        torch.save(saved_content,
                               'checkpoint/{}/{}_best.pth'.format(args.dataset, 'GNN'))
                    else:
                        if not os.path.exists('checkpoint/{}/{}'.format(args.dataset,args.imbalance_ratio)):
                            os.makedirs('checkpoint/{}/{}'.format(args.dataset,args.imbalance_ratio))
                        torch.save(saved_content,
                               'checkpoint/{}/{}/{}_best.pth'.format(args.dataset,args.imbalance_ratio, 'GNN'))
                else:
                    if not os.path.exists('checkpoint/{}/{}'.format(args.dataset,args.confidence)):
                        os.makedirs('checkpoint/{}/{}'.format(args.dataset,args.confidence))
                    torch.save(saved_content,
                           'checkpoint/{}/{}/{}_best.pth'.format(args.dataset,args.confidence, 'GNN'))


        saved = {}
        saved['encoder'] = encoder.state_dict()
        saved['decoder'] = decoder.state_dict()
        saved['classifier'] = classifier.state_dict()
        saved['metric_value'] = metric_values
        if args.confidence is None:
            if args.imbalance_ratio is None:
                torch.save(saved,
                           'checkpoint/{}/{}_final.pth'.format(args.dataset, 'GNN'))
            else:
                if not os.path.exists('checkpoint/{}/{}'.format(args.dataset, args.imbalance_ratio)):
                    os.makedirs('checkpoint/{}/{}'.format(args.dataset, args.imbalance_ratio))
                torch.save(saved,
                           'checkpoint/{}/{}/{}_final.pth'.format(args.dataset, args.imbalance_ratio, 'GNN'))
        else:
            if not os.path.exists('checkpoint/{}/{}'.format(args.dataset, args.confidence)):
                os.makedirs('checkpoint/{}/{}'.format(args.dataset, args.confidence))
            torch.save(saved,
                       'checkpoint/{}/{}/{}_final.pth'.format(args.dataset, args.confidence, 'GNN'))


        return features_new,labels_new,adj_new