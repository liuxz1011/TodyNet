from math import ceil

from layer import *


class GNNStack(nn.Module):
    """ The stack layers of GNN.

    """

    def __init__(self, gnn_model_type, num_layers, groups, pool_ratio, kern_size, 
                 in_dim, hidden_dim, out_dim, 
                 seq_len, num_nodes, num_classes, dropout=0.5, activation=nn.ReLU()):

        super().__init__()
        
        # TODO: Sparsity Analysis
        k_neighs = self.num_nodes = num_nodes
        
        self.num_graphs = groups
        
        self.num_feats = seq_len
        if seq_len % groups:
            self.num_feats += ( groups - seq_len % groups )
        self.g_constr = multi_shallow_embedding(num_nodes, k_neighs, self.num_graphs)
        
        gnn_model, heads = self.build_gnn_model(gnn_model_type)
        
        assert num_layers >= 1, 'Error: Number of layers is invalid.'
        assert num_layers == len(kern_size), 'Error: Number of kernel_size should equal to number of layers.'
        paddings = [ (k - 1) // 2 for k in kern_size ]
        
        self.tconvs = nn.ModuleList(
            [nn.Conv2d(1, in_dim, (1, kern_size[0]), padding=(0, paddings[0]))] + 
            [nn.Conv2d(heads * in_dim, hidden_dim, (1, kern_size[layer+1]), padding=(0, paddings[layer+1])) for layer in range(num_layers - 2)] + 
            [nn.Conv2d(heads * hidden_dim, out_dim, (1, kern_size[-1]), padding=(0, paddings[-1]))]
        )
        
        self.gconvs = nn.ModuleList(
            [gnn_model(in_dim, heads * in_dim, groups)] + 
            [gnn_model(hidden_dim, heads * hidden_dim, groups) for _ in range(num_layers - 2)] + 
            [gnn_model(out_dim, heads * out_dim, groups)]
        )
        
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(heads * in_dim)] + 
            [nn.BatchNorm2d(heads * hidden_dim) for _ in range(num_layers - 2)] + 
            [nn.BatchNorm2d(heads * out_dim)]
        )
        
        self.left_num_nodes = []
        for layer in range(num_layers + 1):
            left_node = round( num_nodes * (1 - (pool_ratio*layer)) )
            if left_node > 0:
                self.left_num_nodes.append(left_node)
            else:
                self.left_num_nodes.append(1)
        self.diffpool = nn.ModuleList(
            [Dense_TimeDiffPool2d(self.left_num_nodes[layer], self.left_num_nodes[layer+1], kern_size[layer], paddings[layer]) for layer in range(num_layers - 1)] + 
            [Dense_TimeDiffPool2d(self.left_num_nodes[-2], self.left_num_nodes[-1], kern_size[-1], paddings[-1])]
        )
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        
        self.softmax = nn.Softmax(dim=-1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.linear = nn.Linear(heads * out_dim, num_classes)
        
        self.reset_parameters()
        
        
    def reset_parameters(self):
        for tconv, gconv, bn, pool in zip(self.tconvs, self.gconvs, self.bns, self.diffpool):
            tconv.reset_parameters()
            gconv.reset_parameters()
            bn.reset_parameters()
            pool.reset_parameters()
        
        self.linear.reset_parameters()
        
        
    def build_gnn_model(self, model_type):
        if model_type == 'dyGCN2d':
            return DenseGCNConv2d, 1
        if model_type == 'dyGIN2d':
            return DenseGINConv2d, 1
        

    def forward(self, inputs: Tensor):
        
        if inputs.size(-1) % self.num_graphs:
            pad_size = (self.num_graphs - inputs.size(-1) % self.num_graphs) / 2
            x = F.pad(inputs, (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
        else:
            x = inputs
            
        adj = self.g_constr(x.device)
        
        for tconv, gconv, bn, pool in zip(self.tconvs, self.gconvs, self.bns, self.diffpool):
            
            x, adj = pool( gconv( tconv(x), adj ), adj )
            
            x = self.activation( bn(x) )
            
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        out = self.global_pool(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
