import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, use_activation=True):
        super().__init__()
        self.use_activation = use_activation
        
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        
        self.encoder = nn.ModuleList(layers)
        
        layers = []
        for i in range(len(hidden_dims) - 1, -1, -1):
            out_dim = input_dim if i == 0 else hidden_dims[i-1]
            layers.append(nn.Linear(hidden_dims[i], out_dim))
        
        self.decoder = nn.ModuleList(layers)
        self.activation = nn.ReLU()
        
    def encode(self, x):
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if self.use_activation and i < len(self.encoder) - 1:
                x = self.activation(x)
        return x
    
    def decode(self, x):
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if self.use_activation and i < len(self.decoder) - 1:
                x = self.activation(x)
        return x
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded

class GraphConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n_relations, gate_type='concat'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gate_type = gate_type
        
        # Relation-specific transformations
        if gate_type == 'concat':
            self.W_r = nn.Parameter(torch.Tensor(n_relations, input_dim * 2, input_dim))
        else:
            self.W_r = nn.Parameter(torch.Tensor(n_relations, input_dim, input_dim))
        
        self.linear = nn.Linear(input_dim * 2, output_dim)
        self.activation = nn.LeakyReLU()
        self.gate_fn = nn.Sigmoid()
        
        nn.init.xavier_uniform_(self.W_r)
        
    def forward(self, x, adj_list, adj_relation):
        n_nodes = x.size(0)
        aggregated = torch.zeros(n_nodes, self.input_dim, device=x.device)
        
        for node in range(n_nodes):
            if node not in adj_list:
                continue
            
            neighbors = adj_list[node]
            relations = adj_relation[node]
            
            if len(neighbors) == 0:
                continue
            
            neighbor_embs = x[neighbors]
            node_emb = x[node].unsqueeze(0).expand(len(neighbors), -1)
            
            # Compute gates based on relations
            gates = []
            for i, rel in enumerate(relations):
                if self.gate_type == 'concat':
                    concat_emb = torch.cat([node_emb[i], neighbor_embs[i]], dim=-1)
                    gate = torch.matmul(concat_emb, self.W_r[rel])
                else:
                    gate = torch.matmul(node_emb[i], self.W_r[rel]) * neighbor_embs[i]
                gates.append(gate)
            
            gates = torch.stack(gates)
            gates = self.gate_fn(gates.sum(dim=-1, keepdim=True))
            
            # Aggregate with gates
            weighted_neighbors = neighbor_embs * gates
            aggregated[node] = weighted_neighbors.mean(dim=0)
        
        # Combine self and neighbor information
        combined = torch.cat([x, aggregated], dim=-1)
        output = self.linear(combined)
        output = self.activation(output)
        
        return output

class GInRecSimple(nn.Module):
    def __init__(self, n_entities, n_users, n_relations, entity_dim=64, 
                 autoencoder_dims=[128, 32], conv_dims=[32, 16], 
                 dropout=0.1, device='cpu'):
        super().__init__()
        
        self.n_entities = n_entities
        self.n_users = n_users
        self.device = device
        
        # Initialize embeddings
        self.entity_embeddings = nn.Parameter(torch.randn(n_entities, entity_dim))
        self.user_embeddings = nn.Parameter(torch.randn(n_users, entity_dim))
        nn.init.xavier_uniform_(self.entity_embeddings)
        nn.init.xavier_uniform_(self.user_embeddings)
        
        # Autoencoder
        self.autoencoder = Autoencoder(entity_dim, autoencoder_dims)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        input_dim = autoencoder_dims[-1]
        for output_dim in conv_dims:
            self.conv_layers.append(GraphConvLayer(input_dim, output_dim, n_relations))
            self.dropouts.append(nn.Dropout(dropout))
            input_dim = output_dim
        
        self.final_dim = sum(conv_dims)
        
    def forward(self, users, items, adj_list, adj_relation):
        # Combine entity and user embeddings
        all_embeddings = torch.cat([self.entity_embeddings, self.user_embeddings], dim=0)
        
        # Encode through autoencoder
        encoded, decoded = self.autoencoder(all_embeddings)
        ae_loss = nn.functional.mse_loss(decoded, all_embeddings)
        
        # Graph convolution
        x = encoded
        layer_outputs = []
        for conv, dropout in zip(self.conv_layers, self.dropouts):
            x = dropout(x)
            x = conv(x, adj_list, adj_relation)
            layer_outputs.append(x)
        
        # Concatenate all layers
        final_embeddings = torch.cat(layer_outputs, dim=-1)
        
        # Get user and item embeddings
        user_nodes = self.n_entities + users
        user_embs = final_embeddings[user_nodes]
        item_embs = final_embeddings[items]
        
        # Compute scores
        scores = (user_embs * item_embs).sum(dim=-1)
        
        return scores, ae_loss
    
    def predict(self, users, adj_list, adj_relation):
        with torch.no_grad():
            all_embeddings = torch.cat([self.entity_embeddings, self.user_embeddings], dim=0)
            encoded, _ = self.autoencoder(all_embeddings)
            
            x = encoded
            layer_outputs = []
            for conv in self.conv_layers:
                x = conv(x, adj_list, adj_relation)
                layer_outputs.append(x)
            
            final_embeddings = torch.cat(layer_outputs, dim=-1)
            
            user_nodes = self.n_entities + users
            user_embs = final_embeddings[user_nodes]
            item_embs = final_embeddings[:self.n_entities]
            
            # Compute all scores
            scores = torch.matmul(user_embs, item_embs.T)
            
        return scores
