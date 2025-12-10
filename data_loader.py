import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

class GInRecDataLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.entities = {}
        self.items = {}
        self.relations = {}
        self.users = {}
        self.kg_triplets = []
        self.train_data = {}
        self.test_data = {}
        
    def load_data(self):
        # Load entities
        with open(self.data_dir / 'entity_list.txt', 'r') as f:
            next(f)  # Skip header
            for line in f:
                # res = line.strip().split()
                # print(res)
                org_id, remap_id = line.strip().split()
                self.entities[org_id] = int(remap_id)
        
        # Load items
        with open(self.data_dir / 'item_list.txt', 'r') as f:
            next(f)  # Skip header
            for line in f:
                org_id, remap_id, freebase_id = line.strip().split()
                self.items[org_id] = {'remap_id': int(remap_id), 'entity_id': freebase_id}
        
        # Load relations
        with open(self.data_dir / 'relation_list.txt', 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split()
                self.relations[parts[0]] = int(parts[1])
        
        # Load users
        with open(self.data_dir / 'user_list.txt', 'r') as f:
            next(f)  # Skip header
            for line in f:
                org_id, remap_id = line.strip().split()
                self.users[org_id] = int(remap_id)
        
        # Load KG
        with open(self.data_dir / 'kg.txt', 'r') as f:
            for line in f:
                h, r, t = map(int, line.strip().split())
                self.kg_triplets.append((h, r, t))
        
        # Load train data
        with open(self.data_dir / 'train.txt', 'r') as f:
            for line in f:
                parts = list(map(int, line.strip().split()))
                user_id = parts[0]
                items = parts[1:]
                self.train_data[user_id] = items
        
        # Load test data
        with open(self.data_dir / 'test.txt', 'r') as f:
            for line in f:
                parts = list(map(int, line.strip().split()))
                user_id = parts[0]
                items = parts[1:]
                self.test_data[user_id] = items
        
        return self
    
    def get_graph_structure(self):
        """Convert KG triplets to adjacency structure"""
        n_entities = len(self.entities)
        n_users = len(self.users)
        n_nodes = n_entities + n_users
        
        # Build adjacency lists
        adj_list = defaultdict(list)
        adj_relation = defaultdict(list)
        
        # Add KG edges
        for h, r, t in self.kg_triplets:
            adj_list[h].append(t)
            adj_relation[h].append(r)
            # Add reverse edges
            adj_list[t].append(h)
            adj_relation[t].append(r)
        
        # Add user-item interactions
        interaction_relation = max(self.relations.values()) + 1 if self.relations else 0
        for user_id, items in self.train_data.items():
            user_node = n_entities + user_id
            for item_id in items:
                adj_list[user_node].append(item_id)
                adj_relation[user_node].append(interaction_relation)
                adj_list[item_id].append(user_node)
                adj_relation[item_id].append(interaction_relation)
        
        return adj_list, adj_relation, n_nodes
    
    def get_train_samples(self, n_neg_samples=1):
        """Generate training samples with negative sampling - OPTIMIZED"""
        print(f"[DEBUG] Generating training samples with {n_neg_samples} negative samples per positive...")
        
        samples = []
        all_items = np.arange(len(self.items))
        
        # Pre-compute for faster sampling
        print(f"[DEBUG] Pre-computing negative item pools...")
        for user_id, pos_items in self.train_data.items():
            pos_items_set = set(pos_items)
            # Create boolean mask for faster filtering
            neg_mask = np.ones(len(self.items), dtype=bool)
            neg_mask[list(pos_items_set)] = False
            neg_pool = all_items[neg_mask]
            
            # Vectorized sampling
            if len(neg_pool) >= n_neg_samples * len(pos_items):
                # Sample all negatives at once for this user
                all_neg_items = np.random.choice(neg_pool, 
                                                 size=n_neg_samples * len(pos_items), 
                                                 replace=False)
                
                for idx, pos_item in enumerate(pos_items):
                    neg_items = all_neg_items[idx * n_neg_samples:(idx + 1) * n_neg_samples]
                    samples.append((user_id, pos_item, neg_items))
            else:
                # Fallback for users with very few negative items
                for pos_item in pos_items:
                    sample_size = min(n_neg_samples, len(neg_pool))
                    neg_items = np.random.choice(neg_pool, sample_size, replace=False)
                    samples.append((user_id, pos_item, neg_items))
        
        print(f"[DEBUG] Generated {len(samples)} training samples")
        return samples
