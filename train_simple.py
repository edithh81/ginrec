import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from data_loader import GInRecDataLoader
from ginrec_simple import GInRecSimple

def train_epoch(model, optimizer, train_samples, adj_list, adj_relation, 
                n_entities, batch_size=1024, ae_weight=0.001):
    model.train()
    np.random.shuffle(train_samples)
    
    total_loss = 0
    total_batches = 0
    
    for i in range(0, len(train_samples), batch_size):
        batch = train_samples[i:i+batch_size]
        
        users = torch.LongTensor([s[0] for s in batch]).to(model.device)
        pos_items = torch.LongTensor([s[1] for s in batch]).to(model.device)
        neg_items = torch.LongTensor([s[2][0] for s in batch]).to(model.device)
        
        # Forward pass
        pos_scores, ae_loss = model(users, pos_items, adj_list, adj_relation)
        neg_scores, _ = model(users, neg_items, adj_list, adj_relation)
        
        # BPR loss
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        loss = bpr_loss + ae_weight * ae_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_batches += 1
    
    return total_loss / total_batches

def evaluate(model, test_data, adj_list, adj_relation, k=20):
    model.eval()
    
    recalls = []
    ndcgs = []
    
    for user_id, test_items in test_data.items():
        if len(test_items) == 0:
            continue
        
        users = torch.LongTensor([user_id]).to(model.device)
        scores = model.predict(users, adj_list, adj_relation)[0]
        
        # Get top-k items
        _, top_k = torch.topk(scores, k)
        top_k = top_k.cpu().numpy()
        
        # Compute metrics
        hits = len(set(top_k) & set(test_items))
        recall = hits / len(test_items)
        recalls.append(recall)
        
        # NDCG
        dcg = sum([1 / np.log2(i + 2) for i, item in enumerate(top_k) if item in test_items])
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(test_items), k))])
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)
    
    return np.mean(recalls), np.mean(ndcgs)

def main(data_dir, n_epochs=100, lr=0.001, batch_size=8):
    # Load data
    loader = GInRecDataLoader(data_dir).load_data()
    adj_list, adj_relation, n_nodes = loader.get_graph_structure()
    
    n_entities = len(loader.entities)
    n_users = len(loader.users)
    
    # Flatten all relations and find max
    all_relations = []
    for rel_list in adj_relation.values():
        all_relations.extend(rel_list)
    
    max_rel_from_adj = max(all_relations) if all_relations else 0
    max_rel_from_list = max(loader.relations.values()) if loader.relations else 0
    n_relations = max(max_rel_from_adj, max_rel_from_list) + 1
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GInRecSimple(n_entities, n_users, n_relations, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training
    train_samples = loader.get_train_samples()
    
    best_ndcg = 0
    for epoch in range(n_epochs):
        loss = train_epoch(model, optimizer, train_samples, adj_list, adj_relation,
                          n_entities, batch_size)
        
        if (epoch + 1) % 5 == 0:
            recall, ndcg = evaluate(model, loader.test_data, adj_list, adj_relation)
            print(f'Epoch {epoch+1}: Loss={loss:.4f}, Recall@20={recall:.4f}, NDCG@20={ndcg:.4f}')
            
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                torch.save(model.state_dict(), 'best_model.pt')
    
    print(f'Best NDCG@20: {best_ndcg:.4f}')

if __name__ == '__main__':
    main('amazon-book')
