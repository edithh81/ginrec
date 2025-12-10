import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from data_loader import GInRecDataLoader
from ginrec_simple import GInRecSimple
from utils import parse_args, set_seed, create_directories, print_model_summary, EarlyStopping, log_metrics

def train_epoch(model, optimizer, train_samples, adj_list, adj_relation, 
                n_entities, batch_size=1024, ae_weight=0.001):
    model.train()
    np.random.shuffle(train_samples)
    
    print(f"[DEBUG] Training with {len(train_samples)} samples, batch_size={batch_size}")
    
    total_loss = 0
    total_batches = 0
    
    pbar = tqdm(range(0, len(train_samples), batch_size), desc="Training batches")
    for i in pbar:
        batch = train_samples[i:i+batch_size]
        
        users = torch.LongTensor([s[0] for s in batch]).to(model.device)
        pos_items = torch.LongTensor([s[1] for s in batch]).to(model.device)
        neg_items = torch.LongTensor([s[2][0] for s in batch]).to(model.device)
        
        print(f"[DEBUG] Batch {total_batches}: users shape={users.shape}, pos_items shape={pos_items.shape}")
        
        # Forward pass
        pos_scores, ae_loss = model(users, pos_items, adj_list, adj_relation)
        neg_scores, _ = model(users, neg_items, adj_list, adj_relation)
        
        print(f"[DEBUG] pos_scores range=[{pos_scores.min().item():.4f}, {pos_scores.max().item():.4f}], ae_loss={ae_loss.item():.4f}")
        
        # BPR loss
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        loss = bpr_loss + ae_weight * ae_loss
        
        print(f"[DEBUG] bpr_loss={bpr_loss.item():.4f}, total_loss={loss.item():.4f}")
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/total_batches:.4f}'})
    
    print(f"[DEBUG] Epoch completed: avg_loss={total_loss / total_batches:.4f}")
    return total_loss / total_batches

def evaluate(model, test_data, adj_list, adj_relation, k=20):
    model.eval()
    
    print(f"[DEBUG] Evaluating on {len(test_data)} users")
    
    recalls = []
    ndcgs = []
    
    pbar = tqdm(test_data.items(), desc="Evaluating users")
    for user_id, test_items in pbar:
        if len(test_items) == 0:
            continue
        
        users = torch.LongTensor([user_id]).to(model.device)
        scores = model.predict(users, adj_list, adj_relation)[0]
        
        print(f"[DEBUG] User {user_id}: test_items={len(test_items)}, score range=[{scores.min().item():.4f}, {scores.max().item():.4f}]")
        
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
        
        pbar.set_postfix({'recall': f'{np.mean(recalls):.4f}', 'ndcg': f'{np.mean(ndcgs):.4f}'})
    
    print(f"[DEBUG] Evaluation completed: avg_recall={np.mean(recalls):.4f}, avg_ndcg={np.mean(ndcgs):.4f}")
    return np.mean(recalls), np.mean(ndcgs)

def main(config):
    print(config)
    
    # Set random seed
    set_seed(config.seed)
    
    # Create directories
    create_directories(config)
    
    print(f"[DEBUG] Starting training with data_dir={config.data_dir}")
    
    # Load data
    print(f"[DEBUG] Loading data from {config.data_dir}...")
    loader = GInRecDataLoader(config.data_dir).load_data()
    print(f"[DEBUG] Data loaded successfully")
    
    print(f"[DEBUG] Building graph structure...")
    adj_list, adj_relation, n_nodes = loader.get_graph_structure()
    
    print(f"[DEBUG] Graph loaded: n_nodes={n_nodes}, adj_list size={len(adj_list)}")
    
    n_entities = len(loader.entities)
    n_users = len(loader.users)
    
    print(f"[DEBUG] n_entities={n_entities}, n_users={n_users}")
    
    # Flatten all relations and find max
    print(f"[DEBUG] Computing number of relations...")
    all_relations = []
    for rel_list in adj_relation.values():
        all_relations.extend(rel_list)
    
    max_rel_from_adj = max(all_relations) if all_relations else 0
    max_rel_from_list = max(loader.relations.values()) if loader.relations else 0
    n_relations = max(max_rel_from_adj, max_rel_from_list) + 1
    
    print(f"[DEBUG] n_relations={n_relations} (max_rel_from_adj={max_rel_from_adj}, max_rel_from_list={max_rel_from_list})")
    
    # Create model
    print(f"[DEBUG] Using device: {config.device}")
    
    print(f"[DEBUG] Creating model...")
    model = GInRecSimple(
        n_entities, 
        n_users, 
        n_relations,
        entity_dim=config.entity_dim,
        autoencoder_dims=config.autoencoder_dims,
        conv_dims=config.conv_dims,
        dropout=config.dropout,
        device=config.device
    ).to(config.device)
    
    # Print model summary
    print_model_summary(model)
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    print(f"[DEBUG] Optimizer created")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=10, mode='max')
    
    # Training
    print(f"[DEBUG] Generating training samples (this may take a while for large datasets)...")
    train_samples = loader.get_train_samples(n_neg_samples=config.n_neg_samples)
    print(f"[DEBUG] Total training samples: {len(train_samples)}")
    
    # Save config
    config.save(f"{config.save_dir}/config.json")
    
    best_ndcg = 0
    for epoch in range(config.n_epochs):
        print(f"\n[DEBUG] ===== Epoch {epoch+1}/{config.n_epochs} =====")
        loss = train_epoch(model, optimizer, train_samples, adj_list, adj_relation,
                          n_entities, batch_size=config.batch_size, ae_weight=config.ae_weight)
        
        if (epoch + 1) % config.eval_interval == 0:
            print(f"[DEBUG] Running evaluation...")
            recall, ndcg = evaluate(model, loader.test_data, adj_list, adj_relation, k=config.top_k)
            
            metrics = {
                'loss': loss,
                f'recall@{config.top_k}': recall,
                f'ndcg@{config.top_k}': ndcg
            }
            
            print(f'Epoch {epoch+1}: Loss={loss:.4f}, Recall@{config.top_k}={recall:.4f}, NDCG@{config.top_k}={ndcg:.4f}')
            
            # Log metrics
            log_metrics(f"{config.save_dir}/{config.log_file}", epoch+1, metrics)
            
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                print(f"[DEBUG] New best NDCG! Saving model...")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ndcg': best_ndcg,
                    'recall': recall,
                }, f"{config.save_dir}/best_model.pt")
            
            # Early stopping check
            if early_stopping(ndcg):
                print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
                break
    
    print(f'\n[DEBUG] Training completed!')
    print(f'Best NDCG@{config.top_k}: {best_ndcg:.4f}')
    
    return best_ndcg

if __name__ == '__main__':
    config = parse_args()
    main(config)
