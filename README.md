# GInRec Simplified - PyTorch Only Implementation

A simplified implementation of GInRec (Graph-Integrated Recommendation) without DGL dependencies, compatible with Python 3.10+.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- tqdm >= 4.65.0

## Installation

```bash
pip install -r requirements.txt
```

## Data Format

Your data directory should contain the following files:

### entity_list.txt
```
org_id remap_id
m.045wq1q 0
m.03_28m 1
```

### item_list.txt
```
org_id remap_id freebase_id
0553092626 0 m.045wq1q
0393316041 1 m.03_28m
```

### kg.txt
```
24915 0 24916
24917 1 5117
```

### relation_list.txt
```
org_id remap_id
http://rdf.freebase.com/ns/type.object.type 0
http://rdf.freebase.com/ns/type.type.instance 1
```

### train.txt
```
0 0 1 2 3 4 5 6 7 8 9
1 32 33 34 35
```

### test.txt
```
0 10 11 12
1 36 37
```

### user_list.txt
```
org_id remap_id
A3RTKL9KB8KLID 0
A38LAIK2N83NH0 1
```

## Usage

### Basic Training

```python
from train_simple import main

# Train with default parameters
main('path/to/your/data', n_epochs=100, lr=0.001, batch_size=8)
```

### Custom Configuration

```python
from data_loader import GInRecDataLoader
from ginrec_simple import GInRecSimple
import torch

# Load data
loader = GInRecDataLoader('amazon-book').load_data()
adj_list, adj_relation, n_nodes = loader.get_graph_structure()

# Create model with custom parameters
model = GInRecSimple(
    n_entities=len(loader.entities),
    n_users=len(loader.users),
    n_relations=10,
    entity_dim=64,
    autoencoder_dims=[128, 32],
    conv_dims=[32, 16],
    dropout=0.1,
    device='cuda'
)

# Train...
```

## Model Architecture

1. **Autoencoder**: Compresses entity/user features into latent space
   - Default: [128, 32] dimensions
   - Reconstructs original features for regularization

2. **Graph Convolution Layers**: Propagates information through the knowledge graph
   - Default: 2 layers with dimensions [32, 16]
   - Relation-aware gating mechanism
   - Aggregates neighbor information

3. **Prediction**: Computes user-item compatibility scores
   - Inner product of user and item embeddings

## Hyperparameters

- `n_epochs`: Number of training epochs (default: 100)
- `lr`: Learning rate (default: 0.001)
- `batch_size`: Batch size for training (default: 8)
- `entity_dim`: Initial entity embedding dimension (default: 64)
- `autoencoder_dims`: Autoencoder layer dimensions (default: [128, 32])
- `conv_dims`: Graph convolution layer dimensions (default: [32, 16])
- `dropout`: Dropout rate (default: 0.1)
- `ae_weight`: Autoencoder loss weight (default: 0.001)

## Evaluation Metrics

- **Recall@20**: Fraction of relevant items in top-20 recommendations
- **NDCG@20**: Normalized Discounted Cumulative Gain at 20

## Output

The model saves the best checkpoint as `best_model.pt` based on NDCG@20 score.

## Differences from Original GInRec

1. No DGL dependency - pure PyTorch implementation
2. Simplified graph convolution using adjacency lists
3. Removed some advanced features for easier reproduction
4. Compatible with Python 3.10+

## Citation

If you use this implementation, please cite the original GInRec paper:

```
@inproceedings{ginrec,
  title={GInRec: Graph-Integrated Recommendation},
  author={...},
  booktitle={...},
  year={...}
}
```
