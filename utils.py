import argparse
import json
import torch
from pathlib import Path

class TrainingConfig:
    """Configuration class for training"""
    def __init__(self, **kwargs):
        # Data parameters
        self.data_dir = kwargs.get('data_dir', 'amazon-book')
        
        # Model parameters
        self.entity_dim = kwargs.get('entity_dim', 64)
        self.autoencoder_dims = kwargs.get('autoencoder_dims', [128, 32])
        self.conv_dims = kwargs.get('conv_dims', [32, 16])
        self.dropout = kwargs.get('dropout', 0.1)
        self.gate_type = kwargs.get('gate_type', 'concat')
        
        # Training parameters
        self.n_epochs = kwargs.get('n_epochs', 100)
        self.batch_size = kwargs.get('batch_size', 1024)
        self.lr = kwargs.get('lr', 0.001)
        self.ae_weight = kwargs.get('ae_weight', 0.001)
        self.n_neg_samples = kwargs.get('n_neg_samples', 1)
        
        # Evaluation parameters
        self.eval_interval = kwargs.get('eval_interval', 5)
        self.top_k = kwargs.get('top_k', 20)
        
        # System parameters
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = kwargs.get('seed', 42)
        self.save_dir = kwargs.get('save_dir', 'checkpoints')
        self.log_file = kwargs.get('log_file', 'training.log')
        
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, filepath):
        """Save config to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[INFO] Config saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        print(f"[INFO] Config loaded from {filepath}")
        return cls(**config_dict)
    
    def __str__(self):
        """Pretty print configuration"""
        lines = ["=" * 60, "Training Configuration", "=" * 60]
        for key, value in self.to_dict().items():
            lines.append(f"  {key:20s}: {value}")
        lines.append("=" * 60)
        return "\n".join(lines)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train GInRec model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='amazon-book',
                        help='Directory containing dataset')
    
    # Model arguments
    parser.add_argument('--entity_dim', type=int, default=64,
                        help='Entity embedding dimension')
    parser.add_argument('--autoencoder_dims', type=int, nargs='+', default=[128, 32],
                        help='Autoencoder hidden dimensions')
    parser.add_argument('--conv_dims', type=int, nargs='+', default=[32, 16],
                        help='Graph convolution layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--gate_type', type=str, default='concat',
                        choices=['concat', 'inner_product', 'none'],
                        help='Gate type for graph convolution')
    
    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--ae_weight', type=float, default=0.001,
                        help='Weight for autoencoder loss')
    parser.add_argument('--n_neg_samples', type=int, default=1,
                        help='Number of negative samples per positive')
    
    # Evaluation arguments
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='Evaluate every N epochs')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Top-K for evaluation metrics')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_file', type=str, default='training.log',
                        help='Log file path')
    parser.add_argument('--config', type=str, default=None,
                        help='Load config from JSON file')
    
    args = parser.parse_args()
    
    # Load from config file if specified
    if args.config:
        config = TrainingConfig.load(args.config)
    else:
        # Convert args to config
        args_dict = vars(args)
        if args_dict['device'] == 'auto':
            args_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = TrainingConfig(**args_dict)
    
    return config


def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"[INFO] Random seed set to {seed}")


def create_directories(config):
    """Create necessary directories"""
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Save directory: {config.save_dir}")


def log_metrics(filepath, epoch, metrics, mode='a'):
    """Log metrics to file"""
    with open(filepath, mode) as f:
        f.write(f"Epoch {epoch}: {metrics}\n")


class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience=10, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        self.monitor_op = lambda x, y: x > y + min_delta if mode == 'max' else x < y - min_delta
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def print_model_summary(model):
    """Print model architecture and parameter count"""
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        print(f"  {name:40s}: {num_params:>10,d} params")
    
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    # Test configuration
    config = TrainingConfig(
        data_dir='amazon-book',
        n_epochs=50,
        lr=0.001
    )
    
    print(config)
    
    # Test save/load
    config.save('test_config.json')
    loaded_config = TrainingConfig.load('test_config.json')
    print(loaded_config)
