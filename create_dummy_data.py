import os
import random
from pathlib import Path

def create_dummy_data(output_dir='dummy', n_users=50, n_items=100, n_entities=200, 
                     n_relations=10, n_kg_triplets=500, items_per_user=5):
    """
    Create dummy data for GInRec testing
    
    Args:
        output_dir: Directory to save dummy data
        n_users: Number of users
        n_items: Number of items (subset of entities)
        n_entities: Total number of entities (must be >= n_items)
        n_relations: Number of relation types
        n_kg_triplets: Number of KG triplets
        items_per_user: Average items per user
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Creating dummy data in {output_dir}/")
    print(f"Settings: {n_users} users, {n_items} items, {n_entities} entities, {n_relations} relations")
    
    # 1. Create entity_list.txt
    print("Creating entity_list.txt...")
    with open(output_path / 'entity_list.txt', 'w') as f:
        f.write("org_id\tremap_id\n")
        for i in range(n_entities):
            f.write(f"entity_{i}\t{i}\n")
    
    # 2. Create item_list.txt (items are subset of entities)
    print("Creating item_list.txt...")
    with open(output_path / 'item_list.txt', 'w') as f:
        f.write("org_id\tremap_id\tfreebase_id\n")
        for i in range(n_items):
            f.write(f"item_{i}\t{i}\tentity_{i}\n")
    
    # 3. Create relation_list.txt
    print("Creating relation_list.txt...")
    with open(output_path / 'relation_list.txt', 'w') as f:
        f.write("relation\tremap_id\n")
        for i in range(n_relations):
            f.write(f"relation_{i}\t{i}\n")
    
    # 4. Create user_list.txt
    print("Creating user_list.txt...")
    with open(output_path / 'user_list.txt', 'w') as f:
        f.write("org_id\tremap_id\n")
        for i in range(n_users):
            f.write(f"user_{i}\t{i}\n")
    
    # 5. Create kg.txt (knowledge graph triplets)
    print("Creating kg.txt...")
    with open(output_path / 'kg.txt', 'w') as f:
        for _ in range(n_kg_triplets):
            h = random.randint(0, n_entities - 1)
            r = random.randint(0, n_relations - 1)
            t = random.randint(0, n_entities - 1)
            if h != t:  # Avoid self-loops
                f.write(f"{h}\t{r}\t{t}\n")
    
    # 6. Create train.txt and test.txt
    print("Creating train.txt and test.txt...")
    train_ratio = 0.8
    
    with open(output_path / 'train.txt', 'w') as train_f, \
         open(output_path / 'test.txt', 'w') as test_f:
        
        for user_id in range(n_users):
            # Generate random items for this user
            n_user_items = random.randint(max(1, items_per_user - 2), items_per_user + 2)
            user_items = random.sample(range(n_items), min(n_user_items, n_items))
            
            # Split into train and test
            n_train = max(1, int(len(user_items) * train_ratio))
            train_items = user_items[:n_train]
            test_items = user_items[n_train:]
            
            # Write train data
            if train_items:
                train_f.write(f"{user_id}")
                for item in train_items:
                    train_f.write(f"\t{item}")
                train_f.write("\n")
            
            # Write test data
            if test_items:
                test_f.write(f"{user_id}")
                for item in test_items:
                    test_f.write(f"\t{item}")
                test_f.write("\n")
    
    print(f"\nDummy data created successfully in {output_dir}/")
    print(f"Files created:")
    print(f"  - entity_list.txt ({n_entities} entities)")
    print(f"  - item_list.txt ({n_items} items)")
    print(f"  - relation_list.txt ({n_relations} relations)")
    print(f"  - user_list.txt ({n_users} users)")
    print(f"  - kg.txt ({n_kg_triplets} triplets)")
    print(f"  - train.txt")
    print(f"  - test.txt")
    
    # Print statistics
    print(f"\nStatistics:")
    with open(output_path / 'train.txt', 'r') as f:
        train_lines = len(f.readlines())
    with open(output_path / 'test.txt', 'r') as f:
        test_lines = len(f.readlines())
    print(f"  - Training users: {train_lines}")
    print(f"  - Test users: {test_lines}")

if __name__ == '__main__':
    # Create small dummy data for quick testing
    create_dummy_data(
        output_dir='dummy',
        n_users=50,
        n_items=100,
        n_entities=200,
        n_relations=10,
        n_kg_triplets=500,
        items_per_user=5
    )
    
    print("\n" + "="*60)
    print("To test with this dummy data, run:")
    print("  python train_simple.py")
    print("And modify main() to use: main('dummy', n_epochs=10, batch_size=16)")
    print("="*60)
