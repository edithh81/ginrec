import time
import numpy as np
from data_loader import GInRecDataLoader

def test_data_loading(data_dir='dummy'):
    print("="*60)
    print("Testing Data Loader")
    print("="*60)
    
    # Test 1: Load data
    print("\n[TEST 1] Loading data files...")
    start = time.time()
    loader = GInRecDataLoader(data_dir).load_data()
    load_time = time.time() - start
    print(f"✓ Data loaded in {load_time:.2f}s")
    print(f"  - Entities: {len(loader.entities)}")
    print(f"  - Items: {len(loader.items)}")
    print(f"  - Relations: {len(loader.relations)}")
    print(f"  - Users: {len(loader.users)}")
    print(f"  - KG triplets: {len(loader.kg_triplets)}")
    print(f"  - Train interactions: {sum(len(items) for items in loader.train_data.values())}")
    print(f"  - Test interactions: {sum(len(items) for items in loader.test_data.values())}")
    
    # Test 2: Build graph structure
    print("\n[TEST 2] Building graph structure...")
    start = time.time()
    adj_list, adj_relation, n_nodes = loader.get_graph_structure()
    graph_time = time.time() - start
    print(f"✓ Graph built in {graph_time:.2f}s")
    print(f"  - Total nodes: {n_nodes}")
    print(f"  - Nodes with edges: {len(adj_list)}")
    
    # Sample some statistics
    edge_counts = [len(neighbors) for neighbors in adj_list.values()]
    print(f"  - Avg edges per node: {np.mean(edge_counts):.2f}")
    print(f"  - Max edges per node: {max(edge_counts)}")
    print(f"  - Min edges per node: {min(edge_counts)}")
    
    # Test 3: Generate training samples
    print("\n[TEST 3] Generating training samples...")
    start = time.time()
    train_samples = loader.get_train_samples(n_neg_samples=1)
    sample_time = time.time() - start
    print(f"✓ Samples generated in {sample_time:.2f}s")
    print(f"  - Total samples: {len(train_samples)}")
    print(f"  - Samples per second: {len(train_samples)/sample_time:.0f}")
    
    # Verify sample format
    print("\n[TEST 4] Verifying sample format...")
    sample = train_samples[0]
    print(f"  - Sample format: (user_id, pos_item, neg_items)")
    print(f"  - Example: user={sample[0]}, pos_item={sample[1]}, neg_items={sample[2]}")
    print(f"  - Negative samples shape: {np.array(sample[2]).shape}")
    
    # Test 5: Verify no overlap between pos and neg
    print("\n[TEST 5] Verifying positive/negative separation...")
    errors = 0
    for user_id, pos_item, neg_items in train_samples[:100]:  # Check first 100
        if pos_item in neg_items:
            errors += 1
    print(f"✓ Checked 100 samples, errors: {errors}")
    
    # Test 6: Check relation IDs
    print("\n[TEST 6] Checking relation IDs...")
    all_relations = []
    for rel_list in adj_relation.values():
        all_relations.extend(rel_list)
    
    if all_relations:
        print(f"  - Min relation ID: {min(all_relations)}")
        print(f"  - Max relation ID: {max(all_relations)}")
        print(f"  - Unique relations: {len(set(all_relations))}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total time: {load_time + graph_time + sample_time:.2f}s")
    print(f"  - Data loading: {load_time:.2f}s")
    print(f"  - Graph building: {graph_time:.2f}s")
    print(f"  - Sample generation: {sample_time:.2f}s")
    print("="*60)
    
    return loader, adj_list, adj_relation, train_samples

def test_sampling_speed_comparison(loader, n_trials=3):
    """Compare different negative sampling strategies"""
    print("\n" + "="*60)
    print("SAMPLING SPEED COMPARISON")
    print("="*60)
    
    for n_neg in [1, 5, 10]:
        times = []
        for trial in range(n_trials):
            start = time.time()
            samples = loader.get_train_samples(n_neg_samples=n_neg)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        print(f"\nNegative samples = {n_neg}")
        print(f"  - Avg time: {avg_time:.2f}s")
        print(f"  - Total samples: {len(samples)}")
        print(f"  - Samples/sec: {len(samples)/avg_time:.0f}")

if __name__ == '__main__':
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'dummy'
    print(f"Testing with dataset: {data_dir}\n")
    
    loader, adj_list, adj_relation, train_samples = test_data_loading(data_dir)
    
    # Optional: Test sampling speed with different parameters
    user_input = input("\nRun sampling speed comparison? (y/n): ")
    if user_input.lower() == 'y':
        test_sampling_speed_comparison(loader)
    
    print("\n✓ All tests completed!")
