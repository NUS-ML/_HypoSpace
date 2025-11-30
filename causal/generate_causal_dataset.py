import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import json
import math
import numpy as np
import networkx as nx
from itertools import combinations
from datetime import datetime
import random
from modules.models import CausalGraph

def _combo_has_unique_perturbed_nodes(combo) -> bool:
    """True iff no two observations perturb the same node."""
    return len({o.perturbed_node for o in combo}) == len(combo)

class PerturbationObservation:
    """Represents a perturbation and its effects."""
    
    def __init__(self, perturbed_node: str, effects: Dict[str, int]):
        """
        Args:
            perturbed_node: The node that was perturbed
            effects: Dictionary mapping node names to binary effects (0 or 1)
        """
        self.perturbed_node = perturbed_node
        self.effects = effects
    
    def to_tuple(self) -> Tuple:
        """Convert to hashable tuple for comparison."""
        sorted_effects = tuple(sorted(self.effects.items()))
        return (self.perturbed_node, sorted_effects)
    
    def to_string(self) -> str:
        """Human-readable string representation."""
        effect_str = " ".join([f"{node}:{val}" for node, val in sorted(self.effects.items())])
        return f"Perturb({self.perturbed_node}) -> {effect_str}"
    
    def __eq__(self, other) -> bool:
        return self.to_tuple() == other.to_tuple()
    
    def __hash__(self) -> int:
        return hash(self.to_tuple())

class CausalDatasetGenerator:
    """Generate datasets with all possible observation combinations."""
    
    @staticmethod
    def get_perturbation_effects(
        graph: CausalGraph,
        perturbed_node: str,
        *,
        desc_map: Optional[Dict[str, Set[str]]] = None
    ) -> PerturbationObservation:
        """
        Effects of perturbing a node:
        - perturbed node -> 0
        - descendants(perturbed) -> 1
        - everyone else -> 0
        """
        G = graph.to_networkx()
        if desc_map is None:
            # Precompute descendants for all nodes if not provided
            desc_map = {n: nx.descendants(G, n) for n in G.nodes}

        effects = {n: (1 if n in desc_map[perturbed_node] else 0) for n in graph.nodes}
        effects[perturbed_node] = 0  # ensure intervention node is 0
        return PerturbationObservation(perturbed_node, effects)
        
    @staticmethod
    def generate_all_dags(nodes: List[str], max_edges: Optional[int] = None) -> List[CausalGraph]:
        """
        Generate ALL possible DAGs with the given nodes.
        
        Args:
            nodes: List of node names
            max_edges: Maximum number of edges (None for no limit)
            
        Returns:
            List of all possible DAGs
        """
        all_dags = []
        all_possible_edges = [(i, j) for i in nodes for j in nodes if i != j]
        
        if max_edges is None:
            max_edges = len(all_possible_edges)
        
        # Try all possible edge combinations
        for edge_count in range(min(max_edges + 1, len(all_possible_edges) + 1)):
            for edge_combo in combinations(all_possible_edges, edge_count):
                # Check if this forms a valid DAG
                test_graph = CausalGraph(nodes=nodes, edges=list(edge_combo))
                G = test_graph.to_networkx()
                
                if nx.is_directed_acyclic_graph(G):
                    all_dags.append(test_graph)
        
        return all_dags
    
    @staticmethod
    def generate_all_observation_combinations(
        nodes: List[str],
        n_observations: int,
        max_edges: Optional[int] = None,
        seed: Optional[int] = None,
        n_samples: Optional[int] = None
    ) -> List[Dict]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 优化 DAG 生成
        if max_edges is None:
            max_edges = min(len(nodes) * 2, len(nodes) * (len(nodes) - 1) // 2)  # 默认限制最大边数
        all_dags = CausalDatasetGenerator.generate_all_dags(nodes, max_edges)

        # 优化兼容性检查
        dag_caches = []
        for dag in all_dags:
            G = dag.to_networkx()
            desc_map = {n: nx.descendants(G, n) for n in G.nodes}
            effects_by_node = {
                n: PerturbationObservation(n, {m: (1 if m in desc_map[n] else 0) for m in dag.nodes})
                for n in dag.nodes
            }
            dag_caches.append((dag, desc_map, effects_by_node))

        # 优化组合生成
        all_possible_observations = {
            effects_by_node[n] for _, _, effects_by_node in dag_caches for n in nodes
        }
        all_possible_observations = sorted(all_possible_observations, key=lambda o: (o.perturbed_node, sorted(o.effects.items())))

        if n_samples is not None:
            observation_combinations = random.sample(
                [combo for combo in combinations(all_possible_observations, n_observations) if _combo_has_unique_perturbed_nodes(combo)],
                n_samples
            )
        else:
            observation_combinations = [
                combo for combo in combinations(all_possible_observations, n_observations)
                if _combo_has_unique_perturbed_nodes(combo)
            ]

        # 生成数据集
        datasets = []
        for obs_combo in observation_combinations:
            compatible_dags = [
                dag for dag, _, effects_by_node in dag_caches
                if all(effects_by_node[o.perturbed_node].effects == o.effects for o in obs_combo)
            ]
            if compatible_dags:
                datasets.append({
                    "observations": [o.to_dict() for o in obs_combo],
                    "ground_truth_graphs": [dag.to_dict() for dag in compatible_dags],
                })

        return datasets
    
    @staticmethod
    def generate_complete_dataset_collection(
        nodes: List[str],
        n_observations: Optional[int] = None,
        fixed: bool = False,
        max_edges: Optional[int] = None,
        seed: Optional[int] = None,
        n_samples: Optional[int] = None
    ) -> Dict:
        """
        Generate complete collection of datasets.
        
        Args:
            nodes: List of node names
            n_observations: Number of observations (None defaults to n_nodes)
            fixed: If True, generate only n_observations exactly. If False, generate 1 to n_observations
            max_edges: Maximum edges in DAGs
            seed: Random seed
            n_samples: If specified, sample this many observation combinations before checking compatibility
            
        Returns:
            Dictionary containing all datasets organized by observation count
        """
        n_nodes = len(nodes)
        
        # Default n_observations to number of nodes if not specified
        if n_observations is None:
            n_observations = n_nodes
            
        print("=" * 60)
        print("GENERATING CAUSAL DATASET COLLECTION")
        print("=" * 60)
        print(f"Nodes: {nodes}")
        print(f"Max edges: {max_edges}")
        
        if fixed:
            print(f"Observations: exactly {n_observations}")
        else:
            print(f"Observations: 1 to {n_observations}")
        print()
        
        # Generate all DAGs once to show hypothesis space
        all_dags = CausalDatasetGenerator.generate_all_dags(nodes, max_edges)
        print(f"Total hypothesis space: {len(all_dags)} DAGs")
        
        if fixed:
            # Generate only for the exact observation count
            print(f"\nGenerating datasets with exactly {n_observations} observations...")
            datasets = CausalDatasetGenerator.generate_all_observation_combinations(
                nodes, n_observations, max_edges, seed, n_samples
            )
            
            result = {
                "metadata": {
                    "nodes": nodes,
                    "n_observations": n_observations,
                    "max_edges": max_edges,
                    "hypothesis_space_size": len(all_dags),
                    "n_datasets": len(datasets),
                    "fixed": True
                },
                "datasets": datasets
            }
        else:
            # Generate for range 1 to n_observations
            collection = {
                "metadata": {
                    "nodes": nodes,
                    "max_n_observations": n_observations,
                    "max_edges": max_edges,
                    "hypothesis_space_size": len(all_dags),
                    "total_observation_sets": 0  # Will be updated
                },
                "datasets_by_n_observations": {}
            }
            
            # Generate datasets for each observation count
            for n_obs in range(1, min(n_observations, n_nodes) + 1):
                print(f"\nGenerating datasets with {n_obs} observation(s)...")
                datasets = CausalDatasetGenerator.generate_all_observation_combinations(
                    nodes, n_obs, max_edges, seed, n_samples
                )
                
                if not datasets:
                    print(f"  - No valid observation sets found for n={n_obs}")
                    continue
                    
                # Statistics
                unique_gt_counts = set()
                for ds in datasets:
                    unique_gt_counts.add(ds["n_compatible_graphs"])
                
                print(f"  - Generated {len(datasets)} unique observation sets")
                print(f"  - Compatible graph counts: {sorted(unique_gt_counts)}")
                
                collection["datasets_by_n_observations"][n_obs] = datasets
            
            # Update total observation sets count
            total_sets = sum(len(datasets) for datasets in collection["datasets_by_n_observations"].values())
            collection["metadata"]["total_observation_sets"] = total_sets
            print(f"\nTotal observation sets across all n_observations: {total_sets}")
            
            result = collection
        
        return result
    
    @staticmethod
    def sample_datasets(
        complete_collection: Dict,
        n_samples: int,
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Sample n datasets from the complete collection.
        
        Args:
            complete_collection: Complete dataset collection
            n_samples: Number of datasets to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of sampled datasets
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Flatten all datasets
        all_datasets = []
        
        # Handle both fixed and range formats
        if "datasets" in complete_collection:
            # Fixed format
            all_datasets = complete_collection["datasets"]
        elif "datasets_by_n_observations" in complete_collection:
            # Range format
            for n_obs, datasets in complete_collection["datasets_by_n_observations"].items():
                all_datasets.extend(datasets)
        
        # Sample
        n_available = len(all_datasets)
        n_to_sample = min(n_samples, n_available)
        
        if n_to_sample < n_samples:
            print(f"Warning: Requested {n_samples} samples but only {n_available} available")
        
        sampled = random.sample(all_datasets, n_to_sample)
        return sampled


def main():
    parser = argparse.ArgumentParser(description="Generate causal dataset collection with all observation combinations")
    parser.add_argument("--nodes", type=int, default=3, help="Number of nodes in graphs")
    parser.add_argument("--n-observations", type=int, default=None, help="Number of observations (default: number of nodes). Without --fixed, generates 1 to n observations")
    parser.add_argument("--fixed", action="store_true", help="With --n-observations, generate only that exact count (not 1 through n)")
    parser.add_argument("--max-edges", type=int, default=None, help="Maximum edges in DAGs (default: no limit)")
    parser.add_argument("--n-samples", type=int, default=None, help="Sample this many observation combinations before checking compatibility (much faster for large node counts)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    
    args = parser.parse_args()
    
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.fixed and args.n_observations:
            args.output = f"causal_datasets_n{args.n_observations}_fixed_{timestamp}.json"
        elif args.n_observations:
            args.output = f"causal_datasets_n{args.n_observations}_{timestamp}.json"
        else:
            args.output = f"causal_datasets_{timestamp}.json"
    
    # Generate node names
    nodes = [chr(65 + i) for i in range(args.nodes)]  # A, B, C, ...
    
    # Generate dataset collection
    result = CausalDatasetGenerator.generate_complete_dataset_collection(
        nodes=nodes,
        n_observations=args.n_observations,
        fixed=args.fixed,
        max_edges=args.max_edges,
        seed=args.seed,
        n_samples=args.n_samples  # Pass n_samples to sample observation combinations early
    )
    
    # Save to file
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nDataset saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()