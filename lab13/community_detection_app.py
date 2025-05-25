#!/usr/bin/env python3
"""
Community Detection Application for Complex Networks

This application implements community detection using:
1. Predefined algorithms from NetworkX
2. Genetic Algorithm based on Pizzuti (2017) paper

Author: AI Lab 10 - Community Detection
"""

import os
import sys
import time
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Any
import argparse
from dataclasses import dataclass
from collections import defaultdict
import community as community_louvain
from networkx.algorithms.community import greedy_modularity_communities, label_propagation_communities
from networkx.algorithms.community import asyn_lpa_communities, girvan_newman
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CommunityResult:
    """Results of community detection"""
    communities: List[List[int]]
    modularity: float
    num_communities: int
    node_to_community: Dict[int, int]
    algorithm: str
    execution_time: float
    parameters: Dict[str, Any]

class NetworkLoader:
    """Handles loading of network data from various formats"""
    
    @staticmethod
    def load_gml(filepath: str) -> nx.Graph:
        """Load graph from GML file"""
        try:
            # Try reading normally first
            G = nx.read_gml(filepath, label='id')
            
            # Ensure undirected graph
            if G.is_directed():
                G = G.to_undirected()
                
            # Remove self loops if any
            G.remove_edges_from(nx.selfloop_edges(G))
                
            print(f"Graph loaded successfully: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            print(f"Error loading GML file {filepath}: {e}")
            print("Trying alternative approach...")
            
            try:
                # Alternative approach: read line by line and build graph manually
                G = nx.Graph()
                
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Simple parser for nodes and edges
                import re
                
                # Extract nodes
                node_pattern = r'node\s*\[\s*id\s+(\d+).*?\]'
                nodes = re.findall(node_pattern, content, re.DOTALL)
                G.add_nodes_from([int(n) for n in nodes])
                
                # Extract edges
                edge_pattern = r'edge\s*\[\s*source\s+(\d+)\s+target\s+(\d+).*?\]'
                edges = re.findall(edge_pattern, content, re.DOTALL)
                
                # Add edges, automatically handles duplicates
                for source, target in edges:
                    G.add_edge(int(source), int(target))
                
                print(f"Graph loaded with manual parser: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                return G
                
            except Exception as e2:
                print(f"Manual parsing also failed: {e2}")
                return None
    
    @staticmethod
    def load_edgelist(filepath: str) -> nx.Graph:
        """Load graph from edge list file"""
        try:
            # Try reading as space/tab separated edge list
            G = nx.read_edgelist(filepath, nodetype=int, comments='#')
            
            print(f"Graph loaded from edge list: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            print(f"Error loading edge list file {filepath}: {e}")
            return None
    
    @staticmethod
    def load_mtx(filepath: str) -> nx.Graph:
        """Load graph from MTX (Matrix Market) file"""
        try:
            G = nx.Graph()
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Skip comment lines and find header
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith('%'):
                    continue
                elif line.strip() and not line.startswith('%'):
                    # This should be the dimensions line
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        nodes = int(parts[0])
                        edges_count = int(parts[2])
                        data_start = i + 1
                        break
            
            # Read edges
            for i in range(data_start, len(lines)):
                line = lines[i].strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        source = int(parts[0])
                        target = int(parts[1])
                        G.add_edge(source, target)
            
            print(f"Graph loaded from MTX: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            print(f"Error loading MTX file {filepath}: {e}")
            return None
    
    @staticmethod
    def get_available_networks() -> Dict[str, str]:
        """Get list of available network files"""
        networks = {}
        
        # Real networks
        real_path = "data/real"
        if os.path.exists(real_path):
            for network_dir in os.listdir(real_path):
                gml_path = os.path.join(real_path, network_dir, f"{network_dir}.gml")
                if os.path.exists(gml_path):
                    networks[network_dir] = gml_path
        
        # Professor networks
        prof_path = "data/data-professor"
        if os.path.exists(prof_path):
            for file in os.listdir(prof_path):
                if file.endswith('.gml'):
                    name = file.replace('.gml', '')
                    networks[name] = os.path.join(prof_path, file)
                elif file.endswith('.txt') and not file.startswith('desktop'):
                    # Add support for .txt edge lists
                    name = file.replace('.txt', '')
                    networks[name] = os.path.join(prof_path, file)
        
        # Student networks (your personal data)
        student_path = "data/data-student"
        if os.path.exists(student_path):
            for file in os.listdir(student_path):
                if file.endswith('.mtx'):
                    name = file.replace('.mtx', '')
                    networks[f"student-{name}"] = os.path.join(student_path, file)
                elif file.endswith('.gml'):
                    name = file.replace('.gml', '')
                    networks[f"student-{name}"] = os.path.join(student_path, file)
                elif file.endswith('.txt') and not file.startswith('desktop'):
                    name = file.replace('.txt', '')
                    networks[f"student-{name}"] = os.path.join(student_path, file)
        
        return networks

class PredefinedAlgorithms:
    """Implementation of predefined community detection algorithms"""
    
    @staticmethod
    def louvain_method(G: nx.Graph) -> CommunityResult:
        """Louvain method for community detection"""
        start_time = time.time()
        
        # Convert to simple graph if needed
        if G.is_directed():
            G = G.to_undirected()
        
        partition = community_louvain.best_partition(G)
        modularity = community_louvain.modularity(partition, G)
        
        # Convert to community format
        communities = defaultdict(list)
        for node, comm in partition.items():
            communities[comm].append(node)
        
        communities_list = list(communities.values())
        execution_time = time.time() - start_time
        
        return CommunityResult(
            communities=communities_list,
            modularity=modularity,
            num_communities=len(communities_list),
            node_to_community=partition,
            algorithm="Louvain",
            execution_time=execution_time,
            parameters={}
        )
    
    @staticmethod
    def greedy_modularity(G: nx.Graph) -> CommunityResult:
        """Greedy modularity optimization"""
        start_time = time.time()
        
        communities = list(greedy_modularity_communities(G))
        
        # Convert to required format
        communities_list = [list(comm) for comm in communities]
        node_to_community = {}
        for i, comm in enumerate(communities_list):
            for node in comm:
                node_to_community[node] = i
        
        modularity = nx.algorithms.community.modularity(G, communities)
        execution_time = time.time() - start_time
        
        return CommunityResult(
            communities=communities_list,
            modularity=modularity,
            num_communities=len(communities_list),
            node_to_community=node_to_community,
            algorithm="Greedy Modularity",
            execution_time=execution_time,
            parameters={}
        )
    
    @staticmethod
    def label_propagation(G: nx.Graph) -> CommunityResult:
        """Label propagation algorithm"""
        start_time = time.time()
        
        communities = list(asyn_lpa_communities(G))
        
        # Convert to required format
        communities_list = [list(comm) for comm in communities]
        node_to_community = {}
        for i, comm in enumerate(communities_list):
            for node in comm:
                node_to_community[node] = i
        
        modularity = nx.algorithms.community.modularity(G, communities)
        execution_time = time.time() - start_time
        
        return CommunityResult(
            communities=communities_list,
            modularity=modularity,
            num_communities=len(communities_list),
            node_to_community=node_to_community,
            algorithm="Label Propagation",
            execution_time=execution_time,
            parameters={}
        )

class GeneticCommunityDetection:
    """
    Genetic Algorithm for Community Detection
    Based on Pizzuti (2017) - "Evolutionary computation for community detection in networks: a review"
    """
    
    def __init__(self, G: nx.Graph, population_size: int = 100, generations: int = 200,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.G = G
        self.nodes = list(G.nodes())
        self.n_nodes = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Precompute adjacency matrix for efficiency
        self.adj_matrix = nx.adjacency_matrix(G, nodelist=self.nodes).toarray()
        self.total_edges = G.number_of_edges()
    
    def create_individual(self) -> np.ndarray:
        """Create a random individual (chromosome)
        
        Each individual is represented as an array where individual[i] = j
        means node i belongs to community j
        """
        # Random assignment with reasonable number of communities
        max_communities = min(self.n_nodes // 2, 20)
        return np.random.randint(0, max_communities, self.n_nodes)
    
    def initialize_population(self) -> List[np.ndarray]:
        """Initialize the population"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def fitness_modularity(self, individual: np.ndarray) -> float:
        """Calculate modularity-based fitness"""
        communities = defaultdict(list)
        for i, comm in enumerate(individual):
            communities[comm].append(self.nodes[i])
        
        # Convert to NetworkX format
        communities_list = [list(comm) for comm in communities.values() if len(comm) > 0]
        
        if len(communities_list) == 0:
            return -1.0
        
        try:
            modularity = nx.algorithms.community.modularity(self.G, communities_list)
            return modularity
        except:
            return -1.0
    
    def fitness_conductance(self, individual: np.ndarray) -> float:
        """Alternative fitness function based on conductance"""
        communities = defaultdict(set)
        for i, comm in enumerate(individual):
            communities[comm].add(i)
        
        total_conductance = 0
        valid_communities = 0
        
        for comm_nodes in communities.values():
            if len(comm_nodes) == 0:
                continue
                
            # Calculate internal and external edges
            internal_edges = 0
            external_edges = 0
            
            for i in comm_nodes:
                for j in range(self.n_nodes):
                    if self.adj_matrix[i][j] > 0:
                        if j in comm_nodes:
                            internal_edges += 1
                        else:
                            external_edges += 1
            
            # Avoid division by zero
            if internal_edges + external_edges > 0:
                conductance = external_edges / (internal_edges + external_edges)
                total_conductance += conductance
                valid_communities += 1
        
        if valid_communities == 0:
            return -1.0
        
        # Return negative average conductance (we want to minimize conductance)
        return -total_conductance / valid_communities
    
    def tournament_selection(self, population: List[np.ndarray], 
                           fitness_scores: List[float], tournament_size: int = 3) -> np.ndarray:
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover operator"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(self.n_nodes):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        
        return child1, child2
    
    def mutation(self, individual: np.ndarray) -> np.ndarray:
        """Mutation operator - random reassignment"""
        mutated = individual.copy()
        unique_communities = list(set(individual))
        
        for i in range(self.n_nodes):
            if random.random() < self.mutation_rate:
                # Assign to existing community or create new one (with low probability)
                if random.random() < 0.1 and len(unique_communities) < self.n_nodes // 3:
                    # Create new community
                    new_comm = max(individual) + 1
                    mutated[i] = new_comm
                    unique_communities.append(new_comm)
                else:
                    # Assign to existing community
                    mutated[i] = random.choice(unique_communities)
        
        return mutated
    
    def evolve(self, fitness_function: str = "modularity") -> CommunityResult:
        """Main evolution loop"""
        start_time = time.time()
        
        # Choose fitness function
        if fitness_function == "modularity":
            fitness_func = self.fitness_modularity
        else:
            fitness_func = self.fitness_conductance
        
        # Initialize population
        population = self.initialize_population()
        best_fitness = -float('inf')
        best_individual = None
        fitness_history = []
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [fitness_func(ind) for ind in population]
            
            # Track best solution
            max_fitness = max(fitness_scores)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_individual = population[fitness_scores.index(max_fitness)].copy()
            
            fitness_history.append(max_fitness)
            
            # Print progress
            if generation % 50 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
            
            # Create new population
            new_population = []
            
            # Elitism - keep best individual
            new_population.append(best_individual.copy())
            
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self.uniform_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            # Keep population size constant
            population = new_population[:self.population_size]
        
        # Convert best solution to result format
        communities = defaultdict(list)
        for i, comm in enumerate(best_individual):
            communities[comm].append(self.nodes[i])
        
        communities_list = [list(comm) for comm in communities.values() if len(comm) > 0]
        node_to_community = {}
        for i, comm_id in enumerate(best_individual):
            node_to_community[self.nodes[i]] = comm_id
        
        execution_time = time.time() - start_time
        
        return CommunityResult(
            communities=communities_list,
            modularity=self.fitness_modularity(best_individual),
            num_communities=len(communities_list),
            node_to_community=node_to_community,
            algorithm=f"Genetic Algorithm ({fitness_function})",
            execution_time=execution_time,
            parameters={
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'fitness_function': fitness_function
            }
        )

class CommunityVisualizer:
    """Visualization utilities for community detection results"""
    
    @staticmethod
    def plot_network_communities(G: nx.Graph, result: CommunityResult, 
                               save_path: Optional[str] = None):
        """Plot network with communities highlighted"""
        plt.figure(figsize=(12, 8))
        
        # Create color map
        colors = plt.cm.Set3(np.linspace(0, 1, result.num_communities))
        node_colors = []
        
        for node in G.nodes():
            comm_id = result.node_to_community[node]
            # Find which community this belongs to
            comm_idx = 0
            for i, comm in enumerate(result.communities):
                if node in comm:
                    comm_idx = i
                    break
            node_colors.append(colors[comm_idx % len(colors)])
        
        # Layout
        if len(G.nodes()) < 100:
            pos = nx.spring_layout(G, seed=42)
        else:
            pos = nx.spring_layout(G, seed=42, k=3, iterations=50)
        
        # Draw network
        nx.draw(G, pos, node_color=node_colors, node_size=50, 
                with_labels=False, edge_color='gray', alpha=0.7)
        
        plt.title(f"{result.algorithm}\nCommunities: {result.num_communities}, "
                 f"Modularity: {result.modularity:.3f}")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_fitness_evolution(fitness_history: List[float], save_path: Optional[str] = None):
        """Plot fitness evolution for genetic algorithm"""
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Genetic Algorithm - Fitness Evolution')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class CommunityDetectionApp:
    """Main application class"""
    
    def __init__(self):
        self.networks = NetworkLoader.get_available_networks()
        self.results = {}
    
    def load_network(self, network_name: str) -> Optional[nx.Graph]:
        """Load a network by name"""
        if network_name not in self.networks:
            print(f"Network '{network_name}' not found. Available networks:")
            for name in self.networks.keys():
                print(f"  - {name}")
            return None
        
        filepath = self.networks[network_name]
        print(f"Loading network from: {filepath}")
        
        # Choose loader based on file extension
        if filepath.endswith('.gml'):
            G = NetworkLoader.load_gml(filepath)
        elif filepath.endswith('.txt'):
            G = NetworkLoader.load_edgelist(filepath)
        elif filepath.endswith('.mtx'):
            G = NetworkLoader.load_mtx(filepath)
        else:
            print(f"Unsupported file format: {filepath}")
            return None
            
        if G is None:
            return None
        
        print(f"Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def run_predefined_algorithms(self, G: nx.Graph, network_name: str):
        """Run all predefined algorithms on the network"""
        print("\n" + "="*50)
        print("RUNNING PREDEFINED ALGORITHMS")
        print("="*50)
        
        n_nodes = G.number_of_nodes()
        
        # Define algorithms based on network size
        if n_nodes > 100000:  # Very large networks
            print(f"Large network detected ({n_nodes} nodes) - using fast algorithms only")
            algorithms = [
                ("Louvain", PredefinedAlgorithms.louvain_method),
                ("Label Propagation", PredefinedAlgorithms.label_propagation)
            ]
        elif n_nodes > 10000:  # Medium-large networks  
            print(f"Medium-large network detected ({n_nodes} nodes) - skipping greedy modularity")
            algorithms = [
                ("Louvain", PredefinedAlgorithms.louvain_method),
                ("Label Propagation", PredefinedAlgorithms.label_propagation)
            ]
        else:  # Small to medium networks
            algorithms = [
                ("Louvain", PredefinedAlgorithms.louvain_method),
                ("Greedy Modularity", PredefinedAlgorithms.greedy_modularity),
                ("Label Propagation", PredefinedAlgorithms.label_propagation)
            ]
        
        results = {}
        
        for name, algorithm in algorithms:
            print(f"\nRunning {name}...")
            try:
                result = algorithm(G)
                results[name] = result
                
                print(f"  Communities found: {result.num_communities}")
                print(f"  Modularity: {result.modularity:.4f}")
                print(f"  Execution time: {result.execution_time:.4f} seconds")
                
            except Exception as e:
                print(f"  Error running {name}: {e}")
        
        self.results[f"{network_name}_predefined"] = results
        return results
    
    def run_genetic_algorithm(self, G: nx.Graph, network_name: str, 
                            fitness_functions: List[str] = ["modularity", "conductance"]):
        """Run genetic algorithm with different fitness functions"""
        print("\n" + "="*50)
        print("RUNNING GENETIC ALGORITHM")
        print("="*50)
        
        # Adjust parameters based on network size
        n_nodes = G.number_of_nodes()
        if n_nodes < 50:
            pop_size, generations = 50, 100
        elif n_nodes < 200:
            pop_size, generations = 100, 200
        else:
            pop_size, generations = 150, 300
        
        # Skip conductance for large networks (too computationally expensive)
        if n_nodes > 200:
            print(f"Network has {n_nodes} nodes - skipping conductance fitness (too computationally expensive)")
            fitness_functions = ["modularity"]
        
        results = {}
        
        for fitness_func in fitness_functions:
            print(f"\nRunning GA with {fitness_func} fitness...")
            
            ga = GeneticCommunityDetection(
                G, population_size=pop_size, generations=generations,
                mutation_rate=0.1, crossover_rate=0.8
            )
            
            try:
                result = ga.evolve(fitness_function=fitness_func)
                results[f"GA_{fitness_func}"] = result
                
                print(f"  Communities found: {result.num_communities}")
                print(f"  Modularity: {result.modularity:.4f}")
                print(f"  Execution time: {result.execution_time:.4f} seconds")
                
            except Exception as e:
                print(f"  Error running GA with {fitness_func}: {e}")
        
        self.results[f"{network_name}_genetic"] = results
        return results
    
    def compare_results(self, network_name: str):
        """Compare results from all algorithms"""
        print("\n" + "="*50)
        print("COMPARISON OF RESULTS")
        print("="*50)
        
        all_results = []
        
        # Collect predefined results
        if f"{network_name}_predefined" in self.results:
            for name, result in self.results[f"{network_name}_predefined"].items():
                all_results.append(result)
        
        # Collect genetic algorithm results
        if f"{network_name}_genetic" in self.results:
            for name, result in self.results[f"{network_name}_genetic"].items():
                all_results.append(result)
        
        if not all_results:
            print("No results to compare")
            return
        
        # Sort by modularity
        all_results.sort(key=lambda x: x.modularity, reverse=True)
        
        print(f"{'Algorithm':<25} {'Communities':<12} {'Modularity':<12} {'Time (s)':<10}")
        print("-" * 65)
        
        for result in all_results:
            print(f"{result.algorithm:<25} {result.num_communities:<12} "
                  f"{result.modularity:<12.4f} {result.execution_time:<10.4f}")
    
    def save_results(self, network_name: str, output_dir: str = "results"):
        """Save results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = []
        
        # Collect all results
        for key in [f"{network_name}_predefined", f"{network_name}_genetic"]:
            if key in self.results:
                all_results.extend(self.results[key].values())
        
        if not all_results:
            print("No results to save")
            return
        
        # Save detailed results
        output_file = os.path.join(output_dir, f"{network_name}_results.txt")
        with open(output_file, 'w') as f:
            f.write(f"Community Detection Results for {network_name}\n")
            f.write("=" * 60 + "\n\n")
            
            for result in all_results:
                f.write(f"Algorithm: {result.algorithm}\n")
                f.write(f"Number of communities: {result.num_communities}\n")
                f.write(f"Modularity: {result.modularity:.6f}\n")
                f.write(f"Execution time: {result.execution_time:.6f} seconds\n")
                f.write(f"Parameters: {result.parameters}\n")
                f.write("\nCommunity assignments:\n")
                
                for i, community in enumerate(result.communities):
                    f.write(f"Community {i}: {community}\n")
                
                f.write("\n" + "-" * 40 + "\n\n")
        
        print(f"Results saved to {output_file}")
    
    def run_full_analysis(self, network_name: str, visualize: bool = True):
        """Run complete analysis on a network"""
        # Load network
        G = self.load_network(network_name)
        if G is None:
            return
        
        # Run predefined algorithms
        predefined_results = self.run_predefined_algorithms(G, network_name)
        
        # Run genetic algorithm
        genetic_results = self.run_genetic_algorithm(G, network_name)
        
        # Compare results
        self.compare_results(network_name)
        
        # Save results
        self.save_results(network_name)
        
        # Visualization
        if visualize and G.number_of_nodes() < 200:  # Only visualize smaller networks
            print("\nGenerating visualizations...")
            
            # Find best result by modularity
            all_results = list(predefined_results.values()) + list(genetic_results.values())
            best_result = max(all_results, key=lambda x: x.modularity)
            
            visualizer = CommunityVisualizer()
            visualizer.plot_network_communities(G, best_result)

def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description='Community Detection Application')
    parser.add_argument('--network', type=str, help='Network name to analyze')
    parser.add_argument('--list-networks', action='store_true', 
                       help='List available networks')
    parser.add_argument('--all', action='store_true', 
                       help='Run analysis on all available networks')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    app = CommunityDetectionApp()
    
    if args.list_networks:
        print("Available networks:")
        for name in app.networks.keys():
            print(f"  - {name}")
        return
    
    if args.all:
        print("Running analysis on all networks...")
        for network_name in app.networks.keys():
            print(f"\n{'='*60}")
            print(f"ANALYZING NETWORK: {network_name}")
            print(f"{'='*60}")
            app.run_full_analysis(network_name, visualize=not args.no_viz)
    
    elif args.network:
        app.run_full_analysis(args.network, visualize=not args.no_viz)
    
    else:
        # Interactive mode
        print("Community Detection Application")
        print("Available networks:")
        for i, name in enumerate(app.networks.keys(), 1):
            print(f"  {i}. {name}")
        
        try:
            choice = input("\nEnter network number or name: ").strip()
            
            if choice.isdigit():
                network_names = list(app.networks.keys())
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(network_names):
                    network_name = network_names[choice_idx]
                else:
                    print("Invalid choice")
                    return
            else:
                network_name = choice
            
            app.run_full_analysis(network_name, visualize=not args.no_viz)
            
        except KeyboardInterrupt:
            print("\nExiting...")

if __name__ == "__main__":
    main() 