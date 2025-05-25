# Community Detection in Complex Networks

This application implements community detection algorithms for complex networks as part of AI Lab 10. It includes both predefined algorithms from specialized libraries and a custom genetic algorithm implementation based on evolutionary computation techniques.

## Overview

The application addresses the problem of identifying communities in complex networks using:

1. **Predefined Algorithms** from NetworkX and specialized libraries:
   - Louvain Method
   - Greedy Modularity Optimization
   - Label Propagation Algorithm

2. **Genetic Algorithm** based on Pizzuti (2017) paper:
   - Modularity-based fitness function
   - Conductance-based fitness function (bonus)
   - Tournament selection
   - Uniform crossover
   - Random mutation

## Features

- **Network Loading**: Supports multiple formats (GML, MTX, TXT)
- **Multiple Data Sources**: Real networks, professor datasets, and student datasets
- **Multiple Algorithms**: Compare predefined vs. evolutionary approaches
- **Performance Metrics**: Modularity, execution time, number of communities
- **Visualization**: Network plots with community highlighting
- **Results Export**: Detailed results saved to files
- **Command Line Interface**: Flexible usage options

## Installation

1. **Clone or download** the repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Available Networks

The application works with the following datasets:

### Real Networks (from `data/real/`):
- **karate**: Zachary's Karate Club (34 nodes)
- **dolphins**: Dolphin social network (62 nodes)
- **football**: American College Football (115 nodes)
- **krebs**: Books about US politics (105 nodes)

### Professor Networks (from `data/data-professor/`):
- **netscience**: Collaboration network of network scientists (1589 nodes)
- **lesmis**: Characters in Les Misérables (77 nodes)

### Student Networks (from `data/data-student/`):
- **student-brock200-1**: Student dataset in MTX format
- **student-ia-enron-only**: Student dataset in MTX format  
- **student-delaunay_n10**: Student dataset in MTX format

> **Note**: Student networks are automatically prefixed with `student-` to distinguish them from other datasets.

## Usage

### Command Line Interface

1. **List available networks** (including student datasets):
   ```bash
   python community_detection_app.py --list-networks
   ```

2. **Analyze a specific network**:
   ```bash
   python community_detection_app.py --network karate
   ```

3. **Analyze student datasets**:
   ```bash
   python community_detection_app.py --network student-ia-enron-only
   ```

4. **Run analysis on all networks**:
   ```bash
   python community_detection_app.py --all
   ```

5. **Disable visualization** (for large networks):
   ```bash
   python community_detection_app.py --network netscience --no-viz
   ```

### Interactive Mode

Simply run without arguments for interactive selection:
```bash
python community_detection_app.py
```

## Algorithm Details

### Predefined Algorithms

1. **Louvain Method**: Fast unfolding algorithm for community detection
   - Time complexity: O(m log n)
   - Optimizes modularity directly

2. **Greedy Modularity**: Greedy optimization of modularity
   - Hierarchical agglomeration approach
   - Good balance between speed and quality

3. **Label Propagation**: Asynchronous label propagation
   - Near-linear time complexity
   - Based on information diffusion

### Genetic Algorithm

Based on Pizzuti (2017) - "Evolutionary computation for community detection in networks: a review"

**Chromosome Representation**: 
- Each individual is an array where `individual[i] = j` means node `i` belongs to community `j`

**Genetic Operators**:
- **Selection**: Tournament selection (size = 3)
- **Crossover**: Uniform crossover (rate = 0.8)
- **Mutation**: Random community reassignment (rate = 0.1)
- **Elitism**: Best individual always survives

**Fitness Functions**:
1. **Modularity**: Q = Σ[Aij - kikj/(2m)]δ(ci,cj)
2. **Conductance**: φ(S) = cut(S,S̄)/min(vol(S),vol(S̄)) (bonus)

**Parameters** (auto-adjusted based on network size):
- Population size: 50-150
- Generations: 100-300
- Adaptive to network complexity

## Output Format

### Input Data
- **Graph**: Network structure (nodes and edges)
- **Parameters**: Algorithm-specific parameters

### Output Data
- **Number of communities**: Integer count
- **Community membership**: Dictionary mapping each node to its community
- **Performance metrics**: Modularity, execution time
- **Detailed results**: Saved to `results/` directory

## Results Structure

For each network analysis, the application generates:

```
results/
└── {network_name}_results.txt
    ├── Algorithm performance comparison
    ├── Detailed community assignments
    ├── Execution times
    └── Parameters used
```

## Evaluation Criteria

Points are awarded based on:

- **Predefined algorithms**: 20 points per network
- **Genetic algorithm**: 50 points per network
- **Bonus fitness functions**: 50 points per additional function

**Maximum possible points**:
- 4 real networks × 70 points = 280 points
- 6 additional networks × 70 points = 420 points
- 2 bonus fitness functions × 50 points = 100 points
- **Total**: 800 points

## Examples

### Sample Output

```
ANALYZING NETWORK: karate
==================================================
Loading network from: data/real/karate/karate.gml
Network loaded: 34 nodes, 78 edges

==================================================
RUNNING PREDEFINED ALGORITHMS
==================================================

Running Louvain...
  Communities found: 4
  Modularity: 0.4198
  Execution time: 0.0045 seconds

Running Genetic Algorithm...
Generation 0: Best fitness = 0.3721
Generation 50: Best fitness = 0.4156
  Communities found: 4
  Modularity: 0.4156
  Execution time: 12.3456 seconds

==================================================
COMPARISON OF RESULTS
==================================================
Algorithm                 Communities   Modularity   Time (s)  
-----------------------------------------------------------------
Louvain                   4             0.4198       0.0045    
Genetic Algorithm         4             0.4156       12.3456   
```

## Technical Implementation

### Key Classes

- `NetworkLoader`: Handles multi-format file loading and network discovery
  - `load_gml()`: GML format support
  - `load_mtx()`: MTX (Matrix Market) format support  
  - `load_edgelist()`: TXT edge list format support
  - `get_available_networks()`: Auto-discovery of all network files
- `PredefinedAlgorithms`: Implements library-based community detection
- `GeneticCommunityDetection`: Custom genetic algorithm implementation
- `CommunityVisualizer`: Network visualization utilities
- `CommunityDetectionApp`: Main application orchestrator

### Performance Optimizations

- Precomputed adjacency matrices for genetic algorithm
- Adaptive parameter selection based on network size
- Efficient fitness evaluation
- Memory-conscious population management

## Dependencies

- `networkx`: Network analysis and predefined algorithms
- `python-louvain`: Louvain method implementation
- `numpy`: Numerical computations
- `matplotlib`: Visualization
- `seaborn`: Enhanced plotting
- `scipy`: Scientific computing utilities

## References

1. Pizzuti, Clara. "Evolutionary computation for community detection in networks: a review." IEEE Transactions on Evolutionary Computation 22.3 (2017): 464-483.

2. Blondel, Vincent D., et al. "Fast unfolding of communities in large networks." Journal of statistical mechanics: theory and experiment 2008.10 (2008): P10008.

3. Newman, Mark EJ. "Modularity and community structure in networks." Proceedings of the national academy of sciences 103.23 (2006): 8577-8582.

## License

This project is developed for educational purposes as part of AI Lab 10 - Community Detection assignment. 

## Supported File Formats

The application supports multiple network file formats:

### GML (Graph Modeling Language)
- Standard format for the real and professor networks
- Contains node and edge definitions with attributes
- Example: `data/real/karate/karate.gml`

### MTX (Matrix Market Format)
- Sparse matrix format commonly used for large networks
- Header format: `%MatrixMarket matrix coordinate pattern symmetric`
- Contains dimensions line followed by edge list
- Example: `data/data-student/ia-enron-only.mtx`

### TXT (Edge List)
- Simple text format with space/tab separated node pairs
- Each line represents an edge: `node1 node2`
- Example: `data/data-professor/com-dblp.ungraph.txt`

## Data Directory Structure

```
data/
├── real/                    # Real-world networks (GML format)
│   ├── karate/
│   ├── dolphins/
│   ├── football/
│   └── krebs/
├── data-professor/          # Professor-provided datasets (GML/TXT)
│   ├── netscience.gml
│   ├── lesmis.gml
│   └── com-dblp.ungraph.txt
└── data-student/           # Student datasets (MTX/GML/TXT)
    ├── brock200-1.mtx
    ├── ia-enron-only.mtx
    └── delaunay_n10.mtx
```

### Recent Updates

**Version 2.0 Features:**
- **MTX Format Support**: Added native support for Matrix Market (.mtx) files
- **Student Dataset Integration**: Automatic discovery of datasets in `data/data-student/`
- **Multi-format Loading**: Unified interface for GML, MTX, and TXT formats
- **Enhanced Network Discovery**: Automatic detection and categorization of all network files
- **Improved Error Handling**: Better error messages and fallback parsing for malformed files 