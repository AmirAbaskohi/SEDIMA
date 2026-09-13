# OpenEvolve Configuration Files

This directory contains configuration files for OpenEvolve with examples for different use cases.

## Configuration Files

### `default_config.yaml`
The main configuration file containing all available options with sensible defaults. This file includes:
- Complete documentation for all configuration parameters
- Default values for all settings
- **Island-based evolution parameters** for proper evolutionary diversity
- **Insight memory parameters** for cross-problem retrieval

Use this file as a template for your own configurations.

### `island_config_example.yaml`
A practical example configuration demonstrating proper island-based evolution setup. Shows:
- Recommended island settings for most use cases
- Balanced migration parameters
- Complete working configuration

### `island_examples.yaml`
Multiple example configurations for different scenarios:
- **Maximum Diversity**: Many islands, frequent migration
- **Focused Exploration**: Few islands, rare migration  
- **Balanced Approach**: Default recommended settings
- **Quick Exploration**: Small-scale rapid testing
- **Large-Scale Evolution**: Complex optimization runs

Includes guidelines for choosing parameters based on your problem characteristics.

## Island-Based Evolution Parameters

The key new parameters for proper evolutionary diversity are:

```yaml
database:
  num_islands: 5                      # Number of separate populations
  migration_interval: 50              # Migrate every N generations  
  migration_rate: 0.1                 # Fraction of top programs to migrate
```

### Parameter Guidelines

- **num_islands**: 3-10 for most problems (more = more diversity)
- **migration_interval**: 25-100 generations (higher = more independence)
- **migration_rate**: 0.05-0.2 (5%-20%, higher = faster knowledge sharing)

### When to Use What

- **Complex problems** → More islands, less frequent migration
- **Simple problems** → Fewer islands, more frequent migration
- **Long runs** → More islands to maintain diversity
- **Short runs** → Fewer islands for faster convergence

## Usage

Copy any of these files as a starting point for your configuration:

```bash
cp configs/default_config.yaml my_config.yaml
# Edit my_config.yaml for your specific needs
```

Then use with OpenEvolve:

```python
from openevolve import OpenEvolve
evolve = OpenEvolve(
    initial_program_path="program.py",
    evaluation_file="evaluator.py", 
    config_path="my_config.yaml"
)
```

## Insight Memory (Three-Stage)

Enable hierarchical memory retrieval with raw evidence, local insights, and clustered guidance:

```yaml
evaluator:
  runs_per_program: 3

insight_memory:
  enabled: true
  memory_name: "insight_memory"
  storage_path: null  # null => ~/.openevolve/chroma/<memory_name>
  embedding_model: "Qwen-embedding-4B"
  embedding_api_base: null
  embedding_api_key: null
  clustering_similarity_threshold: 0.8
  recluster_every_n_steps: 50
  cluster_representation_mode: "average"  # average | similarity_weighted | attention_weighted
  retrieval_top_clusters: 3
  retrieval_top_members_per_cluster: 3
```
