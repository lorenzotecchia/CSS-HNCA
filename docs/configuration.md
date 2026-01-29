# Configuration

## Configuration File

CSS-HNCA uses TOML configuration files. The default configuration is in `config/default.toml`.

## Parameters

### Network Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_neurons` | int | 300 | Number of neurons in the network |
| `box_size` | [float, float, float] | [10.0, 10.0, 10.0] | Bounding box dimensions for neuron positions |
| `radius` | float | 2.0 | Maximum distance for synaptic connections |
| `initial_weight` | float | 0.06 | Initial synaptic weight value |
| `weight_min` | float | 0.0 | Minimum weight for excitatory synapses |
| `weight_max` | float | 0.3 | Maximum weight for excitatory synapses |
| `weight_min_inh` | float | -0.3 | Minimum weight for inhibitory synapses |
| `weight_max_inh` | float | 0.0 | Maximum weight for inhibitory synapses |
| `excitatory_fraction` | float | 1.0 | Fraction of excitatory neurons (0.0-1.0) |
| `leak_rate` | float | 0.08 | LIF membrane potential decay rate |
| `reset_potential` | float | 0.4 | Potential subtracted after firing |

### Learning Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.4 | Firing threshold for membrane potential |
| `learning_rate` | float | 0.01 | STDP potentiation rate (LTP) |
| `forgetting_rate` | float | 0.01 | STDP depression rate (LTD) |
| `decay_alpha` | float | 0.0005 | Baseline weight decay coefficient |
| `oja_alpha` | float | 0.002 | Oja rule decay coefficient |

### Visualization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pygame_enabled` | bool | true | Enable pygame network visualization |
| `matplotlib_enabled` | bool | true | Enable matplotlib analytics plots |
| `window_width` | int | 800 | Visualization window width |
| `window_height` | int | 600 | Visualization window height |
| `fps` | int | 30 | Target frames per second |

### Global Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | 42 | Random seed for reproducibility |

## Example Configuration

```toml
seed = 42

[network]
n_neurons = 300
box_size = [10.0, 10.0, 10.0]
radius = 2.0
initial_weight = 0.06
weight_min = 0.0
weight_max = 0.3
weight_min_inh = -0.3
weight_max_inh = 0.0
excitatory_fraction = 0.8
leak_rate = 0.08
reset_potential = 0.4

[learning]
threshold = 0.4
learning_rate = 0.01
forgetting_rate = 0.01
decay_alpha = 0.0005
oja_alpha = 0.002

[visualization]
pygame_enabled = true
matplotlib_enabled = true
window_width = 800
window_height = 600
fps = 30
```

## Custom Configurations

Create a custom TOML file and pass it to the simulation:

```bash
python main.py -c config/my_config.toml
```

Only override parameters you want to change; others use defaults.
