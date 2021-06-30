
# Tracer library

[Tracer](https://github.com/jjleewustledu/nii_tracer) is a library for studying
tracer kinetics using the KPZ equation, path integrals, renormalization group, 
and graph networks in Tensorflow and Sonnet.

#### What are graph networks?

A graph network takes a graph as input and returns a graph as output. The input
graph has edge- (*E* ), node- (*V* ), and global-level (**u**) attributes. The
output graph has the same structure, but updated attributes. Graph networks are
part of the broader family of "graph neural networks" (Scarselli et al., 2009).

To learn more about graph networks, see Deep Mind's arXiv paper: [Relational inductive
biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261).

![Graph network](https://github.com/deepmind/graph_nets/raw/master/images/graph-network.png)

## Usage example

The following code constructs a simple Tracer module and connects it to data.

```python
import tracer as tr
import sonnet as snt

# Provide your own functions to generate graph-structured data.
input_tracers = get_tracers()

# Create the tracer graph network.
tracer_module = tr.modules.Tracer(
    edge_model_fn=lambda: snt.nets.MLP([32, 32]),
    node_model_fn=lambda: snt.nets.MLP([32, 32]),
    global_model_fn=lambda: snt.nets.MLP([32, 32]))

# Pass the input graphs to the graph network, and return the output graphs.
output_tracers = tracer_module(input_tracers)
```
