
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

## Installation

The Tracer library can be installed from pip.

This installation is compatible with Linux/Mac OS X, and Python 3.8+.

The library will work with both the CPU and GPU version of TensorFlow, but to
allow for that it does not list Tensorflow as a requirement, so you need to
install Tensorflow separately if you haven't already done so.

To install the Tracer library and use it with TensorFlow 2 and Sonnet 2, run:

(CPU)
```shell
$ pip install nii_tracer "tensorflow>=2.1.0-rc1" "dm-sonnet>=2.0.0b0" tensorflow_probability
```

(GPU)
```shell
$ pip install nii_tracer "tensorflow_gpu>=2.1.0-rc1" "dm-sonnet>=2.0.0b0" tensorflow_probability
```

The latest version of the library requires TensorFlow >=2.1.0-rc1. 

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
