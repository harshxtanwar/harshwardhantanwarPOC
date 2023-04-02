# Proof of Concept for PyG - Transpiler Project for Ivy

1. [My understanding with Ivy's transpiler](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#my-understanding-with-ivys-transpiler)
   - [Transpiling functions eagerly](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#1-transpiling-function-eagerly)
   - [Ivy transpiling Functions, Models and Frameworks !](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#2-ivy-transpiling-functions-models-and-frameworks-)
2. [Objective of the project](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#objective-of-the-project)
   - [Main Objective](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#1-main-objective)
   - [Workflow and how exactly I will proceed with the project](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#2-workflow-and-how-exactly-i-will-proceed-with-the-project)
3. [Listing Functions !](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#listing-functions-)
   - [An example of list of functions used in torch_geometric.nn file directory in PyG repository](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#1-an-example-of-list-of-functions-used-in-torch_geometricnn-file-directory-in-pyg-repository)

## My understanding with Ivy's transpiler
In this section, I aim to convey to you my understanding of Ivy's transpiler. I will talk about how
Ivy transpiles functions, modules and frameworks eagerly and lazily !

### 1. Transpiling function eagerly
- step 1: A graph is compiled for the function that Ivy is trying to transpile in the framework that the function is in.  
          For example: in case of a PyG function, we will get a Torch graph.  
- step 2: Each of the nodes in the compiled graph are replaces with the Ivy frontend functions of the corresponding framework  
          For example: in the case of a Torch graph, all of the functions will get replaced by ivy.functional.frontends.torch functions.  
- step 3: All of the arrays in arguments and keyword arguments are converted to the targeted framework.
          For example: if the to argument is given as `to=numpy`, then the arrays of all arguments are converted to numpy arrays.
- step 4: A new optimized graph is compiled using the graph compiled after step 2 and the args and kwargs after step 3 with removed functional wrappings
          For example: In our case, the torch graph will get replaced with numpy functions, and the args and kwargs are already converted to numpy arrays.
          
> Note: that the function demonstrated below is not the real transpiler function of ivy, it is just made to show my basic understanding on how things work 
> with the transpiler
```
def _transpile_function_eagerly(
  fn,
  to: str,
  args: Tuple,
  kwargs: dict,
):
 
"""
fn
  The function to transpile.
to
  The framework to transpile to.
args
  The positional arguments to pass
  to the function for tracing.
kwargs
  The keyword arguments to pass
  to the function for tracing.
"""
  
  # step 1
  source_graph = \
    ivy.compile_graph(fn, *args, **kwargs)
  
  # step 2
  frontend_graph = \
    _replace_nodes_with_frontend_functions(source_graph)
  
  # step 3
  args, kwargs = _src_to_trgt_arrays(args, kwargs)
  
  # step 4 
  target_graph = \
    ivy. compile_graph(frontend_graph, *args, **kwargs, to=to)

  return target_graph

```
### 2. Ivy transpiling Functions, Models and Frameworks !

In reality, the Ivy's transpiler function is quite flexible, and is able to transpile Functions, Models and Frameworks into a  
framework of your choice. 

- Ivy can Transpile either functions, trainable models or importable python modules,
with any number of combinations as many number of times !

- If no “objs” are provided, the function returns a new transpilation function
which receives only one object as input, making it usable as a decorator.

- If neither “args” nor “kwargs” are specified, then the transpilation will occur
lazily, upon the first call of the transpiled function as the transpiler need some arguments
in order to trace all of the functions required to compile a graph, otherwise transpilation is eager.

```
def transpile(
   *objs,
   to: Optional[str] = None,
   args: Optional[Tuple] = None,
   kwargs: Optional[dict] = None,
):
"""
objs
   The functions, models or modules
   to transpile.
to
   The framework to transpile to.
args
   The positional arguments to pass
   to the function for tracing.
kwargs
   The keyword arguments to pass
   to the function for tracing.
"""

# The code below this is made private by ivy, but from the example above for eager transpilation, I covered 
# all of the steps that happen when a user trys to transpile anythinf from Ivy, and not just a function.

```
## Objective of the project
In this section, I aim to convey to you as to what exactly the projects aims to achieve after it's completion and what all things
are exactly required to be done in order to complete this project.

### 1. Main Objective
- The main aim of this project is to make **Pytorch Geometric (PyG)** compatible with all other machine learning frameworks supported by Ivy  
like Numpy, Jax, Paddle and Tensorflow.
- After the successful implementation of this project, users of PyG will be able to transpile the whole 
PyG module to a framework of their choice !
- The users will highly benefit from this as the runtime efficiency is greatly improved when using a JAX backend on TPU, compared  
to the original PyTorch implementation.

### 2. Workflow and how exactly I will proceed with the project
Below are the steps I will follow to work on the project in a chronological order.
- Finding and creating a list of all of the pyTorch function used in the PyG's Module.
- Eliminating the functions from the list which already have been implemented in Ivy's both backend and frontend
- Creating a list of functions that already have been implemented in the backend but don't have any frontend wrapper around them
and creating another list of functions that need to be implemented for all backends in Ivy along with a frontend wrapper.
- You can find a list of such function the section below in my proof of concept.
- After finalising the list, I will start working with multiple pull requests to first finish writing codes for the functions with **missing frontend wrappers** in Ivy's repository.
- Then I will work on multiple pull requests to implement functions that have **both missing backend handlers and frontend wrapper along with the test cases** in Ivy's repository.
- Once this is done, I will make a pull request to PyG's repository where I will implement PyG's framework handlers to enable transpilation to provide PyG's users with functions that can transpile PyG to a framework of their choice.
- I will create Google Colab Demos showcasing how GNNs built using PyG can be used in TensorFlow and JAX projects (or any other framework) for the users of PyG.
- I will also create Google Colab Demos showing how the runtime efficiency is greatly improved when using a JAX backend on TPU, compared to the original PyTorch implementation.

 ## Listing Functions !
 A list of all torch functions used in PyG is to be made, particulary in  
 
> I have listed down all of the torch functions used in torch_geometric.nn, torch_geometric.data, torch_geometric.loader and torch_geometric.sampler below.

- torch_geometric
- torch_geometric.nn
- torch_geometric.data
- torch_geometric.loader
- torch_geometric.sampler
- torch_geometric.datasets
- torch_geometric.transforms
- torch_geometric.utils
- torch_geometric.explain
- torch_geometric.contrib
- torch_geometric.graphgym
- torch_geometric.profile

### 1. An example of list of functions used in torch_geometric.nn file directory in PyG repository  
> this is just a list of all types of functions/modules of the torch used in all of the files combined in the directory. I will further show examples of functions with missing backends and frontends in the the below section  
> (PS: I had to browse through **167 files** inside of the torch_geometric.nn directory in PyG's repository to make a whole list of these functions)

[
torch.nn.parameter , torch.long, torch.no_grad, torch.nn.ModuleList, torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Tanh, torch.cat, 
torch.nn.softplus, torch.nn.Sigmoid, torch.zeros_like, torch.autograd.grad, torch.zeros, torch.enable_grad, torch.nn.GRU, torch.nn.LSTM, torch.nn.MultiheadAttention, torch.stack, torch.mean, torch.bincount, torch.cumsum, torch.sort, torch.arange, torch.float, torch.log, torch.arange, torch.nn.LayerNorm, torch.nn.MultiheadAttention, torch.nn.init.calculate_gain, torch.nn.init.xavier_normal_ torch.nn.functional.normalize, torch.eye, torch.nn.init.kaiming_uniform_, torch.nn.functional.dropout, torch.nn.ReLu, torch.nn.parameter.UninitializedParameter, torch.nn.BatchNorm1d, torch.sparse_csc, torch.matmul, torch.clamp, torch.sqrt, torch.full, torch.nn.functional.gelu, torch.nn.functional.softmax, torch.nn.functional.leaky_relu, torch.nn.GRUCell, torch.nn.Param, torch.jit._overload, torch.sparse_csc, torch.ones, torch.addmm, torch.nn.BatchNorm1d, torch.nn.InstanceNorm1d, torch.nn.Sequential
, torch.nn.Identity, torch.nn.MultiheadAttention, torch.exp, torch.tanh, torch.sum, torch.nn.Embedding, torch.nn.functional.gelu, torch.utils.hooks.RemovableHandle, torch.uint8, torch.int8, torch.int32, torch.int64, torch.sparse_coo, torch.sparse_csr, torch.sparse_csc, torch.jit.unused, torch.utils.data.Dataloader, torch.bincount, torch.atan2, torch.cross, torch.index_select, torch.einsum, torch.bmm, torch.ones_like, torch.mul, torch.ones_like, torch.where, torch.is_floating_point, torch.gather, torch.nn.functional.normalize, torch.no_grad, torch.nn.ELU, torch.nn.Conv1d, torch.norm, torch.nn.init.uniform_, torch.onnx.is_in_onnx_export, torch.nn.utils.rnn.pad_sequence, torch.unique, torch.randint, torch.zeros_like, torch.nn.functional.binary_cross_entropy_with_logits, torch.nn.functional.margin_ranking_loss, torch.nn.init.xavier_uniform_, torch.nn.init.uniform_, torch.linalg.vector_norm, torch.cos, torch.sin, torch.sigmoid, torch.randn_like, torch.mean, torch.device, torch.bool, torch.utils.checkpoint.checkpoint, torch.atan2, torch.from_numpy, torch.linspace, torch.nn.modules.loss._Loss, torch.nn.BCEWithLogitsLoss, torch.nn.functional.logsigmoid, torch.clone, torch.rand, torch.randint, torch.ops.torch_cluster.random_walk, torch.jit.export, torch.jit._overload_method, torch.set_grad_enabled, torch.chunk, torch.pow, torch.load, torch.linspace, torch.nn.functional.nll_loss, torch.randperm, torch.log_softmax, torch.empty, torch.nn.functional.batch_norm, torch.cdist, torch.nn.functional.instance_norm, torch.nn.functional.layer_norm, torch.jit.unused, torch.LongTensor, torch.div, torch.empty_like, torch.nn.KLDivLoss, torch.min, torch.is_tensor, torch.logspace, torch.nn.ModulDict, torch.fx.Graph, torch.fx.GraphModule, torch.fx.Node, torch.fx.map_arg, torch.nn.init.orthogonal_, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR, torch.save, torch.optim.lr_scheduler.ReduceLROnPlateau, torch.optim.lr_scheduler, torch.jit.ScriptModule, torch.max 
]










