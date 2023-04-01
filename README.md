# Proof of Concept for PyG - Transpiler Project for Ivy

1. [My understanding Ivy's transpiler](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#my-understanding-ivys-transpiler)
   - [Transpiling functions eagerly](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#1-transpiling-function-eagerly)
   - [Ivy transpiling Functions, Models and Frameworks !](https://github.com/harshxtanwar/harshwardhantanwarPOC/blob/main/README.md#2-ivy-transpiling-functions-models-and-frameworks-)


## My understanding Ivy's transpiler
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




