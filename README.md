# Proof of Concept for PyG - Transpiler Project for Ivy

1. [My understanding Ivy's transpiler](https://github.com/harshxtanwar/harshwardhantanwarPOC/edit/main/README.md#my-understanding-ivys-transpiler)
   - [Transpiling functions eagerly](https://github.com/harshxtanwar/harshwardhantanwarPOC/edit/main/README.md#1-transpiling-functions-eagerly)

## My understanding Ivy's transpiler
In this section, I aim to convey to you my understanding of Ivy's transpiler. I will talk about how
Ivy transpiles functions, modules and frameworks eagerly and lazily !

### 1. Transpiling function eagerly
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

  source_graph = \
    ivy.compile_graph(fn, *args, **kwargs)

  frontend_graph = \
    _replace_nodes_with_frontend_functions(source_graph)
    
  args, kwargs = _src_to_trgt_arrays(args, kwargs)
  
  target_graph = \
    ivy. compile_graph(frontend_graph, *args, **kwargs, to=to)

  return target_graph

```
 
