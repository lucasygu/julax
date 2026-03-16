# Week 1: Foundations - A Deep Dive for Physics Minds

## Table of Contents
1. [The Physics Analogy: What Are We Really Doing?](#part-1-the-physics-analogy)
2. [Tensors: The Language of Data](#part-2-tensors)
3. [Neural Networks: Universal Function Approximators](#part-3-neural-networks)
4. [The Math: Gradients, Chain Rule, and Backpropagation](#part-4-the-math)
5. [Loss Functions: Measuring "Wrongness"](#part-5-loss-functions)
6. [Optimizers: Finding the Minimum](#part-6-optimizers)
7. [JAX: The Computational Engine](#part-7-jax)
8. [Functional Programming: Why It Matters](#part-8-functional-programming)
9. [PyTrees: JAX's Secret Weapon](#part-9-pytrees)
10. [JULAX base.py: Line-by-Line Annotations](#part-10-basepy)
11. [JULAX core.py: The Heart of the Framework](#part-11-corepy)
12. [MNIST Example: Complete Walkthrough](#part-12-mnist)
13. [How Everything Connects](#part-13-connections)
14. [Exercises and Next Steps](#part-14-exercises)

---

# Part 1: The Physics Analogy

## What Are We Really Doing?

Imagine you're a physicist trying to model a complex system. You have:
- **Experimental data**: Many observations `(x_i, y_i)` where `x` is input and `y` is output
- **Unknown function**: Some function `f` that maps `x → y`
- **Goal**: Find `f` so you can predict `y` for new values of `x`

### Traditional Physics Approach
```
1. Guess the functional form based on theory:
   y = A·sin(ωt + φ) + B·e^(-γt)  (for a damped oscillator)

2. Fit parameters (A, ω, φ, B, γ) using least squares

3. Problem: You need domain knowledge to guess the form
```

### Machine Learning Approach
```
1. Use a "universal approximator" with MANY parameters:
   y = f(x; θ₁, θ₂, ..., θₙ)  where n can be millions

2. The approximator is flexible enough to fit almost any function

3. Find parameters by minimizing prediction error

4. Advantage: No need to guess the functional form
```

## The Universal Approximator Theorem

Here's a profound result: **A neural network with one hidden layer and enough neurons can approximate ANY continuous function to arbitrary precision.**

This is like having a Swiss Army knife that can approximate any tool you need - not perfectly specialized for anything, but good enough for everything.

### Physics Translation

Think of it like a Fourier series:
```
f(x) = Σ (aₙ cos(nx) + bₙ sin(nx))
```

With enough terms, you can approximate any periodic function. Neural networks are similar - they're a particular basis expansion that happens to work well in high dimensions.

---

# Part 2: Tensors

## What is a Tensor?

In physics, you know tensors as multi-indexed objects that transform in specific ways. In ML, we use a simpler definition:

**A tensor is just a multi-dimensional array of numbers.**

```
Rank 0 (Scalar):      42                     # Just a number
Rank 1 (Vector):      [1, 2, 3]              # 1D array
Rank 2 (Matrix):      [[1,2], [3,4]]         # 2D array
Rank 3:               [[[1,2],[3,4]], ...]   # 3D array (like RGB images)
Rank N:               ...                     # Keep nesting
```

### Shape: The Tensor's Dimensions

```python
# A batch of 32 color images, each 28x28 pixels
# Shape: (32, 3, 28, 28)
#         │   │   │   └── width
#         │   │   └────── height
#         │   └────────── color channels (RGB)
#         └────────────── batch size (number of images)
```

### Why "Batch"?

Instead of processing one example at a time, we process many at once:
- **Efficiency**: Matrix operations are parallelized on GPUs
- **Statistics**: Gradients are averaged over the batch (reduces noise)

---

# Part 3: Neural Networks

## The Building Block: A Neuron (Linear Unit)

A single neuron computes:

```
output = activation(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
       = activation(w · x + b)
       = activation(linear transformation)
```

Where:
- `x` = input vector
- `w` = weight vector (learnable)
- `b` = bias (learnable)
- `activation` = nonlinear function

### The Linear Layer

Stack many neurons together:

```
┌─────────────────────────────────────────────────┐
│  Input x          Weights W           Output y  │
│  (size n)        (n × m matrix)      (size m)   │
│                                                 │
│  [x₁]            [w₁₁ w₁₂ ... w₁ₘ]    [y₁]     │
│  [x₂]    ×       [w₂₁ w₂₂ ... w₂ₘ]  + [b₁]     │
│  [...]           [... ... ... ...]    [...]     │
│  [xₙ]            [wₙ₁ wₙ₂ ... wₙₘ]    [yₘ]     │
│                                                 │
│         y = Wx + b                              │
└─────────────────────────────────────────────────┘
```

In physics terms: this is just an affine transformation (rotation + scaling + translation).

### Why Do We Need Nonlinearity?

**Without activation functions, stacking layers is pointless:**

```
y₁ = W₁x + b₁
y₂ = W₂y₁ + b₂ = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)
                                    = W'x + b'
```

Multiple linear layers collapse into ONE linear layer!

**With nonlinearity σ:**
```
y₁ = σ(W₁x + b₁)
y₂ = σ(W₂y₁ + b₂)  ≠ linear function of x
```

Now we can build complex, nonlinear functions.

### Common Activation Functions

```
┌─────────────────────────────────────────────────────────────────┐
│ Sigmoid: σ(x) = 1/(1 + e⁻ˣ)                                     │
│                                                                 │
│ Output range: (0, 1)                                            │
│ Looks like:    ___________                                      │
│               /                                                  │
│          ____/                                                   │
│                                                                 │
│ Problem: Gradients vanish when x is very large or small         │
│ Use for: Output layer when you want probabilities               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ ReLU: f(x) = max(0, x)                                          │
│                                                                 │
│ Output range: [0, ∞)                                            │
│ Looks like:        /                                            │
│                   /                                              │
│          ________/                                               │
│                                                                 │
│ Advantage: Simple, fast, no vanishing gradient for x > 0        │
│ Use for: Hidden layers (the default choice today)               │
└─────────────────────────────────────────────────────────────────┘
```

### A Complete Neural Network

```
Input (784)  →  Linear(784→512)  →  ReLU  →  Linear(512→512)  →  ReLU  →  Linear(512→10)  →  Output (10)
    ↓               ↓               ↓            ↓               ↓            ↓
 [x₁...x₇₈₄]    W₁x + b₁        max(0,·)     W₂h₁ + b₂       max(0,·)     W₃h₂ + b₃
                                    ↓                            ↓
                                   h₁                           h₂           "logits"
```

This is exactly what JULAX's MNIST example does!

---

# Part 4: The Math

## The Objective: Minimize the Loss

Training is an optimization problem:

```
θ* = argmin L(θ)
         θ

Where:
- θ = all learnable parameters (weights and biases)
- L(θ) = loss function measuring prediction error
```

## Gradient Descent: The Algorithm

**Intuition**: The gradient points "uphill". Go the opposite direction to descend.

```
Algorithm:
1. Start with random θ₀
2. Compute gradient ∇L(θ)
3. Update: θ ← θ - η·∇L(θ)    # η = learning rate
4. Repeat until convergence
```

**Physics analogy**: A ball rolling down a potential energy surface.
- Position = parameters θ
- Potential energy = loss L(θ)
- The ball "rolls" opposite to the gradient

### Learning Rate (η)

```
Too small:                    Too large:                  Just right:
      *                            *     *                     *
       \                          /       \                     \
        \                        /         \                     \_
         \                    __/           *                      \_
          \__________        /                \                      \_*
                             *                 *
Slow convergence           Oscillation/diverge      Smooth convergence
```

## The Chain Rule: Backbone of Backpropagation

Remember from calculus:

```
If z = f(g(x)), then dz/dx = (dz/dg)(dg/dx)
```

For neural networks with many layers:

```
L = loss(layer₃(layer₂(layer₁(x))))

∂L/∂W₁ = (∂L/∂layer₃)(∂layer₃/∂layer₂)(∂layer₂/∂layer₁)(∂layer₁/∂W₁)
```

**Key insight**: We compute this efficiently by going BACKWARDS through the network:

```
Forward pass:  x → h₁ → h₂ → h₃ → L    (compute outputs)
Backward pass: x ← h₁ ← h₂ ← h₃ ← L    (compute gradients)
```

Each layer only needs to know:
1. Its local gradient (∂output/∂input)
2. The gradient flowing from above (∂L/∂output)

Multiply them → pass to previous layer. That's backpropagation!

### Why This is Revolutionary

Without backprop, computing gradients for N weights requires N forward passes.
With backprop, ONE forward pass + ONE backward pass gives ALL gradients.

```
Naive:    O(N × cost_of_forward_pass)
Backprop: O(2 × cost_of_forward_pass)  ← MUCH faster!
```

---

# Part 5: Loss Functions

## What Makes a Good Loss Function?

1. **Measures prediction error**: Higher when predictions are wrong
2. **Differentiable**: We need gradients for optimization
3. **Matches the task**: Classification vs regression needs different losses

## Cross-Entropy Loss (for Classification)

When classifying into K categories, the network outputs K "logits" (raw scores).

### Step 1: Softmax (Convert to Probabilities)

```
softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ

Example: logits = [2.0, 1.0, 0.1]
         probs  = [0.659, 0.242, 0.099]  (sums to 1.0)
```

**Why softmax?**
- All outputs are positive
- They sum to 1 (valid probability distribution)
- Preserves ordering (highest logit → highest probability)
- "Soft" version of argmax

### Step 2: Cross-Entropy (Measure Error)

```
H(p, q) = -Σᵢ pᵢ log(qᵢ)

Where:
- p = true distribution (one-hot: [0, 0, 1, 0, 0, ...] for correct class)
- q = predicted distribution (softmax outputs)
```

For one-hot p, this simplifies to:

```
H = -log(q_correct_class)
```

**Intuition**:
- If q_correct = 0.99 → loss = -log(0.99) ≈ 0.01 (low loss, good!)
- If q_correct = 0.01 → loss = -log(0.01) ≈ 4.6 (high loss, bad!)

### Information Theory Connection

Cross-entropy comes from information theory:
- **Entropy**: Expected "surprise" of a distribution
- **Cross-entropy**: Expected surprise when using q instead of p
- **Minimizing cross-entropy** → making q match p

---

# Part 6: Optimizers

## Stochastic Gradient Descent (SGD)

```
θ ← θ - η·∇L(θ)
```

**Why "Stochastic"?**
- Full gradient uses ALL training data (expensive)
- SGD uses a random BATCH (subset) to estimate the gradient
- Cheaper per step, but noisier

### In JULAX:

```python
optimizer = optax.sgd(0.01)  # learning rate = 0.01
```

## The Update Process

```
1. Forward pass:     predictions = model(batch, params)
2. Compute loss:     loss = loss_fn(predictions, labels)
3. Backward pass:    grads = ∇loss w.r.t. params
4. Optimizer step:   updates = optimizer.update(grads)
5. Apply updates:    new_params = params + updates
```

In JULAX's Trainer (core.py:174-181):
```python
@partial(jit, static_argnums=0)
def forward_and_backward(self, x, p, s):
    # Steps 1-3: forward + backward via JAX's value_and_grad
    (_, S), grads = value_and_grad(self.forward, argnums=1, has_aux=True)(x, p, s)

    # Step 4: optimizer computes updates
    updates, S["optimizer"] = self.optimizer.update(grads, S["optimizer"])

    # Step 5: apply updates to parameters
    P = optax.apply_updates(p, updates)
    return P, S
```

---

# Part 7: JAX

## What is JAX?

JAX = **J**ust **A**nother e**X**ecution engine (kind of)

Created by Google, JAX is NumPy + three superpowers:

### Superpower 1: Automatic Differentiation (jax.grad)

```python
import jax
import jax.numpy as jnp

# Define any function
def f(x):
    return x**3 + 2*x**2 + x

# Automatically get its derivative!
df = jax.grad(f)

# Test it
print(f(2.0))    # 2³ + 2(2²) + 2 = 18
print(df(2.0))   # 3(2²) + 4(2) + 1 = 21 ✓
```

This works for ARBITRARILY complex functions, including entire neural networks!

### Superpower 2: JIT Compilation (jax.jit)

```python
@jax.jit
def slow_function(x):
    # Complex computation
    return result

# First call: compiles to optimized XLA code
# Subsequent calls: runs MUCH faster
```

JAX compiles your Python to XLA (Accelerated Linear Algebra), which:
- Runs on CPU, GPU, or TPU
- Fuses operations (fewer memory transfers)
- Optimizes automatically

### Superpower 3: Vectorization (jax.vmap)

```python
# Function for single example
def process_one(x):
    return x @ W + b

# Automatically vectorize over batch dimension!
process_batch = jax.vmap(process_one)

# Now it handles batches efficiently
batch_output = process_batch(batch_input)
```

### Why JAX for Deep Learning?

1. **Composable transforms**: `jit(grad(vmap(f)))` just works
2. **Functional purity**: No hidden state, easier to reason about
3. **First-class derivatives**: `grad` is a core feature, not an afterthought
4. **Hardware agnostic**: Same code runs on CPU/GPU/TPU

---

# Part 8: Functional Programming

## Why Does JULAX Use Functional Programming?

### The Problem with Stateful Code

```python
# PyTorch style (stateful, object-oriented)
class Layer:
    def __init__(self):
        self.W = random_weights()  # State hidden inside object

    def forward(self, x):
        return x @ self.W  # Accesses hidden state

layer = Layer()
y = layer.forward(x)  # What is layer.W? Hard to know!
```

Problems:
- State is hidden, hard to track
- Training modifies objects in place
- Debugging is harder (what's the state NOW?)
- Parallelization is tricky (shared mutable state)

### The Functional Alternative

```python
# JULAX style (functional, explicit state)
def forward(x, params):
    return x @ params['W']

params = {'W': random_weights()}  # State is EXPLICIT
y = forward(x, params)  # Everything is visible!
```

### What is a Pure Function?

A function is **pure** if:
1. **Same inputs → same output** (deterministic)
2. **No side effects** (doesn't change anything outside itself)

```python
# PURE: same input always gives same output
def add(a, b):
    return a + b

# IMPURE: depends on external state
counter = 0
def get_next():
    global counter
    counter += 1  # Side effect!
    return counter
```

### Why Pure Functions Rock

1. **Easy to test**: Just check input → output
2. **Easy to parallelize**: No shared state to worry about
3. **Easy to reason about**: Function is fully described by its code
4. **Composable**: Pure functions can be freely combined

### JULAX's Functional Design

```python
# Layer = configuration (immutable)
layer = Linear(in_dim=784, out_dim=512)

# Parameters = state (passed explicitly)
params, state = layer.init(seed=42)

# Forward = pure function of inputs
output, new_state = layer(input, params, state)
```

Everything is explicit. No hidden state. Pure functions everywhere.

---

# Part 9: PyTrees

## The Challenge: Nested Data Structures

Neural networks have MANY parameters organized hierarchically:

```
Network:
├── encoder:
│   ├── layer1: {W: array, b: array}
│   └── layer2: {W: array, b: array}
└── decoder:
    ├── layer1: {W: array, b: array}
    └── layer2: {W: array, b: array}
```

How do we apply operations (like gradient descent) to ALL these arrays?

## PyTrees to the Rescue

A **PyTree** is JAX's term for nested data structures (dicts, lists, tuples) where the "leaves" are arrays.

```python
# This is a PyTree
params = {
    'encoder': {
        'layer1': {'W': jnp.array(...), 'b': jnp.array(...)},
        'layer2': {'W': jnp.array(...), 'b': jnp.array(...)}
    },
    'decoder': {...}
}
```

### jax.tree.map: Apply to All Leaves

```python
# Apply to every array in the tree!
scaled_params = jax.tree.map(lambda x: x * 0.1, params)

# Works with multiple trees (same structure)
new_params = jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)
```

### Flatten and Unflatten

```python
# Flatten to a list of leaves + structure info
leaves, treedef = jax.tree.flatten(params)
# leaves = [array1, array2, array3, ...]
# treedef = information to reconstruct the tree

# Reconstruct from leaves
reconstructed = jax.tree.unflatten(treedef, leaves)
```

### Why This Matters for JULAX

JULAX uses PyTrees everywhere:
- **Parameters**: Nested dict matching layer structure
- **State**: Nested dict with training state per layer
- **Initialization**: Recursively builds param/state trees
- **Updates**: Apply gradients across entire tree

---

# Part 10: JULAX base.py - Line by Line

Let's annotate every line of `/src/julax/base.py`:

```python
# FILE: src/julax/base.py

from typing import TypeAlias, Any
# TypeAlias: lets us create type shortcuts
# Any: escape hatch for types we can't express

from pydantic import ConfigDict, RootModel
# Pydantic: data validation library
# ConfigDict: configuration for Pydantic models
# RootModel: base class for wrapping simple types

from jax import Array
# Array: JAX's array type (like numpy.ndarray but for JAX)

from jax.sharding import PartitionSpec
# PartitionSpec: tells JAX how to distribute arrays across devices
# e.g., PartitionSpec("data", None) = shard first dim, replicate second

import plum
# plum: library for function overloading (multiple dispatch)
# Allows same function name with different argument types

#########################################################
# TYPE ALIASES - shortcuts for commonly used types
#########################################################

PRNG: TypeAlias = Array
# PRNG = Pseudo-Random Number Generator
# In JAX, randomness is explicit - you pass "keys" around
# This is a JAX array that represents a random seed

PyTree: TypeAlias = Any
# PyTree = any nested structure (dict, list, tuple) of arrays
# Using Any because the actual type is complex

OutShardingType: TypeAlias = PartitionSpec | None
# How to distribute output across devices
# None = don't shard (replicate everywhere)

Dtype: TypeAlias = Any
# Data type (float32, bfloat16, etc.)
# Using Any due to JAX dtype complexity

#########################################################
# DISPATCH - function overloading
#########################################################

dispatch = plum.Dispatcher(warn_redefinition=True)
# Creates a dispatcher for multiple dispatch
# warn_redefinition=True: warn if we accidentally redefine a function
#
# This allows:
# @dispatch
# def foo(x: int): ...
# @dispatch
# def foo(x: str): ...
# foo(5)      # calls first version
# foo("hi")   # calls second version

#########################################################
# FROZENDICT - immutable dictionary
#########################################################

class FrozenDict(RootModel[dict]):
    """An immutable dictionary.

    Why immutable?
    1. Functional programming prefers immutability
    2. JAX likes immutable data (easier to reason about)
    3. Can be used as dict keys (hashable)
    """

    model_config = ConfigDict(frozen=True)
    # frozen=True: instances cannot be modified after creation

    def __getitem__(self, item):
        return self.root[item]
        # self.root is the underlying dict (from RootModel)

    def __iter__(self):
        return iter(self.root)

    def keys(self):
        return self.root.keys()

    def values(self):
        return self.root.values()

    def items(self):
        return self.root.items()

    def __hash__(self):
        return hash(frozenset(self.root.items()))
        # Makes it hashable (can use as dict key)
        # Converts to frozenset for stable hashing

    def __eq__(self, other):
        if isinstance(other, FrozenDict):
            return self.root == other.root
        return self.root == other
```

**Summary of base.py**:
- Defines type aliases for cleaner code
- Sets up multiple dispatch (function overloading)
- Provides FrozenDict for immutable configurations
- Total: 42 lines of foundational infrastructure

---

# Part 11: JULAX core.py - The Heart

Let's understand `/src/julax/core.py`:

```python
# FILE: src/julax/core.py

#########################################################
# IMPORTS
#########################################################

from abc import ABC, abstractmethod
# ABC: Abstract Base Class - can't instantiate directly
# abstractmethod: subclasses MUST implement this

import plum
from functools import partial
from typing import Annotated, Callable, TypeAlias, Any

import optax
# optax: JAX optimization library (SGD, Adam, etc.)

from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
# jit: just-in-time compilation for speed
# value_and_grad: compute output AND gradient in one pass

from julax.base import PRNG, Dtype, OutShardingType, PyTree, dispatch

#########################################################
# MORE TYPE ALIASES
#########################################################

Param: TypeAlias = dict
# Parameters: nested dict of learnable weights

State: TypeAlias = dict
# State: nested dict of mutable state (RNG, batch norm stats, etc.)

#########################################################
# LAYERBASE - the foundation of everything
#########################################################

class LayerBase(BaseModel, ABC):
    """Abstract base class for all layers.

    Inherits from:
    - BaseModel (Pydantic): validation, serialization, immutability
    - ABC: marks as abstract (must be subclassed)
    """

    # Optional: what dtype to use for parameters (float32, bfloat16...)
    param_dtype: Dtype | None = None

    # Optional: how to shard parameters across devices
    param_sharding: OutShardingType = None

    # Optional: how to shard outputs across devices
    out_sharding: OutShardingType = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow JAX arrays etc.
        frozen=True,                   # Layers are immutable configs
        ignored_types=(                # Don't validate these types
            jax.stages.Wrapped,        # JIT-compiled functions
            plum.function.Function,    # Dispatch functions
            optax.GradientTransformation,  # Optimizers
        ),
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        """Called when a subclass is defined.

        Registers the class as a JAX PyTree so JAX transformations
        (jit, grad, vmap) work on instances.
        """
        jax.tree_util.register_dataclass(
            cls,
            data_fields=list(cls.model_fields.keys()),
            meta_fields=[]
        )

    def sublayers(self) -> dict:
        """Find all LayerBase instances nested in this layer.

        This enables recursive initialization: when you init() a layer,
        it automatically finds and inits all nested layers too.

        Returns: dict mapping names to sublayers
        """
        # Flatten this layer's attributes, stopping at LayerBase instances
        attrs_flatten, treedef = jax.tree.flatten(
            dict(self),
            is_leaf=lambda x: isinstance(x, LayerBase)
        )

        # Keep only LayerBase instances (replace others with None)
        masked_sublayers = jax.tree.unflatten(
            treedef,
            [x if isinstance(x, LayerBase) else None for x in attrs_flatten]
        )

        # Filter to non-None entries
        res = {}
        for k, v in masked_sublayers.items():
            if jax.tree.reduce(
                lambda x, y: x or y,
                v,
                None,
                is_leaf=lambda x: isinstance(x, LayerBase),
            ):
                res[k] = v
        return res

    def param(self, rng: PRNG) -> Param:
        """Initialize this layer's OWN parameters.

        Override this in subclasses that have learnable weights.
        Default: no parameters (empty dict).
        """
        return Param()

    def state(self, rng: PRNG) -> State:
        """Initialize this layer's OWN state.

        Override this for layers with mutable state (Dropout, BatchNorm).
        Default: no state (empty dict).
        """
        return State()

    #####################################################
    # INIT - multiple dispatch for flexibility
    #####################################################

    @dispatch
    def init(self, seed: int = 0) -> tuple[Param, State]:
        """Init from integer seed - converts to JAX PRNG key."""
        return self.init(jax.random.key(seed))

    @dispatch
    def init(self, rng: PRNG) -> tuple[Param, State]:
        """Init from PRNG key - the main implementation.

        Recursively initializes all sublayers, then this layer.
        Returns combined (params, state) for entire subtree.
        """
        # Find all sublayers
        sublayers, treedef = jax.tree.flatten(
            self.sublayers(),
            is_leaf=lambda x: isinstance(x, LayerBase)
        )

        # Initialize each sublayer
        sublayer_params_flatten, sublayer_stats_flatten = [], []
        for layer in sublayers:
            if layer is None:
                sublayer_params_flatten.append(None)
                sublayer_stats_flatten.append(None)
            else:
                rng, _rng = jax.random.split(rng)  # Split key for randomness
                p, s = layer.init(_rng)            # Recursively init
                sublayer_params_flatten.append(p)
                sublayer_stats_flatten.append(s)

        # Reconstruct tree structure
        sublayer_params = Param(**jax.tree.unflatten(treedef, sublayer_params_flatten))
        sublayer_states = State(**jax.tree.unflatten(treedef, sublayer_stats_flatten))

        # Initialize THIS layer's own params/state
        rng_p, rng_s = jax.random.split(rng)
        layer_params = self.param(rng_p)
        layer_states = self.state(rng_s)

        # Merge sublayer + own params/state
        return self.init(layer_params, layer_states, sublayer_params, sublayer_states)

    @dispatch
    def init(self, layer_params, layer_states, sublayer_params, sublayer_states):
        """Merge layer's own params with sublayer params."""
        # Check for name collisions
        assert len(layer_params.keys() & sublayer_params.keys()) == 0
        assert len(layer_states.keys() & sublayer_states.keys()) == 0

        # Combine with | (dict union)
        return sublayer_params | layer_params, sublayer_states | layer_states

    #####################################################
    # FORWARD - the actual computation
    #####################################################

    @abstractmethod
    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        """The forward pass computation.

        Args:
            x: input data
            p: parameters for this layer (and sublayers)
            s: state for this layer (and sublayers)

        Returns:
            (output, new_state): computation result and updated state

        MUST be overridden by subclasses.
        """
        ...

    #####################################################
    # CALL - main entry point
    #####################################################

    @dispatch
    def __call__(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        """Call with params and state - just runs forward."""
        return self.forward(x, p, s)

    @dispatch
    def __call__(self, x: PyTree) -> tuple[PyTree, State]:
        """Call without params - auto-initializes (for quick testing)."""
        return self.__call__(x, *self.init())

#########################################################
# LEARNER - model + loss function
#########################################################

class Learner(LayerBase):
    """Wraps a model with a loss function.

    This is what you actually train - it takes labeled data
    and produces a loss value (scalar) to minimize.
    """

    loss_fn: Callable[[PyTree, PyTree], Any]  # (predictions, labels) -> loss
    model: LayerBase                           # The neural network
    agg: Callable = jnp.mean                   # How to aggregate per-sample losses
    feature_name: str = "feature"              # Key for input in data dict
    label_name: str = "label"                  # Key for target in data dict

    def forward(self, input: dict, p: Param, s: State) -> tuple[PyTree, State]:
        """Run model on input and compute loss.

        Args:
            input: dict with 'feature' and 'label' keys
            p: parameters
            s: state

        Returns:
            (loss, new_state)
        """
        x = input[self.feature_name]  # Get input features
        y = input[self.label_name]    # Get true labels

        # Forward pass through model
        ŷ, s["model"] = self.model(x, p["model"], s["model"])

        # Compute per-sample losses
        losses = self.loss_fn(ŷ, y)

        # Aggregate (e.g., mean over batch)
        loss = self.agg(losses)

        return loss, s

#########################################################
# TRAINER - learner + optimizer
#########################################################

class Trainer(LayerBase):
    """Combines Learner with optimizer for training.

    This performs the actual parameter updates:
    forward → compute loss → backprop → optimizer step
    """

    learner: Learner
    optimizer: Any  # optax.GradientTransformation

    def state(self, rng: PRNG) -> State:
        """Initialize trainer state (optimizer state filled later)."""
        return State(optimizer=None, loss=0.0)

    @dispatch
    def init(self, layer_params, layer_states, sublayer_params, sublayer_states):
        """Custom init to set up optimizer state."""
        # Initialize optimizer with learner's parameters
        layer_states["optimizer"] = self.optimizer.init(sublayer_params["learner"])
        return sublayer_params | layer_params, sublayer_states | layer_states

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        """Forward pass - just computes loss (no update)."""
        loss, state = self.learner(x, p["learner"], s["learner"])
        return loss, State(learner=state, optimizer=s["optimizer"], loss=loss)

    @partial(jit, static_argnums=0, donate_argnames=("p", "s"))
    def forward_and_backward(self, x: PyTree, p: Param, s: State) -> tuple[Param, State]:
        """THE TRAINING STEP - forward, backward, update.

        Decorated with @jit for speed.
        donate_argnames tells JAX it can reuse memory for p and s.
        """
        # Compute loss AND gradients in one call
        (_, S), grads = value_and_grad(
            self.forward,
            argnums=1,      # differentiate w.r.t. second arg (p)
            has_aux=True    # forward returns (loss, state), state is auxiliary
        )(x, p, s)

        # Optimizer computes parameter updates from gradients
        updates, S["optimizer"] = self.optimizer.update(grads, S["optimizer"])

        # Apply updates: new_params = old_params + updates
        P = optax.apply_updates(p, updates)

        return P, S

    @dispatch
    def __call__(self, x: PyTree, p: Param, s: State) -> tuple[Param, State]:
        """Calling Trainer = one training step."""
        return self.forward_and_backward(x, p, s)
```

---

# Part 12: MNIST Example - Complete Walkthrough

Let's trace through `/experiments/mnist.py` line by line:

```python
# FILE: experiments/mnist.py

#########################################################
# DEPENDENCIES
#########################################################

import logging
import grain
import jax
from jax.nn.initializers import truncated_normal
import optax
import tensorflow_datasets as tfds

from julax import (
    Chain, DoEveryNSteps, Experiment, Learner,
    Linear, Param, State, Trainer, default_observer, test_mode
)

#########################################################
# LOGGING SETUP
#########################################################

logging.root.setLevel(logging.INFO)
# Show info-level logs (training progress, etc.)

#########################################################
# EVALUATION FUNCTION
#########################################################

def evaluate(x: Experiment, p: Param, s: State):
    """Evaluate model accuracy on test set.

    Called every 100 training steps as an observer.
    """
    # Load MNIST test set
    dataset = (
        grain.MapDataset.source(tfds.data_source("mnist", split="test"))
        .batch(32, drop_remainder=True)
        .map(lambda x: {
            "feature": x["image"].reshape(32, -1),  # Flatten 28x28 → 784
            "label": x["label"],
        })
        .to_iter_dataset()
    )

    # Get model and its params/state from experiment hierarchy
    model = x.trainer.learner.model
    param = p["trainer"]["learner"]["model"]
    state = test_mode(s["trainer"]["learner"]["model"])  # Disable dropout etc.

    # Evaluate accuracy
    n_correct, n_total = 0, 0
    for batch in iter(dataset):
        ŷ, _ = model(batch["feature"], param, state)
        n_correct += (ŷ.argmax(axis=1) == batch["label"]).sum().item()
        n_total += 32

    acc = n_correct / n_total
    logging.info(f"Accuracy at step {s['step']}: {acc}")

#########################################################
# MAIN: DEFINE THE EXPERIMENT
#########################################################

E = Experiment(
    name="mnist",

    #####################################################
    # TRAINER = LEARNER + OPTIMIZER
    #####################################################
    trainer=Trainer(

        #################################################
        # LEARNER = MODEL + LOSS
        #################################################
        learner=Learner(

            #############################################
            # MODEL = 3-layer MLP
            #############################################
            model=Chain(
                # Layer 1: 784 → 512
                Linear(
                    in_dim=784,       # 28×28 image flattened
                    out_dim=512,      # Hidden dimension
                    w_init=truncated_normal(),  # Normal dist, truncated at ±2σ
                ),
                jax.nn.relu,          # Activation: max(0, x)

                # Layer 2: 512 → 512
                Linear(
                    in_dim=512,
                    out_dim=512,
                    w_init=truncated_normal(),
                ),
                jax.nn.relu,

                # Layer 3: 512 → 10 (output)
                Linear(
                    in_dim=512,
                    out_dim=10,       # 10 digit classes
                    w_init=truncated_normal(),
                ),
            ),

            # Loss function: softmax + cross-entropy
            loss_fn=optax.softmax_cross_entropy_with_integer_labels,
            # Takes logits and integer labels, returns per-sample loss
        ),

        # Optimizer: SGD with learning rate 0.01
        optimizer=optax.sgd(0.01),
    ),

    #####################################################
    # DATASET: MNIST training data
    #####################################################
    dataset=grain.MapDataset.source(tfds.data_source("mnist", split="train"))
        .seed(seed=45)              # Reproducibility
        .shuffle()                   # Randomize order
        .batch(32, drop_remainder=True)  # Batch size 32
        .map(lambda x: {
            "feature": x["image"].reshape(32, -1),  # Flatten: (32,28,28) → (32,784)
            "label": x["label"],                    # Labels: (32,)
        })
        .slice(slice(1000))          # Only use first 1000 batches (for speed)
        .to_iter_dataset(),

    #####################################################
    # OBSERVER: what to do during training
    #####################################################
    observer=default_observer()              # Log loss every 10 steps + step time
             * DoEveryNSteps(evaluate, n=100),  # Evaluate every 100 steps
    # The * composes observers together
)

#########################################################
# RUN THE TRAINING
#########################################################

E.run()    # Start training loop
E.close()  # Cleanup (close checkpoint manager, etc.)
```

## Data Flow Visualization

```
MNIST Image          Flatten        Linear           ReLU         Linear
28×28 pixels   →    784 values  →  512 values  →  512 values  →  512 values
                                       │              │
                                       W₁ (784×512)   max(0,·)    W₂ (512×512)
                                       b₁ (512)                    b₂ (512)

    ...→  ReLU  →  Linear  →  Softmax  →  Cross-Entropy  →  Loss
          │         │           │              │
       max(0,·)   W₃ (512×10)  probabilities  compare with
                   b₃ (10)                     true label
```

---

# Part 13: How Everything Connects

## The Full Picture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           YOUR UNDERSTANDING                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
    ┌─────────────────────────────────┼─────────────────────────────────────┐
    │                                 │                                     │
    ▼                                 ▼                                     ▼

MATHEMATICAL                    COMPUTATIONAL                        JULAX
FOUNDATIONS                     INFRASTRUCTURE                     FRAMEWORK

┌─────────────┐                ┌─────────────┐                ┌─────────────────┐
│ Tensors     │ ──────────────▶│ NumPy/JAX   │ ──────────────▶│ LayerBase       │
│ (nD arrays) │                │ Arrays      │                │ (base abstraction)
└─────────────┘                └─────────────┘                └─────────────────┘
       │                              │                              │
       ▼                              ▼                              ▼
┌─────────────┐                ┌─────────────┐                ┌─────────────────┐
│ Linear      │ ──────────────▶│ jnp.einsum  │ ──────────────▶│ Linear layer    │
│ Algebra     │                │ matmul      │                │ in layers.py    │
└─────────────┘                └─────────────┘                └─────────────────┘
       │                              │                              │
       ▼                              ▼                              ▼
┌─────────────┐                ┌─────────────┐                ┌─────────────────┐
│ Gradient    │ ──────────────▶│ jax.grad    │ ──────────────▶│ forward_and_    │
│ Descent     │                │ autodiff    │                │ backward()      │
└─────────────┘                └─────────────┘                └─────────────────┘
       │                              │                              │
       ▼                              ▼                              ▼
┌─────────────┐                ┌─────────────┐                ┌─────────────────┐
│ Chain Rule  │ ──────────────▶│ backprop    │ ──────────────▶│ value_and_grad  │
│ (calculus)  │                │ (auto)      │                │ in Trainer      │
└─────────────┘                └─────────────┘                └─────────────────┘
       │                              │                              │
       ▼                              ▼                              ▼
┌─────────────┐                ┌─────────────┐                ┌─────────────────┐
│ Loss        │ ──────────────▶│ optax       │ ──────────────▶│ Learner         │
│ Functions   │                │ losses      │                │ (model + loss)  │
└─────────────┘                └─────────────┘                └─────────────────┘
       │                              │                              │
       ▼                              ▼                              ▼
┌─────────────┐                ┌─────────────┐                ┌─────────────────┐
│ SGD/Adam    │ ──────────────▶│ optax       │ ──────────────▶│ Trainer         │
│ Optimizers  │                │ optimizers  │                │ (optimizer)     │
└─────────────┘                └─────────────┘                └─────────────────┘
       │                              │                              │
       ▼                              ▼                              ▼
┌─────────────┐                ┌─────────────┐                ┌─────────────────┐
│ Pure        │ ──────────────▶│ Functional  │ ──────────────▶│ Explicit        │
│ Functions   │                │ JAX style   │                │ param/state     │
└─────────────┘                └─────────────┘                └─────────────────┘
       │                              │                              │
       ▼                              ▼                              ▼
┌─────────────┐                ┌─────────────┐                ┌─────────────────┐
│ Nested      │ ──────────────▶│ PyTrees     │ ──────────────▶│ sublayers()     │
│ Structures  │                │ tree_map    │                │ recursive init  │
└─────────────┘                └─────────────┘                └─────────────────┘
```

## The Training Loop in Context

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                          ONE TRAINING ITERATION                                 │
└────────────────────────────────────────────────────────────────────────────────┘

1. GET DATA
   ┌──────────┐
   │  MNIST   │  →  batch = {"feature": (32, 784), "label": (32,)}
   │  Dataset │
   └──────────┘
         │
         ▼
2. FORWARD PASS (compute predictions)
   ┌──────────────────────────────────────────────────────────────────────┐
   │  Chain(                                                              │
   │      Linear(784→512), relu,    ← h₁ = relu(W₁x + b₁)                │
   │      Linear(512→512), relu,    ← h₂ = relu(W₂h₁ + b₂)               │
   │      Linear(512→10)            ← logits = W₃h₂ + b₃                 │
   │  )                                                                   │
   └──────────────────────────────────────────────────────────────────────┘
         │
         ▼
3. COMPUTE LOSS (measure error)
   ┌──────────────────────────────────────────────────────────────────────┐
   │  softmax(logits) → probs                                            │
   │  -log(probs[correct_class]) → per-sample loss                       │
   │  mean(per-sample losses) → scalar loss                              │
   └──────────────────────────────────────────────────────────────────────┘
         │
         ▼
4. BACKWARD PASS (compute gradients)
   ┌──────────────────────────────────────────────────────────────────────┐
   │  jax.value_and_grad(forward, argnums=1)                             │
   │  ∂loss/∂W₃, ∂loss/∂b₃  ← via chain rule                            │
   │  ∂loss/∂W₂, ∂loss/∂b₂  ← via chain rule                            │
   │  ∂loss/∂W₁, ∂loss/∂b₁  ← via chain rule                            │
   └──────────────────────────────────────────────────────────────────────┘
         │
         ▼
5. OPTIMIZER STEP (update parameters)
   ┌──────────────────────────────────────────────────────────────────────┐
   │  updates = -learning_rate * gradients                               │
   │  new_params = old_params + updates                                  │
   │  (For SGD. Other optimizers like Adam are more sophisticated)       │
   └──────────────────────────────────────────────────────────────────────┘
         │
         ▼
6. REPEAT until loss is low / accuracy is high
```

---

# Part 14: Exercises and Next Steps

## Exercises to Test Your Understanding

### Exercise 1: Trace the Shapes
For the MNIST network, write down the shape of the data at each step:
```
Input:     (batch, ??)  = (32, ??)
After L1:  (batch, ??)  = (32, ??)
After R1:  (batch, ??)  = (32, ??)
After L2:  (batch, ??)  = (32, ??)
After R2:  (batch, ??)  = (32, ??)
After L3:  (batch, ??)  = (32, ??)
```

### Exercise 2: Count Parameters
How many learnable parameters does the MNIST network have?
```
Layer 1: W1 is ___×___, b1 is ___ → Total: ___
Layer 2: W2 is ___×___, b2 is ___ → Total: ___
Layer 3: W3 is ___×___, b3 is ___ → Total: ___
Grand total: ___
```

### Exercise 3: Understand the Loss
If the true label is 7, and the network outputs logits:
```
[0.1, -1.0, 0.5, 0.2, -0.3, 0.4, 0.8, 2.0, -0.5, 0.3]
```
1. What is the softmax probability for class 7?
2. What is the cross-entropy loss?
3. Is this a good or bad prediction?

### Exercise 4: Trace the Gradients
Given the chain: `y = relu(W₂ @ relu(W₁ @ x + b₁) + b₂)`

Write out `∂y/∂W₁` using the chain rule (don't compute, just write the structure).

### Exercise 5: Modify the Code
Try modifying `experiments/mnist.py`:
1. Change the hidden dimension from 512 to 256
2. Add a third hidden layer
3. Change ReLU to sigmoid
4. Change learning rate from 0.01 to 0.001

What happens to training speed and final accuracy?

## Resources for Deeper Learning

### Essential Videos
1. [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
   - "But what is a neural network?"
   - "Gradient descent, how neural networks learn"
   - "Backpropagation calculus"

### Essential Reading
2. [JAX Quickstart](https://docs.jax.dev/en/latest/quickstart.html)
3. [JAX Automatic Differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html)
4. [Neural Networks for Physicists (arXiv)](https://arxiv.org/abs/2505.13042)

### Interactive Tutorials
5. [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)
6. [JAX PyTrees Tutorial](https://docs.jax.dev/en/latest/pytrees.html)

## What's Next (Week 2 Preview)

- **layers.py deep dive**: Chain, Branch, Parallel, Skip connections
- **einops.py**: Tensor manipulation with readable syntax
- **Building custom layers**: How to extend JULAX
- **mini_transformer.py**: Attention and modern architectures

---

## Answer Key

### Exercise 1: Shapes
```
Input:     (32, 784)   # 32 images, 784 pixels each
After L1:  (32, 512)   # W₁ is 784×512
After R1:  (32, 512)   # ReLU doesn't change shape
After L2:  (32, 512)   # W₂ is 512×512
After R2:  (32, 512)
After L3:  (32, 10)    # W₃ is 512×10, one logit per class
```

### Exercise 2: Parameters
```
Layer 1: W1 is 784×512, b1 is 512 → 784×512 + 512 = 401,920
Layer 2: W2 is 512×512, b2 is 512 → 512×512 + 512 = 262,656
Layer 3: W3 is 512×10,  b3 is 10  → 512×10 + 10  = 5,130
Grand total: 669,706 parameters
```

### Exercise 3: Loss
```python
import numpy as np
logits = [0.1, -1.0, 0.5, 0.2, -0.3, 0.4, 0.8, 2.0, -0.5, 0.3]
exp_logits = np.exp(logits)
probs = exp_logits / exp_logits.sum()
# probs[7] ≈ 0.44 (highest probability - good!)
# loss = -log(0.44) ≈ 0.82 (relatively low - good prediction!)
```

---

*You made it through Week 1! You now understand the foundations of neural networks, JAX, and JULAX.*

**Sources Used:**
- [Neural Networks for Physicists - arXiv](https://arxiv.org/abs/2505.13042)
- [Machine Learning for Physicists - Erlangen](https://machine-learning-for-physicists.org/)
- [JAX Automatic Differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html)
- [Understanding JAX AutoDiff - Towards Data Science](https://towardsdatascience.com/understanding-automatic-differentiation-in-jax-a-deep-dive-179bbdf01e87/)
- [Backpropagation - Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap2.html)
- [Softmax and Cross Entropy Loss](https://www.parasdahal.com/softmax-crossentropy)
- [Cross-Entropy Loss Guide](https://gombru.github.io/2018/05/23/cross_entropy_loss/)
- [PyTrees - JAX Documentation](https://docs.jax.dev/en/latest/pytrees.html)
- [Mastering JAX PyTrees - CodeSignal](https://codesignal.com/learn/courses/advanced-jax-transformations-for-speed-scale/lessons/nested-data-structures-mastering-jax-pytrees)
- [MNIST Dataset - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/mnist-dataset/)
- [Learning Rate for Neural Networks - ML Mastery](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/)
- [Chain Rule in Neural Networks - Enjoy Algorithms](https://www.enjoyalgorithms.com/blog/chain-rule-of-calculus-for-neural-networks/)
- [3Blue1Brown Backpropagation](https://www.3blue1brown.com/lessons/backpropagation-calculus)
- [ReLU vs Sigmoid - Stack Exchange](https://stats.stackexchange.com/questions/126238/what-are-the-advantages-of-relu-over-sigmoid-function-in-deep-neural-networks)
- [Pure Functions - Wikipedia](https://en.wikipedia.org/wiki/Pure_function)
