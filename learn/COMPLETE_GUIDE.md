# JULAX: A Complete Guide for Physics PhDs (Who Don't Know ML)

## Part 0: The 30-Second Summary

**JULAX** = "Just Layers over JAX"

It's a tool for building "neural networks" (fancy function approximators) using a library called JAX. Think of it like building with LEGO blocks - each block (layer) does a simple thing, and you stack them to do complex things.

**Why does it exist?** To make it easy to build and train neural networks while keeping the code clean, functional (in the math sense), and able to run on many computers/GPUs at once.

---

## Part 1: What Problem Are We Even Solving?

### The Physics Analogy

Imagine you have a complicated function `f(x)` that you don't know the formula for, but you have lots of input-output pairs `(x_i, y_i)`. In physics, you might try to fit a polynomial or use perturbation theory.

**Neural networks** are a different approach: instead of guessing the functional form, you use a very flexible "universal approximator" - a function with many adjustable parameters (like fitting coefficients, but thousands or millions of them).

```
Traditional fitting:    y = ax² + bx + c           (3 parameters)
Neural network:         y = f(x; θ₁, θ₂, ..., θₙ)  (n can be millions)
```

### What is "Training"?

Training = Finding the best values for those parameters (θ) by:
1. Making predictions with current parameters
2. Computing how wrong you are (the "loss")
3. Using calculus (gradients) to adjust parameters to reduce the loss
4. Repeat millions of times

This is literally gradient descent, which you probably know from optimization.

### What is a "Layer"?

A layer is just a function with learnable parameters. The simplest is a "Linear" layer:

```
Linear layer: output = W @ input + b
```

Where `W` (weight matrix) and `b` (bias vector) are the learnable parameters.

Neural networks stack many layers:
```
input → Linear₁ → nonlinearity → Linear₂ → nonlinearity → ... → output
```

The nonlinearities (like `max(0, x)`) are what make neural networks more powerful than just matrix multiplication.

---

## Part 2: What is JAX?

JAX is a Python library from Google that does three main things:

### 1. Automatic Differentiation
```python
# JAX can automatically compute gradients of any function
def f(x):
    return x**2 + 3*x + 2

grad_f = jax.grad(f)  # Now grad_f computes df/dx automatically!
```

This is crucial for training - we need gradients to know how to adjust parameters.

### 2. JIT Compilation
```python
# JAX can compile Python to run really fast
@jax.jit
def slow_function(x):
    # ... complex math ...
    return result

# First call: slow (compiling)
# All subsequent calls: blazing fast
```

### 3. Vectorization & Parallelization
JAX can automatically:
- Run your code on GPUs/TPUs (specialized hardware)
- Distribute computation across multiple devices
- Transform loops into efficient parallel operations

### Why JAX Instead of PyTorch/TensorFlow?

JAX is **functional** - functions are "pure" (no side effects). This means:
- Easier to reason about mathematically
- Better for parallelization
- More like how a physicist thinks about functions

The catch: JAX is low-level. It's like assembly language - powerful but tedious.

**JULAX provides higher-level building blocks on top of JAX.**

---

## Part 3: What JULAX Actually Does

### The Core Insight

JULAX provides a clean way to:
1. **Define** neural network components (layers)
2. **Compose** them into complex architectures
3. **Train** them on data
4. **Scale** across multiple devices

### The Three Pillars

Every layer in JULAX separates three things:

```
┌─────────────────┐
│     LAYER       │  ← Configuration (immutable)
│  (the recipe)   │     e.g., "784 inputs, 512 outputs"
└─────────────────┘
         │
         ├───────────────────┐
         ▼                   ▼
┌─────────────────┐  ┌─────────────────┐
│    PARAMETERS   │  │      STATE      │
│  (the weights)  │  │ (mutable stuff) │
│  learned values │  │  random seeds,  │
│   W, b, etc.    │  │  running stats  │
└─────────────────┘  └─────────────────┘
```

**Why separate these?**

In physics terms: the "Hamiltonian" (layer) is separate from the "state vector" (parameters + state). This makes the code:
- Easier to checkpoint/save
- Easier to parallelize
- Easier to debug
- More mathematically clean

### How Layers Work

```python
# 1. Define the layer (configuration only)
layer = Linear(in_dim=784, out_dim=512)

# 2. Initialize parameters and state
params, state = layer.init(seed=42)

# 3. Run the forward pass
output, new_state = layer(input_data, params, state)
```

Every layer follows this pattern. No hidden state. No surprises.

---

## Part 4: The Codebase Structure

```
julax/
├── src/julax/           # THE CODE (only ~931 lines total!)
│   ├── base.py          # Type definitions, utilities
│   ├── core.py          # LayerBase, Learner, Trainer (the foundation)
│   ├── layers.py        # Neural network building blocks
│   ├── einops.py        # Tensor manipulation helpers
│   ├── experiment.py    # Training orchestration
│   ├── observers.py     # Logging/monitoring during training
│   ├── inputs.py        # Data loading (stub)
│   └── utils.py         # Misc helpers
│
├── experiments/         # EXAMPLES
│   ├── mnist.py         # Handwritten digit recognition
│   └── mini_transformer.py  # Language model (like baby GPT)
│
├── tests/               # Testing
└── docs/                # Documentation
```

---

## Part 5: Core Components Explained

### 5.1 LayerBase (core.py)

The abstract base class that everything inherits from.

```python
class LayerBase:
    def init(self, seed) -> (params, state):
        """Create initial parameters and state"""

    def forward(self, x, params, state) -> (output, new_state):
        """The actual computation"""

    def sublayers(self) -> dict:
        """Return nested layers for automatic initialization"""
```

### 5.2 Building Block Layers (layers.py)

**Data Flow Layers:**
```
┌──────────────────────────────────────────────────────────────┐
│  F(func)     - Wrap any function as a layer                  │
│  Chain       - Sequential: A → B → C                          │
│  Branch      - Split: A → [B, C, D] (one input, many outputs)│
│  Parallel    - Multi: [A,B,C] → [D,E,F] (zip-like)           │
│  SkipConnection - Residual: output = layer(x) + x            │
│  Repeated    - Apply same layer N times                       │
└──────────────────────────────────────────────────────────────┘
```

**Parameterized Layers:**
```
┌──────────────────────────────────────────────────────────────┐
│  Linear      - Matrix multiply + bias: Wx + b                │
│  Embedding   - Lookup table: token_id → vector               │
│  LayerNorm   - Normalize activations (stabilizes training)   │
│  Dropout     - Randomly zero out values (prevents overfitting)│
│  RotaryEmbed - Position encoding for sequences               │
└──────────────────────────────────────────────────────────────┘
```

### 5.3 Learner (core.py)

Wraps a model with a loss function:

```python
learner = Learner(
    model=my_neural_network,
    loss_fn=cross_entropy_loss,  # How to measure "wrongness"
    feature_name="image",        # Input key in data dict
    label_name="label"           # Target key in data dict
)
```

### 5.4 Trainer (core.py)

Wraps a Learner with an optimizer:

```python
trainer = Trainer(
    learner=learner,
    optimizer=optax.sgd(learning_rate=0.01)  # How to update params
)

# One training step:
new_params, new_state = trainer.forward_and_backward(batch, params, state)
```

### 5.5 Experiment (experiment.py)

Full training orchestration:

```python
experiment = Experiment(
    name="my_experiment",
    trainer=trainer,
    dataset=my_dataset,
    observer=LossLogger(),        # Print loss during training
    checkpoint_manager=manager,   # Save progress
    mesh_shape={"data": -1}       # Use all GPUs
)

experiment.run()  # Train until done!
```

---

## Part 6: Data Flow Diagram

```
                    YOUR SCRIPT
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
┌─────────┐      ┌──────────────┐      ┌──────────┐
│ Dataset │      │    Model     │      │ Optimizer│
│ (grain) │      │ (LayerBase)  │      │ (optax)  │
└────┬────┘      └──────┬───────┘      └────┬─────┘
     │                  │                   │
     │                  ▼                   │
     │           ┌──────────────┐           │
     │           │   Learner    │           │
     │           │ model + loss │           │
     │           └──────┬───────┘           │
     │                  │                   │
     │                  ▼                   │
     │           ┌──────────────┐           │
     │           │   Trainer    │◄──────────┘
     │           │learner + opt │
     │           └──────┬───────┘
     │                  │
     │                  ▼
     │           ┌──────────────┐
     └──────────►│  Experiment  │
                 │ orchestrator │
                 └──────┬───────┘
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
    ▼                   ▼                   ▼
┌─────────┐      ┌──────────────┐    ┌───────────┐
│Observers│      │ Checkpointing│    │ Sharding  │
│(logging)│      │   (orbax)    │    │ (devices) │
└─────────┘      └──────────────┘    └───────────┘
```

---

## Part 7: A Concrete Example (MNIST)

The experiments/mnist.py builds a network to recognize handwritten digits:

```
Input image: 28x28 = 784 pixels
              │
              ▼
┌─────────────────────────────┐
│ Linear(784 → 512) + ReLU    │  "Look for features"
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Linear(512 → 512) + ReLU    │  "Combine features"
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Linear(512 → 10)            │  "Score each digit 0-9"
└─────────────────────────────┘
              │
              ▼
Output: 10 scores (which digit is most likely?)
```

The code:
```python
model = Chain(
    F(jnp.ravel),                    # Flatten 28x28 → 784
    Linear(in_dim=784, out_dim=512),
    F(jax.nn.relu),                  # Nonlinearity
    Linear(in_dim=512, out_dim=512),
    F(jax.nn.relu),
    Linear(in_dim=512, out_dim=10),
)
```

Beautiful, right? Just stack building blocks.

---

## Part 8: What Makes JULAX Different?

### Compared to PyTorch

| Aspect | PyTorch | JULAX |
|--------|---------|-------|
| Paradigm | Object-oriented, imperative | Functional, pure |
| Parameters | Hidden inside modules | Explicit, passed around |
| Size | Massive framework | ~931 lines |
| Debugging | Can be tricky (hidden state) | Easy (everything explicit) |
| Distributed | Bolted on | Native (JAX sharding) |

### Compared to Other JAX Libraries

| Library | Focus | Size |
|---------|-------|------|
| Flax | Google's official, comprehensive | Large |
| Equinox | Pytree-based, elegant | Medium |
| Haiku | DeepMind's, transformation-based | Medium |
| **JULAX** | Minimal, composable, learning-friendly | Tiny |

### The Innovation

1. **Radical Minimalism**: Only 931 lines of code for a complete framework
2. **Pure Composition**: Build anything by stacking simple pieces
3. **First-class Sharding**: Distributed training is built-in, not an afterthought
4. **Educational Value**: Small enough to understand completely

---

## Part 9: Things You Should Research

### Essential Background

1. **Neural Network Basics**
   - Search: "3Blue1Brown neural networks" (YouTube series - highly visual)
   - Concepts: neurons, layers, activation functions, backpropagation

2. **Gradient Descent**
   - You probably know this from optimization
   - Key insight: we use gradients to minimize the loss function

3. **Loss Functions**
   - Cross-entropy for classification
   - Mean squared error for regression

### JAX Specific

4. **JAX Quickstart**
   - Search: "JAX quickstart tutorial"
   - Focus on: jax.grad, jax.jit, jax.vmap

5. **PyTrees**
   - JAX's way of handling nested data structures
   - Essential for understanding JULAX's parameter handling

6. **Automatic Differentiation**
   - Search: "automatic differentiation explained"
   - Different from symbolic differentiation (SymPy) and numerical differentiation

### Advanced Topics (For Later)

7. **Transformers**
   - The architecture behind ChatGPT, etc.
   - See mini_transformer.py example

8. **Distributed Training**
   - JAX sharding, device mesh
   - How to train on multiple GPUs/TPUs

---

## Part 10: Learning Roadmap

### Week 1: Foundations
```
□ Watch 3Blue1Brown neural network series
□ Read JAX quickstart guide
□ Run `experiments/mnist.py` and understand each line
□ Read `src/julax/base.py` (~100 lines)
```

### Week 2: Core Mechanics
```
□ Read `src/julax/core.py` (LayerBase, Learner, Trainer)
□ Read `src/julax/layers.py` (understand Chain, Linear)
□ Modify MNIST example: change architecture, hyperparameters
□ Add a new simple layer type
```

### Week 3: Advanced Features
```
□ Read `src/julax/experiment.py` (checkpointing, sharding)
□ Read `src/julax/observers.py` (callback pattern)
□ Study `experiments/mini_transformer.py`
□ Learn about attention mechanism
```

### Week 4: Contribution Ready
```
□ Run tests, understand test patterns
□ Read through all remaining code
□ Identify something to improve or add
□ Make your first contribution!
```

---

## Part 11: Quick Reference Card

### Creating a Model
```python
from julax import Chain, Linear, F
import jax.nn

model = Chain(
    Linear(in_dim=..., out_dim=...),
    F(jax.nn.relu),
    Linear(in_dim=..., out_dim=...),
)
```

### Initializing
```python
params, state = model.init(seed=42)
```

### Forward Pass
```python
output, new_state = model(input_data, params, state)
```

### Training Setup
```python
from julax import Learner, Trainer
import optax

learner = Learner(model=model, loss_fn=my_loss)
trainer = Trainer(learner=learner, optimizer=optax.adam(1e-3))
```

### One Training Step
```python
new_params, new_state = trainer.forward_and_backward(batch, params, state)
```

---

## Part 12: Key Files to Read (In Order)

1. **`src/julax/base.py`** - Type definitions, understand the vocabulary
2. **`src/julax/core.py`** - The heart of everything
3. **`src/julax/layers.py`** - Building blocks
4. **`experiments/mnist.py`** - See it all come together
5. **`src/julax/experiment.py`** - Full training orchestration
6. **`experiments/mini_transformer.py`** - Advanced example

---

## Part 13: Glossary

| Term | Plain English |
|------|---------------|
| **Tensor** | Multi-dimensional array (like a matrix, but can have more dimensions) |
| **Forward pass** | Running input through the network to get output |
| **Backward pass** | Computing gradients for training |
| **Epoch** | One complete pass through the training data |
| **Batch** | Small chunk of data processed together |
| **Learning rate** | How big of steps to take when updating parameters |
| **Loss** | A number measuring how wrong the predictions are |
| **Gradient** | Direction to adjust parameters to reduce loss |
| **Activation function** | Nonlinear function applied after linear layers |
| **PyTree** | JAX's way of handling nested data structures |
| **Sharding** | Distributing data/computation across multiple devices |
| **Checkpoint** | Saved state of training (parameters, optimizer state, etc.) |

---

## Part 14: Is This Project Worth Joining?

### Pros
- **Small and understandable**: You can actually read ALL the code
- **Active development**: Your contributions will matter
- **Good learning vehicle**: Understand deep learning from the ground up
- **Modern tech stack**: JAX is increasingly popular in research
- **Clean code**: Well-structured, functional paradigm

### Cons
- **Early stage**: May break, APIs may change
- **Limited ecosystem**: Not as many tutorials/examples as PyTorch
- **Niche**: Won't directly transfer to industry jobs (which mostly use PyTorch)

### Verdict
**Great for learning, research, and personal projects. Not for production ML systems (yet).**

If you want to deeply understand how modern ML frameworks work, this is an excellent project to study and contribute to.

---

## Part 15: Questions to Ask Your Friend (Jun)

1. What's the long-term vision for JULAX?
2. Are there specific features you want to add next?
3. What's missing compared to Flax/Equinox?
4. Are there performance benchmarks?
5. What would be a good first contribution for a newcomer?

---

*Generated for a physics PhD who doesn't know ML. Good luck!*
