# JULAX Mind Map

## Visual Overview

```
                                    ╔═══════════════════════════════════════╗
                                    ║            JULAX PROJECT              ║
                                    ║    "Just Layers over JAX" - ~931 LOC  ║
                                    ╚═══════════════════════════════════════╝
                                                       │
                    ┌──────────────────────────────────┼──────────────────────────────────┐
                    │                                  │                                  │
                    ▼                                  ▼                                  ▼
        ┌───────────────────┐              ┌───────────────────┐              ┌───────────────────┐
        │   WHAT IT IS      │              │   WHY IT EXISTS   │              │   HOW TO USE IT   │
        └───────────────────┘              └───────────────────┘              └───────────────────┘
                    │                                  │                                  │
    ┌───────────────┼───────────────┐     ┌───────────┼───────────┐      ┌───────────────┼───────────────┐
    │               │               │     │           │           │      │               │               │
    ▼               ▼               ▼     ▼           ▼           ▼      ▼               ▼               ▼
┌─────────┐   ┌─────────┐   ┌─────────┐ ┌─────┐  ┌─────────┐ ┌───────┐ ┌───────┐   ┌─────────┐   ┌─────────┐
│Framework│   │Built on │   │Minimal  │ │JAX  │  │PyTorch  │ │Compose│ │Define │   │Train    │   │Scale    │
│for NN   │   │JAX      │   │~931 LOC │ │is   │  │is OOP   │ │layers │ │model  │   │model    │   │across   │
│         │   │         │   │         │ │func-│  │heavy    │ │easily │ │       │   │         │   │devices  │
└─────────┘   └─────────┘   └─────────┘ │tional│  └─────────┘ └───────┘ └───────┘   └─────────┘   └─────────┘
                                        └─────┘
```

## Architecture Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                     EXPERIMENT                                          │
│   Orchestrates everything: dataset, training loop, checkpointing, distributed compute   │
│                                                                                         │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                    TRAINER                                         │  │
│  │              Combines Learner + Optimizer, performs gradient updates               │  │
│  │                                                                                    │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                                  LEARNER                                     │  │  │
│  │  │                    Combines Model + Loss Function                            │  │  │
│  │  │                                                                              │  │  │
│  │  │  ┌───────────────────────────────────────────────────────────────────────┐  │  │  │
│  │  │  │                              MODEL                                     │  │  │  │
│  │  │  │            Composition of Layers (Chain, Linear, etc.)                 │  │  │  │
│  │  │  │                                                                        │  │  │  │
│  │  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │  │  │  │
│  │  │  │  │ Linear  │→│  ReLU   │→│ Linear  │→│  ReLU   │→│ Linear  │          │  │  │  │
│  │  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘          │  │  │  │
│  │  │  │                                                                        │  │  │  │
│  │  │  └───────────────────────────────────────────────────────────────────────┘  │  │  │
│  │  │                                                                              │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                                    │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## File Structure Map

```
julax/
│
├── src/julax/                    ← ALL THE CODE (~931 lines)
│   │
│   ├── base.py      [~100 LOC]  ← TYPES & UTILITIES
│   │   ├── PRNG, PyTree, Param, State (type aliases)
│   │   ├── FrozenDict (immutable dict)
│   │   └── dispatch (function overloading)
│   │
│   ├── core.py      [~186 LOC]  ← THE FOUNDATION ★★★ START HERE
│   │   ├── LayerBase (abstract base for all layers)
│   │   ├── Learner (model + loss)
│   │   └── Trainer (learner + optimizer)
│   │
│   ├── layers.py    [~284 LOC]  ← BUILDING BLOCKS ★★ THEN HERE
│   │   ├── Control: F, Chain, Branch, Parallel, SkipConnection, Repeated
│   │   └── Neural: Linear, Embedding, LayerNorm, Dropout, RotaryEmbedding
│   │
│   ├── einops.py    [~129 LOC]  ← TENSOR MANIPULATION
│   │   ├── Rearrange (reshape tensors)
│   │   ├── Reduce (aggregate tensors)
│   │   └── EinMix (learnable tensor ops)
│   │
│   ├── experiment.py [~106 LOC] ← TRAINING ORCHESTRATION
│   │   └── Experiment (full training loop, checkpointing, sharding)
│   │
│   ├── observers.py  [~95 LOC]  ← MONITORING
│   │   ├── LossLogger, StepTimeLogger
│   │   └── Composable via * operator
│   │
│   └── utils.py      [~20 LOC]  ← HELPERS
│       └── create_mesh (for distributed compute)
│
├── experiments/                  ← EXAMPLES ★★★ STUDY THESE
│   ├── mnist.py                 ← Simple: digit recognition
│   └── mini_transformer.py      ← Advanced: language model
│
└── tests/
    ├── smoke_test.py            ← Basic sanity
    └── test_einops.py           ← Einops coverage
```

## Layer Composition Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            COMPOSITION PATTERNS                              │
└─────────────────────────────────────────────────────────────────────────────┘

1. SEQUENTIAL (Chain)
   ┌───┐    ┌───┐    ┌───┐    ┌───┐
   │ A │ → │ B │ → │ C │ → │ D │     Chain(A, B, C, D)
   └───┘    └───┘    └───┘    └───┘

2. BRANCHING (Branch)
                    ┌───┐
                 ┌─→│ B │
   ┌───┐        │  └───┘
   │ A │────────┤  ┌───┐              Branch(A, [B, C, D])
   └───┘        ├─→│ C │
                │  └───┘
                │  ┌───┐
                └─→│ D │
                   └───┘

3. PARALLEL (Parallel)
   ┌───┐         ┌───┐
   │ A │ ──────→ │ D │
   └───┘         └───┘
   ┌───┐         ┌───┐              Parallel([A,B,C], [D,E,F])
   │ B │ ──────→ │ E │
   └───┘         └───┘
   ┌───┐         ┌───┐
   │ C │ ──────→ │ F │
   └───┘         └───┘

4. SKIP CONNECTION (Residual)
   ┌─────────────────────────┐
   │                         │
   │    ┌───┐    ┌───┐      │      SkipConnection(Chain(B, C))
   │    │ B │ → │ C │      │      output = B(C(x)) + x
   │    └───┘    └───┘      │
   │         ↓               │
   x ───────[+]────────→ output
```

## The Training Loop Cycle

```
                              ┌──────────────────┐
                              │   INITIALIZE     │
                              │ params, state =  │
                              │ model.init(seed) │
                              └────────┬─────────┘
                                       │
                                       ▼
          ┌────────────────────────────────────────────────────┐
          │                                                    │
          │    ┌─────────────────────────────────────────┐     │
          │    │           GET NEXT BATCH                │     │
          │    │      batch = next(dataset)              │     │
          │    └───────────────────┬─────────────────────┘     │
          │                        │                           │
          │                        ▼                           │
          │    ┌─────────────────────────────────────────┐     │
          │    │          FORWARD PASS                   │     │
          │    │   predictions = model(batch, params)    │     │
          │    └───────────────────┬─────────────────────┘     │
          │                        │                           │
          │                        ▼                           │
          │    ┌─────────────────────────────────────────┐     │
          │    │          COMPUTE LOSS                   │     │
          │    │   loss = loss_fn(predictions, labels)   │     │
          │    └───────────────────┬─────────────────────┘     │
          │                        │                           │
          │                        ▼                           │
REPEAT    │    ┌─────────────────────────────────────────┐     │ REPEAT
          │    │         COMPUTE GRADIENTS               │     │
          │    │   grads = jax.grad(loss)(params)        │     │
          │    └───────────────────┬─────────────────────┘     │
          │                        │                           │
          │                        ▼                           │
          │    ┌─────────────────────────────────────────┐     │
          │    │         UPDATE PARAMETERS               │     │
          │    │   params = params - lr * grads          │     │
          │    └───────────────────┬─────────────────────┘     │
          │                        │                           │
          │                        ▼                           │
          │    ┌─────────────────────────────────────────┐     │
          │    │      CHECKPOINT & LOG (optional)        │     │
          │    │   save(params); log(loss)               │     │
          │    └───────────────────┬─────────────────────┘     │
          │                        │                           │
          └────────────────────────┴───────────────────────────┘
                                   │
                                   ▼
                              ┌──────────────────┐
                              │    FINISHED      │
                              │  trained model   │
                              └──────────────────┘
```

## Dependencies Ecosystem

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DEPENDENCY MAP                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │              JULAX                   │
                    │        Your Application Code        │
                    └─────────────────┬───────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│     JAX       │           │    OPTAX      │           │    EINOPS     │
│ Core compute  │           │  Optimizers   │           │ Tensor ops    │
│ autodiff, jit │           │ SGD, Adam...  │           │ Rearrange...  │
└───────┬───────┘           └───────────────┘           └───────────────┘
        │
        ├──────────────────────┐
        │                      │
        ▼                      ▼
┌───────────────┐    ┌───────────────┐
│    XLA        │    │   NUMPY       │
│ Compilation   │    │ Array ops     │
└───────────────┘    └───────────────┘
        │
        ▼
┌───────────────┐
│  HARDWARE     │
│ CPU/GPU/TPU   │
└───────────────┘

Additional deps:
┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│    GRAIN      │  │    ORBAX      │  │    PLUM       │  │   PYDANTIC    │
│ Data loading  │  │ Checkpointing │  │ Multi-dispatch│  │ Config/valid  │
└───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘
```

## Quick Decision Tree: Which Layer to Use?

```
                                   ┌─────────────────────┐
                                   │ What do you want    │
                                   │ to do?              │
                                   └──────────┬──────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              │                               │                               │
              ▼                               ▼                               ▼
    ┌─────────────────┐             ┌─────────────────┐             ┌─────────────────┐
    │ Transform data  │             │ Combine layers  │             │ Learn patterns  │
    │ (reshape, etc.) │             │ (compose)       │             │ (weights)       │
    └────────┬────────┘             └────────┬────────┘             └────────┬────────┘
             │                               │                               │
    ┌────────┴────────┐         ┌────────────┼────────────┐        ┌────────┴────────┐
    │                 │         │            │            │        │                 │
    ▼                 ▼         ▼            ▼            ▼        ▼                 ▼
┌───────┐       ┌───────┐ ┌───────┐    ┌───────┐   ┌───────┐ ┌───────┐       ┌───────┐
│Rearrange      │Reduce │ │Chain  │    │Branch │   │Parallel │Linear │       │Embed  │
│"b c h w       │"b c ->│ │A→B→C  │    │A→[B,C]│   │[A,B]→   │Wx+b   │       │lookup │
│ -> b (c h)"   │ b"    │ │       │    │       │   │[C,D]    │       │       │table  │
└───────┘       └───────┘ └───────┘    └───────┘   └───────┘ └───────┘       └───────┘
```

## Comparison: JULAX vs Others

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FRAMEWORK COMPARISON                                 │
└─────────────────────────────────────────────────────────────────────────────┘

FEATURE          │ JULAX      │ PyTorch    │ Flax       │ Equinox
─────────────────┼────────────┼────────────┼────────────┼────────────
Paradigm         │ Functional │ OOP        │ Functional │ Functional
Size (LOC)       │ ~931       │ ~500k      │ ~50k       │ ~10k
Backend          │ JAX        │ LibTorch   │ JAX        │ JAX
Learning curve   │ Low        │ Medium     │ High       │ Medium
Industry use     │ None       │ High       │ Medium     │ Low
Documentation    │ Minimal    │ Extensive  │ Good       │ Good
State handling   │ Explicit   │ Implicit   │ Explicit   │ Mixed
GPU/TPU support  │ Native     │ Native     │ Native     │ Native
Distributed      │ Built-in   │ Add-on     │ Built-in   │ Manual

BEST FOR:
─────────────────┼────────────┼────────────┼────────────┼────────────
                 │ Learning   │ Production │ Research   │ Research
                 │ Research   │ Industry   │ Google     │ Elegance
```

## Innovation Score Card

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         IS THIS PROJECT INNOVATIVE?                          │
└─────────────────────────────────────────────────────────────────────────────┘

INNOVATION ASPECTS:

[████████████████████] 100%  Minimalism (smallest complete framework)
[████████████████░░░░]  80%  Clean API design
[████████████████░░░░]  80%  Composition patterns
[██████████████░░░░░░]  70%  First-class sharding
[████████████░░░░░░░░]  60%  Educational value
[████████░░░░░░░░░░░░]  40%  Novelty (similar ideas exist)
[████░░░░░░░░░░░░░░░░]  20%  Production readiness
[██░░░░░░░░░░░░░░░░░░]  10%  Ecosystem/community

VERDICT:
┌─────────────────────────────────────────────────────────────────────────────┐
│  INTERESTING FOR: Learning JAX, understanding ML frameworks, small projects │
│  NOT FOR: Production ML systems, large teams, industry jobs                 │
│                                                                             │
│  UNIQUE VALUE: Read and understand ALL the code in one afternoon!           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Your Learning Path

```
                                START HERE
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   WEEK 1: FOUNDATIONS         │
                    │   □ Watch 3B1B neural nets    │
                    │   □ JAX quickstart            │
                    │   □ Run mnist.py              │
                    │   □ Read base.py              │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   WEEK 2: CORE MECHANICS      │
                    │   □ Read core.py              │
                    │   □ Read layers.py            │
                    │   □ Modify MNIST example      │
                    │   □ Add a new layer           │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   WEEK 3: ADVANCED            │
                    │   □ Read experiment.py        │
                    │   □ Read observers.py         │
                    │   □ Study mini_transformer    │
                    │   □ Learn attention           │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   WEEK 4: CONTRIBUTION        │
                    │   □ Run all tests             │
                    │   □ Read remaining code       │
                    │   □ Pick an improvement       │
                    │   □ Submit first PR!          │
                    └───────────────────────────────┘
                                    │
                                    ▼
                              YOU'RE IN! 🎉
```
