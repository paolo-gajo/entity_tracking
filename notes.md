2026-03-04

i still wanna see if i can do MML using LEACE (https://arxiv.org/pdf/2306.03819) to remove positional information instead of using GRL

2026-03-03

need to figure out what's the correct direction for the edges, from the earlier step to the later step, or the opposite, i.e. the edge w.r.t. the entailment u \prec v.

diff = H_steps.unsqueeze(0) - H_steps.unsqueeze(1)

vs

diff = H_steps.unsqueeze(1) - H_steps.unsqueeze(0)

2026-03-02

- need to check how much correlation there is between the AUC obtained by sims.py on the `permutations` samples and the f1 macro score obtained by the regression probe on the cat-bench samples. if there is a strong correlation, then this could be an indication of the fact that the model's reasoning is reflected in the topology of its latent space.
- also need to try with non-negative activations. it seems right now i'm getting a lot of instability and the AUC can actually even flip and below 0.5, which is not bad per se, but it probably means the model is flipping the cone inclusion.

2026-02-24

my model should be able to tell whether a recipe has a valid topological order or if it's invalid. i want the topology of the step embeddings to reflect this difference.

or is that actually my desiderata? maybe what i want is that the pair-wise directed energy of the model's embeddings doesn't depend on the order of the steps, but just on what steps are present? so in that way the model actually knows what comes before what based on content, and the order i'm seeing printed out doesn't actually matter?

2026-02-21

i'm calculating MML positive/negative loss always on the same order: BAD BAD BAD

2026-02-20

CaT-Bench doesn't just have adjacent nodes, the questions are made from the transitive closure step-level adjacency matrix, meaning it asks questions about nodes linked by a path of any length, not just length = 1. So to eval on CaT-Bench I need to produce links between all nodesa, i.e. the transitive closure. I should also probably use transitive closure for `sims.py` as well, but not just on A, also on S.

2026-02-16

Evaluation idea:
[s_1, s_2, ..., s_N] + [s_i] + [s_j] --> calculate loss on [s_i] + [s_j]
[s_1, s_2, ..., s_N] + [s_i] + [s_j] --> calculate loss on [s_j] + [s_i]

and check whether they are contained in the same cone (how?)

2026-02-13
new losses:
- gated topo loss
$$\mathcal{L}_{geo} = \sum_{i < j} \alpha_{j \to i} \cdot || \text{ReLU}(H_j - H_i) ||^2$$
Where $\alpha_{j \to i}$ is the average attention weight from tokens in Step $j$ to tokens in Step $i$.
- contrastive order loss
$$\mathcal{L} = \max(0, E(H_i, H_j) - E(H_j, H_i) + \lambda)$$

tried losses:
- causal LM
- kl
- cone topological loss (order loss)

new training:
- batched k, 1 positive, k-1 negatives, causal LM

Pre-training:
- ablation: pre-train on just the unshuffled recipes
- ablation: pre-train on just the shuffled recipes

ideas:
- thinking vs non-thinking w/ pretraining vs non-thinking w/o pretraining or anything --> similarity evaluation of embeddings
- plot with topological orders on x axis, pick one model, e.g. the one at 120k steps
- block world: try to see if you can train a model to emit parameters for the creation of block world problems
