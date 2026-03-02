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

Unshuffling experiment:

We use two datasets, RecipeNLG and CaT-Bench. RecipeNLG consists of 2.2M samples, where each sample contains a list of strings, which are steps of a culinary recipe. CaT-Bench has a train set of 20,802 samples and a test set of 4260 samples. Each sample in CaT-Bench has a list of steps of recipes, like RecipeNLG, but also has questions e.g. "Must Step 10 happen after Step 8?" where the model needs to answer a binary question about the causal dependence of two of the steps.

I am pre-training a GPT2 model on prompts such as:

Minimal prompt:
```
In a heavy 2-quart saucepan, mix brown sugar, nuts, evaporated milk and butter or margarine. Let stand until firm, about 30 minutes. Stir in vanilla and cereal; mix well. Stir over medium heat until mixture bubbles all over top. Using 2 teaspoons, drop and shape into 30 clusters on wax paper. Boil and stir 5 minutes more. Take off heat. In a heavy 2-quart saucepan, mix brown sugar, nuts, evaporated milk and butter or margarine. Stir over medium heat until mixture bubbles all over top. Boil and stir 5 minutes more. Take off heat. Stir in vanilla and cereal; mix well. Using 2 teaspoons, drop and shape into 30 clusters on wax paper. Let stand until firm, about 30 minutes.<|endoftext|>
```
Natural language prompt:
```
Below is a jumbled list of recipe steps. Put them in the correct order.

Input:
- In a heavy 2-quart saucepan, mix brown sugar, nuts, evaporated milk and butter or margarine.
- Let stand until firm, about 30 minutes.
- Stir in vanilla and cereal; mix well.
- Stir over medium heat until mixture bubbles all over top.
- Using 2 teaspoons, drop and shape into 30 clusters on wax paper.
- Boil and stir 5 minutes more. Take off heat.

Correct order:
1. In a heavy 2-quart saucepan, mix brown sugar, nuts, evaporated milk and butter or margarine.
2. Stir over medium heat until mixture bubbles all over top.
3. Boil and stir 5 minutes more. Take off heat.
4. Stir in vanilla and cereal; mix well.
5. Using 2 teaspoons, drop and shape into 30 clusters on wax paper.
6. Let stand until firm, about 30 minutes.<|endoftext|>
```

In the `full_input` setting, all the prompt is used for loss calculation. In the `completion_only`, only the unshuffled (second) part is used.


We want the model to learn to produce different representations for each step, based on the underlying graph. Using sims.py we calculate the similarity between the step representations and compare them to the adjacency matrix A by calculating the auc(S, A) between A and the step similarity matrix S.


Results:

When we feed shuffled recipes the AUC is ~0.5, but with unshuffled recipes and recipes shuffled with valid topological orders the AUC is 0.65-0.67. This suggests that the model internally learns the step topology of the recipes.

    We want this skill to transfer to the CaT-Bench benchmark:

    ```
    [
        {
            "plan_idx": 0,
            "title": "spicy-tomato-anchovy-pasta",
            "question_idx": 0,
            "steps": [
                "Heat 6 tablespoons olive oil in a large frying pan over medium heat, then stir in garlic, broccoli and mushrooms;",
                "cook until lightly browned.",
                "Add anchovies and water, cover and simmer for 4 to 5 minutes.",
                "Stir in spring onions, tomatoes and parsley and cover again, simmering until vegetables are soft, about 3 to 4 minutes.",
                "While the vegetables are cooking, bring a large pot of water and one teaspoon of olive oil to the boil.",
                "Add linguine and cook until al dente, about 7 to 8 minutes;",
                "drain.",
                "Toss with anchovy mixture and chilli flakes.",
                "If desired, season with black pepper.",
                "Serve immediately."
            ],
            "question_type": "dependent_real_after",
            "step_pair_idx_asked_about": [
                7,
                9
            ],
            "binary_question": "Must Step 10 happen after Step 8?",
            "why_question": "Why must Step 10 happen after Step 8?",
            "label": 1,
            "type": "real",
            "direction": "after"
        }
    ]
    ```

    where recipes are assigned binary questions on the dependence of steps. Note that CaT-Bench uses data from English Flowgraph Recipe Corpus by Yamakata et al. 2020. RecipeNLG contains some of these, but we have already filtered matches so there is no overlap between pre-training and evaluation.
