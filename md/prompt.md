# Research questions

We have a dataset of procedural texts in which every sample is a list of strings $[S_1, S_2, \dots, S_N]$, $N \in \{4, \dots, 12\}$, $S_i = \{\text{token}_j\}_{j=0}^{|S_i|}$. We do not have any annotations, but we do know that these are procedural texts and so the annotations would be DAGs, where the sink would usually be towards the end of the list, realistically almost always the last. We cannot guarantee however that the leaves would be towards the start. We want a sequence model such as a causal attention Transformer (e.g. GPT2) to learn the dependencies between steps, i.e. the directed $N \times N$ adjacency matrix, and more specifically the reachability matrix.

Currently, our approach is making the model learn to produce its hidden states so that the pooled step embeddings abide by the order-embedding loss from `ORDER-EMBEDDINGS OF IMAGES AND LANGUAGE`, Vendrov et al. 2016:

$$E(x,y) = \lVert\max(0, y-x)\rVert^2$$

$$\sum_{(u,v) \in \mathcal{P}} E(f(u), f(v)) + \sum_{(u',v') \in \mathcal{N}} \max\{0, \alpha - E(f(u'), f(v')) \}$$

We also want to verify if causal LM implicitly learns a topology similar to that of order embeddings.

After training, the model should still be able to work as a standard LLM. We want it to learn knowledge about reasoning about causal dependencies, while retaining its LM mechanics so that it can be evaluated on causal reasoning benchmarks.

# Data

Unsupervised data from RecipeNLG ($\sim 2.2\text{M}$ samples) used in `src/pretrain.py`:

```json
[
    {
        "title":"Oatmeal Breakfast Cookies",
        "directions":[
            "Blend together all wet ingredients.",
            "Mix the flour, soda, bran and salt.",
            "Mix into wet ingredients.",
            "Blend in oatmeal.",
            "Drop by large tablespoons.",
            "Bake at 375F (190C) F for 10 to 12 minutes."
        ],
        "link":"recipeland.com\/recipe\/v\/oatmeal-breakfast-cookies-33408",
        "source":"Recipes1M",
    }
]
```

ERFGC data from Yamakata et al. 2020:

```json
[
    {
        "words": ["Preheat", "oven", "to", "160", "C", "/", "Gas", "mark", "3", ".", "Cream", "butter", "and", "brown", "sugar", ".", "Add", "400g", "to", "500g", "flour", ".", "Mix", "well", ".", "Sprinkle", "board", "with", "the", "remaining", "flour", ".", "Knead", "for", "5", "minutes", ",", "adding", "enough", "flour", "to", "make", "a", "soft", "dough", ".", "Roll", "to", "1cm", "(", "1/2", "in", ")", "thickness", ".", "Cut", "into", "7x2cm", "(", "3x1", "in", ")", "strips", ".", "Prick", "with", "fork", "and", "place", "on", "ungreased", "baking", "trays", ".", "Bake", "in", "preheated", "oven", "for", "20", "to", "25", "minutes", "."],
        "head_indices": [77, 1, 0, 1, 0, 0, 1, 0, 0, 0, 17, 11, 0, 11, 0, 0, 23, 21, 0, 0, 17, 0, 33, 0, 0, 33, 26, 0, 0, 31, 26, 0, 38, 0, 33, 0, 0, 41, 40, 38, 47, 0, 0, 45, 41, 0, 56, 0, 54, 0, 49, 0, 0, 47, 0, 65, 0, 56, 0, 58, 0, 0, 56, 0, 69, 0, 65, 0, 75, 0, 72, 69, 0, 0, 0, 0, 75, 77, 0, 75, 0, 0, 0, 0],
        "step_indices": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    },
]
```

CaT-Bench benchmark ($\sim 20\text{k}$ samples):

```json
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

For the CaT-Bench example above, the desiderata for the model $f$ is that after being fed steps $S$ and producing the last hidden state $H = f(S)$, $H_{8}$ and $H_{10}$ should have embeddings that can be used to say $s(H_{8}, H_{10}) = 1$, while the score for the opposite direction is $s(H_{10}, H_{8}) = 0$.

# Pre-training task (pooled causal LM)

Given a dataset $\mathcal{D} = \{S^{(i)}\}_{i=1}^{|\mathcal{D}|}$, and given a sequence of steps $S^{(i)} = \{S^{(i)}_1, \dots, S^{(i)}_k\}$, with $j \in \{1, \dots, k\}$, and the original ordering $S_{orig}^{(i)} = [S_1, S_2, \dots, S_k]$ s.t. $j < j + 1$ for all $i \in \{1, \dots, k\}$, we sample a random ordering $S_{shuf}^{(i)}$ and feed as input to the model the concatenation $X^{(i)} = S_{shuf}^{(i)} \oplus S_{orig}^{(i)}$. For example, given $S^{(1)}$ and its original ordering $S_{orig}^{(1)} = [S_1, S_2, S_3, S_4]$, one shuffled version could be $S_{shuf}^{(1)} = [S_4, S_1, S_3, S_2]$ so that $X^{(1)} = [S_4, S_1, S_3, S_2] \oplus [S_1, S_2, S_3, S_4]$. We train a causal Transformer (e.g. GPT2) on inputs $X^{(i)}$ by calculating a causal language modeling loss only on the second part of the prompt and masking the loss for the first, shuffled portion of the predicted output. This is because it is not desirable for the model to learn a distribution over tokens belonging to randomly shuffled steps. In the unmasked ordered sequence, we average-pool the loss for all tokens within the same step, i.e. $\mathcal{L_{clm}^{(i)}} = \sum_{j=1}^{k} - \mathbb{E}_{x_t \sim S_{j}} \text{log}~p(x_t | x_{<t})$. The pooling is done so that the model's latent space is modified for the step as a whole, and not just on the first few tokens of the sequence. The reason for this is that predicting the first few tokens of the step entails the model correctly classifying the next step, while the subsequent tokens can just be copied by the induction heads \cite{anthropic2022inductionheads} by attending to the step in the shuffled side, matching the already predicted tokens. Note that the model is still able to attend to the shuffled sequence to compute the embeddings of the ordered portion of its hidden states. The masking is only done on the labels at the computation of the loss.

We wish to verify whether this unshuffling task enables the model to learn representations that can be used for inference on the CaT-Bench test set shown above.

# Pre-training task (simple causal LM)

Given a dataset $\mathcal{D} = \{S^{(i)}\}_{i=1}^{|\mathcal{D}|}$, and given a sequence of steps $S^{(i)} = \{S^{(i)}_1, \dots, S^{(i)}_k\}$, with $j \in \{1, \dots, k\}$, and the original ordering $S_{orig}^{(i)} = [S_1, S_2, \dots, S_N]$ s.t. $j < j + 1$ for all $i \in \{1, \dots, N\}$, we sample a random ordering $S_{shuf}^{(i)}$ and produce the input into the model as the concatenation $X^{(i)} = S_{shuf}^{(i)} \oplus S_{orig}^{(i)}$. For example, given $S^{(1)}$ and its original ordering $S_{orig}^{(1)} = [S_1, S_2, S_3, S_4]$, one shuffled version could be $S_{shuf}^{(1)} = [S_4, S_1, S_3, S_2]$ so that $X^{(1)} = [S_4, S_1, S_3, S_2] \oplus [S_1, S_2, S_3, S_4]$. We train a causal Transformer (e.g. GPT2) on inputs $X^{(i)}$ by calculating a causal language modeling loss only on the second part of the prompt, i.e. $\mathcal{L_{clm}^{(i)}} = - \sum \text{log}~p(x_t | x_{n, ..., t-1})$, where $N = |S_{shuf}^{(i)}|$. Note that the model is still able to attend to the shuffled sequence to compute the embeddings of the ordered portion of its hidden states. The masking is only done on the labels at the computation of the loss.

We wish to verify whether this unshuffling task enables the model to learn representations that can be used for inference on the CaT-Bench test set shown above.




## On-going discussion of the pre-training task

Given $[S_4, S_1, S_3, S_2]$, when we try to predict $S_1$ the model needs to pick what the first step is, so there is no problem. However, subsequent steps $S_j$ with $j > i$ are easier to predict given also the already predicted $S_i$.

The current proposed training paradigm entails using batches of samples $X^{(i)}_{j}$, $j \in \{1, \dots, M\}$ such as the following:

$X^{(1)}_{0} = [S_4, S_1, S_3, S_2] \oplus [S_1]$

$X^{(1)}_{1} = [S_4, S_1, S_3, S_2] \oplus [\text{<MASK>}] \oplus [S_2]$

...

$X^{(1)}_{M} = [S_4, S_1, S_3, S_2] \oplus [\text{<MASK>}, \text{<MASK>}, \dots, \text{<MASK>}] \oplus [S_N]$

The loss would just be causal LM loss on the predicted step and would be ignored on the $\text{<MASK>}$ tokens.

Even though we are masking at the step level, the model should learn to unshuffle the recipe since the first few tokens of each step entail classifying the next step of the sequence. Of course, the subsequent tokens of that step will be trivial to predict, since the induction heads of the model will simply be able to copy the sequence. The idea is that the classification done through the prediction of the first few tokens should be enough to teach the model to separate the embeddings of the steps in its latent space via classification of those first tokens.

We need to evaluate whether this is a sensible pre-training strategy.