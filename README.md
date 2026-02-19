# Research questions

We have a dataset of procedural texts where every sample is a list of strings [step_1, step_2, ..., step_N], N \in {4, ..., 12}, step_i = {token_j}_j=0^|step_i|. We do not have any annotations, but we do know that these are procedural texts and so the annotations would be DAGs, where the sink would usually be towards the end of the list, realistically almost always the last. We cannot guarantee however that the leaves would be towards the start. We want a sequence model such as a causal attention Transformer (e.g. GPT2) to learn the dependencies between steps, i.e. the directed NxN adjacency matrix. Currently, our approach is making the model learn to produce its hidden states so that the poooled step embeddings abide by the order-embedding loss from `ORDER-EMBEDDINGS OF IMAGES AND LANGUAGE`, Vendrov et al. 2016.

We also want to verify if causal LM implicitly learns a topology similar to that of order embeddings.

After training, the model should still be able to work as a standard LLM. i want it to learn knowledge about reasoning about causal dependencies, but it still has to be an LM after that that i can evaluate on causal reasoning benchmarks.

# Data

Unsupervised data from RecipeNLG (~2.2M samples) used in `src/pretrain.py`:

```
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
    },
...
]
```

ERFGC data from Yamakata et al. 2020 that we use in `src/sims.py`:

```
[
    {
        "words": ["Combine", "tea", ",", "spinach", ",", "kiwi", ",", "avocado", ",", "banana", "and", "ginger", "in", "a", "blender", ".", "Blend", "until", "smooth", "."],
        "pos_tags": ["B-Ac", "B-F", "O", "B-F", "O", "B-Ac", "O", "B-F", "O", "B-F", "O", "B-F", "O", "O", "B-T", "O", "B-Ac", "O", "B-Sf", "O"],
        "head_tags": ["t", "t", "root", "t", "root", "t", "root", "t", "root", "t", "root", "t", "root", "root", "d", "root", "root", "root", "v-tm", "root"],
        "head_indices": [17, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 17, 0],
        "step_indices": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "sent_indices": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
        "word_sent_indices": [1, 9, 13, 15, 23, 25, 30, 32, 40, 42, 49, 53, 60, 63, 65, 73, 1, 7, 13, 20]
    },
...
]
```

CaT-Bench benchmark (~20k samples) used in `src/eval_zeroshot.py`:

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
    },
...
]
```

Note that CaT-Bench samples are built from ERFGC.
