# Results

Corpus: `data/input.txt` (1,115,394 characters, vocab 65); model: 159,937 parameters, context 32, 3 layers x 4 heads, n_embd 64.

| Model | Validation loss (nats/char) | Perplexity |
|---|---|---|
| Uniform guess (ln vocab) | 4.174 | 65.0 |
| Unigram character frequencies | 3.347 | 28.4 |
| **Mini-GPT (this repo)** | **1.802** | **6.1** |

Samples (top-k 20):

Temperature 0.5:
```

Countent the senation this sound he present the father,
And in this with be sound the sent of great brother waster.

POMPEY:
I shall not here not he shall with him with his brother,
I wake my than name his true that is the brother and that with on wi
```

Temperature 0.8:
```

But with this move, for your master:
In this to he have creeter of my big then own eachard.

COMIO:
In him it
I will will will to speak is the die should this find
More news me bound and, for himselves hear own sin
With be is hear. Go hour traitors t
```

Temperature 1.0:
```


Come, diwn, this gave, your cattales friend the wrong intake haste of so cheeknates friends,
I call go, a fingue done, that and by suake so,
To wife be lieve witch be damnight
With bay shall notire fink thy light my tyrant up his come
To that sould 
```
