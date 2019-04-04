# Arithmetic Hash
> This project was developed upon [Justin](https://github.com/justin-labry)'s idea.

[Evaluation](https://nbviewer.jupyter.org/gist/q0115643/d5e2aee3089cca8f64791285eda23ba7)

## Background: SHA-1


```
A good hash function should map the expected inputs as evenly as possible over its output range. - Wikipedia/Hash_Function
```

Hash algorithm we use for bittorrent is SHA-1, which succeeds to map the encoded outputs in absolute evenly with given range.
But this is earned by losing the ability to decode and distinguish the original key. (This project does not consider about the 'security' side of SHA-1)


## Improvement Direction

If we can get the original key from hashed output, we will be able to easily search through all the info_hash (hashed keys) to find the file we want to download.
For this purpose the Arithmetic Hashing is developed.
(1) To hash keys in known algorithm in the way that we can decode, and (2) to map the encoded outputs evenly to its output range.


The basic method is like this, suppose that our key is "apple" and our output range is ```[0, 1]```.
By expected probability if the probability of ```a``` being the first letter is 10%, than the 'apple' goes in 0 to 0.1 section.
And if the probability of ```b```s appearance as second letter after ```a```is 5%, ```apple``` goes in ```[x, x+0.005]```.
So on, we can specify the position ```apple``` shold go in.


## Recent Work

Recently, one paper jumped into this idea with 3-order Markov Chain. (I will link the paper as soon as I get it back)
But in our hypothesis, 3-order Markov Chain would have descending performance as the key (query) gets longer than one simple word.
That's why we are trying to solve this issue with RNN(LSTM).

## Results

You can see the experiment flow, results, and the evaluation from
[this link](https://nbviewer.jupyter.org/gist/q0115643/d5e2aee3089cca8f64791285eda23ba7) of the jupyter notebook.



![](https://raw.githubusercontent.com/q0115643/my_blog/master/assets/images/arithmetic-hash/0.png)

standard deviation results from brown corpus (1 word long keys)


![](https://raw.githubusercontent.com/q0115643/my_blog/master/assets/images/arithmetic-hash/1.png)

standard deviation results from 2gram corpus (equal or more than 2 words long keys)



with keys with short length, Markov Chain showed better performance, but RNN becomes better as the key length increases.
The results were as we desired, and encoding-decoding experiments also went well.

