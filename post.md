LucaProt is a bioinformatic tool published in a 2024 Cell article "[Using artificial intelligence to document the hidden RNA virosphere](https://www.cell.com/cell/fulltext/S0092-8674\(24\)01085-7)" It's a binary classifier which detects RNA dependent RNA polymerases (RdRps) \- proteins essential to the lifecycle of a large proportion of viruses, including SARS-CoV-2[^1].

Viruses only have one goal \- to make more of themselves, and to this end they require only two things: to replicate their genetic material and create a capsid to enclose it[^2]. RdRps perform the first function for the subset of viruses which use RNA as their genetic material. Since every RNA virus has to have a version of this protein, it makes it an attractive identification target for finding novel viruses.

The standard approach is homology search \- find things that look like known RdRps. This approach is sound, but it can only find RdRps which are similar to previously studied ones. LucaProt's pitch is that a machine learning model can learn to recognize novel RdRps. 

The model architecture is as follows: they take the pretrained protein language model ESM-2 3B, which was trained to predict protein sequences and contains some biology knowledge inside it. They then run their sequences through it, then take the output from its final layer. Completely parallel to this, they have their own transformer architecture that they also run the candidate sequence through. They then pool each output with a learned attention layer, concatenate the two vectors and train a logistic regression classifier on top.

LucaProt was pointed at 10,487 metatranscriptomes and detected 161,979 putative viral species classified into 180 supergroups \- of which only 21 contain anything currently classified by the ICTV. This is an 8.6-fold expansion of RNA virus diversity at the supergroup level.

What first sparked my interest was page 23, figure S2, part of which I show here:\!\[\[Pasted image 20260813002647.png|333\]\] Notice what's wrong?

If you're not familiar with machine learning then this might not mean much to you, so I'm going to spend a little while explaining what train/validation/test splits are.

## ML basics

When you train a ML model, you typically need to do three things:

1. Train the weights for your model, aka the parameters, in linear regression this would be the coefficients.  
2. Find good hyperparameters \- these are parameters which determine how your model works, but aren't learned in training, things like learning rate, regularization strength, max depth for tree models and so on.  
3. Get a final estimation of the loss / accuracy of your model

To achieve these tasks, you split your training data into three sets \- train/validation/test, with something like a 8 : 1 : 1 ratio. You tune your weights on the train set, optimize hyperparameters on the validation set and finally get the final estimate of performance on the test set. Importantly the test number is the only one that hasn't been trained against and is treated as representative of the actual performance of the model.

In a classifier, we mainly care about two things

1. The true positive rate (TPR) \- how many of the positives were correctly identified as such.  
2. The false positive rate (FPR) \- how many of the true negatives were incorrectly identified.

AUC is a way to combine both of them, by plotting TPR against FPR at different thresholds for classification and taking the integral. More information [here](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc).

LucaProt's split was 190k / 22k / 22k. On the test set it gets 539 of 540 positives, and 0 false positives out of 22k negatives. Accuracy 0.99+, AUC 0.99+. Six of their ten cross-validation folds make zero mistakes, the rest make 1\. This looks like excellent performance.

## This is suspicious

Some time ago, my physics teacher told me that if you get a nice line of experimental data, and one massive outlier, the first thing you should do is not email Nature or Science with your discovery, but instead rerun the experiment\! It's a lot more likely that you've screwed up somewhere than that you have discovered new physics.

Drawing a loose analogy, if someone was training a classifier and saw an accuracy of 100% on the test set, they should check their setup for problems.

The main problem these researchers have is that they used a random, homology unaware split.

## Quantifying the problem

85.6% of test positives had a corresponding protein which was at least 30% similar in the train set. Median identity is 54%. \!\[\[Pasted image 20260813183255.png\]\] This means almost all of the test positives have a similar protein in the train set, inflating the models performance.

Homology leakage isn't the only problem here \- the positives they chose are also much longer than the negatives \- median \~1.6k vs \~300 amino acids. Classifying just on sequence length alone gets you an AUC of **0.93**.

To give the researchers some credit, constructing a proper train / test split for this problem is not trivial. Choosing positives is relatively easy \- just use a dataset of well known RdRps. However, what do you do about the negatives? You could just choose random cellular proteins and be done with it, but then your classifier could just learn to differentiate viral and non-viral proteins and have perfect performance. You need to choose negatives which are representative of the negatives you're likely to encounter in the "deployment"[^3] environment. The only problem is: that environment here is literally every other possible protein group on the planet that can appear in an environmental sample\! They chose reasonably \- using mainly proteins which are highly similar to RdRps in function, like other polymerases, eukaryotic RdRps, reverse transcriptases, and 170k random cellular proteins for good measure.

The only problem is that best I can tell, in addition to the random split problem above, they accidentally put most of the hard negatives exclusively in the train set. All 10,000 sequences labelled `rt` (reverse transcriptase) and all 2,000 labelled `dna` (DNA virus proteins) are in training, and none are in test. The test set's hard negatives are 160 non-RdRP viral proteins against 21,584 cellular ones.

And reverse transcriptases are the ones that matter here \- it's a common protein in environmental data, performs a very similar function[^4] , and has homologous domains. They are, on priors, the hardest negative. Alas, if you actually test RTs that aren't homologous to the ones in the train set they turn out to be easy to correctly categorize.

The cross validation also doesn't save them, since by looking at the counts of sequences in train/test you can see that it's short about 12k \- almost exactly how many were added after the fact.

## The false positive rate

What matters most here is the false positive rate. If the recall isn't perfect that just means missing some results, not getting false ones. They ran LucaProt and a more conventional homology based tool (ClstrSearch) against 144M sequences, and after filtering out results from ClstrSearch, only 444 contigs remain out of which they build their unique viral supergroups.

Despite the fact that they show zero false positives on the test set[^5], they report a FPR of 0.014%. Against 144.6 million proteins that's roughly 20,000 expected false positives \- about 46 times the number of unique contigs the AI-only claim rests on. This isn't immediate ironclad evidence that all of them are fake \- these 444 have survived a couple of rounds of other filtering, but it does raise doubts.

In a different figure they report sequencing 50 samples themselves and then labeling those as true positives by running 5 detection tools, including LucaProt, against them. If any of them return true then it's a true positive. **This is a circular argument**, so the statistics which use the TP count aren't useful. Thankfully when constructing the true negative set they did it reasonably, by checking which sequences definitely can't be RdRps and measuring against those. This is the best number the paper offers, and it is 0.023% \- 33,000 expected false positives, a lot more than 444\.

However, this is using their own reported FPR. Since their evaluations were leaky and shouldn't be trusted, we have to conduct our own investigation.

## Benchmarking

At this point in the investigation I wasn't even sure their tool worked at all. They had done other validations which we will get to in a bit, but I wanted to verify for myself. They tested their tool against other studies which reported finding novel RdRps, but once again did not check for homology against their train set. So I took all of those sequences, kept the ones that didn't have any homology to the train set ones and ran their model against these "orphan" sequences.

And their model works\! Somehow, after running against 2.4k sequences, it has a recall of 97.7% \- this is good. Based on this I update that RdRp detection isn't very hard as an ML problem, since if it was I would expect the model to memorize/overfit a lot more given the leakage.

To get some insight into the FPR, I ran their model against a bunch of subsets of proteins:

| class | n | false positives | FPR |
| :---- | :---- | :---- | :---- |
| cellular, short (\<2,500) | 16,671 | **0** | 0.000% |
| bacteriophage | 7,951 | **0** | 0.000% |
| prokaryotic mobile-element RT | 3,728 | **0** | 0.000% |
| **all prokaryotic \+ phage** | **28,350** | **0** | **0.000%** |
| cellular, long (≥2,500) | 6,090 | 24 | 0.394% |
| eukaryotic DNA virus, 300–999 | 9,565 | 133 | 1.390% |
| eukaryotic RNA virus non-RdRP, short | 4,471 | 202 | 4.518% |
| eukaryotic DNA virus, 1,000–2,499 | 4,941 | 252 | 5.100% |

The FPR is zero for phages and short cellular proteins. But DNA viruses have a rate of **1.4 \- 5.1%**, that is huge. The false positives are also not ones you would predict. Top annotation words among the firing DNA-virus proteins are *helicase* and *initiator protein*. Among long cellular false positives: 6 ubiquitinyl hydrolases, 4 SecA-family, **only 3 actual polymerases**. I take this as evidence that something weird is going on with the model \- it has learned some weird proxy for being a RdRp instead of the actual catalytic core. If it had learned something closer to the real underlying biology of what makes a protein an RdRp I would expect more mistakes with other polymerases which are more similar in the catalytic core.

This 1.4% rate isn't comparable to the 0.023% rate, since that one is based on the full metagenomic dataset which unsurprisingly is not composed exclusively of DNA virus material. It is mostly evidence towards the fact that this model is not doing what it’s supposed to do.

As a sanity check, I ran a simple regex script which looks for the typical protein domains that comprise the catalytic core in RdRps:

| set | n | With A/B/C domains |
| :---- | :---- | :---- |
| positive control — their 4,900 training positives | 4,900 | 49.9% |
| positive control — orphan RdRPs | 2,874 | 23.1% |
| cellular/phage/RT, definitely not RdRP | 39,098 | **0.13%** |
| **their 513,134 claimed RdRPs** | 513,134 | **31.9%** |

Their set definitely contains real RdRps, with the characteristic A/B/C domains. This means the whole procedure wasn't a sham, and at least the ClstrSearch I trust.

Sadly I wasn't able to recover from the published data which of the contigs/supergroups were the ones exclusively identified by LucaProt, and thus unable to do a thorough investigation of them.

## The model

Their model architecture struck me as a little odd, incorporating a pretrained protein language model alongside your own architecture. They report that if you remove ESM-2, the recall drops to 44% on the test set \- pretty bad. So it seems the ESM-2 part is necessary. What about their own architecture part? They do not report this, but I ran the experiment myself and it turns out logistic regression on the frozen mean-pooled ESM-2 embeddings matches LucaProt to within 0.01 percentage points on the zero-homology set \- 97.74% against their 97.73%. The other half of the model, best I can tell, is useless.

You can even run the logistic regression with L1 regularization enabled to find out how many of the 2,560 dimensions that ESM-2 works with are necessary for prediction \- turns out about 14 of them get you within 2.4 points of the full model. Their full model is quite expensive to run inference on, all of the inference I needed for this project ran me 38 euros on Modal.

The hardest thing for me to explain is that in their repo they have a baseline like this set up \- but they never report it in the paper. My best guess at what happened is that they ran an xgboost baseline, but they had set iterations to only 50, which is far too low for convergence and results in poor performance. I ran it with more iterations and it converged to a similar performance of \~97%.

This is incredibly weird. In general, you should start with the smallest / simplest possible model and move up, if for no other reason than it's cheaper to run/train. It also becomes easier to interpret, since there are less total parameters. It should not be the case that you can discard half of your entire tool and have it still work just as well.

## What does all of this mean

To be clear: **I do not claim that these researchers committed any kind of fraud.** All of this is explainable by normal “ML is confusing and hard” reasons. It could be that many similar papers make similar mistakes. But they did make some minor to major mistakes \- did this affect their results in any way?

99.9% of the viral contigs were found by both LucaProt and ClstrSearch, the conventional homology-based pipeline that runs alongside it. The headline finding is true, but it does not depend on LucaProt. It does still feel weird that a paper which is so heavily framed around this new tool doesn't actually depend on the tool being good.

However, the 23 supergroups which were identified by LucaProt only are based on only 444 contigs, far below the expected number of false positives. The paper does run RT-PCR validation, but only on 5 groups, and it is unclear whether any of those were the groups that only LucaProt identified.

The model architecture problem is quite egregious, the fact that the model can be replaced by simple logistic regression on layer 36 of ESM-2 is not great. The former is a lot easier to work with and build on than their split architecture. I am open on being wrong about this, I've hosted the code needed to reproduce this finding on [github](http://linky) \<- link add. Feel free to correct me if I'm wrong.

I think there is interesting work to be done here, mainly in training a better model. Extracting what the 12 ESM-2 dimensions mean by doing ablation studies and seeing which parts of the protein are relevant for it would also be really interesting. I did some preliminary work to check if any of them encode length and they all correlate with length quite strongly, which helps explain the false positive results on long bacterial proteins.

The moral of the story is that if you see a 1.0 AUC you should probably check your setup.

---

[^1]: the causative agent of covid-19

[^2]: Though even this has exceptions since some virus families like *Mitoviridae* don't have capsids.

[^3]: deployment as in where your tool will actually be used

[^4]: Instead of RNA \-\> RNA replication, it does RNA \-\> DNA

[^5]: Not sure what happened here, guessing the number is from some different evaluation