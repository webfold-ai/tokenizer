# H-Net, and how this tokenizer compares

An intake of Cartesia's **H-Net** (Dynamic Chunking for End-to-End Hierarchical Sequence Modeling,
`arxiv.org/abs/2507.07955`, and the Cartesia blog series "From Tokens to Bytes I/II"), against this
tokenizer. The question that decides the comparison is not "which is better at perplexity" - it is
**whether the accumulated statistics survive a change of representation**.

## What H-Net is

A lightweight encoder, a **learned dynamic chunking** of the raw byte stream, a main network that
runs over the shorter chunk sequence, and a decoder that returns to the original resolution.
Chunk boundaries depend on content and context, and they are learned **jointly with the prediction
task**. Stages nest. The premise it attacks is exactly ours: a fixed BPE vocabulary is not
fundamental, and the useful units can be learned from raw bytes.

## The claim, with its scope kept

| claim | scope |
|---|---|
| ~10x less compute and data | a **two-stage** H-Net matching a *selected* open-source 3B model on its benchmark set - one comparison, not a universal multiplier |
| ~1.28x compute efficiency | controlled **one-stage** comparison, validation loss, against BPE |
| ~2.06x | controlled one-stage, OLMES downstream aggregate, against BPE |
| other modalities | DNA in the paper; biology and vision discussed at a different maturity in the newer work - not "every modality is text bytes" |

The 10x figure is the headline of a blog post, and at least one commentary notes that the
benchmarks behind it are not published. Quote it with the comparison it came from, or not at all.

## This tokenizer

Two tables and a count (`README.md`):

| face | holds | operation |
|---|---|---|
| Set | the vocabulary | `mark` |
| Map | token to id | `put` |
| Count | the n-gram counter: context to next-token counts | `add` |

Encoding folds the byte stream into **contexts the counter has seen**; decoding folds the ids back.
The count table *is* the model: there is no training phase beyond counting the corpus, and the
per-token cost is a fixed number of probes.

## Side by side

| | this tokenizer | H-Net |
|---|---|---|
| input | raw bytes | raw bytes |
| units | fixed-order byte contexts | learned, content-dependent, variable-length chunks |
| how they are learned | **counting** the corpus - no gradient | gradient descent, jointly with the LM task |
| hierarchy | one level | nested stages |
| do the units move during training? | **no** - the counts grow, the context order is fixed | **yes** - the chunker trains while the model trains |
| cost per token | fixed probes | the encoder's forward pass |

## The algebraic consequence - the part that matters for a fold

Let one level represent features by `z`, with accumulated state `G = sum z z^T` and
`B = sum z y^T`. If the representation changes by a **known linear map** `z' = T z`, then

```
G' = T G T^T        B' = T B
```

so the statistics transport **exactly**, with no replay of the data and no re-fold.

That identity is the whole difference between the two designs:

- **This tokenizer's coordinates are fixed once the corpus is counted.** The contexts are a
  coordinate system, and the state lives in it. The map from bytes to contexts is a *lookup*, not a
  linear map - so a **retrained** tokenizer does not transport the state either; it voids it. The
  discipline is therefore: **count once, freeze the codes, fold in those coordinates.** The codes
  are the coordinate system of the sufficient statistic, exactly as the calibration is the coordinate
  system of a Gram prior.
- **H-Net's chunker is content-dependent and moves while the model trains.** There is no global
  linear `T` from the old chunks to the new ones, so `G' = T G T^T` does not apply: the accumulated
  statistics cannot be transported through a chunker change, and the state has to be re-folded.

## What would be needed to have both

Learnable units **and** a transportable state. The condition is exactly the linear one above, so
the designs have to meet at one of three points:

1. the chunker is trained first and then **frozen** - then the state lives in a fixed coordinate
   system and merges exactly;
2. the chunker changes only by a **known linear map** of the feature space (`T` known, invertible
   or not) - then the state transports exactly, `G' = T G T^T`, `B' = T B`;
3. the chunker moves and the state is **re-folded** - exact, but it costs the data again, which is
   the thing the fold exists to avoid.

Note the asymmetry that the algebra makes plain: **a rank-reducing `T` still transports exactly**,
but it does not preserve everything the old representation carried. "Transportable" is not
"lossless", and the two should not be used as synonyms.

## What this tokenizer does not claim

It is not a drop-in for a pretrained model's tokenizer: the vocabulary is whatever the corpus
produced, and the ids line up with nobody else's. It is also not a hierarchical model - it is the
*representation* layer that a hierarchy could sit on, with the property that its coordinates do not
move once counted.
