# Method (draft for the final paper, v3)

Each diary covers one day from 4:00 a.m. to 4:00 a.m., which we cut into 288 five-minute slots. Every slot holds
one of 13 activity categories. We also give each slot its clock time as a sine and a cosine of the time of day,
so that slots just before and just after midnight are treated as close together.

The encoder turns each slot into a learned code of 16 numbers for its activity, joined with the two clock
numbers. A one-layer bidirectional GRU with 64 units in each direction reads the 288 slots forwards and
backwards, so each slot's output of 128 numbers reflects what happened before and after it. We average these
outputs into one 128-number summary of the day. Two separate pairs of linear layers turn that summary into two
Gaussian distributions, each described by a mean and a log variance. One gives the amount vector, with 8
numbers, and the other gives the timing vector, with 16 numbers.

> **[Insert equation: the two Gaussian distributions for the amount and timing vectors, with their dimensions]**

During training we draw one sample from each distribution using the reparameterization trick, which adds
scaled random noise to the mean so that gradients can still flow through the sampling step. After training we
use the means directly, so every day always gets the same two vectors.

> **[Insert equation: reparameterization, sample = mean + standard deviation × noise, with noise from a standard normal]**

The decoder rebuilds the day one slot at a time. For each slot it receives the amount vector, the timing vector
and the clock position of that slot, given as the sine and cosine of the first four harmonics of the time of
day. A two-layer network with 128 units per layer turns these into a probability for each of the 13
activities. The same network is used for every slot, so the clock features are the only thing that tells one
slot from another. The decoder never sees the original day, so anything it rebuilds correctly has to come
through the two vectors. A small separate network, the amount head, reads only the amount vector and predicts
what share of the day went to each activity.

To give the model a reason to separate the two vectors, every day in each training batch is paired with a
freshly rearranged copy. With equal probability, the copy either shifts the whole day by 30 minutes to 2 hours
in either direction, wrapping around the window, or swaps two activity blocks inside the day, never moving the
blocks that touch 4:00 a.m. Every copy is checked to have exactly the same number of slots for each activity as
the original, and copies that come out unchanged are rejected. A copy always stays in the same data split as
its original.

The loss has four parts, each logged on its own. The first is the reconstruction error: the average
cross-entropy between the decoder's probabilities and the true activities over all 288 slots. The second is the
KL divergence that pulls each distribution toward a standard normal. We divide it by the number of slots so
that it stays on the same per-slot scale as the reconstruction error. The third is the invariance term: the
squared distance between the amount means of a day and its rearranged copy, which should be close because only
the arrangement changed. The fourth is the amount prediction term: the cross-entropy between the true activity
shares and the shares predicted by the amount head. Without it, the amount vector could satisfy the invariance
term by carrying no information at all. The reconstruction, KL and amount terms are averaged over both the
originals and the copies.

> **[Insert equation: total loss = reconstruction + β(KL amount + KL timing)/T + λ_inv × invariance + λ_amt × amount prediction, with β = 1, λ_inv = 0.3, λ_amt = 1]**

We train with Adam at a learning rate of 0.001 and a batch size of 32 for 40 epochs. The weight on the KL term
rises linearly over the first three epochs, and we keep the checkpoint with the lowest validation loss after
that warmup. We train six models on the 2022 training split, using seeds 42 to 47. All settings were chosen on
the 2022 validation data.

To rebuild a day, the simplest approach is to pick the most likely activity in each slot, but short activities
such as meals rarely win any single slot and disappear. Instead, we first add up each activity's probability
across all 288 slots to get the number of slots the decoder expects for it. We round these to whole numbers
that add up to 288, giving each activity a budget.

> **[Insert equation: expected slots for each activity = sum of its probabilities over all slots, rounded so the budgets sum to 288]**

We then assign activities to slots so that the total log probability is as high as possible while every
activity gets exactly its budget. This is a standard linear assignment problem, which we solve exactly with the
Hungarian method. The budgets come only from the model's own output, never from the true day.

> **[Insert equation: maximize the sum of log probabilities of the assigned activities, subject to each activity using exactly its budget of slots]**

To group days, we join the timing means of three models trained with different seeds into one 48-number vector
for each day and run k-means with six groups and 50 restarts, fitted on 2022 training days only. We build a
second ensemble from three other seeds in the same way, so that we can check whether two independent sets of
models produce the same groups.

The building blocks of this design are standard: the sequence VAE, the reparameterization trick, the KL penalty
toward a standard normal prior, linear assignment and k-means. What we changed is how they are put together. We
use two vectors that both describe the whole day, one for amount and one for timing, instead of a single vector
or the split between fixed and changing content in Li and Mandt (2018). We train on rearranged copies whose
totals are known to match, similar in spirit to the paired training of Locatello et al. (2020), and we tie the
amount vectors of each pair together directly. We add the amount head so that this tie cannot be met by an
empty vector. We replace slot-by-slot decoding with budgeted decoding to fix the oversmoothing, and we combine
several models to make the groups depend less on any single random seed.

We compare the model with k-means on activity totals, k-means on hand-built timing features, a model with a
single joint vector of 24 numbers, and the same two-vector model without the invariance term. The hand-built
features are whether each activity occurs, its time of day on a circle, when it starts and ends, how many
blocks it has, and how many activity changes the day contains. All comparisons use the same data splits, and
the neural models share the same decoder, total vector size and training budget.

---

*Note: these settings are for the v3 models tested on 2025, which trained for 40 epochs. If the paper also
reports the v1 results from 2023, those models trained for 80 epochs.*
