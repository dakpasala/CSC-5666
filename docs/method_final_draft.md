# Method (final model draft)

Edited from the team's current Method text. Sizes and training data are updated to the final model (M_big2 in the
repo), typos are fixed, and two parts are new: the training data paragraph and "Choosing the model size and data".
All numbers in the selection section come from 2022 validation runs (`docs/v3_capacity_results.md`,
`docs/v3_latent_results.md`, `docs/v3_round2_results.md` to `docs/v3_round5_results.md`).

---

Each diary records a day from 4:00 a.m. to 4:00 a.m. the following morning. This window is divided into 288
five-minute slots, and each slot is assigned to one of the 13 activity categories. Each slot also receives the sine
and cosine of its time of day. This circular representation keeps times on either side of midnight close to each
other.

The encoder represents each activity with a 32-dimensional embedding that is learned during training and joins it
with the two clock features. A two-layer bidirectional GRU with 192 units in each direction processes the sequence
from the front and the back. Its output at each slot is a vector of 384 numbers that carries information from both
earlier and later in the day. We average these outputs to form a single 384-dimensional summary of the day. Linear
layers then predict the mean and variance of two Gaussian distributions: an eight-dimensional distribution for
activity amounts and a sixteen-dimensional distribution for timing.

[Insert equation: the two Gaussian distributions for the amount and timing vectors, including their dimensions]

During training, we draw one sample from each distribution using the reparameterization trick. This writes the
sample as the mean plus random noise scaled by the standard deviation, which lets gradients pass through the
sampling step. When extracting representations after training, we use only the means, so each diary always gets the
same pair of vectors.

[Insert equation: sample = mean + standard deviation × noise, where the noise follows a standard normal distribution]

The decoder uses the amount vector, the timing vector and each slot's clock features to predict an activity
distribution for that slot. The clock features given to the decoder are the sine and cosine of the first eight
time-of-day harmonics. A network with three hidden layers of 256 units outputs probabilities for the 13 activities.
The same network processes every slot, and the clock features are what identify the slot's position in the day.
Because the decoder is never given the original activity sequence, everything it needs has to come through the two
latent vectors. A separate, smaller network, the amount head, uses the amount vector to predict what share of the
day was spent on each activity. The full model has about 1.09 million parameters.

For every diary in a training batch, we create a rearranged copy. Half of the copies shift the entire schedule by 30
minutes to two hours in either direction. The rest swap two activity blocks, never moving the blocks that touch the
4:00 a.m. boundary. We confirm that each copy contains the same number of slots for each activity as the original
day. Each copy stays in the same data split as its original diary.

The training loss has four terms. The reconstruction term is the cross-entropy between the predicted and observed
activities, averaged over the 288 slots. The KL term encourages each latent distribution to stay close to a standard
normal distribution. It is divided by the sequence length so that it stays on the same per-slot scale as the
reconstruction term.

The invariance term is the squared distance between the amount means of an original diary and its rearranged copy.
These means should stay close because the rearrangement changes only the timing, while the activity totals stay the
same. The amount prediction term is the cross-entropy between the observed activity shares and the shares predicted
by the amount head. The reconstruction, KL and amount prediction terms are averaged over both the original diaries
and their copies.

[Insert equation: total loss = reconstruction + β(KL amount + KL timing)/T + λ_inv × invariance + λ_amt × amount prediction, with final weights β = 1, λ_inv = 0.3 and λ_amt = 1]

We train on the 2022 training split together with every diary from 2023 and 2024, 22,726 days in total. Model
selection uses only the 2022 validation split. The 2023 and 2024 diaries had served as held-out test years for
earlier versions of the model, and we added them to training only after those tests were complete. Results for the
final model on those years are therefore not held-out results.

We train each model for 40 epochs with Adam, a learning rate of 0.001 and batches of 32 diaries. The KL weight rises
linearly over the first three epochs until it reaches its final value. After this warmup, we keep the checkpoint
with the lowest validation loss. We train six models with random seeds 42 through 47, on CPU.

The simplest way to reconstruct a diary is to choose the most likely activity in each slot. This removes short
activities such as meals, because a short activity can receive real probability in several slots without being the
most likely choice in any of them. We therefore use a budgeted decoding procedure. We first sum each activity's
predicted probabilities across the 288 slots to estimate how many slots it should occupy, and then convert these
estimates into integer budgets that sum to 288.

[Insert equation: each activity's expected slot count is the sum of its probabilities across time, converted to integer budgets that sum to 288]

Each activity is then assigned to slots so that the total log probability is as high as possible while every
activity receives exactly its budget. This is a linear assignment problem, which we solve with the Hungarian method.
The budgets come from the decoder's own predictions; the true activity totals are never used.

[Insert equation: maximize the sum of assigned log probabilities, with one activity per slot and each activity assigned exactly its budget]

To form groups, we concatenate the timing means of three models trained with different seeds, which gives a
48-dimensional representation of each diary. We fit k-means with six clusters and 50 restarts using only the 2022
training diaries. We repeat the procedure with the remaining three models to form a second ensemble. Comparing the
two ensembles shows whether independent sets of training runs produce similar groups.

### Choosing the model size and data

We chose each setting on the 2022 validation split, training three seeds per setting for 40 epochs and changing one
thing at a time. The starting point was the smaller model we used earlier in the project (a 16-dimensional
embedding, a one-layer GRU with 64 units per direction, two decoder layers of 128 and four clock harmonics, about
62,000 parameters). Table [X] reports the validation reconstruction cross-entropy for each change, along with the
two separation checks that the size of the latent vectors affects most.

| Change from the smaller model | Parameters | Val. cross-entropy | Timing information in amount vector | Total information in timing vector |
| --- | ---: | ---: | ---: | ---: |
| Smaller model (starting point) | 62K | 0.540 | 0.014 | 0.288 |
| Embedding 32 | | 0.522 | 0.003 | 0.299 |
| Decoder 3 × 256 | | 0.527 | 0.022 | 0.284 |
| Eight clock harmonics | | 0.540 | 0.012 | 0.333 |
| GRU 128 per direction | | 0.475 | 0.018 | 0.285 |
| Two GRU layers | | 0.494 | 0.014 | 0.263 |
| All of the above (GRU 128 × 2) | 581K | 0.421 | 0.002 | 0.233 |
| GRU 192 × 2 | 1.09M | 0.385 | 0.003 | 0.221 |
| GRU 256 × 2 | 1.80M | 0.384 | 0.006 | 0.204 |
| GRU 128 × 2, trained for 80 epochs | 581K | 0.372 | 0.006 | 0.222 |
| GRU 192 × 2 with a GRU decoder | 1.08M | 0.395 | -0.006 | 0.298 |
| Timing vector 32 (smaller model) | | 0.527 | 0.012 | 0.371 |
| Amount vector 12 (smaller model) | | 0.560 | 0.039 | 0.281 |
| Both larger vectors (smaller model) | | 0.526 | 0.071 | 0.331 |
| Timing vector 32 (GRU 128 × 2) | | 0.383 | 0.031 | 0.356 |
| **GRU 192 × 2, trained on 2022 to 2024 (final)** | **1.09M** | **0.314** | **-0.004** | **0.233** |

*Information columns are probe R² values on 2022 validation; lower is cleaner. Our pass limits were 0.05 for timing
in the amount vector and 0.38 for totals in the timing vector.*

The encoder mattered most. Widening the GRU to 128 units per direction gave the largest single improvement, and a
second GRU layer helped further. Enlarging the decoder or the embedding helped only slightly, and adding clock
harmonics made no measurable difference on its own. We kept the larger embedding, decoder and harmonics because the
combined model, with all changes together, reached a lower error than any single change.

We then increased the GRU size. Moving from 128 to 192 units per direction lowered the validation error from 0.421 to
0.385, but 256 units, with 1.8 million parameters, gave no further gain (0.384). We therefore kept 192 units. Training
the smaller combined model for 80 epochs instead of 40 reached a similar error (0.372), and its validation loss was
nearly flat by the end, so longer training was not the main lever.

We also tested a decoder that reads the day in order. We replaced the per-slot network with a bidirectional GRU
of the same total size, which reads the latent vectors and clock features in sequence but never sees the true
activities. It did not lower the error (0.395 against 0.385), and the decoder relied less on the latent vectors:
replacing a diary's amount or timing vector with another diary's raised the error less than with the per-slot
decoder. We kept the per-slot decoder.

Enlarging the latent vectors did not pay off. A 32-dimensional timing vector improved reconstruction only slightly,
and the timing vector began to carry more information about the activity totals. A 12-dimensional amount vector let
timing information leak into it, and enlarging both vectors pushed that leakage past our limit (0.071 against 0.05).
Because a clean separation is the main goal of the model, we kept eight dimensions for amount and sixteen for timing.

The larger models did not overfit: their training and validation errors stayed close throughout. That suggested the
amount of data, rather than model size, was the remaining limit. Training the same 1.09-million-parameter model on
the 2022 training split plus all of 2023 and 2024 lowered the validation error from 0.385 to 0.314, the largest
improvement in the search, while keeping the separation clean. Part of this gain may come from the larger number of
training steps that the extra data provides. However, training twice as long on 2022 alone helped much less, which
suggests that most of the gain comes from the additional days.

None of the larger models produced more stable groups. Agreement between the groups of models with different seeds
stayed between 0.11 and 0.21 for every setting, compared with 0.17 for the smaller model.

---

Most of the main components of this approach are established methods: sequence VAEs, reparameterization, KL
regularization toward a normal distribution, linear assignment and k-means. Our design combines them to separate
activity amounts from timing. Both latent vectors describe the entire day, unlike a single joint representation or
the split between constant and changing content used by Li and Mandt (2018).

The model is compared against four alternatives: k-means on activity totals, k-means on hand-built timing features,
a sequence model with one 24-dimensional joint vector, and the two-vector model trained without the invariance term.
The hand-built features describe the presence of each activity, its time of day, its start and end times, its
number of blocks and the number of activity changes in the day.
