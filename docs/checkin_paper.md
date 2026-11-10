# CSC 5666 Final Project Prototype Check-In Paper

Braeden Alonge, Kevin Beltran, Nickaan Jahadi, Dakshesh Pasala

## Separating Schedules using a Variational Autoencoder

### Abstract

Daily activity totals hide the shape of a day. Two people can sleep eight hours and work eight hours and still
live very different days, because those hours can fall at different times, in a different order, and in a
different number of blocks. We built a two-part sequence model on American Time Use Survey (ATUS) diaries. An
encoder reads a full day and writes two vectors: an amount vector for how much time goes to each activity and
a timing vector for when the activities happen. A decoder rebuilds the day from both. We train it on pairs of
days that have the same totals but a rearranged schedule, and we penalize the amount vector when it changes
across a pair. We checked whether the model can rebuild held-out days, whether the amount vector stays fixed
when only the arrangement changes, and how the result compares with simpler features. On a held-out year
(2023), the split works: the amount vector carries the activity totals and almost no timing, and the timing
vector carries timing beyond the totals. Rebuilt days are reasonably accurate but too smooth. The timing
vector does not hold more timing information than a single joint vector, and groups built from it are less
stable than groups built from hand-made timing features. Whether the timing vector adds value beyond those
simpler features is still open.

### 1. Question

Two days can have the same minutes of sleep or work and still be organized differently. One person might work
from 9 to 5 and sleep from 11 to 7, while another works an evening shift and sleeps late into the morning.
Daily totals treat these two days as the same. Our question is whether a sequence model can learn to split
"how much" from "when and in what order" if we train it on pairs of days that keep the activity totals fixed
and change only the arrangement.

A negative answer would still be useful. Simple hand-built timing features, such as when each activity starts
and how many blocks it has, may group days just as well as a learned representation, and we want to know if
that is the case. We also want to be clear about scope. Each ATUS diary covers one day for one person, so any
patterns we find describe single days. They are not lifestyle types for people, and they are not health advice.

### 2. Related work

Li and Mandt (2018) built a sequence model that separates the part of a sequence that stays fixed from the
part that changes over time. We borrow the idea of giving a model separate vectors for separate factors, but
not their video model. In our case both vectors describe the whole day, and the split is between amount and
timing instead of between static and changing content. Hsu et al. (2017) separated sequence-level information
from segment-level information, which helped us think about what each of our vectors should hold.

Locatello et al. (2019) showed that a model will not separate factors on its own unless the training gives it
a reason to, which is why we add guidance. Locatello et
al. (2020) learned from pairs of examples that share a factor. Our pairs work the same way, except that the
shared factor is known: two days in a pair have the same activity totals. Because we know what should stay the
same, we penalize changes in the amount vector directly.

### 3. Data

We use the American Time Use Survey (ATUS) from the U.S. Bureau of Labor Statistics. Each record is one
person's diary of a single day: a list of activity episodes with durations. The diary window runs from 4:00
a.m. to 4:00 a.m. the next day. We downloaded the 2022 and 2023 activity and respondent files from the BLS year
pages and left the raw files unchanged. Survey weights, the day of the week, and demographics are separated
into another table so that they are not inputs to the model. In 2022, the activity file has 146,393 episodes
and 8,136 diaries, one diary per respondent. We tested and ensured that episodes are in order, each duration is
at least one minute, there are no gaps or overlaps in the days, and there are no duplicate rows. The same
checks pass for all 8,548 diaries in 2023.

We map the official activity codes to 13 categories: sleep, personal care, eating and drinking, household work,
caregiving, paid work, education, shopping and services, leisure and socializing, exercise, travel, other, and
unknown. Refused or unclear codes go to unknown, which is 0.75% of 2022 minutes. Each day is then cut into 288
bins of five minutes. A bin receives the category that fills most of that bin, and if two categories tie, the
one that appears first in the bin wins. About 93% of 2022 diaries start in sleep, and about 96% end in sleep.
Sleep that crosses 4:00 a.m. is cut by the window on both sides.

The 2022 diaries are split 80/20 by respondent with a fixed seed: 6,509 for training and 1,627 for validation.
A rearranged copy of a diary stays in that diary's split. The 2023 diaries are labeled test. They are prepared
and checked, and they are scored only after the modeling choices are fixed.

### 4. Method

The encoder reads the whole day. Each five-minute bin becomes a short learned code for its activity, plus two
numbers for the time of day, and a bidirectional GRU reads the 288 bins forwards and backwards. We average its
outputs into one summary of the day and pass that summary to two separate heads. One head produces the amount
vector (8 numbers) and the other produces the timing vector (16 numbers). Because the model is a variational
autoencoder, each head outputs a mean and a variance, and during training we sample a vector from that range
before decoding. When we use the trained model, we take the mean.

The decoder rebuilds the day one bin at a time. For every bin it receives both vectors and the clock position
of that bin, and it outputs a probability for each of the 13 activities. It never sees the original day, so
anything it rebuilds correctly has to come through the two vectors.

To give the model a reason to split the vectors, every training day is paired with a rearranged copy that has
the same totals. The copy either shifts the whole day by 30 minutes to 2 hours, or swaps two activity blocks
inside the day. The loss has four parts, and we log each one on its own. The first is the error in rebuilding
the day. The second keeps each vector close to a simple prior, which is the usual variational term. The third
is the distance between the amount vectors of a day and its rearranged copy, which should be close because
only the arrangement changed. The fourth asks a small head to predict the activity totals from the amount
vector. Without this last term, the amount vector could stay fixed simply by carrying no information at all.

We compare the model against four alternatives: activity totals alone, hand-built timing features, a model
with one joint vector instead of two, and the same two-vector model without the term that keeps the amount
vector fixed across a pair. The settings we use (a weight of 0.3 on the pair term and 80 training epochs) were
chosen on the 2022 validation split.

### 5. What the prototype already does

Our current prototype uses the command line as the pipeline. It first downloads the ATUS files, builds the
episode table and the 288 slices of each schedule, draws example days, fits the simpler baselines, trains the
two-vector model, reloads a saved checkpoint, and writes both vectors plus the evaluation numbers. Three
baselines run on the full 2022 split. The first baseline uses only how many minutes went to each activity.
The second baseline only uses when the activities happen and how the day is broken into blocks: whether an
activity occurs, its time of day on a circle, when it starts and ends, how many blocks it has, and how many
activity changes the day contains. The third baseline uses both. If we rearrange a day without changing the
totals, the timing features change and the minute totals stay the same. The evaluation can therefore see a
change in organization when the totals do not change.

The model side runs end to end on real diaries. It trains on the 2022 training split, reloads a saved
checkpoint, and produces the same vectors and scores after reloading. It rebuilds held-out days, and we plot
several original and rebuilt timelines next to each other. We use the pairs to test whether the amount vector
moves when only the arrangement changes. We also use simple prediction tests, called probes, to check what
each vector carries: whether the activity totals can be read from the amount vector, and whether timing
information can be read from the timing vector beyond what the totals already tell us. Finally, we group days
by their timing vectors and compare how stable those groups are with groups from the simpler baselines.

Each model is trained three times with different random seeds, and we report the average and the spread. Every
model goes through the same checks. We rearrange held-out days by shifting the whole day or swapping two
blocks, and we measure how far each vector moves. We also take two days with similar totals, give the decoder
one day's amount vector and the other day's timing vector, and see which day the rebuilt timing follows. The
two-vector model is compared with an untrained network of the same shape, with a single joint vector, and with
the same model trained without the pair term. Before we scored the 2023 diaries, we wrote down the criteria
for success, had them approved, and stored checksums of the code and trained models, so the test could not be
adjusted after seeing the results. Automated tests check the data preparation, the rearrangements, the model,
and checkpoint reloading.

### 6. Early evidence

The pipeline runs end to end on real 2022 diaries, and we fixed the modeling choices on validation data before
scoring the 2023 diaries once. The numbers below are from 2023, averaged over three trained models. The model
met 8 of the 11 criteria we set in advance, the same ones it met on the validation data.

The split between the two vectors works. When a day is shifted or has blocks swapped, the amount vector barely
moves (a change of 0.02 to 0.03 relative to the typical difference between two days), while the timing vector
moves clearly. Without the pair term, the amount vector moves about twice as much. The amount vector predicts
the activity totals well (R² of 0.85, against 0.71 for an untrained network of the same shape) and carries
almost no timing information beyond the totals. Predicting the totals from the amount vector is what keeps it
from going blank. The timing vector carries timing information beyond the totals (a gain of 0.24 in R² over
the totals alone). For two days with similar totals, if we combine one day's amount vector with the other
day's timing vector, the rebuilt day's timing follows the day that supplied the timing vector 79% of the time.

Rebuilding the day is decent. The model labels about 84% of the five-minute bins correctly (macro-F1 0.67), but
it oversmooths: a rebuilt day has about 7 activity changes, against about 15 in a real day, and short
activities such as meals often disappear. Most of this seems to come from how we read the decoder's output,
since we pick the most likely activity in each bin and short activities rarely win a single bin. We are testing
a different way of reading the output that gives each activity the number of minutes the decoder expects.

Three criteria were not met. The timing vector does not clearly beat a single joint vector, which holds about
the same amount of timing information (0.23 against 0.24). The learned groups are readable, but they are much
less stable than groups from the hand-built timing features. Groups from different training runs agree at an
adjusted Rand index of about 0.26, against about 0.76 for the hand-built features, and their split of days with
similar totals tracks timing differences only weakly. One explanation we are looking into is that timing may
vary along a continuum instead of falling into separate types, which would make any hard grouping unstable.

### 7. Still to do before the final paper

We froze the main choices on validation data and scored the 2023 test year once, so the rest of the work builds
on that result. We plan to fix the oversmoothing by changing how we read the decoder's output, and to test
whether combining several trained models makes the groups steadier. We also want to check whether days are
better described by a few continuous timing scales than by hard groups, and whether simple patterns such as
night work or naps can be read from the timing vector. If time allows, we will compare one or two alternative
designs, for example a model that builds the split into its structure instead of learning it from pairs. Any of
these results come from the 2022 validation data, since 2023 has already been used, and we will label them
that way.

The final paper needs one results table, one model diagram, and a few example days that show original and
rebuilt timelines. It should say plainly what failed, what the simpler baselines already do as well as the
model, and the limits of the study. Each diary is one day for one person, so the results describe days and not
people. The rearranged days are synthetic and are only used for training and testing. Our counts are
unweighted, so they are not national rates, and any statement about how common something is would need the
ATUS survey weights.

### Oral check-in (12 minutes)

We will run through the prototype in one pass, from downloading the diaries to rebuilding a held-out day. Then
we will explain why the model has two vectors and what the pair term does. We will show one result that worked
(the amount vector stays fixed when only the arrangement changes) and one that did not (the timing vector does
not beat a single joint vector, and its groups are unstable). We will finish with what is left before December
18.

### References

Hsu, W.-N., Zhang, Y., & Glass, J. (2017). Unsupervised learning of disentangled and interpretable
representations from sequential data. Advances in Neural Information Processing Systems 30, 1876-1887.

Li, Y., & Mandt, S. (2018). Disentangled sequential autoencoder. Proceedings of the 35th International
Conference on Machine Learning, PMLR 80, 5670-5679.

Locatello, F., et al. (2019). Challenging common assumptions in the unsupervised learning of disentangled
representations. Proceedings of the 36th International Conference on Machine Learning, PMLR 97, 4114-4124.

Locatello, F., et al. (2020). Weakly-supervised disentanglement without compromises. Proceedings of the 37th
International Conference on Machine Learning, PMLR 119, 6348-6359.
