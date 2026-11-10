# v1 cheat sheet

## How the model works

1. A day is 288 five-minute slots (24 hours × 12). Each slot holds one of 13 activities.
2. Each slot becomes numbers: a learned code for its activity plus its clock time.
3. A bidirectional GRU reads the day forwards and backwards and gives each slot a description that includes
   what came before and after it.
4. The 288 descriptions are averaged into one summary of the day.
5. Two output layers turn the summary into two distributions (a mean and a spread for each number): an amount
   vector of 8 numbers and a timing vector of 16 numbers.
6. During training, one sample is drawn from each distribution (mean + spread × random noise). After training,
   we use the means.
7. The decoder gets those 24 numbers plus each slot's clock time and outputs the probability of each of the 13
   activities in every slot.
8. A small amount head looks only at the amount vector and predicts each activity's share of the day.

## How we teach the split

- Every training day is paired with a rearranged copy: the whole day shifted 30 min to 2 h, or two activity
  blocks swapped. The totals (minutes per activity) stay exactly the same.
- Pair rule: the amount vectors of a day and its copy should match, since only the timing changed.
- Amount head: the amount vector must still predict the real totals, so it cannot cheat the pair rule by
  being the same for every day.
- The KL penalty pulls each distribution toward a standard bell curve (mean 0, spread 1). It keeps the space
  smooth, so similar days get similar codes, and stops the model from memorizing.
- Settings: pair rule weight 0.3, amount head weight 1, 80 epochs, three seeds, choices fixed on 2022
  validation data.

## Tests, in plain words

| Test | What it asks |
| --- | --- |
| Rearrangement test | Shift or swap a day without changing totals. Does the amount vector stay put while the timing vector moves? |
| Swap test | Give the decoder day A's amount vector and day B's timing vector (similar totals). Does the rebuilt timing follow B? |
| Probes | Can simple models read the totals from the amount vector, and timing beyond the totals from the timing vector? |
| Untrained network | Same architecture with random weights. It shows what the architecture gives for free, so a trained result must beat it. |
| Joint vector | One combined 24-number vector instead of two. Does the split hold more timing information? |
| Group stability | Retrain the model and regroup the same days. Do they land in the same groups? |

## v1 results on 2023 (held out, criteria approved before scoring; 8 of 11 met)

| Result | Value | Comparison | Meaning |
| --- | --- | --- | --- |
| Slots rebuilt correctly | 84% (macro-F1 0.67) | | Decent rebuild |
| Activity changes per rebuilt day | about 7 | real days: about 15 | Too smooth; short activities like meals disappear |
| Totals read from amount vector (R²) | 0.85 | untrained: 0.71 | Amount vector holds the amounts |
| Timing hidden in amount vector | 0.008 | limit 0.05 | Amount vector holds almost no timing |
| Timing beyond totals in timing vector | 0.24 | untrained: 0.002 | Timing vector holds the "when" |
| Amount vector movement when rearranged | 0.019 to 0.028 | no pair rule: 0.043 to 0.060 | The pair rule works |
| Swap test: timing follows B | 79% | no guidance: 67% | The two vectors control different things |
| Timing info: split vs one vector | 0.24 | joint vector: 0.23 | Failed: the split organizes but adds no information |
| Group agreement after retraining (ARI) | 0.26 | hand-built features: 0.76 | Failed: groups are not reproducible |
| Groups split same-total days by timing (AUC) | 0.58 | hand-built features: 0.72 | Failed: groups follow timing only weakly |

## The story in three lines

1. The split works and held up on years the model never saw.
2. Hard groups of timing are unstable, because timing behaves like a continuum.
3. The timing vector still captures real patterns, such as night work and naps.

Three numbers worth remembering: 84% of slots rebuilt, amount R² about 0.85, and group agreement 0.26 against
0.76 for hand-built features.

## Built on v1 afterwards

- Better decoding: give each activity the minutes the decoder expects. About 13 activity changes per day instead
  of 6, and the error in activity minutes drops from 26 to 10 (2025 test).
- Ensembles: combining three trained models roughly doubles group agreement (about 0.18 to about 0.37).
- Development results on 2022 validation: five stable day types by what the day is spent on, a reproducible
  earlier-to-later rhythm axis, and night work detected from the timing vector at AUC 0.98.
