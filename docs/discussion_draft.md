# Discussion (draft for the final paper)

Target: about 500 words, roughly three quarters of a NeurIPS page.

The main claim held up. Training on rearranged copies with the same totals, together with the amount head,
produced an amount vector that records the totals and ignores the arrangement, and a timing vector that
records when things happen. This held on three years the models never saw, with almost no change from
validation. The weaker part of the claim did not hold. Splitting the representation organized the information
but did not add any, since a single joint vector carried the same amount of timing. The split is useful
because each part has a clear meaning, not because it sees more.

The model's failures are systematic rather than random. Slot-by-slot decoding removes short activities in
every model we trained, because a short activity rarely has the highest probability in any single slot. Giving
each activity its expected number of slots fixed most of this, which shows the problem was in how we read the
decoder's output and not in what the model learned. The timing vector also misses fragmentation. It
separates night work, naps and late risers from days with the same totals, but it adds nothing for days broken
into many short blocks, such as a split work day. A whole-day vector built by averaging seems to keep clock
position better than counts of blocks. The rearranged copies may contribute too, since they shift and swap
whole blocks but rarely split or merge them.

The unstable groups are the most informative negative result. Several approaches failed in the same way:
other numbers of groups, longer training, model ensembles and a prior with built-in groups. The prior with
built-in groups was even more reproducible on simulated data with no groups at all. Meanwhile, a continuous
early-to-late axis reproduced well. Together these suggest that daily timing varies along a continuum, so any
hard grouping cuts it in somewhat arbitrary places. Stable types do exist, but they come from what a day is
spent on, not when. A better model is unlikely to change this. Data with several days per person could show
whether people, rather than days, have stable rhythms.

The data impose clear limits. Each ATUS diary is one day of one person, so every result describes days and not
people. Five-minute slots remove about 2% of episodes, mostly short travel and household tasks. Sleep is cut at
4:00 a.m. on both ends, so we do not treat sleep blocks as a measure of sleep quality. Our results are
unweighted. The timing results barely changed with survey weights, but the activity amounts did, so
statements about the population need weights. The 2025 test year has no October data.

The model and its approximations have their own limits. The posterior is a diagonal Gaussian, and we
summarize each day by its mean. The decoder treats each slot separately once the vectors are given, which
favors smooth days. The rearranged copies are synthetic: whole-day shifts wrap activities around the 4:00 a.m.
boundary, and the model may learn some of these artifacts.

The evaluation has limits too. Probe results depend on the probe: the timing vector scores about 0.24 with
nearest-neighbor probes and about 0.62 with tree probes, so only comparisons within one probe type are meaningful.
The hand-built features score well on our timing checks partly because those checks use similar features.
Group reliability rests on a few seeds, and all follow-up analyses after the test years were done on 2022
validation data, so they still need a fresh test year.
