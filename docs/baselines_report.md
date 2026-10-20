# Checkpoint 2 baselines report

Run: `outputs/checkpoint2/20261019-184825`. Data: ATUS 2022, train n=6509, validation n=1627. Unweighted.
All scaling, imputation, PCA and k-means fitted on train only; k and diagnostics on validation.
2023 was not used. These are baseline results, not final findings.

## k selection (validation)

Rule (fixed before inspecting profiles): best val silhouette with bootstrap ARI >= 0.8 and smallest val cluster >= 3%; else most stable k. Chosen: {'totals': 2, 'timing_only': 3, 'totals_plus_timing': 2}. A fixed comparison k is also reported for every baseline (`<baseline>@k<k>`); it was added after seeing this table and is for like-for-like comparison, not a selected k.

| baseline           |   k |   train_inertia |   val_silhouette |   val_davies_bouldin |   bootstrap_ari_mean |   bootstrap_ari_sd |   smallest_val_cluster_share |
|:-------------------|----:|----------------:|-----------------:|---------------------:|---------------------:|-------------------:|-----------------------------:|
| totals             |   2 |       75608.675 |            0.154 |                2.254 |                0.966 |              0.025 |                        0.317 |
| totals             |   3 |       70043.541 |            0.141 |                2.490 |                0.736 |              0.198 |                        0.293 |
| totals             |   4 |       64847.755 |            0.164 |                2.214 |                0.638 |              0.141 |                        0.022 |
| totals             |   5 |       59971.377 |            0.170 |                1.894 |                0.676 |              0.115 |                        0.022 |
| totals             |   6 |       55742.562 |            0.190 |                1.776 |                0.774 |              0.095 |                        0.022 |
| totals             |   7 |       52026.001 |            0.203 |                1.688 |                0.831 |              0.058 |                        0.022 |
| totals             |   8 |       47992.933 |            0.220 |                1.500 |                0.835 |              0.070 |                        0.022 |
| totals             |   9 |       44470.714 |            0.228 |                1.419 |                0.906 |              0.026 |                        0.022 |
| totals             |  10 |       41384.096 |            0.235 |                1.322 |                0.915 |              0.024 |                        0.004 |
| timing_only        |   2 |      453713.639 |            0.077 |                3.957 |                0.513 |              0.403 |                        0.315 |
| timing_only        |   3 |      435374.997 |            0.059 |                3.687 |                0.806 |              0.210 |                        0.274 |
| timing_only        |   4 |      422033.441 |            0.062 |                3.121 |                0.746 |              0.109 |                        0.022 |
| timing_only        |   5 |      409817.497 |            0.048 |                3.055 |                0.632 |              0.139 |                        0.022 |
| timing_only        |   6 |      399706.403 |            0.063 |                2.864 |                0.705 |              0.099 |                        0.022 |
| timing_only        |   7 |      389525.014 |            0.054 |                2.784 |                0.679 |              0.093 |                        0.022 |
| timing_only        |   8 |      381644.176 |            0.055 |                2.691 |                0.692 |              0.116 |                        0.012 |
| timing_only        |   9 |      374778.154 |            0.056 |                2.641 |                0.718 |              0.105 |                        0.022 |
| timing_only        |  10 |      362607.432 |            0.067 |                2.415 |                0.652 |              0.083 |                        0.011 |
| totals_plus_timing |   2 |      530643.955 |            0.081 |                3.634 |                0.929 |              0.035 |                        0.301 |
| totals_plus_timing |   3 |      510894.279 |            0.062 |                3.756 |                0.822 |              0.175 |                        0.277 |
| totals_plus_timing |   4 |      493304.275 |            0.065 |                3.184 |                0.779 |              0.152 |                        0.026 |
| totals_plus_timing |   5 |      479869.529 |            0.068 |                2.985 |                0.683 |              0.104 |                        0.026 |
| totals_plus_timing |   6 |      468093.980 |            0.057 |                3.009 |                0.673 |              0.111 |                        0.025 |
| totals_plus_timing |   7 |      456731.544 |            0.055 |                2.775 |                0.713 |              0.104 |                        0.013 |
| totals_plus_timing |   8 |      449640.863 |            0.060 |                2.649 |                0.646 |              0.103 |                        0.024 |
| totals_plus_timing |   9 |      438782.879 |            0.055 |                2.566 |                0.707 |              0.079 |                        0.012 |
| totals_plus_timing |  10 |      427820.447 |            0.060 |                2.546 |                0.711 |              0.099 |                        0.013 |

## Can the evaluation see timing when totals match?

### Total-preserving transforms of validation diaries

`rel dist` = median distance between a diary and its transformed copy, in that baseline's scaled feature space, divided by the median distance between two random validation diaries. `reassigned` = share of diaries whose cluster changes. Shifts wrap across 04:00 (artificial boundary effect); swaps exchange two non-boundary runs.

| transform   |   timing_only reassigned |   timing_only@k6 reassigned |   totals reassigned |   totals@k6 reassigned |   totals_plus_timing reassigned |   totals_plus_timing@k6 reassigned |   timing_only rel dist |   timing_only@k6 rel dist |   totals rel dist |   totals@k6 rel dist |   totals_plus_timing rel dist |   totals_plus_timing@k6 rel dist |
|:------------|-------------------------:|----------------------------:|--------------------:|-----------------------:|--------------------------------:|-----------------------------------:|-----------------------:|--------------------------:|------------------:|---------------------:|------------------------------:|---------------------------------:|
| shift_+12   |                    0.088 |                       0.107 |                   0 |                      0 |                           0.014 |                              0.108 |                  0.184 |                     0.186 |                 0 |                    0 |                         0.169 |                            0.17  |
| shift_+24   |                    0.186 |                       0.217 |                   0 |                      0 |                           0.026 |                              0.243 |                  0.365 |                     0.365 |                 0 |                    0 |                         0.334 |                            0.335 |
| shift_+6    |                    0.05  |                       0.06  |                   0 |                      0 |                           0.01  |                              0.05  |                  0.091 |                     0.092 |                 0 |                    0 |                         0.085 |                            0.084 |
| shift_-12   |                    0.067 |                       0.103 |                   0 |                      0 |                           0.01  |                              0.087 |                  0.187 |                     0.187 |                 0 |                    0 |                         0.173 |                            0.173 |
| shift_-24   |                    0.125 |                       0.171 |                   0 |                      0 |                           0.027 |                              0.14  |                  0.403 |                     0.399 |                 0 |                    0 |                         0.368 |                            0.365 |
| shift_-6    |                    0.036 |                       0.055 |                   0 |                      0 |                           0.005 |                              0.056 |                  0.093 |                     0.092 |                 0 |                    0 |                         0.086 |                            0.085 |
| swap        |                    0.103 |                       0.121 |                   0 |                      0 |                           0.023 |                              0.103 |                  0.252 |                     0.252 |                 0 |                    0 |                         0.233 |                            0.233 |

### Real diaries with near-identical totals

Each validation diary is paired with its nearest neighbour by category counts; pairs are kept if totals differ by at most the threshold (minutes reassigned between categories). `timing dist` is the median timing-feature distance relative to random pairs (1.0 = as different as random diaries). Same-cluster rates: matched pairs / random pairs.

| Max totals diff (min) | Pairs | timing dist | totals | totals@k6 | timing_only | timing_only@k6 | totals_plus_timing | totals_plus_timing@k6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | 7 | 0.60 | 1.00 / 0.57 | 1.00 / 0.25 | 1.00 / 0.35 | 1.00 / 0.23 | 1.00 / 0.58 | 1.00 / 0.22 |
| 30 | 32 | 0.50 | 1.00 / 0.57 | 1.00 / 0.25 | 1.00 / 0.35 | 0.97 / 0.23 | 1.00 / 0.58 | 1.00 / 0.22 |
| 60 | 179 | 0.59 | 0.99 / 0.57 | 0.98 / 0.25 | 0.83 / 0.35 | 0.82 / 0.23 | 0.99 / 0.58 | 0.85 / 0.22 |
| 120 | 740 | 0.73 | 0.99 / 0.57 | 0.95 / 0.25 | 0.76 / 0.35 | 0.66 / 0.23 | 0.98 / 0.58 | 0.71 / 0.22 |

### How much of each timing feature do totals explain? (ridge, validation R^2)

| group       |   count |   mean |   50% |    min |   max |
|:------------|--------:|-------:|------:|-------:|------:|
| bouts       |      12 |  0.429 | 0.417 |  0.137 | 0.814 |
| center      |      24 |  0.059 | 0.032 |  0.003 | 0.276 |
| first       |      12 |  0.106 | 0.1   |  0.017 | 0.302 |
| last        |      12 |  0.073 | 0.063 |  0.006 | 0.174 |
| present     |      12 |  0.327 | 0.334 | -0.018 | 0.761 |
| transitions |       1 |  0.209 | 0.209 |  0.209 | 0.209 |

## Cluster profiles (validation diaries)

Minutes are mean minutes/day from binned sequences. `center_*` is the circular mean clock time of the activity among diaries that contain it. Descriptions, not value judgements.

### totals (k=2)

|   cluster |    n |   share |   min_sleep |   min_work |   min_leisure_social |   min_household |   min_eating_drinking |   min_travel |   min_education |   min_shopping_services |   min_caregiving |   min_other |   pct_with_work | center_sleep   | center_work   | center_eating_drinking   |
|----------:|-----:|--------:|------------:|-----------:|---------------------:|----------------:|----------------------:|-------------:|----------------:|------------------------:|-----------------:|------------:|----------------:|:---------------|:--------------|:-------------------------|
|         0 | 1111 |   0.683 |     567.200 |     10.600 |              393.100 |         170.100 |                75.900 |       51.600 |           4.600 |                  30.500 |           41.200 |      24.500 |           9.000 | 02:59          | 12:48         | 14:29                    |
|         1 |  516 |   0.317 |     488.400 |    459.500 |              151.800 |          49.000 |                63.400 |       81.000 |          21.300 |                  12.900 |           30.100 |       7.500 |          91.900 | 02:19          | 12:30         | 15:30                    |

### totals@k6 (k=6)

|   cluster |   n |   share |   min_sleep |   min_work |   min_leisure_social |   min_household |   min_eating_drinking |   min_travel |   min_education |   min_shopping_services |   min_caregiving |   min_other |   pct_with_work | center_sleep   | center_work   | center_eating_drinking   |
|----------:|----:|--------:|------------:|-----------:|---------------------:|----------------:|----------------------:|-------------:|----------------:|------------------------:|-----------------:|------------:|----------------:|:---------------|:--------------|:-------------------------|
|         0 | 120 |   0.074 |     530.500 |     51.500 |              169.600 |         147.800 |                74.900 |       55.700 |           3.400 |                  13.400 |          317.200 |      12.400 |          20.800 | 02:34          | 12:51         | 15:04                    |
|         1 | 437 |   0.269 |     486.400 |    515.500 |              151.700 |          48.900 |                62.100 |       64.700 |           1.600 |                  11.900 |           21.900 |       4.500 |          99.500 | 02:15          | 12:28         | 15:28                    |
|         2 | 486 |   0.299 |     585.900 |     12.600 |              578.300 |          92.700 |                62.100 |       29.600 |           1.900 |                  14.000 |            8.700 |       7.900 |           7.600 | 03:13          | 12:50         | 14:01                    |
|         3 |  36 |   0.022 |     567.900 |     17.100 |              187.600 |          46.400 |                72.500 |       74.900 |         369.400 |                  11.500 |           11.700 |       2.100 |           5.600 | 03:20          | 19:46         | 15:43                    |
|         4 | 475 |   0.292 |     550.500 |     16.500 |              257.000 |         255.400 |                91.400 |       87.500 |           1.300 |                  52.700 |           17.500 |       8.900 |          13.900 | 02:49          | 12:54         | 14:47                    |
|         5 |  73 |   0.045 |     538.700 |     39.000 |              253.100 |          98.000 |                64.200 |       74.700 |           1.900 |                  20.500 |            8.800 |     266.700 |          12.300 | 02:50          | 12:31         | 14:19                    |

### timing_only (k=3)

|   cluster |   n |   share |   min_sleep |   min_work |   min_leisure_social |   min_household |   min_eating_drinking |   min_travel |   min_education |   min_shopping_services |   min_caregiving |   min_other |   pct_with_work | center_sleep   | center_work   | center_eating_drinking   |
|----------:|----:|--------:|------------:|-----------:|---------------------:|----------------:|----------------------:|-------------:|----------------:|------------------------:|-----------------:|------------:|----------------:|:---------------|:--------------|:-------------------------|
|         0 | 714 |   0.439 |     582.100 |     37.100 |              401.000 |         161.700 |                74.200 |       40.100 |          14.400 |                  11.900 |           34.600 |      17.200 |          12.200 | 03:17          | 16:56         | 14:46                    |
|         1 | 467 |   0.287 |     539.100 |     29.400 |              341.000 |         154.700 |                75.900 |       80.100 |           5.300 |                  53.600 |           47.200 |      33.200 |          13.300 | 02:36          | 14:30         | 13:59                    |
|         2 | 446 |   0.274 |     481.900 |    467.900 |              155.700 |          59.700 |                64.000 |       74.100 |           7.400 |                  15.700 |           32.400 |       7.400 |          95.300 | 02:10          | 12:11         | 15:45                    |

### timing_only@k6 (k=6)

|   cluster |   n |   share |   min_sleep |   min_work |   min_leisure_social |   min_household |   min_eating_drinking |   min_travel |   min_education |   min_shopping_services |   min_caregiving |   min_other |   pct_with_work | center_sleep   | center_work   | center_eating_drinking   |
|----------:|----:|--------:|------------:|-----------:|---------------------:|----------------:|----------------------:|-------------:|----------------:|------------------------:|-----------------:|------------:|----------------:|:---------------|:--------------|:-------------------------|
|         0 | 205 |   0.126 |     533.000 |    164.300 |              247.400 |         137.500 |                75.500 |       86.600 |           5.000 |                  58.600 |           37.100 |      16.400 |          40.500 | 03:16          | 12:31         | 15:45                    |
|         1 | 566 |   0.348 |     577.800 |     42.200 |              430.200 |         166.200 |                75.900 |       32.500 |           3.700 |                   3.400 |           37.700 |       6.200 |          12.900 | 03:12          | 16:17         | 14:31                    |
|         2 | 151 |   0.093 |     557.600 |     31.700 |              329.600 |         126.500 |                67.500 |       63.500 |           2.300 |                  22.400 |           36.500 |     126.100 |          12.600 | 02:32          | 14:22         | 14:02                    |
|         3 | 364 |   0.224 |     487.600 |    483.300 |              158.300 |          59.900 |                61.500 |       67.000 |           1.500 |                   8.400 |           35.900 |       4.400 |          96.700 | 02:02          | 12:07         | 15:37                    |
|         4 | 305 |   0.187 |     540.800 |     31.400 |              347.800 |         163.000 |                76.500 |       83.000 |           1.000 |                  64.400 |           43.500 |       9.100 |          14.400 | 02:40          | 14:42         | 14:01                    |
|         5 |  36 |   0.022 |     537.200 |     27.500 |              203.800 |          40.000 |                73.300 |      100.700 |         325.400 |                  14.000 |           13.900 |      22.200 |           8.300 | 03:06          | 22:10         | 15:35                    |

### totals_plus_timing (k=2)

|   cluster |    n |   share |   min_sleep |   min_work |   min_leisure_social |   min_household |   min_eating_drinking |   min_travel |   min_education |   min_shopping_services |   min_caregiving |   min_other |   pct_with_work | center_sleep   | center_work   | center_eating_drinking   |
|----------:|-----:|--------:|------------:|-----------:|---------------------:|----------------:|----------------------:|-------------:|----------------:|------------------------:|-----------------:|------------:|----------------:|:---------------|:--------------|:-------------------------|
|         0 |  489 |   0.301 |     484.500 |    470.200 |              153.000 |          54.200 |                64.100 |       73.100 |          11.100 |                  13.700 |           33.900 |       7.200 |          95.100 | 02:19          | 12:20         | 15:50                    |
|         1 | 1138 |   0.699 |     567.100 |     16.600 |              386.800 |         165.000 |                75.300 |       55.700 |           9.400 |                  29.700 |           39.300 |      24.200 |           9.600 | 02:59          | 17:34         | 14:22                    |

### totals_plus_timing@k6 (k=6)

|   cluster |   n |   share |   min_sleep |   min_work |   min_leisure_social |   min_household |   min_eating_drinking |   min_travel |   min_education |   min_shopping_services |   min_caregiving |   min_other |   pct_with_work | center_sleep   | center_work   | center_eating_drinking   |
|----------:|----:|--------:|------------:|-----------:|---------------------:|----------------:|----------------------:|-------------:|----------------:|------------------------:|-----------------:|------------:|----------------:|:---------------|:--------------|:-------------------------|
|         0 | 493 |   0.303 |     583.200 |     13.700 |              469.700 |         174.000 |                74.400 |       23.000 |           2.600 |                   1.700 |           31.400 |       2.600 |           6.500 | 03:05          | 13:24         | 14:10                    |
|         1 | 247 |   0.152 |     541.000 |    117.700 |              268.500 |         150.200 |                78.800 |       85.100 |           2.200 |                  43.700 |           53.900 |       9.700 |          38.900 | 03:39          | 15:06         | 16:04                    |
|         2 | 169 |   0.104 |     554.300 |     31.200 |              321.500 |         122.000 |                70.600 |       66.500 |           2.000 |                  22.000 |           34.600 |     141.400 |          11.200 | 02:35          | 14:11         | 14:10                    |
|         3 | 285 |   0.175 |     541.900 |     23.900 |              338.200 |         170.100 |                74.700 |       89.000 |           0.600 |                  72.800 |           47.900 |       3.500 |          12.300 | 02:38          | 14:42         | 14:03                    |
|         4 |  40 |   0.025 |     546.900 |     24.800 |              195.800 |          43.800 |                74.100 |       93.200 |         329.100 |                  14.100 |           17.000 |      20.000 |           7.500 | 03:12          | 22:10         | 15:39                    |
|         5 | 393 |   0.242 |     486.300 |    508.700 |              149.100 |          52.300 |                62.800 |       67.100 |           1.400 |                   9.800 |           31.200 |       4.400 |          99.000 | 02:03          | 12:10         | 15:34                    |

Figures (git-ignored, in the run directory): `cluster_profiles_<baseline>.png`, `representatives_<baseline>.png`.
