# [Missing data](@id guide-missing-data)

Survey respondents skip questions. This page shows how the package handles missing
responses in the indicators: what it assumes, how the data are prepared and fitted, what
the posterior probabilities of incomplete respondents look like, and how the bootstrap
procedures treat the missing responses. The estimator is described in
[Missing data](@ref missing-data) on the methodology page.

## What the package assumes

A missing response is treated as *missing at random* (MAR) [rubin1976](@cite): whether a
respondent skips an item may depend on their answers to the other items (and, with
covariates, on their covariates), but not on the answer they would have given. Under MAR
the likelihood of the observed responses is the right thing to maximise, and the package
does exactly that: a respondent who skipped an item is classified on the items they did
answer, and the response probabilities of an item are estimated from the respondents who
answered it. This uses every partially observed row, unlike listwise deletion, which is
unbiased only under the stronger assumption that the missing responses are missing
*completely* at random. Neither assumption can be tested from the data.

## Data with missing responses

Two classes, six yes/no items, 500 respondents. We fit the complete data first so that
we can compare later, then blank out 20% of the answers to items 2 and 5 and every answer
of the first two respondents.

```@example missing
using LatentClassAnalysis, DataFrames, Random, StableRNGs, Statistics

rng = StableRNG(2024)
n = 500
truth = rand(rng, 1:2, n)
p = [0.85 0.80 0.90 0.75 0.85 0.80;     # class 1: mostly 1
     0.15 0.25 0.10 0.20 0.15 0.30]     # class 2: mostly 0
items = [Symbol("item", j) for j in 1:6]
df = DataFrame([items[j] => [Int(rand(rng) < p[c, j]) for c in truth] for j in 1:6]...)
d_full = prepare_data(df, items)
m_full = fit(LCAModel, d_full, 2; rng = StableRNG(1))
nothing # hide
```

```@example missing
df_m = copy(df)
allowmissing!(df_m)
rng = StableRNG(7)
for item in (:item2, :item5)
    df_m[randsubseq(rng, 1:n, 0.20), item] .= missing
end
df_m[1:2, :] .= missing                  # two respondents who answered nothing
first(df_m, 4)
```

[`prepare_data`](@ref) codes `missing` as `0` and keeps the rows. Its printed summary
shows how many respondents answered each item, and it warns (log messages are not shown on
this page) that two rows have every indicator missing:

```@example missing
d_m = prepare_data(df_m, items)
```

[`hasmissing`](@ref) and [`nmissing`](@ref) report the missing responses per item:

```@example missing
hasmissing(d_m), nmissing(d_m)
```

## Fitting

Nothing changes in the call. The E-step skips the missing responses of every row, and the
M-step estimates the response probabilities of each item from the rows where it is
observed:

```@example missing
m_m = fit(LCAModel, d_m, 2; rng = StableRNG(1))
```

The class sizes and the response probabilities are close to those from the complete data;
the ones for the items with missing responses are estimated from fewer respondents and
therefore differ a little more:

```@example missing
round.(m_m.class_probs; digits = 3), round.(m_full.class_probs; digits = 3)
```

```@example missing
round.(m_m.item_probs[2]; digits = 3), round.(m_full.item_probs[2]; digits = 3)
```

The number of free parameters is unchanged: missing responses reduce the information, not
the model, so `dof`, and with it AIC, BIC and sBIC, are computed exactly as for complete
data and the fit statistics of models with different numbers of classes remain comparable.

```@example missing
dof(m_m) == dof(m_full)
```

For comparison, listwise deletion — the only option in a package without missing-data
support — would discard every respondent with at least one missing answer, here more than
a third of the sample:

```@example missing
d_cc = prepare_data(dropmissing(df_m), items)
m_cc = fit(LCAModel, d_cc, 2; rng = StableRNG(1))
nobs(d_cc), round.(m_cc.class_probs; digits = 3)
```

## Posterior probabilities for incomplete rows

[`predict`](@ref) returns the posterior class probabilities of every row, computed from
the items the row does have. A respondent who answered nothing carries no information, so
their posterior is the prior — the class sizes:

```@example missing
post = predict(m_m)
round.(post[1:2, :]; digits = 3)
```

A respondent with two missing answers is classified on the other four. Here is the first
such row (`0` marks the missing responses in the coded data):

```@example missing
i = findfirst(i -> count(iszero, d_m.y[i, :]) == 2, 1:n)
d_m.y[i, :], round.(post[i, :]; digits = 3)
```

The fewer items a respondent answered, the less certain their assignment. Summarising the
largest posterior probability by the number of missing items makes this visible:

```@example missing
df_m.n_missing = count.(iszero, eachrow(d_m.y))
df_m.max_posterior = vec(maximum(post; dims = 2))
combine(groupby(df_m, :n_missing), nrow => :n, :max_posterior => mean => :mean_max_posterior)
```

[`classify`](@ref) assigns the fully missing rows to class 1, the largest class, because
that is the argmax of the prior; whether such rows should be assigned at all is a
substantive decision, and the posterior probabilities are the honest summary. The same
rules apply to new data: a table with `missing` entries is coded with the training levels
and predicted item by item.

```@example missing
new = DataFrame(item1 = [1, missing], item2 = [missing, 0], item3 = [1, 0],
                item4 = [missing, missing], item5 = [1, 0], item6 = [1, missing])
round.(predict(m_m, new); digits = 3)
```

## Bootstrap and the bootstrap likelihood-ratio test

Both resampling procedures respect the pattern of missing responses. The non-parametric
[`bootstrap`](@ref) resamples rows, each with its own missing responses, so every replicate
has about the same amount of missing data as the original sample. The parametric bootstrap
(`parametric = true`) and the [`bootstrap_lrt`](@ref) simulate complete data from the
fitted model and then re-apply the observed pattern of missing responses cell by cell,
which is what [`simulate`](@ref) does with a `missing_mask`:

```@example missing
d_sim = simulate(m_m; rng = StableRNG(2), missing_mask = m_m.data.y .== 0)
nmissing(d_sim) == nmissing(d_m)
```

Re-applying the observed pattern is exact when the responses are missing completely at
random and is the standard practice under MAR. The bootstrap standard errors below are of
the first three parameters on the logit scale (the class log-odds and two response logits),
next to those from the observed information matrix; see
[Standard errors and the bootstrap](@ref guide-inference) for how to read them.

```@example missing
b = bootstrap(m_m; n_boot = 50, rng = StableRNG(3))
coefnames(m_m)[1:3], round.(stderror(b)[1:3]; digits = 3), round.(stderror(m_m)[1:3]; digits = 3)
```

## What the package does not do

- **Missing not at random.** If the probability of skipping an item depends on the answer
  itself — refusing to report an income because it is high, skipping a question about
  drug use because the answer is yes — the MAR estimates are biased, and no software can
  fix that from the data alone. Sensitivity analyses or explicit models of the
  missingness mechanism are outside the scope of the package. One pragmatic alternative
  is to treat "not answered" as a response category of its own: replace `missing` by a
  label such as `"not answered"` before calling [`prepare_data`](@ref), which adds one
  category to the item and lets the classes differ in their non-response.
- **Imputation.** The package does not impute missing responses, and it does not need
  to: the likelihood-based treatment above is what multiple imputation approximates for
  a latent class model.
- **Missing covariates.** Covariates of the class-membership model must be complete;
  [`prepare_data`](@ref) throws an error for a covariate with a `missing` value, and the
  rows must be dropped or the covariate imputed before preparing the data. See
  [Covariates](@ref guide-covariates).
- **Survey weights.** Weighted estimation is not supported.
