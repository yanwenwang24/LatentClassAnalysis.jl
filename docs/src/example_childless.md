# Example: childlessness in Singapore

This page replicates the latent class analysis in [wang2024](@cite), which
identifies distinct pathways to permanent childlessness among older adults in
Singapore from indicators of their partnership, education, and work histories in
their twenties and thirties. The file `examples/childless_df.arrow` bundled with
the package is a de-identified extract of the analytic sample (493 respondents,
15 columns) provided so that the analysis can be reproduced. The same analysis is
available as a script in `examples/example_childless.jl`.

<!-- TODO(author): survey name, sampling, permission/license statement -->

## The data

```@example childless
using LatentClassAnalysis, Arrow, DataFrames, StableRNGs, Statistics

df = Arrow.Table(joinpath(pkgdir(LatentClassAnalysis), "examples", "childless_df.arrow")) |> DataFrame
size(df)
```

The eight indicators used in the analysis are listed below. The remaining columns
(`id`, `weights`, `female`, `age`, `race`, `nativity`, `sibs`) are not used as
indicators; `female` (0/1) and `age` (in years) serve as covariates of class membership
further down.

| Column        | Domain      | Meaning                                    | Values                                                        |
|:--------------|:------------|:-------------------------------------------|:--------------------------------------------------------------|
| `age_fmarry`  | Partnership | Timing of first marriage                   | `"no"`, `"early"`, `"norm"`, `"late"`                          |
| `marry_end`   | Partnership | Marriage dissolved                         | 0/1                                                            |
| `infertility` | Partnership | Infertility reported                       | 0/1                                                            |
| `edu`         | Education   | Highest education                          | `"low"`, `"middle"`, `"high"`                                  |
| `ocp20s`      | Work        | Occupation in the twenties                 | `"Unemployed"`, `"Blue-collared"`, `"Semi-professional"`, `"Professional"` |
| `ocp30s`      | Work        | Occupation in the thirties                 | same as `ocp20s`                                               |
| `flexible`    | Work        | Flexible work arrangements available       | 0/1                                                            |
| `familyleave` | Work        | Generous family leave available            | 0/1                                                            |

```@example childless
indicators = [:age_fmarry, :marry_end, :infertility, :edu, :ocp20s, :ocp30s, :flexible, :familyleave]
first(df[:, indicators], 5)
```

String columns are coded in alphabetical order unless told otherwise, so we spell out
the substantive order of the four string indicators. The order affects only how the
profiles are displayed, not the fit.

```@example childless
occupations = ["Unemployed", "Blue-collared", "Semi-professional", "Professional"]
levels = Dict(:age_fmarry => ["no", "early", "norm", "late"],
              :edu => ["low", "middle", "high"],
              :ocp20s => occupations,
              :ocp30s => occupations)
nothing # hide
```

## Choosing the number of classes

`fit(LCAModel, table, items, ks)` prepares the table and fits one model per class
count, each from 20 random starts of which the 4 best are continued to convergence.
We fit models with two to six classes with a fixed random number generator so that
the page is reproducible.

```@example childless
models = fit(LCAModel, df, indicators, 2:6; levels = levels, rng = StableRNG(1024))
selection = DataFrame(diagnostics(models))
```

If you run this yourself, every fit prints a warning (log messages are not captured
on this page) that some response probabilities were estimated at exactly 0 or 1;
with 493 respondents and rare responses such as unemployment in the twenties, some
cells are empty within a class, and their probabilities go to the boundary. The
number of such cells grows with the number of classes.

BIC, which penalises each parameter most heavily, is minimised at the five-class
model, while AIC and sBIC keep decreasing as classes are added. We select by BIC, as
recommended in [Choosing the number of classes](@ref choosing-k).

```@example childless
best = models[argmin(selection.bic)]
```

Printing the model shows its fit, the class sizes (classes are numbered by decreasing
size), and the fit flags; here the only flag is the boundary count discussed above.

### Was the maximum found?

Each model was fitted from 20 random starts. `start_loglik` holds the log-likelihood
reached by every start — the final value for the four starts that were continued to
convergence and the value after 50 iterations for the others — so sorting it shows
whether the best solution was reached more than once:

```@example childless
sort(best.start_loglik; rev = true)[1:6]
```

All four continued starts reached the same log-likelihood, and
`best.flags.best_ll_replicated` records this:

```@example childless
best.flags.best_ll_replicated
```

Had the best value appeared only once, `fit` would have warned and the model should
be refitted with larger `n_starts` and `n_final`.

### Testing the number of classes

BIC chose five classes while AIC and sBIC kept decreasing. The bootstrap
likelihood-ratio test ([`bootstrap_lrt`](@ref), see the
[model selection](@ref guide-model-selection) guide) offers a third opinion: it simulates
data sets from the smaller model and asks whether the observed gain of the extra class is
more than such data produce by chance. `models` holds the fits for two to six classes, so
`models[3]` and `models[4]` are the four- and five-class models. Nineteen replicates keep
the page quick; they resolve the p-value only down to 0.05.

```@example childless
test45 = bootstrap_lrt(models[3], models[4]; n_boot = 19, rng = StableRNG(2024))
```

```@example childless
test56 = bootstrap_lrt(models[4], models[5]; n_boot = 19, rng = StableRNG(2025))
```

Both tests reject: the fifth class improves the log-likelihood far more than any data set
simulated from four classes, and so does the sixth relative to five (with 99 replicates
the second p-value is 0.01, again at the floor). The test therefore sides with AIC and
sBIC rather than with BIC. Two caveats temper this. The test assumes that the smaller
model is exactly right, including local independence, and with real survey data small
departures from the model are enough to make an extra class "significant"; and the fits
with many classes are increasingly fragile — the number of response probabilities
estimated at exactly 0 or 1 grows from four in the two-class model to 38 in the six-class
model:

```@example childless
[m.flags.n_boundary for m in models]
```

With 493 respondents, a sixth class is mostly a class of empty cells. We keep the
five-class solution selected by BIC and note that the statistical evidence does not rule
out a sixth; the substantive interpretability of the classes, discussed below, is the
final arbiter.

## Class profiles

```@example childless
show_profiles(best;
              var_names = ["Marriage timing", "Marriage dissolved", "Infertility", "Education",
                           "Occupation in 20s", "Occupation in 30s", "Flexible work", "Family leave"])
```

## Class assignments

[`classify`](@ref) gives every respondent's most likely class, which we attach to the
data and summarise by sex and age. The mean of the largest posterior probability
per class says how confidently its members are assigned.

```@example childless
df.class = classify(best)
df.max_posterior = vec(maximum(predict(best); dims = 2))
combine(groupby(df, :class), nrow => :n,
        :female => mean => :share_female, :age => mean => :mean_age,
        :max_posterior => mean => :mean_max_posterior)
```

## Interpretation

Read each table column by column: within a class, the entries for an item give the
probability of each response. Four of the five classes consist mainly of people who
never married, and they are separated by education and occupation; the fifth is the
only one whose members married.

- **Class 1**, the largest, never married with middle or high education and
  semi-professional work in both decades.
- **Class 2** never married with low education and blue-collar work, and had the
  least access to flexible work and family leave.
- **Class 3** never married with high education and professional occupations.
- **Class 4** never married with low education and unemployment or blue-collar work in
  the twenties, yet reports flexible work arrangements and family leave almost
  universally.
- **Class 5**, the smallest, is the only class whose members married, typically late,
  and it carries most of the reported infertility and marital dissolution in the
  sample.

The availability of flexible work and family leave distinguishes class 4 but differs
little among the other classes. Class numbers follow class size, so identify the
classes by these profiles rather than by their numbers. Readers can compare these
profiles with the named pathways in [wang2024](@cite).

## Who follows which pathway? Covariates

The table of class composition above is descriptive. A latent class regression makes
the relation between respondents' characteristics and their pathway part of the model:
the class membership probabilities become a multinomial-logit function of covariates
(see the [Covariates](@ref guide-covariates) guide). We use sex and age, the two
covariates in the data that need no recoding (`race` and `nativity` are numeric codes and
would have to be dummy-coded first).

```@example childless
d_cov = prepare_data(df, indicators; levels = levels, covariates = [:female, :age])
m_cov = fit(LCAModel, d_cov, 5; rng = StableRNG(1024))
```

The class sizes and profiles of the covariate model are nearly identical to those of the
unconditional five-class model, so the classes have kept their meaning and the
coefficients can be read against the profiles above:

```@example childless
round.(m_cov.class_probs; digits = 3), round.(best.class_probs; digits = 3)
```

```@example childless
coeftable(m_cov; which = :class)
```

Every coefficient is a log-odds against class 1, the largest class (never married,
middle or high education, semi-professional work). Women are much less likely than men to
be in class 2 (never married, low education, blue-collar work: the odds ratio is
``\exp(-1.85) \approx 0.16``), and also less likely to be in classes 3 and 4; the
respondents in class 2 are older (about 5% higher odds per year of age) and those in class
3 (never married, high education, professional work) younger. Class 5, the only class
whose members married, does not differ from class 1 by sex or age. The eight slopes
together improve the log-likelihood by a wide margin:

```@example childless
m_nocov = fit(LCAModel, d_cov, 5; rng = StableRNG(1024), covariates = false)
2 * (loglikelihood(m_cov) - loglikelihood(m_nocov)), dof(m_cov) - dof(m_nocov)
```

Three caveats. This is a one-step model, so the covariates take part in defining the
classes; the check above that the class sizes did not move is the reason to trust the
comparison with the unconditional profiles. In a cross-section, age is also birth cohort,
and the age coefficients mix the two. And the standard errors of many item-response
probabilities are undefined because those probabilities are on the boundary (the fit
warns; `coeftable(m_cov)` shows the `NaN` rows), while every class-membership coefficient
has a standard error.

## Differences from the published analysis

- **Number of classes.** With random restarts, the four continued starts of every
  model agree on the maximum, so the selection above is a property of the data and
  the criteria rather than of a seed; BIC selects five classes here. Compare with the
  solution reported in the article before drawing conclusions from any single
  criterion, and consider the substantive interpretability of the extra class, as
  discussed in [Choosing the number of classes](@ref choosing-k).
- **No survey weights.** The `weights` column is ignored; the package does not
  support weighted estimation.
- **Category order.** By default [`prepare_data`](@ref) orders the levels of a string
  column alphabetically (education would be listed as `high`, `low`, `middle`). The
  `levels` keyword, used above through `fit`, fixes the order per item; the same
  keyword works with [`prepare_data`](@ref) directly:

```@example childless
d = prepare_data(df, indicators; levels = levels)
```

- **Covariates.** The latent class regression above uses `female` and `age` only, in a
  one-step model; any analysis in the article that relates the pathways to other
  characteristics is not reproduced. The columns `race`, `nativity` and `sibs` are in
  the data set for that purpose (dummy-code the first two before passing them to
  `covariates`).
- **Standard errors.** Those of the profiles are available from `profiles(best)` and
  printed by `show_profiles`; the article's are not compared here because of the
  weights.
