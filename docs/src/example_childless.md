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
indicators on this page.

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

- **No covariates or standard errors.** Any analysis in the article that relates
  class membership to other characteristics, or that reports uncertainty around the
  estimates, is not reproduced here; both are coming in the 0.3.0 release, see
  [What is coming in the 0.3.0 release](@ref roadmap). The columns `female`, `age`,
  `race`, `nativity` and `sibs` are in the data set for that purpose.
