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
using LatentClassAnalysis, Arrow, DataFrames, Random, Statistics

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

## Preparing the data

```@example childless
data, n_categories = prepare_data(df, indicators...)
n_categories
```

Two of the indicators have four categories, one has three, and five are binary.

## Choosing the number of classes

We fit models with two to six classes, seeding the random number generator before
each model so that the page is reproducible, and collect the fit statistics in a
table.

```@example childless
results = DataFrame(k = Int[], loglik = Float64[], npar = Int[],
                    AIC = Float64[], BIC = Float64[], sBIC = Float64[], entropy = Float64[])
models = Dict{Int,LCAModel}()

for k in 2:6
    Random.seed!(1024)
    model = LCAModel(k, size(data, 2), n_categories)
    ll = fit!(model, data)
    d = diagnostics!(model, data, ll)
    npar = (k - 1) + k * sum(n_categories .- 1)
    push!(results, (k, ll, npar, d.aic, d.bic, d.sbic, d.entropy))
    models[k] = model
end
results
```

BIC, which penalises each parameter most heavily, is minimised at the four-class
model in this run, while AIC and sBIC keep decreasing as classes are added. We
select by BIC, as recommended in [Choosing the number of classes](@ref choosing-k).

```@example childless
best_k = results.k[argmin(results.BIC)]
```

## Class profiles

```@example childless
best = models[best_k]
show_profiles(best, df, indicators;
              var_names = ["Marriage timing", "Marriage dissolved", "Infertility", "Education",
                           "Occupation in 20s", "Occupation in 30s", "Flexible work", "Family leave"])
```

## Class assignments

```@example childless
assignments, posterior = predict(best, data)
df.class = assignments
combine(groupby(df, :class), nrow => :n, :female => mean => :share_female, :age => mean => :mean_age)
```

## Interpretation

Read each table column by column: within a class, the entries for an item give the
probability of each response. The classes that come out of this run are
distinguished mainly by marriage timing, education, and occupation, while the
availability of flexible work and family leave differs little between them. One
class combines never marrying with low education and blue-collar or no employment;
another combines never marrying with high education and professional occupations;
a third also consists largely of the never married but with middle or high
education and semi-professional occupations, and is the largest class; and the
remaining, smallest, class is the only one whose members mostly married, typically
late, and it carries most of the marital dissolution and reported infertility in
the sample. Class numbers are arbitrary and can change with the seed, so identify
the classes by these profiles rather than by their numbers. Readers can compare
these profiles with the named pathways in [wang2024](@cite).

## Differences from the published analysis

- **Single random start.** Each model above is fitted once from the starting
  values drawn after `Random.seed!(1024)`. Other seeds can reach a different local
  maximum with a slightly different log-likelihood, as the table below shows; for a
  publishable analysis, refit from many seeds and keep the best fit (see
  [Caveats](@ref)).

```@example childless
seeds = [1, 2, 3, 42, 1024]
fits = map(seeds) do seed
    Random.seed!(seed)
    m = LCAModel(best_k, size(data, 2), n_categories)
    (seed = seed, loglik = fit!(m, data))
end
DataFrame(fits)
```

- **No survey weights.** The `weights` column is ignored; the package does not
  support weighted estimation yet.
- **Category order.** [`prepare_data`](@ref) orders the levels of a string column
  alphabetically, which is why the tables above list education as
  `high`, `low`, `middle` and marriage timing as `early`, `late`, `no`, `norm`. The
  order does not affect the fit, only the display. To show the categories in a
  substantive order, convert the column to a `CategoricalArray` with explicit levels
  before calling `prepare_data`:

```julia
using CategoricalArrays
df.edu = categorical(df.edu; levels = ["low", "middle", "high"])
df.age_fmarry = categorical(df.age_fmarry; levels = ["no", "early", "norm", "late"])
```

- **No covariates or standard errors.** Any analysis in the article that relates
  class membership to other characteristics, or that reports uncertainty around the
  estimates, is not reproduced here; see
  [Limitations of the current version and roadmap](@ref limitations).
