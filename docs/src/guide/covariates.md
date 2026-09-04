# [Covariates](@id guide-covariates)

Latent class analysis often has two questions: what are the classes, and who is in them?
The second is answered by a *latent class regression*, in which covariates such as age or
sex predict class membership. This page shows how to fit one, how to read its
coefficients, how to compare it with the unconditional model, and what can go wrong. The
estimator is described in [Covariates on class membership](@ref covariates-model) on the
methodology page.

## The model

Without covariates every respondent has the same prior probability ``\pi_k`` of belonging
to class ``k``. With covariates ``x`` (a vector with a leading 1 for the intercept) the
prior becomes a multinomial-logit function of ``x``, the *concomitant-variable* model of
[dayton1988](@cite):

```math
\log \frac{\pi_k(x)}{\pi_1(x)} = x' \beta_k, \qquad k = 2, \dots, K,
```

with class 1 as the reference. The coefficients ``\beta_k`` have the same reading as in a
multinomial logistic regression: a one-unit increase in a covariate multiplies the odds of
class ``k`` against class 1 by ``\exp(\beta_{k})`` for that covariate. The item-response
probabilities are unchanged: the covariates explain who is in each class, not what the
classes look like. Because classes are numbered by decreasing size, class 1 — the
reference — is always the largest class.

## Simulated data

800 respondents, five yes/no items, two classes. Age is drawn between 25 and 65, and sex is
a `Bool` column. The log-odds of class 2 against class 1 are ``-0.8 + 0.06\,(\text{age} -
45) + 0.8\,\text{female}``, so older respondents and women are more likely to be in class
2, which is the smaller class (about 38% of the sample).

```@example covariates
using LatentClassAnalysis, DataFrames, StableRNGs

rng = StableRNG(2024)
n = 800
age = Float64.(rand(rng, 25:65, n))
female = rand(rng, Bool, n)
η = -0.8 .+ 0.06 .* (age .- 45) .+ 0.8 .* female        # log-odds of class 2 vs class 1
class = [rand(rng) < 1 / (1 + exp(-η[i])) ? 2 : 1 for i in 1:n]
p = [0.85 0.80 0.90 0.80 0.85;                            # class 1: mostly 1
     0.15 0.20 0.10 0.20 0.15]                            # class 2: mostly 0
items = [Symbol("item", j) for j in 1:5]
df = DataFrame([items[j] => [Int(rand(rng) < p[c, j]) for c in class] for j in 1:5]...)
df.age = age
df.female = female
first(df, 3)
```

Covariates are named in the `covariates` keyword of [`prepare_data`](@ref). They must be
numeric (`Real` or `Bool`) columns without `missing` values; an intercept is added
automatically.

```@example covariates
d = prepare_data(df, items; covariates = [:age, :female])
```

[`fit`](@ref) uses the covariates of the data by default. Printing the model shows the
coefficient block in addition to the usual summary:

```@example covariates
m = fit(LCAModel, d, 2; rng = StableRNG(1))
```

## Reading the coefficients

[`coeftable`](@ref) with `which = :class` restricts the coefficient table to the
class-membership block: one row per covariate (and the intercept) and per class other than
class 1, with standard errors from the observed information matrix, Wald ``z`` tests and
confidence intervals.

```@example covariates
coeftable(m; which = :class)
```

The row `class2: age` says that each additional year of age multiplies the odds of being
in class 2 rather than class 1 by ``\exp(0.058) \approx 1.06``, and `class2: female` that
women have ``\exp(0.78) \approx 2.2`` times the odds of men. Both are estimated close to
the simulated values (0.06 and 0.8). The intercept is the log-odds at age 0 for a man,
``-0.8 - 0.06 \times 45 = -3.5`` in the simulation; centring a covariate before
[`prepare_data`](@ref) gives the intercept a more useful meaning. The coefficients are
also available as the matrix `m.beta` (one row per covariate, intercept first; one column
per class from class 2 on), and [`coefnames`](@ref) labels them.

With covariates the class sizes are not parameters but sample averages of the
covariate-specific membership probabilities, and that is what `m.class_probs` holds:

```@example covariates
round.(m.class_probs; digits = 3)
```

With three or more classes the table has one block of rows per class from class 2 on,
each against class 1; a coefficient that is positive for class 2 and negative for class 3
means that the covariate shifts respondents from class 3 towards class 2 relative to
class 1.

## Prediction needs the covariates

The posterior probability of a respondent combines their covariate-specific prior with
their responses, so [`predict`](@ref) and [`classify`](@ref) on new data need the same
covariate columns as the training data, and refuse a table without them:

```@repl covariates
predict(m, df[:, items])
```

```@example covariates
new = DataFrame(item1 = [1, 0], item2 = [1, 0], item3 = [1, 1], item4 = [1, 0], item5 = [1, 0],
                age = [30.0, 60.0], female = [false, true])
round.(predict(m, new); digits = 3)
```

The second respondent answers like class 2 and, being an older woman, has a prior that
leans the same way, so the posterior is nearly certain.

## Is the covariate model an improvement?

`covariates = false` fits the unconditional model on the same `LCAData`, which gives a
nested comparison: the unconditional model is the covariate model with every slope set to
zero, so its log-likelihood is never larger. The model-selection table puts the two side by
side:

```@example covariates
m0 = fit(LCAModel, d, 2; rng = StableRNG(1), covariates = false)
DataFrame(diagnostics([m0, m]))
```

The likelihood-ratio statistic and its degrees of freedom (the number of slopes) are

```@example covariates
2 * (loglikelihood(m) - loglikelihood(m0)), dof(m) - dof(m0)
```

Because the restriction is on interior parameters (slopes equal to zero, not a class
removed), the usual chi-squared reference distribution applies: with 2 degrees of freedom
the 5% critical value is 5.99, and the covariates are clearly needed. BIC says the same.
This test compares two models with the *same* number of classes; comparing numbers of
classes is the job of the bootstrap likelihood-ratio test on the
[model selection](@ref guide-model-selection) page. Note that the two models rank the same
respondents slightly differently, because their priors differ; the response profiles,
however, should be nearly identical, and a large change in the profiles when covariates
are added is a warning sign discussed below.

```@example covariates
round.(m0.item_probs[1]; digits = 3), round.(m.item_probs[1]; digits = 3)
```

## Separation

A covariate that determines class membership almost perfectly — because it is nearly a
function of the class, or because it is itself one of the indicators under another name —
has no finite maximum-likelihood coefficient. The Newton steps then push the coefficient
towards infinity; `fit` stops it at a large value, raises the `coef_divergence` flag,
warns about quasi-complete separation, and reports `NaN` standard errors. In the
simulation we know the true classes, so we can provoke this deliberately:

```@example covariates
df.perfect = Float64.(class .== 2)
d_sep = prepare_data(df, items; covariates = [:perfect])
m_sep = fit(LCAModel, d_sep, 2; rng = StableRNG(1))
m_sep.flags.coef_divergence, round.(m_sep.beta; digits = 1)
```

The remedy is to drop the covariate, or to use it as an indicator if it describes the
classes rather than predicts them.

## Categorical covariates: dummy-code them first

Covariate columns must be numeric. A string or categorical column is an error, because the
package would have to choose a coding for it:

```@repl covariates
df.region = rand(StableRNG(5), ["north", "south", "east"], n);
prepare_data(df, items; covariates = [:region])
```

Create the dummy variables in the table before calling [`prepare_data`](@ref), leaving one
level out as the reference. `Bool` columns are accepted and coded 0/1:

```@example covariates
df.south = df.region .== "south"
df.east = df.region .== "east"
d_region = prepare_data(df, items; covariates = [:age, :female, :south, :east])
m_region = fit(LCAModel, d_region, 2; rng = StableRNG(1))
coeftable(m_region; which = :class)
```

The region coefficients are the log-odds of class 2 against class 1 for respondents in the
south and the east relative to those in the north. (The region was drawn at random here,
so nothing substantive should be read into them.) Interaction terms, squared terms and
similar transformations are made the same way, as columns of the table. A covariate must
not be constant, and the covariates must not be collinear (a full set of dummies plus the
intercept, for instance); both are errors.

## Standard errors of the class sizes

The class sizes of a covariate model are sample averages of the covariate-specific
membership probabilities, for which the delta method used by [`profiles`](@ref) has no
standard error to offer, so the `se`, `lower` and `upper` columns of the class-size rows
are `NaN`:

```@example covariates
first(DataFrame(profiles(m; classes = true)), 2)
```

The bootstrap does provide them, since it simply averages the membership probabilities of
every replicate:

```@example covariates
b = bootstrap(m; n_boot = 50, rng = StableRNG(3))
first(DataFrame(profiles(b; classes = true)), 2)
```

`coeftable(b; which = :class)` gives the coefficients with bootstrap standard errors and
percentile intervals; see [Standard errors and the bootstrap](@ref guide-inference).

## One-step versus three-step

The package fits the *one-step* model: the class-membership regression and the
item-response probabilities are estimated simultaneously, by maximum likelihood, and the
standard errors of the coefficients account for the uncertainty of the classification.
This is statistically the most efficient approach, but it has a well-known drawback
[vermunt2010](@cite): the covariates take part in defining the classes, so adding or
removing a covariate can change the response profiles, and the meaning of "class 2"
with covariates is not guaranteed to be the meaning of "class 2" without them. Always fit
the unconditional model first, then the covariate model, and compare their profiles
(`show_profiles` on both) and class sizes; a covariate model whose classes have drifted
away from the unconditional solution is answering a different question.

The alternative *three-step* approach — fit without covariates, classify, then regress
the assigned class on the covariates — keeps the classes fixed but underestimates the
effects, because the assigned classes contain classification error, unless the
bias-adjusted corrections of [vermunt2010](@cite) are applied. The package does not
implement the three-step procedure or its corrections; the ingredients for the naive
version are [`classify`](@ref) or [`predict`](@ref) and any regression package.

Two practical notes. Covariates are standardised internally for the Newton steps, so a
covariate measured in dollars and one measured in thousands of dollars give the same fit
and coefficients that differ by exactly a factor of 1000; the reported `beta` and standard
errors are always on the raw scale of the columns. And rows with a missing covariate must
be dropped before [`prepare_data`](@ref): missing values are supported in the indicators
only (see [Missing data](@ref guide-missing-data)).
