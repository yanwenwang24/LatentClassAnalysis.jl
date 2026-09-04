# Base.show for the package types and the printed class profiles.

function Base.show(io::IO, d::LCAData)
    n, J = size(d.y)
    if get(io, :compact, false)
        print(io, "LCAData(", n, " × ", J, ")")
        return nothing
    end
    nm = nmissing(d)
    print(io, "LCAData with ", n, " observations and ", J, " items")
    total_missing = sum(nm)
    total_missing > 0 && print(io, " (", total_missing, " missing responses)")
    maxshow = 25
    for j in 1:min(J, maxshow)
        levels = d.item_levels[j]
        levstr = length(levels) <= 6 ? join(levels, ", ") : join(levels[1:5], ", ") * ", …"
        print(io, "\n  ", d.item_names[j], ": ", d.n_categories[j], " levels (", levstr, "), ",
              n - nm[j], " observed")
    end
    J > maxshow && print(io, "\n  … and ", J - maxshow, " more items")
    print(io, "\n  covariates: ",
          hascovariates(d) ? join(string.(d.covariate_names[2:end]), ", ") : "none")
    return nothing
end

# "1 class", "2 classes"
_plural(n::Integer, word::AbstractString, plural::AbstractString=word * "s") =
    string(n, " ", n == 1 ? word : plural)

function Base.show(io::IO, m::LCAModel)
    K, J, n = m.n_classes, m.n_items, nobs(m)
    if get(io, :compact, false)
        print(io, "LCAModel(", _plural(K, "class", "classes"), ", ", _plural(J, "item"), ", n = ", n, ")")
        return nothing
    end
    print(io, "LCAModel with ", _plural(K, "class", "classes"), ", ", _plural(J, "item"),
          " and ", _plural(n, "observation"))
    @printf(io, "\n  log-likelihood: %.4f   dof: %d   BIC: %.4f", m.loglik, dof(m), bic(m))
    if K == 1
        print(io, "\n  single class: closed-form solution")
    else
        n_starts = length(m.start_loglik)
        status = m.converged ? "converged after $(m.iterations) iterations" :
                 "not converged after $(m.iterations) iterations"
        print(io, "\n  ", status, "; best of ", n_starts, " start(s)")
    end
    print(io, "\n  class sizes: ", join([@sprintf("%.3f", p) for p in m.class_probs], "  "))
    if hascovariates(m)
        print(io, "\n  covariates: ", join(string.(m.data.covariate_names[2:end]), ", "))
        K > 1 && _show_coefficients(io, m)
    end
    hasmissing(m) && print(io, "\n  missing responses: ", sum(nmissing(m)))
    if m.vcov === nothing
        print(io, "\n  standard errors: none (se = :none)")
    else
        n_nan = count(i -> isnan(m.vcov[i, i]), 1:size(m.vcov, 1))
        print(io, "\n  standard errors: observed information",
              n_nan == 0 ? "" : " (NaN for $n_nan of $(dof(m)) parameters)")
    end
    msgs = _flag_messages(m.flags, m.options)
    print(io, "\n  fit flags: ", isempty(msgs) ? "none" : join(msgs, "; "))
    return nothing
end

# Coefficient table of the class-membership model: rows = covariates, columns = classes 2..K.
function _show_coefficients(io::IO, m::LCAModel)
    K = m.n_classes
    names = string.(m.data.covariate_names)
    w = max(9, maximum(length, names))
    print(io, "\n  class-membership coefficients (log-odds against class 1):")
    print(io, "\n    ", rpad("", w), join([lpad("class $k", 11) for k in 2:K]))
    for p in eachindex(names)
        print(io, "\n    ", rpad(names[p], w),
              join([@sprintf("%11.4f", m.beta[p, k - 1]) for k in 2:K]))
    end
    return nothing
end

function Base.show(io::IO, d::ModelDiagnostics)
    print(io, "ModelDiagnostics(n_classes = ", d.n_classes, ", nobs = ", d.nobs, ", dof = ", d.dof)
    @printf(io, ", ll = %.3f, aic = %.3f, bic = %.3f, sbic = %.3f, entropy = %.3f, converged = %s)",
            d.ll, d.aic, d.bic, d.sbic, d.entropy, d.converged)
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", v::AbstractVector{ModelDiagnostics})
    if isempty(v)
        print(io, "ModelDiagnostics[]")
        return nothing
    end
    print(io, length(v), "-element Vector{ModelDiagnostics}:")
    @printf(io, "\n  %7s %8s %5s %14s %14s %14s %14s %8s %9s",
            "classes", "nobs", "dof", "loglik", "AIC", "BIC", "sBIC", "entropy", "converged")
    for d in v
        @printf(io, "\n  %7d %8d %5d %14.3f %14.3f %14.3f %14.3f %8.3f %9s",
                d.n_classes, d.nobs, d.dof, d.ll, d.aic, d.bic, d.sbic, d.entropy, d.converged)
    end
    return nothing
end

"""
    show_profiles(m::LCAModel; var_names=nothing, var_labels=nothing, digits=3, io=stdout)

Print the latent class profiles: the size of every class and, for every item, the
probability of each response level within each class (as percentages). Item names and
level labels default to those stored in `m.data`.

When the model carries a covariance matrix (fitted with `se=:hessian`, the default),
every percentage is followed by `±` its delta-method standard error in percentage points
(see [`profiles`](@ref)); an undefined standard error (a probability on the boundary, or
the class sizes of a model with covariates) prints as `NaN`.

# Arguments
- `m::LCAModel`: fitted model
- `var_names`: display names of the items (default: the item names of the data)
- `var_labels`: display labels of the levels of every item, one vector per item (default:
  the level labels of the data)
- `digits::Integer=3`: decimal places of the printed percentages and standard errors
- `io::IO=stdout`: output stream

# Returns
- `nothing`

The 0.2 form `show_profiles(m, df, cols; kwargs...)` is deprecated and ignores `df` and
`cols`. See [`profiles`](@ref) for the same numbers as a table.
"""
function show_profiles(m::LCAModel;
                       var_names::Union{Nothing,AbstractVector}=nothing,
                       var_labels::Union{Nothing,AbstractVector}=nothing,
                       digits::Integer=3, io::IO=stdout)
    names = var_names === nothing ? string.(m.data.item_names) : String.(collect(var_names))
    length(names) == m.n_items ||
        throw(ArgumentError("var_names has $(length(names)) entries, expected $(m.n_items)"))
    labels = var_labels === nothing ? m.data.item_levels :
             [String.(collect(l)) for l in var_labels]
    length(labels) == m.n_items ||
        throw(ArgumentError("var_labels has $(length(labels)) entries, expected $(m.n_items)"))
    for j in 1:m.n_items
        length(labels[j]) == m.n_categories[j] || throw(ArgumentError(
            "var_labels[$j] has $(length(labels[j])) labels but item $(names[j]) has $(m.n_categories[j]) categories"))
    end
    digits >= 0 || throw(ArgumentError("digits must be non-negative, got $digits"))

    withse = m.vcov !== nothing
    se_class, se_items = withse ? _profile_se(m) : (Float64[], Matrix{Float64}[])
    fmt = Printf.Format("%.$(digits)f%%")
    sefmt = Printf.Format("%.$(digits)f")
    cell(p, se) = Printf.format(fmt, p * 100) * (withse ? " ±" * Printf.format(sefmt, se * 100) : "")
    colw = withse ? max(12, 2 * digits + 12) : 12
    println(io)
    println(io, "Latent Class Profiles")
    println(io, "="^80)
    println(io, "Class Sizes:")
    for k in 1:m.n_classes
        print(io, "  Class $k: $(rpad(@sprintf("%.1f", m.class_probs[k] * 100), 6))%")
        withse && @printf(io, " ±%.1f", se_class[k] * 100)
        println(io)
    end
    println(io, "-"^80)

    max_label_length = maximum(maximum(length, l) for l in labels)
    for (j, var) in enumerate(names)
        println(io, "\n$var:")
        print(io, " "^(max_label_length + 2))
        for k in 1:m.n_classes
            print(io, rpad("Class $k", colw))
        end
        println(io)
        for (c, label) in enumerate(labels[j])
            print(io, rpad("$label:", max_label_length + 2))
            for k in 1:m.n_classes
                se = withse ? se_items[j][k, c] : NaN
                print(io, rpad(cell(m.item_probs[j][k, c], se), colw))
            end
            println(io)
        end
    end
    println(io, "\n" * "-"^80)
    return nothing
end
