# Tables.jl data layer: prepare_data, level/coding helpers, covariate matrix builder.

"""
    prepare_data(table, items; covariates=Symbol[], levels=nothing,
                 drop_unused_levels=true) -> LCAData

Build an [`LCAData`](@ref) from any Tables.jl source (a `DataFrame`, a `NamedTuple` of
vectors, a vector of `NamedTuple`s, ...).

Each item column is recoded to consecutive integer codes `1, 2, …, C_j` following the
order of its levels; `missing` becomes the code `0`. Levels are taken from
`DataAPI.levels`: sorted distinct values for plain vectors (`[false, true]` for `Bool`),
the level order of a `CategoricalArray`. Levels that do not occur in the column are dropped
unless `drop_unused_levels=false`; every item must keep at least two levels.

# Arguments
- `table`: any Tables.jl table
- `items::AbstractVector{<:Union{Symbol,AbstractString}}`: names of the indicator columns
- `covariates`: names of numeric (`Real`/`Bool`) columns to use as covariates of the
  class-membership model. Missing values are not allowed in covariates: drop those rows.
- `levels`: a `Dict` (or any `AbstractDict`) mapping an item name to its level vector, to
  fix the level order or to include levels absent from the column. Values are matched to
  the supplied levels by their string representation (`"1"`, `"true"`, ...); a value not
  in the list is an error.
- `drop_unused_levels::Bool=true`: drop levels that do not occur in the column

Rows whose indicators are all missing are kept (they contribute nothing to the item
parameters) and counted in a warning.

# Returns
- [`LCAData`](@ref) with `item_names`, `item_levels` and, if requested, the covariate
  matrix `X` (intercept first) and `covariate_names`.

# Example
```jldoctest
julia> using LatentClassAnalysis

julia> table = (a = [0, 1, 1, 0], b = ["no", "yes", "yes", missing], c = [1, 3, 3, 1]);

julia> d = prepare_data(table, [:a, :b, :c]);

julia> d.y
4×3 Matrix{Int64}:
 1  1  1
 2  2  2
 2  2  2
 1  0  1

julia> d.n_categories, d.item_levels
([2, 2, 2], [["0", "1"], ["no", "yes"], ["1", "3"]])
```
"""
function prepare_data(table, items::AbstractVector{<:Union{Symbol,AbstractString}};
                      covariates::AbstractVector{<:Union{Symbol,AbstractString}}=Symbol[],
                      levels::Union{Nothing,AbstractDict}=nothing,
                      drop_unused_levels::Bool=true)
    Tables.istable(table) || throw(ArgumentError(
        "expected a Tables.jl table (for example a DataFrame or a NamedTuple of vectors), got $(typeof(table))"))
    isempty(items) && throw(ArgumentError("items must name at least one column"))
    item_names = Symbol.(items)
    allunique(item_names) || throw(ArgumentError("items contains duplicate names"))
    covariate_names = Symbol.(covariates)
    allunique(covariate_names) || throw(ArgumentError("covariates contains duplicate names"))

    cols = Tables.columns(table)
    available = collect(Tables.columnnames(cols))
    user_levels = levels === nothing ? Dict{Symbol,Any}() :
                  Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(levels))

    J = length(item_names)
    n = -1
    y = Matrix{Int}(undef, 0, 0)
    n_categories = Vector{Int}(undef, J)
    item_levels = Vector{Vector{String}}(undef, J)
    for (j, name) in enumerate(item_names)
        col = _getcolumn(cols, name, available)
        if n < 0
            n = length(col)
            y = Matrix{Int}(undef, n, J)
        end
        length(col) == n || throw(ArgumentError("column $name has $(length(col)) rows, expected $n"))
        codes, labels = _code_column(col, name, get(user_levels, name, nothing), drop_unused_levels)
        y[:, j] = codes
        n_categories[j] = length(labels)
        item_levels[j] = labels
    end

    X, cnames = _covariate_matrix(cols, covariate_names, available, n)

    n_all_missing = count(i -> all(iszero, view(y, i, :)), 1:n)
    if n_all_missing > 0
        @warn "$n_all_missing row(s) have all $J indicators missing; they are kept but carry no information about the item-response probabilities"
    end
    return LCAData(y, n_categories, item_names, item_levels, X, cnames)
end

function _getcolumn(cols, name::Symbol, available)
    name in available || throw(ArgumentError(
        "column $name not found; available columns: $(join(string.(available), ", "))"))
    return Tables.getcolumn(cols, name)
end

# String label of a level; integer-valued floats print without the trailing ".0".
_level_label(x) = string(x)
_level_label(x::AbstractFloat) = (isinteger(x) && abs(x) < 1e15) ? string(Int(x)) : string(x)

# Recode one column to 1-based codes (0 = missing). Returns (codes, labels).
function _code_column(col, name::Symbol, user_levels, drop_unused::Bool)
    n = length(col)
    codes = Vector{Int}(undef, n)
    if user_levels === nothing
        levs = [DataAPI.unwrap(l) for l in DataAPI.levels(col)]
        if drop_unused
            used = Set{Any}()
            for v in col
                ismissing(v) || push!(used, DataAPI.unwrap(v))
            end
            levs = filter(in(used), levs)
        end
        length(levs) >= 2 || throw(ArgumentError(
            "item $name has $(length(levs)) distinct non-missing value(s); at least two are required"))
        allunique(levs) || throw(ArgumentError("the levels of item $name are not unique"))
        code = Dict{Any,Int}(l => c for (c, l) in enumerate(levs))
        for (i, v) in enumerate(col)
            if ismissing(v)
                codes[i] = 0
            else
                c = get(code, DataAPI.unwrap(v), 0)
                c == 0 && throw(ArgumentError("value $(repr(v)) of item $name is not among its levels"))
                codes[i] = c
            end
        end
        return codes, [_level_label(l) for l in levs]
    else
        labels = [_level_label(DataAPI.unwrap(l)) for l in user_levels]
        allunique(labels) || throw(ArgumentError("the levels supplied for item $name are not unique"))
        if drop_unused
            used = Set{String}()
            for v in col
                ismissing(v) || push!(used, _level_label(DataAPI.unwrap(v)))
            end
            kept = filter(in(used), labels)
        else
            kept = labels
        end
        length(kept) >= 2 || throw(ArgumentError(
            "item $name has $(length(kept)) used level(s); at least two are required"))
        code = Dict{String,Int}(l => c for (c, l) in enumerate(kept))
        for (i, v) in enumerate(col)
            if ismissing(v)
                codes[i] = 0
            else
                c = get(code, _level_label(DataAPI.unwrap(v)), 0)
                c == 0 && throw(ArgumentError(
                    "value $(repr(v)) of item $name is not among the supplied levels $(labels)"))
                codes[i] = c
            end
        end
        return codes, kept
    end
end

# Covariate matrix with a leading intercept column.
function _covariate_matrix(cols, names::Vector{Symbol}, available, n::Integer)
    X = ones(n, length(names) + 1)
    for (i, name) in enumerate(names)
        col = _getcolumn(cols, name, available)
        length(col) == n || throw(ArgumentError("column $name has $(length(col)) rows, expected $n"))
        for (r, v) in enumerate(col)
            ismissing(v) && throw(ArgumentError(
                "covariate $name has a missing value in row $r; drop rows with missing covariates (missing values are supported in indicators only)"))
            v isa Real || throw(ArgumentError(
                "covariate $name must be numeric (Real or Bool), got a value of type $(typeof(v)) in row $r"))
            x = Float64(v)
            isfinite(x) || throw(ArgumentError("covariate $name has a non-finite value in row $r"))
            X[r, i + 1] = x
        end
    end
    return X, [:intercept; names]
end
