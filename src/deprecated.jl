# 0.2 compatibility shims. Included last; scheduled for removal in 0.4.0.

"""
    prepare_data(df, cols::Symbol...; zero_based=nothing) -> (Matrix{Int}, Vector{Int})

Deprecated 0.2 form of [`prepare_data`](@ref): returns the code matrix and the number of
categories of every column instead of an [`LCAData`](@ref). Use
`prepare_data(df, [cols...])` instead. `zero_based` is ignored (its length is still
validated).
"""
function prepare_data(df, cols::Symbol...;
                      zero_based::Union{Nothing,AbstractVector{Bool}}=nothing)
    if zero_based !== nothing && length(zero_based) != length(cols)
        throw(ArgumentError("Length of zero_based must match number of columns"))
    end
    Base.depwarn("`prepare_data(df, cols...)` returning `(data, n_categories)` is deprecated; " *
                 "use `prepare_data(df, [cols...])`, which returns an `LCAData`", :prepare_data)
    d = prepare_data(df, collect(cols))
    return d.y, d.n_categories
end

Base.@deprecate diagnostics!(model::LCAModel, data, ll) diagnostics(model)

@doc """
    diagnostics!(model::LCAModel, data, ll) -> ModelDiagnostics

Deprecated 0.2 form of [`diagnostics`](@ref): `data` and `ll` are ignored, the statistics
are computed from the fitted model. Use `diagnostics(model)` instead.
""" diagnostics!

Base.@deprecate show_profiles(model::LCAModel, data, cols; kwargs...) show_profiles(model; kwargs...)

"""
    LCAModel(n_classes, n_items, n_categories)

The 0.2 constructor. Models are now created by [`fit`](@ref)`(LCAModel, data, k)`, which
returns the fitted model; this method always throws an `ArgumentError`.

```jldoctest
julia> using LatentClassAnalysis

julia> LCAModel(2, 3, [2, 2, 2])
ERROR: ArgumentError: LCAModel(n_classes, n_items, n_categories) was replaced by fit(LCAModel, data, k) in v0.3; see CHANGELOG.md
[...]
```
"""
function LCAModel(n_classes::Integer, n_items::Integer, n_categories::AbstractVector{<:Integer})
    throw(ArgumentError("LCAModel(n_classes, n_items, n_categories) was replaced by fit(LCAModel, data, k) in v0.3; see CHANGELOG.md"))
end
