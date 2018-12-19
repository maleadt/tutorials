using Weave

input = joinpath(dirname(@__DIR__), "src")
output = joinpath(dirname(@__DIR__), "public")
mkpath(output)
for file in readdir(input)
    if endswith(file, ".jl")
        @info "Rendering $file"
        weave(joinpath(input, file), out_path=output, doctype = "md2html")
    else
        @info "Copying $file"
        cp(joinpath(input, file), joinpath(output, file))
    end
end
