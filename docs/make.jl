using ESMFold
using Documenter

DocMeta.setdocmeta!(ESMFold, :DocTestSetup, :(using ESMFold); recursive=true)

makedocs(;
    modules=[ESMFold],
    authors="Ben Murrell",
    sitename="ESMFold.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/ESMFold.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/ESMFold.jl",
    devbranch="main",
)
