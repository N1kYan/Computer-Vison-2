In order to use this IJulia notebook:

1) Add some packages:
    julia> import Pkg
    julia> Pkg.add("Images")
    julia> Pkg.add("PyPlot")
    julia> Pkg.add("JLD")
    julia> Pkg.add("IJulia")
    julia> Pkg.add("DSP")

2) Running IJulia Notebook:
    julia> using IJulia
    julia> notebook()

    ( If asked whether to install Jupyter via Conda, y/n? [y]:  You can savely pick (y)es )

3) Navigate to the corresponding ipynb file and run cells:
    e.g. Cell -> Run All
