# Assignment 1 - Problem 4
ENV["MPLBACKEND"]="tkagg"
using PyPlot
using Statistics
pygui(true)

# Function to map disparity map to picture
function shift_disparity(I, d)
    if !(size(I) == size(d))
        print("Disparity map size does not match image size.\n")
    end
    return I+d
end

# Gaussian likelihood function
function gaussian_lh(I0, I1d, mu, sigma)
    #TODO
end

# Negative gaussian log likelihood function
function gaussian_nllh(I0, I1d, mu, sigma)
    # TODO
end

# Negative laplacian log likelihood function
function laplacian_nllh(I0, I1d, mu, s)
    # TODO
end

# Convert image to Float64 grayscale image
function convert_to_grayscale(image)
    gray_image = zeros(Float64,(size(image)[1],size(image)[2]))
    for i = 1:size(image)[1]
        for j = 1:size(image)[2]
            gray_image[i, j] = mean(image[i, j, :])
        end
    end
    return gray_image
end

# Load images into FLoat64 arrays and disparity map
function load_data()
    i0_path = string(@__DIR__,"/skeleton/i0.png")
    i0 = imread(i0_path)
    i0 = convert_to_grayscale(i0)
    i1_path = string(@__DIR__,"/skeleton/i1.png")
    i1 = imread(i1_path)
    i1 = convert_to_grayscale(i1)
    gt_path = string(@__DIR__,"/skeleton/gt.png")
    gt = imread(gt_path)
    print("Image data loaded.\n")
    return (i0, i1, gt)
end
