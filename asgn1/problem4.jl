# Assignment 1 - Problem 4
ENV["MPLBACKEND"]="tkagg"
using PyPlot
using Statistics
using Distributions
pygui(true)

# Function to map disparity map to picture
function shift_disparity(I, d)
    if !(size(I) == size(d))
        print("Disparity map size does not match image size.\n")
    end
    I_d = zeros(Float64, size(I))
    for a = 1:size(I)[1]
        for b = 1:size(I)[2]
            I_d[a, b + convert(Int64, d[a, b])] = I[a, b]
        end
    end
    return I_d
end

# Gaussian likelihood function
function gaussian_lh(I0, I1d, mu, sigma)
    gauss = Normal(mu, sigma)
    product = 1
    for a = 1:size(I0)[1]
        for b = 1:size(I0)[2]
            product = product * pdf(gauss, (I0[a, b]-I1d[a, b]))
        end
    end
    return product
end

# Negative gaussian log likelihood function
function gaussian_nllh(I0, I1d, mu, sigma)
    gauss = Normal(mu, sigma)
    sum = 0
    for a = 1:size(I0)[1]
        for b = 1:size(I0)[2]
            sum = sum + log(pdf(gauss, (I0[a, b]-I1d[a, b])))
        end
    end
    return -sum
end

# Negative laplacian log likelihood function
function laplacian_nllh(I0, I1d, mu, s)
    function lap(x, mu, s)
        y = (1/2*s)*exp(-abs(x-mu)/s)
    sum = 0
    for a = 1:size(I0)[1]
        for b = 1:size(I0)[2]
            sum = sum + lap(I0[a, b]-I1d[a, b])
        end
    end
    return -sum
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
    gt64 = convert(Array{Float64,2}, imread(gt_path)*255)
    return (i0, i1, gt64)
end

# Crop image
function crop(I0, I1d, gt)
    #TODO
end

# Add some noise to brightness of image at p% of its pixels
function make_noise(I, p)
    # TODO
end


clearconsole()
print("Assignment 1 - Problem 4:\n")
(i0, i1, gt64) = load_data()
i_d = shift_disparity(i0, gt64)
figure()
imshow(i0, cmap="gray")
title("Original image")
figure()
imshow(i_d, cmap="gray")
title("Shifted image")
show()
print(gaussian_lh(i0, i_d, 1, 10))
print(gaussian_nllh(i0, i_d, 1, 10))
print(laplacian_nllh(i0, i_d, 1, 10))
