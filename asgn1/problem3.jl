# Assignment 1 - Exercise 3
ENV["MPLBACKEND"]="tkagg"
using PyPlot
using Statistics
pygui(true)

clearconsole()
print("Exercise 3:\n")

# Read Image
img_path = string(@__DIR__,"/a1p3.png")
img = imread(img_path)

# Convert to Float64 grayscale
img=convert(Array{Float64,3}, img)
gray_img = 0.2989*img[:,:,1] + 0.5870*img[:,:,2] + 0.1140*img[:,:,3]

# Plot without axes; adding colorbar
imshow(gray_img, cmap="gray")
axis("off")
colorbar()
show()

# Compute min, max and mean pixel value
min_value = minimum(gray_img)
print("Minimum grayscale value: ",min_value, "\n")
mean_value = mean(gray_img)
print("Mean grayscale value: ",mean_value, "\n")
max_value = maximum(gray_img)
print("Maximum grayscale value: ",max_value)
