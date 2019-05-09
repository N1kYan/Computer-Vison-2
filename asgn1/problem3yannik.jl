# Assignment 1 - Problem 3
ENV["MPLBACKEND"]="tkagg"
using PyPlot
using Statistics
pygui(true)

print("\n\n\nExercise 3:\n")

# Read Image
img_path = string(@__DIR__,"/skeleton/a1p3.png")
img = imread(img_path)

# Convert to Float64 grayscale
gray_img = zeros(Float64,(size(img)[1],size(img)[2]))
for i = 1:size(img)[1]
    for j = 1:size(img)[2]
        gray_img[i, j] = mean(img[i, j, :])
    end
end

# Plot without axes; adding colorbar
imshow(gray_img, cmap="gray")
axis("off")
colorbar()
show()

# Compute min, max and mean pixel value
min_value = minimum(gray_img)
print("Minimum grayscale value:",min_value, "\n")
mean_value = mean(gray_img)
print("Mean grayscale value:",mean_value, "\n")
max_value = maximum(gray_img)
print("Maximum grayscale value:",max_value)
