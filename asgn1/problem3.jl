# Assignment 1 - Exercise 3
import FileIO
using PyPlot


print("Exercise 3:\n")
# Read Image
img_path = string(@__DIR__,"/skeleton/a1p3.png")
img = FileIO.load(img_path)
# Convert to Float64 grayscale
gray_img = Gray.(img)
plot(gray_img)
