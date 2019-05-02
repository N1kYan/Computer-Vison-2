# Assignment 1
import FileIO
using Plots, Colors
pyplot()


# Exercise 3
print("Exercise 3:\n")
# Read Image
img_path = string(@__DIR__,"/skeleton/a1p3.png")
img = FileIO.load(img_path)
# Convert to Float64 grayscale
gray_img = Gray.(img)
figure()
plot(gray_img)
