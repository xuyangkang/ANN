#!/usr/bin/Rscript
f <- function(x, y) { (1 - x) ** 2 + 100 * (y - x ** 2) ** 2}
df.dx <- function(x, y) { x * 2 - 2 - 400 * y + 400 * x ** 3}
df.dy <- function(x, y) { 200 * y - 200 }

x <- runif(1, -100, 100)
y <- runif(1, -100, 100)
z = f(x, y)
learning.rate = 1E-6

while (TRUE) {
  new.x = x - df.dx(x, y) * learning.rate
  new.y = y - df.dy(x, y) * learning.rate
  new.z = f(new.x, new.y)
  if (abs(new.z - z) < 1E-10) break
  x = new.x
  y = new.y
  z = new.z
  print(c(x, y, z))
}
