#!/usr/bin/Rscript
f <- function(x) { x * x + 10 + 2 * x }
df <- function(x) { 2 * x  + 2 }

x <- runif(1, -1000, 1000)
y <- f(x)
learning.rate <- 0.3

while (TRUE) {
  new.x = x - df(x) * learning.rate
  new.y = f(new.x)
  if (abs(y - new.y) < 1E-6) break
  x = new.x
  y = new.y
  print(c(x, y))
}
