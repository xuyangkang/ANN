#!/usr/bin/Rscript
model <- read("model.txt")
layer.size <- model$layer.size
neuron.size <- model$neuron.size
synapse <- model.synapse

pf.in <- file("test-images", "rb")
pf.out <- file("test-labels", "rb")

#check magic
magic <-readBin(pf.in, integer(), 1, endian = "big")
stopifnot(magic == 2051)
magic <-readBin(pf.out, integer(), 1, endian = "big")
stopifnot(magic == 2049)

#read and check train data numbers
n <- readBin(pf.in, integer(), 1, endian = "big")[1]
stopifnot(n == readBin(pf.out, integer(), 1, endian = "big"))

# read number of rows and columns
m <- readBin(pf.in, integer(), 1, endian = "big")[1] * readBin(pf.in, integer(), 1, endian = "big")[1]

# number of right answers
right <- 0

for (ii in 1:n) {
  vec.in <- readBin(pf.in, integer(), n = m, size = 1)
  vec.in <- (vec.in + 256) %% 256 / 256.0
  vec.in <- c(vec.in, 1)
  ac <- list(t(as.matrix(vec.in)))
  for (i in 2:layer.size) {
    ac.next <- ac[[i - 1]] %*% synapse[[i - 1]]
    #active function : x > 0 ? x : 0
    rectifier <- function(x) { (abs(x) + x) * 0.5}
    ac.next <- rectifier(ac.next)
    ac.next[neuron.size[i]] <- 1
    ac <- c(ac, list(ac.next))
  }

  #do softmax for output layer
  result <- exp(ac[[layer.size]][1:10])
  normalizer <- function(x) { x / sum(x) }
  result <- normalizer(result)
  
  class <- 1
  for (i in 2:10) {
      if (result[class] < result[i]) class <- i
  }

  label <- readBin(pf.out, integer(), size = 1) + 1
  if (class == label) right <- right + 1
}

print(right)