#!/usr/bin/Rscript
layer.size <- 4
neuron.size <- c(28 * 28, 200, 200, 10) + 1
synapse <- list()
for (i in 2:layer.size) {
  rand.data <- rnorm(n = neuron.size[i - 1] * neuron.size[i], mean = 0, sd = 0.1)
  synapse <- c(synapse, list(matrix(data = rand.data, nrow = neuron.size[i - 1], ncol = neuron.size[i])))
}

pf.in <- file("train-images", "rb")
pf.out <- file("train-labels", "rb")

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

# average error for all train data
avg <- 0

for (ii in 1:n) {
  # learning rate, a magic number...
  learning.rate <- 0.009 * (n - ii) / n / 3 + 0.002
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

  label <- readBin(pf.out, integer(), size = 1) + 1
  error <- rep(0.0, 10)
  error[label] <- 1.0

  # I don't know why.... My GF tell me to do so and it works....
  # Maybe I should learn some more ML...
  result <- result - error
  avg <- (avg * (ii - 1) + sum(error * result)) / ii

  if (ii %% 100 == 0) print(avg)
  if (is.nan(avg)) break
  error <- as.matrix(c(result, 0))    
  
  for (i in layer.size:2) {
    delta <- t(error %*% ac[[i - 1]]) + synapse[[i - 1]] * 0.0001
    new.error <- t(synapse[[i - 1]] %*% error) 
    synapse[[i - 1]] <- synapse[[i - 1]] - delta * learning.rate
    for (j in 1:ncol(new.error)) {
      if (ac[[i - 1]][j] == 0) new.error[j] <- 0
    }
    new.error[neuron.size[i]] <- 0
    error <- t(new.error)
  }
}
#!/usr/bin/Rscript
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