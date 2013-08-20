layer_size <- 4
neuron_size <- c(28 * 28, 200, 200, 10) + 1
synapse <- list()
for (i in 2:layer_size) {
    rand_data <- rnorm(n = neuron_size[i - 1] * neuron_size[i], mean = 0, sd = 0.1)
    synapse <- c(synapse, list(matrix(data = rand_data, nrow = neuron_size[i - 1], ncol = neuron_size[i])))
}
pf_in <- file("TrainX.dat", "rb")
pf_out <- file("TrainY.dat", "rb")
n <- readBin(pf_in, integer(), 1)[1]
m <- readBin(pf_in, integer(), 1)[1]

n = readBin(pf_out, integer(), 2)[1]
avg = 0
tot = 0
for (ii in 1:n) {
    learning_rate <- 0.009 * (n - ii) / n / 3 + 0.002
    vec_in <- c(readBin(pf_in, double(), m) , 1)
    ac <- list(t(as.matrix(vec_in)))
    for (i in 2:layer_size) {
    	ac_next <- ac[[i - 1]] %*% synapse[[i - 1]]
	rectifier <- function(x) { (abs(x) + x) / 2}
	ac_next <- rectifier(ac_next)
	ac_next[neuron_size[i]] <- 1
   	ac <- c(ac, list(ac_next))
    }

    result <- ac[[layer_size]][1:10]
    result <- exp(result)
    normalizer <- function(x) { x / sum(x) }
    result <- normalizer(result)
    error <- readBin(pf_out, double(), 10) 
    result <- result - error
    avg <- (avg * (ii - 1) + sum(error * result)) / ii
    error <- result
    if (ii %% 100 == 0) print(avg)
    if (is.nan(avg)) break
    error <- as.matrix(c(error, 0))    

    for (i in layer_size:2) {
	delta <- t(error %*% ac[[i - 1]])
	delta <- delta + synapse[[i - 1]] * 0.0001
	new_error = t(error) %*% t(synapse[[i - 1]]) 
	synapse[[i - 1]] <- synapse[[i - 1]] - delta * learning_rate
	for (j in 1:ncol(new_error)) {
	    if (ac[[i - 1]][j] == 0) new_error[j] = 0
	}
	error = new_error
    	error[neuron_size[i]] = 0
	error = t(error)
    }
}