# Generating conditionally linear Gaussian graphs and sampling synthetic data in accordance to them

# project folder, this file is supposed to be in first level of project folder
# setwd("~/csl-experiments")

# create folders to store results about DGPs and adjacency matrices
dir.create("data")
dir.create("data/dgps")
dir.create("data/true_amat")

# install packages if necessary 
#if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
#BiocManager::install("graph")
#install.packages('pcalg')

# load package 'pcalg'
library(pcalg)

# set seed
set.seed(44)

# number of nodes for small to large graph
size <- c(10, 20, 50, 100)

# probabilities for each pair of nodes to share an edge
prob_i <- c(2/9, 2/19, 2/49, 2/99)
prob_ii <- c(3/9, 3/19, 3/49, 3/99)
prob_iii <- c(4/9, 4/19, 4/49, 4/99)
prob_iv <- c(5/9, 5/19, 5/49, 5/99)
prob_v <- c(6/9, 6/19, 6/49, 6/99)

# names of graphs (add prob to graph name)
size_tokens <- c("s", "sm", "m", "l")

# sample size of sampled data sets
n <- 1000000


for (i in c(1:3)){
  probs <- c(prob_i[i], prob_ii[i], prob_iii[i], prob_iv[i], prob_v[i])
  for (prob in probs){  
    d <- size[i]
    token <- size_tokens[i]
  
    # randomly generate DAG
    graph <- r.gauss.pardag(d, prob=prob, top.sort = FALSE, normalize = TRUE,
                            lbe = 0.1, ube = 1, neg.coef = TRUE, labels = as.character(1:d),
                            lbv = 0.5, ubv = 1)
  
    # retrieve and store info about DGP (weight matrix and error variance)
    proba <- round(prob,digits=5) 
    weights <- graph$weight.mat()
    error_var <- graph$err.var()
    weights_filename <- paste("data/dgps/dag_", token,"_", proba ,"_weights.csv", sep="")
    write.csv(weights, weights_filename, row.names = FALSE)
    variance_filename <- paste("data/dgps/dag_", token, "_", proba, "_error_var.csv", sep="")
    write.csv(error_var, variance_filename, row.names = FALSE)
  
    # retrieve Boolean adjacency matrix and store it
    amat <- as(graph, "matrix")
    amat_name <- paste("data/true_amat/dag_", token, "_", proba, ".csv", sep="")
    write.csv(amat, amat_name, row.names = FALSE)
  
    # create and store data
    data <- graph$simulate(n)
    filename <- paste("data/dag_", token, "_", proba, ".csv", sep="")
    write.csv(data, filename, row.names = FALSE)
  
  }
}
