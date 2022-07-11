# data generation from bn object as from .rda files (cf. https://www.bnlearn.com/bnrepository/)

# project folder, this file is supposed to be in first level of sub-folder 
setwd("~/Desktop/csl_sage/scripts/csl-experiments")

# create folders to store adjacency matrices, folder data supposed to exist
# dir.create("data")
dir.create("data/true_amat")

# install package if necessary
# install.packages('bnlearn')

# load package 'bnlearn'
library('bnlearn')

# set seed
set.seed(1902)

names <- c("alarm", "asia", "hepar", "sachs")

for (i in names){
  
  env_name <- paste("data/env/", i, ".rda", sep="")
  load(env_name)

  data <- rbn(bn, n=1000)
  
  filename <- paste("data/", i, ".csv", sep="")

  write.csv(data, filename, row.names = FALSE)
    
  adj_mat <- amat(bn)
  amat_file <- paste("data/true_amat/", i, ".csv", sep="")
  write.csv(adj_mat, file=amat_file, row.names = FALSE)
}
