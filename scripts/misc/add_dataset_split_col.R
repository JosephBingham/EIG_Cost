#!/usr/bin/env Rscript

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript add_dataset_split_col.R soft_eval_data.csv soft_eval_data_with_split.csv 3")
}

library(ggplot2)
library(dplyr)
library(readr)

input_path <- args[1]
output_path <- args[2]
dataset_split <- as.integer(args[3])

input_df <- read_csv(input_path)
input_df$dataset_split <- dataset_split
write_csv(input_df, output_path)
