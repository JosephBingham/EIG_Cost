# Takes a list of csv files with soft budget evaluation results and combines them into a single csv file.
# Every csv file must follow the format described in docs/dataframe_formats.md (soft budget case).
# Usage: Rscript combine_soft_budget_results.R file1.csv file2.csv combined.csv

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript combine_soft_budget_results.R file1.csv file2.csv ... combined.csv")
}

library(dplyr)

output_path <- args[length(args)]
input_files <- args[-length(args)]

# Required columns for soft budget results
required_cols <- c(
  "method",
  "training_seed",
  "cost_parameter",
  "dataset",
  "features_chosen",
  "predicted_label_builtin",
  "predicted_label_external",
  "true_label"
)

dfs <- lapply(input_files, function(f) {
  df <- read.csv(f, stringsAsFactors = FALSE)
  missing <- setdiff(required_cols, colnames(df))
  if (length(missing) > 0) {
    stop(sprintf("File %s is missing columns: %s", f, paste(missing, collapse = ", ")))
  }
  df
})

# Stack by rows
combined <- do.call(rbind, dfs)

write.csv(combined, file = output_path, row.names = FALSE)
