library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)

# Which datasets we want to plot F1 for instead of accuracy
f1_datasets <- c("dataset_B")

results_path <- "eval_results.csv"
plot_path <- "plot.png"

# Specify your expected columns and types
expected_types <- cols(
  method = col_factor(),
  training_seed = col_integer(),
  cost_parameter = col_double(),
  dataset = col_factor(),
  dataset_split = col_integer(),
  features_chosen = col_integer(),
  acquisition_cost = col_double(),
  predicted_label_builtin = col_integer(),
  predicted_label_external = col_integer(),
  true_label = col_integer()
)

results <- read_csv(results_path, col_types = expected_types)

# Check for unspecified columns
unspecified_cols <- setdiff(names(results), names(expected_types$cols))
if (length(unspecified_cols) > 0) {
  stop(
    paste(
      "The following columns were not specified in col_types:",
      paste(unspecified_cols, collapse = ", ")
    )
  )
}

# Tidy data
df <- results %>%
  pivot_longer(
    cols = c(predicted_label_external, predicted_label_builtin),
    names_to = "prediction_type",
    names_prefix = "predicted_label_",
    values_to = "predicted_label"
  ) %>%
  filter(!is.na(predicted_label)) # Remove rows where builtin is NA


# Summarize accuracy and features chosen
df_summarized <- df %>%
  mutate(
    correct = (predicted_label == true_label),
    tp = (predicted_label == 1 & true_label == 1),
    fp = (predicted_label == 1 & true_label == 0),
    tn = (predicted_label == 0 & true_label == 0),
    fn = (predicted_label == 0 & true_label == 1)
  ) %>%
  group_by(method, prediction_type, dataset, dataset_split, cost_parameter, training_seed) %>%
  summarize(
    accuracy = mean(correct),
    avg_features_chosen = mean(features_chosen),
    avg_acquisition_cost = mean(acquisition_cost),
    tp = sum(tp),
    fp = sum(fp),
    tn = sum(tn),
    fn = sum(fn),
    precision = ifelse(tp + fp == 0, 0, tp / (tp + fp)),
    recall = ifelse(tp + fn == 0, 0, tp / (tp + fn)),
    f1 = ifelse(precision + recall == 0, 0, 2 * (precision * recall) / (precision + recall)),
    .groups = "keep"
  )

# Some datasets use accuracy, others use F1
df_summarized <- df_summarized %>%
  mutate(
    metric_type = ifelse(dataset %in% f1_datasets, "f1", "accuracy"),
    metric_value = ifelse(metric_type == "f1", f1, accuracy)
  )

# Mean and sd over dataset splits and training seeds
df_summary <- df_summarized %>%
  group_by(method, prediction_type, dataset, cost_parameter, metric_type) %>%
  summarize(
    avg_metric = mean(metric_value),
    sd_metric = sd(metric_value),
    mean_avg_features_chosen = mean(avg_features_chosen),
    sd_avg_features_chosen = sd(avg_features_chosen),
    mean_avg_acquisition_cost = mean(avg_acquisition_cost),
    sd_avg_acquisition_cost = sd(avg_acquisition_cost),
    .groups = "drop"
  )

df_labels <- df_summary %>%
  group_by(dataset) %>%
  summarize(
    metric_label = ifelse(first(metric_type) == "f1", "F1", "Accuracy"),
    x = min(mean_avg_acquisition_cost),
    # x = min(mean_avg_features_chosen),
    y = 1
  )

# Create the plot
p <- ggplot(df_summary, aes(
  # x = mean_avg_features_chosen,
  x = mean_avg_acquisition_cost,
  y = avg_metric,
  color = method
)) +
  geom_point() +
  geom_line() +
  geom_errorbar(
    aes(
      ymin = avg_metric - sd_metric,
      ymax = avg_metric + sd_metric
    ),
    width = 0
  ) +
  geom_errorbarh(
    aes(
      xmin = mean_avg_acquisition_cost - sd_avg_acquisition_cost,
      xmax = mean_avg_acquisition_cost + sd_avg_acquisition_cost
    ),
    height = 0
  ) +
  geom_text(
    data = df_labels,
    aes(x = x, y = y, label = metric_label),
    inherit.aes = FALSE,
    hjust = 0, vjust = 1, fontface = "bold"
  ) +
  facet_grid(rows = vars(prediction_type), cols = vars(dataset), scales = "free_x") +
  coord_cartesian(ylim = c(0, 1)) +
  labs(
    title = "Metric vs avg. acquisition cost",
    x = "Avg. acquisition cost",
    y = "Metric",
  ) +
  theme_bw()
p

# Save the plot
ggsave(plot_path, p, width = 10, height = 6, dpi = 300)

auc1 <- df_summary %>%
  group_by(method, dataset, metric_type) %>%
  summarise(
    auc = {
      df_sorted <- arrange(pick(everything()), mean_avg_features_chosen)
      x <- df_sorted$mean_avg_features_chosen
      y <- df_sorted$avg_metric
      sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
    },
    .groups = "drop"
  )

# Plot of n_feature_chosen vs cost_param
ggplot(
  df_summary,
  aes(x = mean_avg_features_chosen, y = cost_parameter)
) +
  geom_point() +
  facet_grid(rows = vars(method), cols = vars(dataset), scales = "free")
