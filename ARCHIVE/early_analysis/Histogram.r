# Indoor Walk Test - Time Series Histogram & Data Prep
#
# Reads the raw NR FR1 "Top N Signal" scan CSVs (per-measurement, not
# coordinate-averaged), georeferences each point onto the floor plan using
# the same affine fit as the Python notebook, and stamps every row with
# elapsed walk time (t_sec) computed from System Date + System Time.
#
# Outputs:
#   - timeseries_data.js         per-point JSON consumed by Frontend_Data_Display.html
#   - Histogram_Metric_Distributions.png
#   - Histogram_Measurements_Over_Time.png

required_packages <- c("jsonlite", "ggplot2")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}
library(jsonlite)
library(ggplot2)

ROOT <- getwd()
CSV_DIR <- file.path(ROOT, "CSV")
TAB_PATH <- file.path(ROOT, "7th_Floor_2nd_Indoor_Walk_Test_V2.2.TAB")
OUT_JS <- file.path(ROOT, "timeseries_data.js")

UNITS <- c(rsrp = "dBm", rsrq = "dB", cinr = "dB", rssi = "dBm")

# ---- Georeferencing (mirrors methods_floor_analysis.ipynb) ----------------

read_tab_gcps <- function(tab_path) {
  lines <- readLines(tab_path, warn = FALSE)
  pattern <- "\\((-?[0-9.]+),(-?[0-9.]+)\\)\\s*\\(([0-9]+),([0-9]+)\\)"
  hits <- regmatches(lines, regexec(pattern, lines))
  hits <- hits[lengths(hits) == 5]
  rows <- lapply(hits, function(h) as.numeric(h[2:5]))
  gcps <- as.data.frame(do.call(rbind, rows))
  colnames(gcps) <- c("longitude", "latitude", "px", "py")
  gcps
}

fit_affine <- function(gcps) {
  model_x <- lm(px ~ longitude + latitude, data = gcps)
  model_y <- lm(py ~ longitude + latitude, data = gcps)
  rmse <- sqrt(mean(residuals(model_x)^2 + residuals(model_y)^2))
  cat(sprintf("affine fit rmse: %.2f px over %d GCPs\n", rmse, nrow(gcps)))
  list(model_x = model_x, model_y = model_y)
}

add_pixel_coords <- function(df, affine) {
  df$px <- predict(affine$model_x, newdata = df)
  df$py <- predict(affine$model_y, newdata = df)
  df
}

# ---- Time parsing -----------------------------------------------------------
# System Time looks like "14:02:01:622" (H:M:S:millis). Combine with
# System Date to get an absolute POSIXct so elapsed time reflects the true
# order measurements were collected across separate band-scan files.

parse_datetime <- function(date_str, time_str) {
  parts <- strsplit(time_str, ":")
  seconds_of_day <- vapply(parts, function(p) {
    if (length(p) < 4 || any(is.na(suppressWarnings(as.numeric(p[1:4]))))) return(NA_real_)
    h <- as.numeric(p[1]); m <- as.numeric(p[2]); s <- as.numeric(p[3]); ms <- as.numeric(p[4])
    h * 3600 + m * 60 + s + ms / 1000
  }, numeric(1))
  date_base <- as.POSIXct(strptime(date_str, format = "%m/%d/%Y", tz = "UTC"))
  date_base + seconds_of_day
}

# ---- Read + reduce NR Top N Signal CSVs ------------------------------------

read_nr_topn <- function(path) {
  df <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  out <- data.frame(
    latitude = as.numeric(df[["Latitude"]]),
    longitude = as.numeric(df[["Longitude"]]),
    pci = as.numeric(df[["Cell ID"]]),
    freq = as.numeric(df[["Channel Frequency"]]),
    band = as.numeric(df[["Band"]]),
    rsrp = as.numeric(df[["SSS_RP"]]),
    rsrq = as.numeric(df[["SSS_RQ"]]),
    cinr = as.numeric(df[["SS_CINR"]]),
    rssi = as.numeric(df[["SSB RSSI"]]),
    datetime = parse_datetime(df[["System Date"]], df[["System Time"]]),
    source = tools::file_path_sans_ext(basename(path)),
    stringsAsFactors = FALSE
  )
  out$network <- "nr_fr1"
  out
}

csv_files <- list.files(CSV_DIR, pattern = "NR_FR1.*nr Top N Signal.*\\.CSV$", full.names = TRUE)
csv_files <- csv_files[!grepl("_BestServing", csv_files)]
stopifnot(length(csv_files) > 0)

records_df <- do.call(rbind, lapply(csv_files, read_nr_topn))
records_df <- records_df[stats::complete.cases(records_df[, c("latitude", "longitude", "datetime")]), ]
records_df <- records_df[order(records_df$datetime), ]

t0 <- min(records_df$datetime)
records_df$t_sec <- as.numeric(difftime(records_df$datetime, t0, units = "secs"))

gcps <- read_tab_gcps(TAB_PATH)
affine <- fit_affine(gcps)
records_df <- add_pixel_coords(records_df, affine)

# ---- Export per-point JSON for the HTML frontend --------------------------

export_cols <- c("network", "band", "pci", "freq", "rsrp", "rsrq", "cinr", "rssi",
                  "latitude", "longitude", "px", "py", "t_sec")
export_df <- records_df[, export_cols]
# Rounding keeps the exported JSON file (thousands of rows) from ballooning
# with full float64 precision that the dashboard has no use for.
export_df[] <- lapply(export_df, function(col) if (is.numeric(col)) round(col, 4) else col)

json_text <- toJSON(export_df, dataframe = "rows", auto_unbox = TRUE, na = "null")
# timeseriesStartTime is the wall-clock time of the first measurement (t_sec = 0);
# the HTML frontend adds each row's t_sec to this to show real clock time in
# the Time Elapsed Playback control (military/standard toggle).
start_iso <- strftime(t0, "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC")
writeLines(c(
  paste0("const timeseriesStartTime = \"", start_iso, "\";"),
  paste0("const timeseriesRecords = ", json_text, ";")
), con = OUT_JS)
cat(sprintf("wrote %s (%d rows, %.1f min span)\n", OUT_JS, nrow(export_df), max(export_df$t_sec) / 60))

# ---- Histogram 1: signal metric distributions ------------------------------

metrics <- c("rsrp", "rsrq", "cinr", "rssi")
long_rows <- lapply(metrics, function(m) {
  data.frame(metric = toupper(m), value = export_df[[m]], unit = UNITS[[m]])
})
long_df <- do.call(rbind, long_rows)
long_df <- long_df[is.finite(long_df$value), ]

p_metric <- ggplot(long_df, aes(x = value)) +
  geom_histogram(bins = 40, fill = "#0a7f7a", color = "white") +
  facet_wrap(~ metric, scales = "free", ncol = 2) +
  labs(title = "Signal Metric Distributions - NR FR1",
       subtitle = sprintf("%d measurements across bands %s",
                           nrow(export_df),
                           paste(sort(unique(export_df$band)), collapse = ", ")),
       x = NULL, y = "Measurement count") +
  theme_minimal(base_size = 12)

ggsave(file.path(ROOT, "Histogram_Metric_Distributions.png"), p_metric, width = 10, height = 7, dpi = 150)

# ---- Histogram 2: measurements collected over elapsed walk time -----------

p_time <- ggplot(export_df, aes(x = t_sec / 60)) +
  geom_histogram(binwidth = 1, fill = "#3d6fae", color = "white") +
  labs(title = "Measurements Collected Over Elapsed Walk Time",
       subtitle = sprintf("Total span: %.1f minutes", max(export_df$t_sec) / 60),
       x = "Elapsed time (minutes)", y = "Measurements per minute") +
  theme_minimal(base_size = 12)

ggsave(file.path(ROOT, "Histogram_Measurements_Over_Time.png"), p_time, width = 10, height = 5, dpi = 150)

# Quick sanity check when running interactively in RStudio: sample count and
# average RSRP per band, to eyeball that nothing dropped out unexpectedly.
cat("\nSummary:\n")
print(aggregate(rsrp ~ band, data = export_df, FUN = function(x) c(n = length(x), mean = mean(x, na.rm = TRUE))))
