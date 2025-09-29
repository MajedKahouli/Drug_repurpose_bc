#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(stringr)
  library(ggplot2)
})

say <- function(...) cat(format(Sys.time(), "%H:%M:%S"), "-", ..., "\n")

# -------- parse args --------
opt_list <- list(
  make_option("--deg_dir",      type="character", help="Folder with DEGs_*.tsv"),
  make_option("--expr_matrix",  type="character", default=NA,
              help="(Optional) expression matrix TSV for QC"),
  make_option("--sample_sheet", type="character", default=NA,
              help="(Optional) sample sheet CSV/TSV"),
  make_option("--expr_cutoff",  type="double", default=1,
              help="(Optional) CPM cutoff (only reported)"),
  make_option("--outdir",       type="character", default="qc_out",
              help="Output directory [default %default]")
)
opt <- parse_args(OptionParser(option_list = opt_list))

if (is.null(opt$deg_dir)) stop("--deg_dir is required")
dir.create(opt$outdir, showWarnings = FALSE, recursive = TRUE)

say("deg_dir     :", opt$deg_dir)
say("expr_matrix :", opt$expr_matrix)
say("sample_sheet:", opt$sample_sheet)
say("outdir      :", opt$outdir)

# -------- read DEGs --------
files <- list.files(opt$deg_dir, pattern="^DEGs_.*\\.tsv$", full.names=TRUE)
if (!length(files)) stop("No DEG files found in: ", opt$deg_dir)

say("Found", length(files), "DEG files")

read_deg <- function(f) {
  # expect columns like: gene, logFC, direction OR a column that lets us sign
  df <- suppressMessages(readr::read_tsv(f, show_col_types = FALSE))
  df$.file <- basename(f)
  # Infer direction if not present: sign(logFC)
  if (!("direction" %in% names(df))) {
    if ("logFC" %in% names(df)) {
      df$direction <- ifelse(df$logFC > 0, "up",
                        ifelse(df$logFC < 0, "down", "zero"))
    } else {
      stop("DEG file lacks 'direction' and 'logFC': ", f)
    }
  }
  df
}

all_deg <- dplyr::bind_rows(lapply(files, read_deg))

# patient id from filename: DEGs_<PATIENT>.tsv
all_deg$patient_id <- str_replace(all_deg$.file, "^DEGs_(.*)\\.tsv$", "\\1")

# summarize per patient
deg_counts <- all_deg %>%
  mutate(direction = ifelse(direction %in% c("up","down"), direction, "zero")) %>%
  count(patient_id, direction) %>%
  tidyr::pivot_wider(names_from = direction, values_from = n, values_fill = 0) %>%
  mutate(n_total = up + down) %>%
  arrange(patient_id)

out_tsv <- file.path(opt$outdir, "deg_counts.tsv")
readr::write_tsv(deg_counts, out_tsv)
say("Wrote:", out_tsv)

# -------- plot distribution (no cairo needed) --------
plot_path <- file.path(opt$outdir, "qc_gene_distribution.png")
save_png <- function(path, expr) {
  if (requireNamespace("ragg", quietly = TRUE)) {
    ragg::agg_png(path, width = 1800, height = 1200, res = 200)
    on.exit(grDevices::dev.off(), add = TRUE)
    eval(expr)
  } else {
    png(path, width = 1800, height = 1200, res = 200)
    on.exit(grDevices::dev.off(), add = TRUE)
    eval(expr)
  }
}

say("Creating plot:", plot_path)
save_png(plot_path, quote({
  p <- ggplot(deg_counts, aes(x = n_total)) +
    geom_histogram(binwidth = max(1, floor(diff(range(n_total))/30)),
                   color = "grey30", fill = "grey75") +
    geom_vline(aes(xintercept = median(n_total)), linetype = 2) +
    labs(x = "Total DEGs per patient", y = "Count of patients",
         title = "Distribution of per-patient DEGs") +
    theme_minimal(base_size = 14)
  print(p)
}))
say("Wrote:", plot_path)

# -------- optional notes if expr/sample provided --------
if (!is.na(opt$expr_matrix)) {
  say("Note: expr_matrix provided -> record cutoff to text file")
  writeLines(
    c(paste0("expr_matrix: ", opt$expr_matrix),
      paste0("sample_sheet: ", opt$sample_sheet),
      paste0("expr_cutoff: ", opt$expr_cutoff)),
    file.path(opt$outdir, "filtering_summary.txt")
  )
  say("Wrote:", file.path(opt$outdir, "filtering_summary.txt"))
}

say("Done.")
