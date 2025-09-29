# qc_gene_qc_min.R  -- base R only

say <- function(...) cat(format(Sys.time(), "%H:%M:%S"), "-", ..., "\n")

# --- tiny arg parser (base R) ---
args <- commandArgs(trailingOnly = TRUE)
kv <- strsplit(args, "=")
arglist <- list()
for (x in kv) {
  if (length(x) == 2) arglist[[sub("^--", "", x[[1]])]] <- x[[2]]
}

required <- c("deg_dir")
for (k in required) if (is.null(arglist[[k]])) stop("Missing --", k)

deg_dir     <- arglist$deg_dir
expr_matrix <- arglist$expr_matrix %||% NA
sample_sheet<- arglist$sample_sheet %||% NA
expr_cutoff <- as.numeric(arglist$expr_cutoff %||% "1")
outdir      <- arglist$outdir %||% "qc_out"

`%||%` <- function(a,b) if (is.null(a)) b else a

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

say("deg_dir     :", deg_dir)
say("expr_matrix :", expr_matrix)
say("sample_sheet:", sample_sheet)
say("expr_cutoff :", expr_cutoff)
say("outdir      :", outdir)

# --- read DEGs ---
files <- list.files(deg_dir, pattern="^DEGs_.*\\.tsv$", full.names=TRUE)
if (!length(files)) stop("No DEG files found in ", deg_dir)

say("Found", length(files), "DEG files")

read_deg <- function(f) {
  df <- try(read.delim(f, check.names = FALSE), silent = TRUE)
  if (inherits(df, "try-error")) stop("Cannot read: ", f)
  # Expect either a 'direction' column or a 'logFC' column
  if (!("direction" %in% names(df))) {
    if ("logFC" %in% names(df)) {
      df$direction <- ifelse(df$logFC > 0, "up",
                        ifelse(df$logFC < 0, "down", "zero"))
    } else {
      stop("File lacks 'direction' and 'logFC': ", f)
    }
  }
  df$patient_id <- sub("^DEGs_(.*)\\.tsv$", "\\1", basename(f))
  df
}

all <- do.call(rbind, lapply(files, read_deg))

# count per patient
tab <- aggregate(list(n = all$direction),
                 by = list(patient_id = all$patient_id, direction = all$direction),
                 FUN = length)

# spread to wide (up/down/zero)
dirs <- c("up","down","zero")
wide <- data.frame(patient_id = unique(all$patient_id), up=0, down=0, zero=0)
for (i in seq_len(nrow(tab))) {
  r <- tab[i,]
  if (r$direction %in% dirs) {
    wide[wide$patient_id == r$patient_id, as.character(r$direction)] <-
      wide[wide$patient_id == r$patient_id, as.character(r$direction)] + r$n
  }
}
wide$n_total <- wide$up + wide$down
wide <- wide[order(wide$patient_id), ]

# write TSV
out_tsv <- file.path(outdir, "deg_counts.tsv")
write.table(wide, out_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
say("Wrote:", out_tsv)

# plot (base graphics only)
png(file.path(outdir, "qc_gene_distribution.png"), width=1800, height=1200, res=200)
hist(wide$n_total, breaks="FD", col="grey80", border="grey40",
     main="Distribution of per-patient DEGs", xlab="Total DEGs per patient")
abline(v = median(wide$n_total, na.rm=TRUE), lty=2)
dev.off()
say("Wrote:", file.path(outdir, "qc_gene_distribution.png"))

# optional note file
if (!is.na(expr_matrix)) {
  sink(file.path(outdir, "filtering_summary.txt"))
  cat("expr_matrix:", expr_matrix, "\n")
  cat("sample_sheet:", sample_sheet, "\n")
  cat("expr_cutoff:", expr_cutoff, "\n")
  sink()
  say("Wrote:", file.path(outdir, "filtering_summary.txt"))
}

say("Done.")
