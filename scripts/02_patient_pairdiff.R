# scripts/02_patient_pairdiff.R
# Per-patient paired differences (tumor - normal) with z-scores vs normal-sample variability
# Usage:
#   Rscript scripts/02_patient_pairdiff.R \
#     --expr C:/Projects/drug-repurpose-bc/data/geo/gse15852/expr_gene_level.tsv \
#     --meta C:/Projects/drug-repurpose-bc/data/geo/gse15852/sample_sheet_fixed.csv \
#     --out  C:/Projects/drug-repurpose-bc/work/degs_per_patient

# --- tiny args parser ---
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default=NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0 || hit == length(args)) return(default)
  args[hit + 1]
}
expr_path <- get_arg("--expr")
meta_path <- get_arg("--meta")
out_dir   <- get_arg("--out")
stopifnot(!is.null(expr_path), !is.null(meta_path), !is.null(out_dir))

# --- read expression (genes x samples) ---
X <- read.table(expr_path, sep = "\t", header = TRUE, check.names = FALSE,
                quote = "", comment.char = "", stringsAsFactors = FALSE)
# first column is gene symbol -> rownames
rownames(X) <- X[[1]]
X[[1]] <- NULL
X <- as.matrix(data.frame(lapply(X, function(col) as.numeric(as.character(col))),
                          row.names = rownames(X)))
storage.mode(X) <- "double"

# --- read metadata ---
meta <- read.csv(meta_path, stringsAsFactors = FALSE)
colnames(meta) <- tolower(colnames(meta))
meta$sample_id  <- trimws(meta$sample_id)
meta$condition  <- tolower(trimws(meta$condition))
meta$patient_id <- trimws(meta$patient_id)

# keep samples present in expression
meta <- meta[meta$sample_id %in% colnames(X), ]
X    <- X[, meta$sample_id, drop = FALSE]

# check pairing
tab <- table(meta$patient_id, meta$condition)
have_norm <- if ("normal" %in% colnames(tab)) tab[, "normal"] else 0
have_tum  <- if ("tumor"  %in% colnames(tab)) tab[, "tumor"]  else 0
valid_patients <- rownames(tab)[(have_norm == 1) & (have_tum == 1)]
valid_patients <- valid_patients[!is.na(valid_patients) & valid_patients != ""]
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("[INFO] Valid paired patients:", length(valid_patients), "\n")

# --- gene-wise variability from normals (across all patients) ---
normal_samples <- meta$sample_id[meta$condition == "normal"]
if (length(normal_samples) < 2) {
  stop("Need at least 2 normal samples to estimate variability; found ", length(normal_samples))
}
Xnorm <- X[, normal_samples, drop = FALSE]
# robust-ish SD: standard deviation across normal samples per gene
sd_norm <- apply(Xnorm, 1, sd, na.rm = TRUE)
# epsilon to avoid division by zero
eps <- 1e-6
sd_norm[is.na(sd_norm)] <- median(sd_norm, na.rm = TRUE)
sd_norm[sd_norm < eps]  <- eps

summary_path <- file.path(out_dir, "DEG_summary.tsv")
summary_rows <- list()

for (pid in valid_patients) {
  sub <- meta[meta$patient_id == pid, ]
  s_norm <- sub$sample_id[sub$condition == "normal"][1]
  s_tum  <- sub$sample_id[sub$condition == "tumor"][1]
  if (!(s_norm %in% colnames(X) && s_tum %in% colnames(X))) next

  v_norm <- X[, s_norm]
  v_tum  <- X[, s_tum]
  logFC  <- v_tum - v_norm

  # z, p, FDR using Normal approximation
  z      <- as.numeric(logFC / sd_norm)
  pval   <- 2 * pnorm(-abs(z))
  fdr    <- p.adjust(pval, method = "BH")

  out <- data.frame(
    gene   = rownames(X),
    logFC  = logFC,
    z      = z,
    P.Value    = pval,
    adj.P.Val  = fdr,
    stringsAsFactors = FALSE
  )

  out_file <- file.path(out_dir, paste0("DEGs_", pid, ".tsv"))
  write.table(out, out_file, sep = "\t", row.names = FALSE, quote = FALSE)
  cat("[OK]", pid, ": wrote", nrow(out), "genes ->", out_file, "\n")

  summary_rows[[length(summary_rows)+1]] <- data.frame(
    patient_id = pid,
    genes_tested = nrow(out),
    sig_deg_FDR_0_05 = sum(out$adj.P.Val < 0.05, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}

if (length(summary_rows) > 0) {
  summary_df <- do.call(rbind, summary_rows)
  write.table(summary_df, summary_path, sep = "\t", row.names = FALSE, quote = FALSE)
  cat("[OK] Summary ->", summary_path, "\n")
} else {
  cat("[WARN] No outputs written; check pairing in sample sheet.\n")
}
