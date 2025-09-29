# scripts/02_patient_deg_limma.R
# Per-patient paired DEGs with limma (tumor vs normal)
# Usage:
#   Rscript scripts/02_patient_deg_limma.R \
#     --expr C:/Projects/drug-repurpose-bc/data/geo/gse15852/expr_gene_level.tsv \
#     --meta C:/Projects/drug-repurpose-bc/data/geo/gse15852/sample_sheet.csv \
#     --out  C:/Projects/drug-repurpose-bc/work/degs_per_patient

suppressPackageStartupMessages({
  library(limma)
})

# --- tiny args parser (no extra deps) ---
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0 || hit == length(args)) return(default)
  args[hit + 1]
}
expr_path <- get_arg("--expr")
meta_path <- get_arg("--meta")
out_dir   <- get_arg("--out")

stopifnot(!is.null(expr_path), !is.null(meta_path), !is.null(out_dir))

# --- load expression (genes x samples) ---
X <- read.table(expr_path, sep = "\t", header = TRUE, check.names = FALSE, quote = "", comment.char = "", stringsAsFactors = FALSE)
# first column should be gene symbols -> rownames
if (tolower(colnames(X)[1]) %in% c("gene", "genes", "gene_symbol", "symbol", "")) {
  rownames(X) <- X[[1]]
  X[[1]] <- NULL
} else {
  # still treat first column as rownames
  rownames(X) <- X[[1]]
  X[[1]] <- NULL
}
# coerce to numeric
X <- as.matrix(data.frame(lapply(X, function(col) as.numeric(as.character(col))), row.names = rownames(X)))
storage.mode(X) <- "double"

# --- load metadata ---
meta <- read.csv(meta_path, stringsAsFactors = FALSE)
needed <- c("sample_id","condition","patient_id")
missing_cols <- setdiff(needed, tolower(colnames(meta)))
if (length(missing_cols) > 0) {
  stop("Meta must contain columns: sample_id, condition, patient_id (missing: ", paste(missing_cols, collapse=", "), ")")
}
# normalize colnames to lower
colnames(meta) <- tolower(colnames(meta))
meta$condition <- tolower(trimws(meta$condition))
meta$patient_id <- trimws(meta$patient_id)
meta$sample_id  <- trimws(meta$sample_id)

# keep only samples present in expression
meta <- meta[meta$sample_id %in% colnames(X), ]
X    <- X[, meta$sample_id, drop = FALSE]

# only patients with exactly one tumor + one normal
tab <- table(meta$patient_id, meta$condition)
valid_patients <- rownames(tab)[apply(tab[, c("normal","tumor"), drop = FALSE], 1, function(v) all(c("normal","tumor") %in% names(v)) && all(v[c("normal","tumor")] == 1))]
valid_patients <- valid_patients[!is.na(valid_patients) & valid_patients != ""]
if (length(valid_patients) == 0) stop("No valid paired patients found (need exactly one tumor and one normal per patient).")

# prepare out dir
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("[INFO] Patients with clean pairs:", length(valid_patients), "\n")

per_patient_summary <- data.frame(patient_id=character(), genes_tested=integer(), sig_deg_FDR_0_05=integer(), stringsAsFactors = FALSE)

for (pid in valid_patients) {
  sub <- meta[meta$patient_id == pid, ]
  # order: normal first, tumor second (so contrast is tumor - normal)
  if (!all(c("normal","tumor") %in% sub$condition)) next
  s_norm <- sub$sample_id[sub$condition == "normal"][1]
  s_tum  <- sub$sample_id[sub$condition == "tumor"][1]
  cols   <- c(s_norm, s_tum)
  cols   <- cols[cols %in% colnames(X)]
  if (length(cols) != 2) next

  Y <- X[, cols, drop = FALSE]

  group <- factor(c("normal","tumor"), levels = c("normal","tumor"))
  design <- model.matrix(~ 0 + group)                 # columns: groupnormal, grouptumor
  colnames(design) <- c("normal","tumor")
  contrast.mat <- makeContrasts(tumor - normal, levels = design)

  fit <- lmFit(Y, design)
  fit2 <- contrasts.fit(fit, contrast.mat)
  fit2 <- eBayes(fit2)

  tt <- topTable(fit2, coef = 1, number = Inf, sort.by = "none")
  tt$gene <- rownames(tt)
  tt <- tt[, c("gene","logFC","AveExpr","t","P.Value","adj.P.Val","B")]

  out_file <- file.path(out_dir, paste0("DEGs_", pid, ".tsv"))
  write.table(tt, out_file, sep = "\t", row.names = FALSE, quote = FALSE)
  cat("[OK]", pid, ": wrote", nrow(tt), "genes ->", out_file, "\n")

  per_patient_summary <- rbind(per_patient_summary,
                               data.frame(patient_id = pid,
                                          genes_tested = nrow(tt),
                                          sig_deg_FDR_0_05 = sum(tt$adj.P.Val < 0.05, na.rm = TRUE),
                                          stringsAsFactors = FALSE))
}

# write a small summary
summary_path <- file.path(out_dir, "DEG_summary.tsv")
write.table(per_patient_summary, summary_path, sep = "\t", row.names = FALSE, quote = FALSE)
cat("[OK] Summary ->", summary_path, "\n")
