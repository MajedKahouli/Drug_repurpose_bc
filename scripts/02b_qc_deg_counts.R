# scripts/02b_qc_deg_counts.R
# Quick QC on per-patient DEGs with stricter thresholds

args <- commandArgs(trailingOnly = TRUE)
deg_dir <- ifelse(length(args) >= 1, args[1], "work/degs_per_patient")

files <- list.files(deg_dir, pattern="^DEGs_.*\\.tsv$", full.names=TRUE)
if (length(files) == 0) stop("No DEG files found in ", deg_dir)

qc <- data.frame(patient_id=character(),
                 sig_FDR0.05=integer(),
                 sig_FDR0.05_LFC0.5=integer(),
                 stringsAsFactors = FALSE)

for (f in files) {
  d <- read.delim(f, stringsAsFactors=FALSE)
  pid <- sub("^DEGs_(.*)\\.tsv$", "\\1", basename(f))
  n1 <- sum(d$adj.P.Val < 0.05, na.rm=TRUE)
  n2 <- sum(d$adj.P.Val < 0.05 & abs(d$logFC) >= 0.5, na.rm=TRUE)
  qc <- rbind(qc, data.frame(patient_id=pid,
                             sig_FDR0.05=n1,
                             sig_FDR0.05_LFC0.5=n2))
}

out <- file.path(deg_dir, "DEG_QC_counts.tsv")
write.table(qc, out, sep="\t", row.names=FALSE, quote=FALSE)
cat("[OK] Wrote stricter QC summary ->", out, "\n")

# print first few for sanity
print(head(qc))
