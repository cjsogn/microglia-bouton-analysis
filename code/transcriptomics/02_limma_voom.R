#!/usr/bin/env Rscript

# Biological-sample differential expression for every prepared contrast.  TMM-normalized limma-voom
# is used for integer bulk/pseudobulk counts and for fractional Kallisto estimated counts.  FDR is
# calculated over all retained genes, never only over the selected panel.

suppressPackageStartupMessages({
    library(edgeR)
    library(limma)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
    stop("Usage: 02_limma_voom.R <contrast_manifest.tsv> <output_dir>")
}

manifest_path <- args[[1]]
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

manifest <- read.delim(manifest_path, stringsAsFactors = FALSE, check.names = FALSE)
qc_rows <- list()

for (i in seq_len(nrow(manifest))) {
    rec <- manifest[i, ]
    message(sprintf("[%d/%d] %s", i, nrow(manifest), rec$contrast_id))
    con <- if (grepl("\\.gz$", rec$counts_path))
        gzfile(rec$counts_path, "rt") else rec$counts_path
    counts_df <- read.delim(con, stringsAsFactors = FALSE, check.names = FALSE)
    if (inherits(con, "connection"))
        close(con)
    sample_df <- read.delim(rec$samples_path, stringsAsFactors = FALSE, check.names = FALSE)

    if (!"gene_id" %in% colnames(counts_df))
        stop("gene_id missing: ", rec$contrast_id)
    gene_id <- as.character(counts_df$gene_id)
    if (anyDuplicated(gene_id))
        stop("duplicate gene IDs: ", rec$contrast_id)
    counts <- as.matrix(counts_df[, setdiff(colnames(counts_df), "gene_id"), drop = FALSE])
    storage.mode(counts) <- "double"
    rownames(counts) <- gene_id
    if (anyDuplicated(sample_df$sample_id) || anyDuplicated(colnames(counts)) || !setequal(sample_df$sample_id,
        colnames(counts)))
        stop("sample mismatch: ", rec$contrast_id)
    counts <- counts[, sample_df$sample_id, drop = FALSE]
    if (any(!is.finite(counts)) || any(counts < 0))
        stop("invalid counts: ", rec$contrast_id)

    group <- factor(sample_df$group, levels = c("Control", "Treatment"))
    if (any(is.na(group)))
        stop("invalid groups: ", rec$contrast_id)
    design <- model.matrix(~0 + group)
    colnames(design) <- c("Control", "Treatment")
    rownames(design) <- sample_df$sample_id
    design_rank <- qr(design)$rank
    residual_df <- nrow(design) - design_rank
    if (design_rank != ncol(design))
        stop("rank-deficient design: ", rec$contrast_id)
    if (residual_df < 2)
        stop("fewer than two residual degrees of freedom: ", rec$contrast_id)

    y <- DGEList(counts = counts, samples = sample_df, genes = data.frame(gene_id = gene_id))
    keep <- filterByExpr(y, design = design, min.count = 5, min.total.count = 15)
    if (sum(keep) < 100)
        stop("too few genes retained: ", rec$contrast_id)
    y <- y[keep, , keep.lib.sizes = FALSE]
    y <- calcNormFactors(y, method = "TMM")
    v <- voom(y, design, plot = FALSE, save.plot = TRUE)
    diag_dir <- file.path(output_dir, "diagnostics", rec$contrast_id)
    dir.create(diag_dir, recursive = TRUE, showWarnings = FALSE)
    write.table(data.frame(gene_id = gene_id, retained = keep), file.path(diag_dir, "filtering.tsv"),
        sep = "\t", quote = FALSE, row.names = FALSE)
    write.table(data.frame(sample_id = rownames(design), design, check.names = FALSE), file.path(diag_dir,
        "design.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
    write.table(data.frame(sample_id = sample_df$sample_id, group = group, input_library_size = colSums(counts),
        retained_library_size = y$samples$lib.size, TMM_factor = y$samples$norm.factors, effective_library_size = y$samples$lib.size *
            y$samples$norm.factors), file.path(diag_dir, "sample_qc.tsv"), sep = "\t", quote = FALSE,
        row.names = FALSE)
    write.table(data.frame(x = v$voom.xy$x, y = v$voom.xy$y), file.path(diag_dir, "voom_points.tsv"),
        sep = "\t", quote = FALSE, row.names = FALSE)
    write.table(data.frame(x = v$voom.line$x, y = v$voom.line$y), file.path(diag_dir, "voom_trend.tsv"),
        sep = "\t", quote = FALSE, row.names = FALSE)
    pc <- prcomp(t(v$E), center = TRUE, scale. = FALSE)
    write.table(data.frame(sample_id = sample_df$sample_id, group = group, pc$x, check.names = FALSE),
        file.path(diag_dir, "sample_pca.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
    write.table(data.frame(component = seq_along(pc$sdev), variance_fraction = pc$sdev^2/sum(pc$sdev^2)),
        file.path(diag_dir, "pca_variance.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
    writeLines(capture.output(sessionInfo()), file.path(diag_dir, "session_info.txt"))
    fit <- lmFit(v, design)
    contrast <- makeContrasts(Treatment - Control, levels = design)
    fit <- contrasts.fit(fit, contrast)
    fit <- eBayes(fit, robust = TRUE)
    tab <- topTable(fit, number = Inf, sort.by = "none", confint = 0.95)
    tab$gene_id <- rownames(tab)
    tab$se_log2fc <- ifelse(is.finite(tab$t) & abs(tab$t) > 0, abs(tab$logFC/tab$t), NA_real_)
    tab$contrast_id <- rec$contrast_id
    tab$study <- rec$study
    tab$injury <- rec$injury
    tab$region <- rec$region
    tab$regimen_or_time <- rec$regimen_or_time
    tab$n_control <- rec$n_control
    tab$n_treatment <- rec$n_treatment
    tab$ci95_low <- tab$CI.L
    tab$ci95_high <- tab$CI.R
    tab <- tab[, c("contrast_id", "study", "injury", "region", "regimen_or_time", "n_control", "n_treatment",
        "gene_id", "logFC", "se_log2fc", "ci95_low", "ci95_high", "AveExpr", "t", "P.Value", "adj.P.Val",
        "B")]
    out_path <- file.path(output_dir, paste0(rec$contrast_id, "_limma_voom.tsv.gz"))
    gz <- gzfile(out_path, "wt")
    write.table(tab, gz, sep = "\t", quote = FALSE, row.names = FALSE)
    close(gz)

    lib <- colSums(counts)
    qc_rows[[length(qc_rows) + 1]] <- data.frame(contrast_id = rec$contrast_id, study = rec$study, injury = rec$injury,
        n_control = sum(group == "Control"), n_treatment = sum(group == "Treatment"), n_genes_input = nrow(counts),
        n_genes_tested = sum(keep), min_library_size = min(lib), median_library_size = median(lib), max_library_size = max(lib),
        normalization = "edgeR TMM", design_rank = design_rank, residual_df = residual_df, model = "limma-voom; treatment-control; empirical Bayes robust=TRUE",
        fdr_family = "all filterByExpr-retained genes", result_path = out_path, stringsAsFactors = FALSE)
}

qc <- do.call(rbind, qc_rows)
write.table(qc, file.path(output_dir, "limma_voom_contrast_qc.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
message("Completed ", nrow(manifest), " biological-sample contrasts")
