#!/usr/bin/env Rscript
# GSE243639 donor-level PD/control contrast. The clinical covariates are fixed before inspecting DE
# results. No stepwise selection or removal of covariates.
suppressPackageStartupMessages({
    library(edgeR)
    library(limma)
})
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("Usage: 02_human_limma.R prepared_dir output_dir")
input <- args[1]
out <- args[2]
input <- normalizePath(input, mustWork = TRUE)
out <- normalizePath(out, mustWork = FALSE)
if (dir.exists(out)) stop("Use a fresh output directory")
if (identical(input, out) || startsWith(out, paste0(input, "/"))) stop("Output must be outside inputs")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
write_gz <- function(x, path) {
    con <- gzfile(path, "wt")
    on.exit(close(con))
    write.table(x, con, sep = "\t", quote = FALSE, row.names = FALSE)
}
panel <- c("C3", "C3AR1", "C1QA", "TNF", "IL1B", "P2RY12", "TMEM119", "CX3CR1", "TREM2", "AXL", "MERTK")
jobs <- list(primary = list(selection = "broad", mincells = 20, minage = 0), final_annotation = list(selection = "final",
    mincells = 20, minage = 0), min50_nuclei = list(selection = "broad", mincells = 50, minage = 0),
    age_overlap = list(selection = "broad", mincells = 20, minage = 57))
qc <- list()
panels <- list()
for (name in names(jobs)) {
    spec <- jobs[[name]]
    m <- read.delim(file.path(input, spec$selection, "samples.tsv"), check.names = FALSE)
    use <- m$n_cells >= spec$mincells & m$age >= spec$minage
    m$included <- use
    folder <- file.path(out, name)
    dir.create(folder, recursive = TRUE, showWarnings = FALSE)
    write.table(m, file.path(folder, "inclusion.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
    m <- m[use, , drop = FALSE]
    cc <- read.delim(gzfile(file.path(input, spec$selection, "counts.tsv.gz")), check.names = FALSE)
    if (anyDuplicated(cc$gene_id) || anyDuplicated(m$donor_id))
        stop("Duplicate identifiers")
    counts <- as.matrix(cc[, m$donor_id, drop = FALSE])
    rownames(counts) <- cc$gene_id
    if (any(!is.finite(counts)) || any(counts < 0) || any(counts != round(counts)))
        stop("Invalid raw counts")
    m$group <- factor(m$group, levels = c("Control", "PD"))
    m$sex <- factor(m$sex)
    if (anyNA(m[, c("group", "sex", "age", "pmi_hours", "rin")]))
        stop("Missing covariates")
    design <- model.matrix(~age + sex + pmi_hours + rin + group, data = m)
    rownames(design) <- m$donor_id
    if (qr(design)$rank != ncol(design))
        stop("Non-estimable specified model")
    y <- DGEList(counts = counts)
    keep <- filterByExpr(y, design = design, min.count = 5, min.total.count = 15)
    y <- calcNormFactors(y[keep, , keep.lib.sizes = FALSE], method = "TMM")
    v <- voom(y, design, plot = FALSE, save.plot = TRUE)
    fit <- eBayes(lmFit(v, design), robust = TRUE)
    tab <- topTable(fit, coef = "groupPD", number = Inf, sort.by = "none", confint = 0.95)
    tab$gene <- rownames(tab)
    write_gz(tab, file.path(folder, "all_genes.tsv.gz"))
    p <- merge(data.frame(gene = panel), tab, by = "gene", all.x = TRUE, sort = FALSE)
    p <- p[match(panel, p$gene), ]
    p$analysis <- name
    p$tested <- !is.na(p$P.Value)
    p$measured <- p$gene %in% rownames(counts)
    p$n_control <- sum(m$group == "Control")
    p$n_pd <- sum(m$group == "PD")
    p$n_tested_genes <- sum(keep)
    panels[[name]] <- p
    expr <- data.frame(gene = rownames(v$E), v$E, check.names = FALSE)
    write_gz(expr, file.path(folder, "normalized_logcpm.tsv.gz"))
    write.table(data.frame(donor_id = m$donor_id, design, check.names = FALSE), file.path(folder, "design.tsv"),
        sep = "\t", quote = FALSE, row.names = FALSE)
    write.table(data.frame(gene = rownames(counts), tested = keep), file.path(folder, "filtering.tsv"),
        sep = "\t", quote = FALSE, row.names = FALSE)
    pc <- prcomp(t(v$E), center = TRUE, scale. = FALSE)
    write.table(data.frame(donor_id = m$donor_id, group = m$group, pc$x), file.path(folder, "pca.tsv"),
        sep = "\t", quote = FALSE, row.names = FALSE)
    write.table(data.frame(component = seq_along(pc$sdev), variance = pc$sdev^2/sum(pc$sdev^2)), file.path(folder,
        "pca_variance.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
    write.table(data.frame(x = v$voom.xy$x, y = v$voom.xy$y), file.path(folder, "voom_points.tsv"), sep = "\t",
        quote = FALSE, row.names = FALSE)
    write.table(data.frame(x = v$voom.line$x, y = v$voom.line$y), file.path(folder, "voom_trend.tsv"),
        sep = "\t", quote = FALSE, row.names = FALSE)
    write.table(data.frame(donor_id = m$donor_id, group = m$group, n_cells = m$n_cells, raw_count_total = colSums(counts),
        filtered_count_total = y$samples$lib.size, tmm_factor = y$samples$norm.factors), file.path(folder,
        "sample_qc.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
    qc[[name]] <- data.frame(analysis = name, n_control = sum(m$group == "Control"), n_pd = sum(m$group ==
        "PD"), n_microglia = sum(m$n_cells), n_tested_genes = sum(keep), n_fdr_genes = sum(tab$adj.P.Val <
        0.05), n_panel_fdr = sum(p$adj.P.Val < 0.05, na.rm = TRUE), design_rank = qr(design)$rank, residual_df = nrow(design) -
        qr(design)$rank, formula = "~age+sex+pmi_hours+rin+group")
}
write.table(do.call(rbind, panels), file.path(out, "panel_all_analyses.tsv"), sep = "\t", quote = FALSE,
    row.names = FALSE)
write.table(do.call(rbind, qc), file.path(out, "analysis_qc.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out, "session_info.txt"))
print(do.call(rbind, qc))
