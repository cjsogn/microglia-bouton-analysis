# Independent Xu microglia. Donors, not nuclei, are the biological replicates.  Primary model fixed
# before inspecting target-gene differential expression: age + sex + diagnosis. PMI adjustment and
# annotation agreement are sensitivities.
suppressPackageStartupMessages({
    library(edgeR)
    library(limma)
})
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: 05_donor_pseudobulk.R annotation_input_dir fresh_output_dir")
input <- normalizePath(args[1], mustWork = TRUE)
out <- normalizePath(args[2], mustWork = FALSE)
if (dir.exists(out)) stop("Use a fresh output directory")
if (identical(input, out) || startsWith(out, paste0(input, "/"))) stop("Output must be outside inputs")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
script_file <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
panel_file <- file.path(dirname(normalizePath(script_file)), "../../../config/transcript_panel.tsv")
panel <- toupper(read.delim(panel_file, check.names = FALSE)$gene)
stopifnot(length(panel) == 25L, !anyDuplicated(panel))
jobs <- list(primary = list(selection = "primary", formula = ~age_years + sex + group), pmi_adjusted = list(selection = "primary",
    formula = ~age_years + sex + pmi_hours + group), unadjusted = list(selection = "primary", formula = ~group),
    harmony_annotation = list(selection = "harmony", formula = ~age_years + sex + group), concordant_annotation = list(selection = "concordant",
        formula = ~age_years + sex + group))
donors <- read.delim(file.path(input, "primary", "samples.tsv"))$donor_id
for (d in donors) jobs[[paste0("omit_", d)]] <- list(selection = "primary", formula = ~age_years + sex +
    group, omit = d)
panels <- list()
summaries <- list()
write_tsv <- function(x, path) write.table(x, path, sep = "\t", row.names = FALSE, quote = FALSE)
for (name in names(jobs)) {
    job <- jobs[[name]]
    folder <- file.path(out, name)
    dir.create(folder, showWarnings = FALSE)
    m <- read.delim(file.path(input, job$selection, "samples.tsv"))
    m$included <- m$n_cells >= 20 & !(m$donor_id %in% job$omit)
    write_tsv(m, file.path(folder, "inclusion.tsv"))
    m <- m[m$included, , drop = FALSE]
    m$group <- factor(m$diagnosis, levels = c("CTRL", "PD"))
    m$sex <- factor(m$sex)
    stopifnot(!anyNA(m[, c("age_years", "pmi_hours", "sex", "group")]), min(table(m$group)) >= 3, !anyDuplicated(m$donor_id))
    raw <- read.delim(file.path(input, job$selection, "counts.tsv.gz"), check.names = FALSE)
    counts <- as.matrix(raw[, m$donor_id, drop = FALSE])
    rownames(counts) <- raw$gene
    stopifnot(!anyDuplicated(rownames(counts)), all(is.finite(counts)), all(counts >= 0), all(counts ==
        round(counts)))
    design <- model.matrix(job$formula, data = m)
    stopifnot(qr(design)$rank == ncol(design), nrow(design) - ncol(design) >= 2)
    y <- DGEList(counts)
    keep <- filterByExpr(y, design = design, min.count = 5, min.total.count = 15)
    y <- calcNormFactors(y[keep, , keep.lib.sizes = FALSE], method = "TMM")
    v <- voom(y, design, plot = FALSE, save.plot = TRUE)
    fit <- eBayes(lmFit(v, design), robust = TRUE)
    tab <- topTable(fit, coef = "groupPD", number = Inf, sort.by = "none", confint = 0.95)
    tab$gene <- rownames(tab)
    stopifnot(isTRUE(all.equal(tab$adj.P.Val, p.adjust(tab$P.Value, "BH"), tolerance = 1e-12)))
    con <- gzfile(file.path(folder, "all_genes.tsv.gz"), "wt")
    write_tsv(tab, con)
    close(con)
    p <- merge(data.frame(gene = panel), tab, by = "gene", all.x = TRUE, sort = FALSE)
    p <- p[match(panel, p$gene), ]
    p$analysis <- name
    p$tested <- !is.na(p$P.Value)
    p$measured <- p$gene %in% rownames(counts)
    p$n_control <- sum(m$group == "CTRL")
    p$n_pd <- sum(m$group == "PD")
    p$n_tested_genes <- sum(keep)
    panels[[name]] <- p
    write_tsv(data.frame(gene = rownames(counts), tested = keep), file.path(folder, "filtering.tsv"))
    write_tsv(data.frame(donor_id = m$donor_id, design, check.names = FALSE), file.path(folder, "design.tsv"))
    write_tsv(data.frame(donor_id = m$donor_id, group = m$group, n_cells = m$n_cells, raw_total = colSums(counts),
        filtered_total = y$samples$lib.size, tmm_factor = y$samples$norm.factors), file.path(folder,
        "sample_qc.tsv"))
    con <- gzfile(file.path(folder, "normalized_logcpm.tsv.gz"), "wt")
    write_tsv(data.frame(gene = rownames(v$E), v$E, check.names = FALSE), con)
    close(con)
    # Residuals and mean-variance data for model diagnostics, not outcome selection.
    residual <- v$E - fit$coefficients %*% t(design)
    write_tsv(data.frame(donor_id = m$donor_id, residual_rms = sqrt(colMeans(residual^2))), file.path(folder,
        "residual_qc.tsv"))
    pc <- prcomp(t(v$E))
    write_tsv(data.frame(donor_id = m$donor_id, group = m$group, pc$x), file.path(folder, "pca.tsv"))
    write_tsv(data.frame(x = v$voom.xy$x, y = v$voom.xy$y), file.path(folder, "voom_points.tsv"))
    write_tsv(data.frame(x = v$voom.line$x, y = v$voom.line$y), file.path(folder, "voom_trend.tsv"))
    summaries[[name]] <- data.frame(analysis = name, n_control = sum(m$group == "CTRL"), n_pd = sum(m$group ==
        "PD"), n_microglia = sum(m$n_cells), n_tested_genes = sum(keep), n_fdr_genes = sum(tab$adj.P.Val <
        0.05), n_panel_fdr = sum(p$adj.P.Val < 0.05, na.rm = TRUE), residual_df = nrow(design) - ncol(design),
        formula = paste(deparse(job$formula), collapse = ""))
}
write_tsv(do.call(rbind, panels), file.path(out, "panel_all_analyses.tsv"))
write_tsv(do.call(rbind, summaries), file.path(out, "analysis_qc.tsv"))
writeLines(capture.output(sessionInfo()), file.path(out, "R_session_info.txt"))
print(do.call(rbind, summaries))
