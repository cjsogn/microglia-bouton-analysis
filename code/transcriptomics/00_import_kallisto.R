#!/usr/bin/env Rscript
# Joint import for GSE244652. Published MARS-seq chemistry supports original gene counts without a
# length offset. Length-scaled counts are retained as a sensitivity comparator because the GEO
# library description conflicts.
suppressPackageStartupMessages({
    library(tximport)
    library(rhdf5)
})
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Usage: 00_import_kallisto.R samples.tsv tx2gene.tsv output_dir")
samples <- read.delim(args[1], check.names = FALSE, stringsAsFactors = FALSE)
map <- read.delim(args[2], check.names = FALSE, stringsAsFactors = FALSE)
out <- args[3]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
stopifnot(all(c("sample_id", "input_path", "group") %in% names(samples)))
if (anyDuplicated(samples$sample_id) || !all(file.exists(samples$input_path))) stop("Duplicate or missing samples")
map <- unique(data.frame(tx = as.character(map[[1]]), gene = as.character(map[[2]])))
map <- map[!is.na(map$gene) & nzchar(map$gene) & map$gene != "nan", ]
if (anyDuplicated(map$tx)) stop("A transcript maps to multiple gene names")
base <- sub("\\.[0-9]+$", "", map$tx)
base_map <- split(map$gene, base)
ambiguous <- names(Filter(function(z) length(unique(z)) > 1L, base_map))
files <- setNames(samples$input_path, samples$sample_id)
ids <- as.character(h5read(files[1], "aux/ids"))
if (anyDuplicated(ids)) stop("Duplicate quantified transcript IDs")
idx <- match(ids, map$tx)
gene <- map$gene[idx]
unmatched <- which(is.na(gene))
keys <- sub("\\.[0-9]+$", "", ids[unmatched])
if (any(keys %in% ambiguous)) stop("Ambiguous version-free transcript mapping")
fallback <- vapply(base_map, function(z) unique(z)[1], character(1))
gene[unmatched] <- unname(fallback[match(keys, names(fallback))])
resolved <- data.frame(tx = ids[!is.na(gene)], gene = gene[!is.na(gene)])
write.table(data.frame(transcript_id = ids, gene_name = gene, mapping = ifelse(!is.na(idx), "exact",
    ifelse(!is.na(gene), "version_fallback", "unmapped"))), file.path(out, "transcript_mapping_audit.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE)
qc <- vector("list", length(files))
for (i in seq_along(files)) {
    these_ids <- as.character(h5read(files[i], "aux/ids"))
    count <- as.numeric(h5read(files[i], "est_counts"))
    effective <- as.numeric(h5read(files[i], "aux/eff_lengths"))
    if (!identical(ids, these_ids))
        stop("Transcript index/order mismatch: ", names(files)[i])
    if (length(count) != length(ids) || length(effective) != length(ids))
        stop("Quantification length mismatch")
    if (any(!is.finite(count) | count < 0) || any(!is.finite(effective) | effective <= 0))
        stop("Invalid count/effective length")
    qc[[i]] <- data.frame(sample_id = names(files)[i], total_estimated_counts = sum(count), mapped_estimated_counts = sum(count[!is.na(gene)]),
        unmapped_estimated_counts = sum(count[is.na(gene)]), n_transcripts = length(ids), n_unmapped_transcripts = sum(is.na(gene)))
}
# tximport selects its HDF5 reader by the exact basename abundance.h5.  GEO renames the files.
# Temporary links retain the original bytes and permit use of the official reader, without
# converting or modifying quantifications.
link_root <- tempfile("gse244652_kallisto_")
dir.create(link_root)
import_files <- setNames(file.path(link_root, names(files), "abundance.h5"), names(files))
for (i in seq_along(files)) {
    dir.create(dirname(import_files[i]))
    if (!file.symlink(normalizePath(files[i]), import_files[i]))
        stop("Cannot create import link")
}
# Both imports use the official reader and identical transcript mapping.
raw <- tximport(import_files, type = "kallisto", tx2gene = resolved, countsFromAbundance = "no", dropInfReps = TRUE)
length_scaled <- tximport(import_files, type = "kallisto", tx2gene = resolved, countsFromAbundance = "lengthScaledTPM",
    dropInfReps = TRUE)
if (!identical(dimnames(raw$counts), dimnames(length_scaled$counts))) stop("Import dimension mismatch")
if (any(!is.finite(length_scaled$counts) | length_scaled$counts < 0)) stop("Invalid length-scaled counts")
if (!isTRUE(all.equal(colSums(raw$counts), colSums(length_scaled$counts), tolerance = 1e-10))) stop("Mapped library totals changed")
write_matrix <- function(m, filename) {
    con <- gzfile(file.path(out, filename), "wt")
    on.exit(close(con))
    write.table(data.frame(gene_id = rownames(m), m, check.names = FALSE), con, sep = "\t", quote = FALSE,
        row.names = FALSE)
}
options(digits = 17)
write_matrix(raw$counts, "counts.tsv.gz")
write_matrix(length_scaled$counts, "length_scaled_counts_sensitivity.tsv.gz")
write_matrix(length_scaled$abundance, "gene_abundance_tpm.tsv.gz")
write_matrix(length_scaled$length, "gene_effective_lengths.tsv.gz")
qc <- do.call(rbind, qc)
qc$imported_gene_count <- nrow(length_scaled$counts)
qc$length_scaled_count_total <- colSums(length_scaled$counts)
qc$countsFromAbundance <- "no"
write.table(qc, file.path(out, "import_qc.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out, "import_session_info.txt"))
message("Original-count primary and length-scaled sensitivity imports complete: ", nrow(raw$counts),
    " genes, ", ncol(raw$counts), " samples")
