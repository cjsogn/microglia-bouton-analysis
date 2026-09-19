# Read the published donor table without changing the source workbook.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: 03_extract_donor_metadata.R source_workbook output_csv")
source_workbook <- normalizePath(args[1], mustWork = TRUE)
output_csv <- normalizePath(args[2], mustWork = FALSE)
dir.create(dirname(output_csv), recursive = TRUE, showWarnings = FALSE)
library(readxl)
x <- as.data.frame(read_excel(source_workbook, sheet = "Table_S1_Patient_information",
    col_names = FALSE))
header <- which(x[[1]] == "Case")
stopifnot(length(header) == 1L)
x <- x[seq.int(header + 1L, nrow(x)), , drop = FALSE]
stopifnot(nrow(x) == 12L, ncol(x) == 10L)
names(x) <- c("donor", "diagnosis", "sex", "age_years", "pmi_hours", "cause_of_death", "braak_nft", "braak_amyloid_beta",
    "braak_lewy_body", "cdr")
for (column in c("age_years", "pmi_hours", "braak_lewy_body", "cdr")) {
    x[[column]] <- as.numeric(x[[column]])
    stopifnot(!anyNA(x[[column]]))
}
x$geo_sample_label <- gsub("-", "", x$donor, fixed = TRUE)
x$source <- "43587_2023_363_MOESM3_ESM.xlsx, Table_S1_Patient_information"
stopifnot(!anyDuplicated(x$geo_sample_label))
write.csv(x, output_csv, row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(dirname(output_csv), "metadata_R_session_info.txt"))
print(x[, c("donor", "diagnosis", "sex", "age_years", "pmi_hours", "braak_lewy_body")], row.names = FALSE)
