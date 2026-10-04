"""Reconcile analysis sample labels and clinical covariates with primary records.

Supply unpacked GEO family SOFT records and the published putamen donor table.
The audit is read-only with respect to analysis inputs and saves field-level
comparisons. Publication-level collection times are documented separately.
"""

from collections import defaultdict
from pathlib import Path
import argparse
import json
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from mg_paths import BUNDLE, DERIVED, require_writable_output


def read_soft(path):
    samples = {}
    current = None
    for line in path.read_text().splitlines():
        if line.startswith("^SAMPLE = "):
            current = line.split(" = ", 1)[1]
            samples[current] = defaultdict(list)
        elif line.startswith("^"):
            current = None
        elif current and " = " in line:
            key, value = line.split(" = ", 1)
            samples[current][key].append(value)
    if not samples:
        raise ValueError(f"No sample metadata in {path}")
    return samples


def characteristics(record):
    fields = {}
    for value in record.get("!Sample_characteristics_ch1", []):
        key, val = value.split(":", 1)
        if key.lower().strip() in fields:
            raise ValueError(f"Repeated characteristic: {key}")
        fields[key.lower().strip()] = val.strip()
    return fields


def audit(geo, donor_table):
    comparisons = []

    def check(accession, sample, field, analysed, primary):
        equal = str(analysed).lower().strip() == str(primary).lower().strip()
        comparisons.append(
            dict(
                accession=accession,
                sample=sample,
                field=field,
                analysis_value=analysed,
                source_value=primary,
                status="PASS" if equal else "MISMATCH",
                source_url=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={sample}",
            )
        )

    accessions = [
        "GSE75246",
        "GSE213100",
        "GSE244652",
        "GSE263094",
        "GSE307796",
        "GSE243639",
        "GSE161045",
    ]
    records = {acc: read_soft(geo / (acc + ".soft")) for acc in accessions}
    manifest = pd.read_csv(DERIVED / "transcriptomics/contrast_manifest.tsv", sep="\t")
    for contrast in manifest.itertuples(index=False):
        samples = pd.read_csv(
            DERIVED / "transcriptomics/contrasts" / contrast.contrast_id / "samples.tsv", sep="\t"
        )
        for rec in samples.itertuples(index=False):
            gsm = re.search(r"GSM\d+", rec.sample_id).group()
            source = records[contrast.study][gsm]
            char = characteristics(source)
            title = source["!Sample_title"][0]
            if contrast.study == "GSE75246":
                check(
                    contrast.study,
                    gsm,
                    "group",
                    rec.group,
                    {"lps": "Treatment", "vehicle": "Control"}[char["treatment"].lower()],
                )
                check(contrast.study, gsm, "cell_type", rec.cell_type, char["cell type"])
            elif contrast.study == "GSE213100":
                check(
                    contrast.study,
                    gsm,
                    "group",
                    rec.group,
                    {"MPTP": "Treatment", "saline": "Control"}[char["treatment"]],
                )
                check(contrast.study, gsm, "cell_type", rec.cell_type, char["cell type"])
                check(
                    contrast.study,
                    gsm,
                    "region",
                    rec.region,
                    {"substantia nigra": "SN", "striatum": "Striatum"}[char["tissue"]],
                )
                protocol = " ".join(source["!Sample_treatment_protocol_ch1"])
                check(
                    contrast.study,
                    gsm,
                    "six_hours_after_final_dose",
                    True,
                    "Six hours after last" in protocol,
                )
            elif contrast.study == "GSE244652":
                check(
                    contrast.study,
                    gsm,
                    "group",
                    rec.group,
                    "Control"
                    if "_Ctrl" in title
                    else "Treatment"
                    if "_MPTP" in title
                    else "UNRESOLVED",
                )
                check(contrast.study, gsm, "region", rec.region, char["tissue"])
                check(
                    contrast.study,
                    gsm,
                    "CD11b_selection",
                    True,
                    char["cell type"] == "CD11b+ cells",
                )
                check(
                    contrast.study,
                    gsm,
                    "regimen",
                    rec.regimen,
                    "subacute"
                    if title.startswith("sMPTP_")
                    else "chronic_probenecid"
                    if title.startswith("cMPTP_")
                    else "UNRESOLVED",
                )
            else:
                check(
                    contrast.study,
                    gsm,
                    "condition",
                    rec.condition_detail,
                    char["treatment"].replace(" ", ""),
                )
                check(contrast.study, gsm, "genotype", "WT", char["genotype"])
                check(
                    contrast.study,
                    gsm,
                    "group",
                    rec.group,
                    "Control" if char["treatment"] == "PBS" else "Treatment",
                )
        for group, column in [("Control", "n_control"), ("Treatment", "n_treatment")]:
            if int(samples.group.eq(group).sum()) != getattr(contrast, column):
                raise ValueError(f"Manifest/sample count mismatch: {contrast.contrast_id}")
    for gsm, treatment, days in [
        ("GSM9231553", "PBS", 3),
        ("GSM9231549", "LPS (2mg/kg)", 3),
        ("GSM9231551", "LPS (2mg/kg)", 30),
    ]:
        rec = records["GSE307796"][gsm]
        char = characteristics(rec)
        check("GSE307796", gsm, "treatment", treatment, char["treatment"])
        check(
            "GSE307796",
            gsm,
            "days_after_injection",
            days,
            int(re.search(r"(\d+) days post injection", rec["!Sample_title"][0]).group(1)),
        )
        check("GSE307796", gsm, "tissue", "Whole Brain", char["tissue"])
    human = pd.read_csv(BUNDLE / "corrected/human_pd/donor_metadata.tsv", sep="\t")
    lookup = {
        re.search(r"s_\d+", rec["!Sample_title"][0]).group().replace("_", "."): (gsm, rec)
        for gsm, rec in records["GSE243639"].items()
    }
    if set(human.donor_id) != set(lookup):
        raise ValueError("SNpc donor set differs from GEO")
    for row in human.itertuples(index=False):
        gsm, rec = lookup[row.donor_id]
        char = characteristics(rec)
        check("GSE243639", gsm, "diagnosis", row.diagnosis, char["clinical diagnosis"])
        check("GSE243639", gsm, "sex", row.sex, char["sex"])
        for col, key in [("age", "age"), ("pmi_hours", "pmi hours"), ("rin", "rin measure")]:
            check("GSE243639", gsm, col, float(getattr(row, col)), float(char[key]))
    # Donor covariates are in the publication supplement, not the GEO sample fields.
    book = pd.read_excel(donor_table, sheet_name="Table_S1_Patient_information", header=None)
    header = book.index[book.iloc[:, 0].eq("Case")].item()
    donor_rows = book.iloc[header + 1 :].copy()
    donor_rows.columns = [
        "donor",
        "diagnosis",
        "sex",
        "age_years",
        "pmi_hours",
        "cause_of_death",
        "braak_nft",
        "braak_amyloid_beta",
        "braak_lewy_body",
        "cdr",
    ]
    donor_rows["donor_id"] = donor_rows.donor.str.replace("-", "", regex=False)
    donor_rows = donor_rows.set_index("donor_id")
    putamen = pd.read_csv(
        BUNDLE / "xu_putamen/results/independent_annotation/primary/samples.tsv", sep="\t"
    )
    lookup = {
        rec["!Sample_title"][0].replace("Putamen_", ""): (gsm, rec)
        for gsm, rec in records["GSE161045"].items()
    }
    diagnoses = {"control": "CTRL", "Parkinson disease (PD)": "PD"}
    expected = {
        d for d, (_, rec) in lookup.items() if characteristics(rec)["diagnosis"] in diagnoses
    }
    if set(putamen.donor_id) != expected:
        raise ValueError("Putamen PD/control donor set differs from GEO")
    for row in putamen.itertuples(index=False):
        gsm, rec = lookup[row.donor_id]
        char = characteristics(rec)
        check("GSE161045", gsm, "diagnosis", row.diagnosis, diagnoses[char["diagnosis"]])
        check("GSE161045", gsm, "tissue", "Putamen", char["tissue"])
        published = donor_rows.loc[row.donor_id]
        for col in ["sex", "age_years", "pmi_hours"]:
            val = getattr(row, col)
            actual = published[col]
            if col != "sex":
                val, actual = float(val), float(actual)
            check("GSE161045", gsm, col, val, actual)
            comparisons[-1]["source_url"] = (
                "https://doi.org/10.1038/s43587-023-00363-8 (Supplementary Table 1)"
            )
    return pd.DataFrame(comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geo-dir", type=Path, required=True)
    parser.add_argument("--putamen-donor-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = require_writable_output(args.out)
    if output.exists():
        raise FileExistsError("Use a fresh audit output directory")
    result = audit(args.geo_dir, args.putamen_donor_table)
    output.mkdir(parents=True)
    result.to_csv(output / "geo_metadata_reconciliation.tsv", sep="\t", index=False)
    summary = {
        "field_comparisons": len(result),
        "mismatches": int(result.status.ne("PASS").sum()),
        "unique_samples": result[["accession", "sample"]]
        .drop_duplicates()
        .groupby("accession")
        .size()
        .to_dict(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["mismatches"]:
        raise SystemExit("Metadata mismatch: inspect the saved field-level comparisons")


if __name__ == "__main__":
    main()
