"""Check PCA arithmetic against a separate summation path after BLAS warnings."""

import json
import warnings
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

from xu_paths import WORK as ROOT

a = sc.read_h5ad(ROOT / "results/independent_annotation/immune.h5ad")
b = a[:, a.var.highly_variable].copy()
sc.pp.scale(b, max_value=10)
x = np.asarray(b.X, dtype=np.float64)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    pca = PCA(n_components=30, svd_solver="arpack", random_state=20260910).fit(x)
    got = pca.transform(x)
expected = np.einsum("ij,kj->ik", x - pca.mean_, pca.components_, optimize=False)
max_error = float(np.max(np.abs(got - expected)))
assert np.isfinite(got).all() and np.isfinite(expected).all()
assert np.allclose(got, expected, rtol=1e-10, atol=1e-10)
result = {
    "check": "sklearn_PCA_transform_vs_independent_einsum",
    "status": "PASS",
    "max_absolute_difference": max_error,
    "n_rows": len(x),
    "n_features": x.shape[1],
    "warnings": [str(w.message) for w in caught],
}
(ROOT / "results/numeric_backend_check.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
