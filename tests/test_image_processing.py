"""Processing and geometry checks using synthetic image fixtures."""

from pathlib import Path
import importlib.util
import unittest

import numpy as np
from skimage.morphology import disk, white_tophat


ROOT = Path(__file__).resolve().parents[1] / "upstream_imaging"


def load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


segmentation = load("tested_segmentation", "vendor/03_synapse_segmentation.py")
microglia = load("tested_microglia", "vendor/02_microglia_segmentation.py")
geometry = load("tested_geometry", "vendor/04_contact_geometry.py")
preprocessing = load("tested_preprocessing", "vendor/01_preprocessing.py")


class PhysicalGeometryTests(unittest.TestCase):
    def test_shared_marker_detection_is_bitwise_equivalent(self):
        z, y, x = np.mgrid[:3, :31, :31]
        image = (
            5.0
            + 300.0 * np.exp(-((z - 1) ** 2 / 0.8 + (y - 10) ** 2 / 5 + (x - 11) ** 2 / 5))
            + 220.0 * np.exp(-((z - 1) ** 2 / 0.8 + (y - 21) ** 2 / 7 + (x - 20) ** 2 / 7))
        ).astype(np.float32)
        params = {"n_scales": 3, "sigma_range_um": (0.12, 0.45)}
        independent = segmentation.segment_synapses(
            image,
            image,
            "vglut1",
            voxel_xy=0.0884,
            voxel_z=0.38,
            params=params,
        )
        shared = segmentation.prepare_shared_detection(
            image,
            "vglut1",
            voxel_xy=0.0884,
            voxel_z=0.38,
            params=params,
        )
        reused = segmentation.segment_synapses(
            image,
            image,
            "vglut1",
            voxel_xy=0.0884,
            voxel_z=0.38,
            params=params,
            shared_detection=shared,
        )
        np.testing.assert_array_equal(independent["labels"], reused["labels"])
        np.testing.assert_array_equal(
            independent["log_response"],
            reused["log_response"],
        )
        self.assertEqual(independent["n_seeds"], reused["n_seeds"])

    def test_fast_tophat_is_exactly_equivalent_to_skimage(self):
        rng = np.random.default_rng(20260713)
        image = rng.normal(200, 30, size=(2, 57, 61)).astype(np.float32)
        image[:, :3, :4] += 400
        radius = 5
        observed = preprocessing.white_tophat_background_subtraction(
            image,
            radius_px=radius,
        )
        expected = np.stack([white_tophat(section, disk(radius)) for section in image])
        np.testing.assert_array_equal(observed, expected)

    def test_physical_log_has_zero_constant_response(self):
        image = np.ones((5, 13, 13), dtype=np.float32)
        response, _ = segmentation.compute_log_response(
            image,
            sigma_range_um=(0.12, 0.6),
            n_scales=4,
            voxel_xy=0.0884,
            voxel_z=0.38,
        )
        self.assertLessEqual(float(np.max(np.abs(response))), 1e-6)

    def test_sub_axial_closing_radius_does_not_bridge_z(self):
        footprint = microglia._physical_ball(0.30, 0.0884, 0.38)
        centre_z = footprint.shape[0] // 2
        outside_centre_slice = footprint.copy()
        outside_centre_slice[centre_z] = False
        self.assertFalse(bool(outside_centre_slice.any()))

    def test_marker_sub_axial_ellipsoid_does_not_bridge_z(self):
        footprint = segmentation._physical_ellipsoid(
            0.15,
            0.15,
            voxel_xy=0.0884,
            voxel_z=0.38,
        )
        centre_z = footprint.shape[0] // 2
        outside_centre_slice = footprint.copy()
        outside_centre_slice[centre_z] = False
        self.assertFalse(bool(outside_centre_slice.any()))

    def test_process_sized_iba1_component_survives_final_filter(self):
        binary = np.zeros((3, 20, 20), dtype=bool)
        binary[1, 3:13, 3:13] = True  # 100 voxels ~= 0.30 um3
        filtered, n_objects, _ = microglia.filter_by_volume(
            binary,
            min_volume_um3=0.15,
            voxel_xy=0.0884,
            voxel_z=0.38,
        )
        self.assertEqual(n_objects, 1)
        self.assertEqual(int(filtered.sum()), 100)

    def test_primary_iba1_refinement_preserves_enclosed_voids(self):
        shell = np.ones((5, 9, 9), dtype=bool)
        shell[2, 3:6, 3:6] = False
        refined = microglia.morphological_refinement(
            shell,
            closing_radius_um=0.0,
            min_fragment_voxels=0,
            voxel_xy=0.1,
            voxel_z=0.4,
        )
        self.assertFalse(bool(refined[2, 4, 4]))

    def test_capped_iba1_hole_fill_is_physical(self):
        shell = np.ones((5, 9, 9), dtype=bool)
        shell[2, 2, 2] = False  # 0.004 um3
        shell[2, 4:7, 4:7] = False  # 0.036 um3
        capped = microglia.fill_holes_by_volume(
            shell,
            max_hole_volume_um3=0.01,
            voxel_xy=0.1,
            voxel_z=0.4,
        )
        self.assertTrue(bool(capped[2, 2, 2]))
        self.assertFalse(bool(capped[2, 5, 5]))

    def test_iba1_dependent_candidate_filter_fails_loudly(self):
        labels = np.zeros((2, 5, 5), dtype=np.int32)
        labels[0, 2, 2] = 1
        with self.assertRaisesRegex(ValueError, "Iba1-dependent"):
            segmentation.filter_objects(
                labels,
                labels.astype(float),
                labels.astype(float),
                "vglut1",
                voxel_xy=0.0884,
                voxel_z=0.38,
                microglia_mask=np.zeros_like(labels, dtype=bool),
            )

    def test_single_voxel_candidate_is_below_support_floor(self):
        labels = np.zeros((3, 7, 7), dtype=np.int32)
        labels[1, 3, 3] = 1
        raw = np.zeros_like(labels, dtype=float)
        raw[1, 3, 3] = 100.0
        filtered, removed, _ = segmentation.filter_objects(
            labels,
            raw,
            raw,
            "vglut1",
            voxel_xy=0.0884,
            voxel_z=0.38,
            min_volume_um3=0.0,
            max_volume_um3=10.0,
            marker_params={"min_support_voxels": 2},
            microglia_mask=None,
        )
        self.assertEqual(int(filtered.max()), 0)
        self.assertEqual(removed["too_small"], 1)

    def test_fixed_floor_controls_rescue_threshold(self):
        """A fixed cross-group floor must also govern late bright rescue."""
        background = np.array([0.0, 1.0, 10000.0])
        floor = segmentation.compute_rescue_base_floor(
            background,
            is_axon_marker=True,
            marker_params={
                "fixed_noise_floor": 250.0,
                "global_intensity_factor": 0.5,
                "intensity_sigma_k": 99.0,
            },
        )
        self.assertEqual(floor, 125.0)

    def test_precomputed_floor_controls_object_and_rescue_acceptance(self):
        labels = np.zeros((3, 9, 9), dtype=np.int32)
        labels[1, 3, 3:5] = 1
        labels[1, 6, 5:7] = 2
        raw = np.zeros_like(labels, dtype=float)
        raw[labels == 1] = [110.0, 80.0]
        raw[labels == 2] = [95.0, 90.0]
        filtered, removed, floor = segmentation.filter_objects(
            labels,
            raw,
            raw,
            "vglut1",
            voxel_xy=0.1,
            voxel_z=0.4,
            min_volume_um3=0.0,
            max_volume_um3=10.0,
            marker_params={
                "min_support_voxels": 2,
                "precomputed_noise_floor": 100.0,
                "global_intensity_factor": 0.5,
            },
            microglia_mask=None,
        )
        self.assertEqual(floor, 100.0)
        self.assertEqual(int(filtered.max()), 1)
        self.assertEqual(removed["too_dim"], 1)
        rescue = segmentation.compute_rescue_base_floor(
            np.array([0.0, 1.0, 10000.0]),
            is_axon_marker=True,
            marker_params={
                "precomputed_rescue_base_floor": 321.0,
                "global_intensity_factor": 1.0,
            },
        )
        self.assertEqual(rescue, 321.0)

    def test_operational_geometry_categories(self):
        mask = np.zeros((3, 7, 9), dtype=bool)
        mask[1, 3, 2] = True
        labels = np.zeros(mask.shape, dtype=np.int32)
        labels[1, 3, 2] = 1  # complete inclusion
        labels[1, 3, 3] = 2  # <= 0.1 um proximity
        labels[1, 3, 4] = 3  # farther than 0.1 um
        labels[1, 2, 2] = 4  # partial overlap object
        labels[1, 2, 3] = 4
        mask[1, 2, 2] = True
        labels[0, 5, 7] = 5  # truncated complete-looking object
        mask[0, 5, 7] = True
        distance = geometry.compute_distance_to_microglia(
            mask,
            voxel_xy=0.0884,
            voxel_z=0.38,
        )
        frame = geometry.classify_contacts(
            labels,
            mask,
            distance,
            contact_distance_um=0.1,
            full_threshold=1.0,
        ).set_index("label_id")
        self.assertEqual(frame.loc[1, "contact_category"], "complete_inclusion")
        self.assertEqual(frame.loc[2, "contact_category"], "proximity_without_overlap")
        self.assertEqual(frame.loc[3, "contact_category"], "no_contact")
        self.assertEqual(frame.loc[4, "contact_category"], "partial_overlap")
        self.assertFalse(bool(frame.loc[5, "complete_inclusion_eligible"]))
        self.assertEqual(frame.loc[5, "contact_category"], "boundary_truncated_overlap")


if __name__ == "__main__":
    unittest.main()
