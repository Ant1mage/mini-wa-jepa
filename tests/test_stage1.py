import unittest

import numpy as np

from configs.stage1 import Stage1Config
from scripts.verify_stage1 import validate_image


class ImageSmokeTests(unittest.TestCase):
    def test_valid_semantic_image(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[10:30] = [255, 255, 255]
        image[30:40] = [80, 80, 80]
        self.assertEqual(validate_image(image, (64, 64, 3))["unique_colors"], 3)

    def test_rejects_black_frame_wrong_shape_and_float(self):
        for image in (np.zeros((64, 64, 3), dtype=np.uint8),
                      np.zeros((32, 64, 3), dtype=np.uint8),
                      np.zeros((64, 64, 3), dtype=np.float32)):
            with self.assertRaises(AssertionError):
                validate_image(image, (64, 64, 3))

    def test_invalid_simulation_config(self):
        with self.assertRaises(ValueError):
            Stage1Config(horizon=0)
        with self.assertRaises(ValueError):
            Stage1Config(bev_pixels_per_metre=-1)

    def test_msaa_adapter_does_not_mutate_callers_properties(self):
        from direct.filter.FilterManager import FilterManager
        from panda3d.core import FrameBufferProperties
        from env.render_compat import limit_filter_msaa

        original, original_alias = FilterManager.createBuffer, FilterManager.create_buffer
        seen = []

        def receiver(self, name, xsize, ysize, texgroup, depthbits=True, fbprops=None):
            seen.append(fbprops.getMultisamples())
            return "buffer"

        FilterManager.createBuffer = receiver
        try:
            changes, restore = limit_filter_msaa(8)
            try:
                props = FrameBufferProperties()
                props.setMultisamples(16)
                self.assertEqual(FilterManager.createBuffer(None, "test", 64, 64, [], fbprops=props), "buffer")
                self.assertEqual(props.getMultisamples(), 16)
                self.assertEqual(seen, [8])
                self.assertEqual(len(changes), 1)
            finally:
                restore()
            self.assertIs(FilterManager.createBuffer, receiver)
        finally:
            FilterManager.createBuffer, FilterManager.create_buffer = original, original_alias


if __name__ == "__main__":
    unittest.main()
