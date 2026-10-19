import unittest

import numpy as np

from env.semantic_bev import SemanticBEVSpec,semantic_channels


class SemanticChannelTests(unittest.TestCase):
    def test_four_channels_visibility_and_ego_exclusion(self):
        from metadrive.constants import Semantics as S
        rgb=np.zeros((128,128,3),np.uint8)
        rgb[20,20]=S.ROAD.color;rgb[30,30]=S.SIDEWALK.color
        rgb[40,40]=S.TRUCK.color;rgb[63:65,63:65]=S.CAR.color
        bev=semantic_channels(rgb,4.515,1.852)
        self.assertEqual(bev.shape,(4,128,128));self.assertEqual(bev.dtype,np.uint8)
        self.assertEqual(bev[0,20,20],255);self.assertEqual(bev[1,30,30],255)
        self.assertEqual(bev[2,40,40],255);self.assertEqual(bev[2,63:65,63:65].sum(),0)
        self.assertTrue(np.all(bev[3]==255))
        self.assertEqual(SemanticBEVSpec().extent_m/SemanticBEVSpec().size,0.5)


if __name__=='__main__':unittest.main()
