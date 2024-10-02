import torch, unittest


class Test0100(unittest.TestCase):
    def test_import(self):
        import sde_bbdm

        bbdm_version = sde_bbdm.VERSION
        self.assertGreaterEqual(bbdm_version, "0.1a")
        pass
