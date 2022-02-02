import unittest
import torch, jax
import numpy as np
from memory_efficient_attention import efficient_dot_product_attention_pt, efficient_dot_product_attention_jax
from flax.linen.attention import dot_product_attention

efficient_dot_product_attention_jax = jax.jit(efficient_dot_product_attention_jax)

jax.config.update('jax_log_compiles', True)

class ComputationTest(unittest.TestCase):
    @staticmethod
    def data():
        b = 8
        Qb = np.random.rand(1, b, 128, 16, 8).astype("float32")
        Kb = np.random.rand(1, b, 128, 16, 8).astype("float32")
        Vb = np.random.rand(1, b, 128, 16, 8).astype("float32")
        Mb = np.random.rand(1, b, 16, 128, 128) > 0.5
        Bb = np.random.rand(1, b, 16, 128, 128).astype("float32") / 100
        return Qb, Kb, Vb, Mb, Bb

    @staticmethod
    def datasets():
        # plain data
        Qb, Kb, Vb, Mb, Bb = ComputationTest.data()

        def MBchunker(tensor, requires_grad):
            def get_range_jax(query_start, query_size, key_start, key_size):
                return jax.lax.dynamic_slice(tensor, tuple([0] * (tensor.ndim - 2)) + (query_start, key_start),
                        slice_sizes=tuple(tensor.shape[:-2]) + (query_size, key_size))
            def get_range_torch(query_start, query_size, key_start, key_size):
                return torch.tensor(tensor[:,:,:,query_start:query_start+query_size,key_start:key_start+key_size], requires_grad=requires_grad)
            return dict(jax=get_range_jax, torch=get_range_torch)

        # mask broadcasting
        for Mb_variant in (Mb, Mb[:,:,:1,:,:], Mb[:,:,:,:1,:], Mb[:,:,:,:,:1], MBchunker(Mb, False)):
            # bias broadcasting
            for Bb_variant in (Bb, Bb[:,:,:1,:,:], Bb[:,:,:,:1,:], Bb[:,:,:,:,:1], MBchunker(Bb, True)):
                yield Qb, Kb, Vb, Mb_variant, Bb_variant

    @staticmethod
    def calc_pt(data):
        Qb, Kb, Vb, Mb, Bb = data
        Qbt = torch.tensor(Qb, requires_grad=True)
        Kbt = torch.tensor(Kb, requires_grad=True)
        Vbt = torch.tensor(Vb, requires_grad=True)
        if type(Bb) is dict:
            Bbt = Bb['torch']
        else:
            Bbt = torch.tensor(Bb, requires_grad=True)
        if type(Mb) is dict:
            Mbt = Mb['torch']
        else:
            Mbt = torch.tensor(Mb)
        return efficient_dot_product_attention_pt(Qbt, Kbt, Vbt, Mbt, Bbt).detach().numpy()

    @staticmethod
    def calc_jax(data):
        Qb, Kb, Vb, Mb, Bb = data
        if type(Mb) is dict:
            Mb = Mb['jax']
        if type(Bb) is dict:
            Bb = Bb['jax']
        return np.asarray(efficient_dot_product_attention_jax(Qb, Kb, Vb, Mb, Bb))

    @staticmethod
    def calc_flax(data):
        Qb, Kb, Vb, Mb, Bb = data
        if type(Mb) is dict:
            Mb = Mb['jax'](0, Qb.shape[-3], 0, Kb.shape[-3])
        if type(Bb) is dict:
            Bb = Bb['jax'](0, Qb.shape[-3], 0, Kb.shape[-3])
        return np.asarray(dot_product_attention(Qb, Kb, Vb, Bb, Mb))

    def test_pt(self):
        for data in ComputationTest.datasets():
            res_pt = ComputationTest.calc_pt(data)
            res_flax = ComputationTest.calc_flax(data)
            self.assertTrue(np.allclose(res_pt, res_flax))

    def test_jax(self):
        for data in ComputationTest.datasets():
            res_jax = ComputationTest.calc_jax(data)
            res_flax = ComputationTest.calc_flax(data)
            self.assertTrue(np.allclose(res_jax, res_flax))

    def test_jax_and_pt(self):
        for data in ComputationTest.datasets():
            res_pt = ComputationTest.calc_pt(data)
            res_jax = ComputationTest.calc_jax(data)
            res_flax = ComputationTest.calc_flax(data)
            self.assertTrue(np.allclose(res_pt, res_jax))
            self.assertTrue(np.allclose(res_pt, res_flax))
            self.assertTrue(np.allclose(res_jax, res_flax))


if __name__ == '__main__':
    unittest.main()
