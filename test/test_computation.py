import unittest
import torch
import numpy as np
from memory_efficient_attention import efficient_dot_product_attention_pt, efficient_dot_product_attention_jax
import math
#from flax.linen.attention import dot_product_attention

def dot_product_attention(query, key, value, bias, mask, return_weights = False):
    query = torch.from_numpy(query)
    key = torch.from_numpy(key)
    value = torch.from_numpy(value)
    weights = torch.einsum('...qhd,...khd->...qhk', query / math.sqrt(key.shape[-1]), key)
    if bias is not None:
        bias = torch.from_numpy(bias)
        bias = torch.einsum('...hqk->...qhk', bias)
        weights += bias
    if mask is not None:
        mask = torch.from_numpy(mask)#.to(torch.bool)
        big_neg = torch.finfo(weights.dtype).min
        big_neg = torch.tensor(big_neg, dtype=torch.float32)
        mask = torch.einsum('...hqk->...qhk', mask)
        weights = torch.where(mask, weights, big_neg)
    postweights = torch.nn.functional.softmax(weights, dim=-1)
    if return_weights:
        return torch.einsum('...vhf,...qhv->...qhf', value, postweights), weights
    else:
        return torch.einsum('...vhf,...qhv->...qhf', value, postweights)
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
    def calc_pt(data, **kwparams):
        Qb, Kb, Vb, Mb, Bb = data
        Qbt = torch.tensor(Qb, requires_grad=True)
        Kbt = torch.tensor(Kb, requires_grad=True)
        Vbt = torch.tensor(Vb, requires_grad=True)
        Bbt = torch.tensor(Bb, requires_grad=True)
        Mbt = torch.tensor(Mb)
        return efficient_dot_product_attention_pt(Qbt, Kbt, Vbt, Mbt, Bbt, **kwparams).detach().numpy()

    @staticmethod
    def calc_jax(data):
        Qb, Kb, Vb, Mb, Bb = data
        return np.asarray(efficient_dot_product_attention_jax(Qb, Kb, Vb, Mb, Bb))

    @staticmethod
    def calc_flax(data):
        Qb, Kb, Vb, Mb, Bb = data
        return np.asarray(dot_product_attention(Qb, Kb, Vb, Bb, Mb))

    def test_pt(self):
        data = ComputationTest.data()
        res_pt = ComputationTest.calc_pt(data)
        res_flax = ComputationTest.calc_flax(data)
        self.assertTrue(np.allclose(res_pt, res_flax))
        res_pt, weights_pt = ComputationTest.calc_pt(data, return_weights=True)
        res_flax, weights_flax = ComputationTest.calc_flax(data, return_weights=True)
        self.assertTrue(np.allclose(res_pt, res_flax))
        self.assertTrue(np.allclose(weights_pt, weights_flax))

    def test_jax(self):
        data = ComputationTest.data()
        res_jax = ComputationTest.calc_jax(data)
        res_flax = ComputationTest.calc_flax(data)
        self.assertTrue(np.allclose(res_jax, res_flax))

    def test_jax_and_pt(self):
        data = ComputationTest.data()
        res_pt = ComputationTest.calc_pt(data)
        res_jax = ComputationTest.calc_jax(data)
        res_flax = ComputationTest.calc_flax(data)
        self.assertTrue(np.allclose(res_pt, res_jax))
        self.assertTrue(np.allclose(res_pt, res_flax))
        self.assertTrue(np.allclose(res_jax, res_flax))


if __name__ == '__main__':
    unittest.main()
