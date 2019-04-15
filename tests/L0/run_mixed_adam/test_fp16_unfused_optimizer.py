import unittest
import torch
import apex
import os

class DummyModel(torch.nn.Module):
    def __init__(self, D_in, D_out, D2_out, D3_out):
        super(DummyModel, self).__init__()
        self.dense1 = torch.nn.Linear(D_in, D_out)
        self.dense2 = torch.nn.Linear(D_out, D2_out)
        self.dense3 = torch.nn.Linear(D2_out, D3_out)

    def forward(self, x):
        output = self.dense1(x)
        output = self.dense2(output)
        output = self.dense3(output)
        return output
        
class TestFP16UnfusedOptimizer(unittest.TestCase):
    def setUp(self, max_abs_diff=1e-3, max_rel_diff=1, iters=7):
        self.max_abs_diff = max_abs_diff
        self.max_rel_diff = max_rel_diff
        self.iters = iters
        torch.cuda.manual_seed(13337)

        N, D_in, D_out, D2_out, D3_out = 64,256, 256, 256, 256
        self.N = N
        self.D_in = D_in
        self.D_out = D_out
        self.D2_out = D2_out
        self.D3_out = D3_out
        self.x = torch.randn((N, D_in), dtype=torch.float16, device='cuda')
        self.ref_model = DummyModel(D_in, D_out, D2_out, D3_out).cuda().half()
        self.tst_model = DummyModel(D_in, D_out, D2_out, D3_out).cuda().half()
        for p,q in zip(self.tst_model.parameters(), self.ref_model.parameters()):
            p.data.copy_(q.data)

    def get_max_diff(self, ref_param, tst_param):
        max_abs_diff = max_rel_diff = 0
        for p_ref, p_tst in zip(ref_param, tst_param):
            max_abs_diff_p = (p_ref - p_tst).abs().max().item()
            max_rel_diff_p = ((p_ref - p_tst) / p_ref).abs().max().item()

            if max_abs_diff_p > max_abs_diff:  max_abs_diff = max_abs_diff_p
            if max_rel_diff_p > max_rel_diff:  max_rel_diff = max_rel_diff_p

        return max_abs_diff, max_rel_diff

    def test_fp16_optimizer(self):
        ref_optim = apex.optimizers.FusedLamb(self.ref_model.parameters())
        print("Ref Opt Param Group", ref_optim.param_groups)
        ref_optim = apex.optimizers.FP16_Optimizer(ref_optim, verbose=False)
        print("\n Ref Opt Param Group after passing through FP16 Opt", ref_optim.param_groups)
        
        tst_optim = apex.optimizers.FusedLamb(self.tst_model.parameters())
        print("\n \n Test Opt Param group", tst_optim.param_groups)
        tst_optim = apex.optimizers.FP16_UnfusedOptimizer(tst_optim)
        print("\n Test opt Param Group after passing though FP16 Opt", tst_optim.optimizer.param_groups)
        for i in range(self.iters):
            ref_loss = self.ref_model(self.x).sum()
            ref_optim.backward(ref_loss)
            print("\n Ref Step")
            ref_optim.step()

            print("\n Test Step")
            tst_loss = self.tst_model(self.x).sum()
            tst_optim.backward(tst_loss)
            tst_optim.step()

            max_abs_diff, max_rel_diff = self.get_max_diff(self.ref_model.parameters(), self.tst_model.parameters())
            self.assertLessEqual(max_abs_diff, self.max_abs_diff)
            self.assertLessEqual(max_rel_diff, self.max_rel_diff)


    def test_loss_scaling(self):

        ref_optim = apex.optimizers.FusedLamb(self.ref_model.parameters())
        ref_optim = apex.fp16_utils.FP16_Optimizer(ref_optim, static_loss_scale=128.0, verbose=False)

        tst_optim = apex.optimizers.FusedLamb(self.tst_model.parameters())
        tst_optim = apex.optimizers.FP16_UnfusedOptimizer(tst_optim, static_loss_scale=128.0)

        for i in range(self.iters):
            ref_loss = self.ref_model(self.x).sum()
            ref_optim.backward(ref_loss)
            ref_optim.step()

            tst_loss = self.tst_model(self.x).sum()
            tst_optim.backward(tst_loss)
            tst_optim.step()

            max_abs_diff, max_rel_diff = self.get_max_diff(self.ref_model.parameters(), self.tst_model.parameters())
            self.assertLessEqual(max_abs_diff, self.max_abs_diff)
            self.assertLessEqual(max_rel_diff, self.max_rel_diff)

    def test_parameter_groups(self):

        ref_groups = [{'params': [self.ref_model.weight]},{'params': [self.ref_model.bias]}]
        ref_optim = apex.optimizers.FusedLamb(ref_groups)
        ref_optim = apex.optimizers.FP16_Optimizer(ref_optim, verbose=False)

        tst_groups = [{'params': [self.tst_model.weight]},{'params': [self.tst_model.bias]}]
        tst_optim = apex.optimizers.FusedLamb(tst_groups)
        tst_optim = apex.optimizers.FP16_UnfusedOptimizer(tst_optim)

        for i in range(self.iters):
            ref_loss = self.ref_model(self.x).sum()
            ref_optim.backward(ref_loss)
            ref_optim.step()

            tst_loss = self.tst_model(self.x).sum()
            tst_optim.backward(tst_loss)
            tst_optim.step()

            max_abs_diff, max_rel_diff = self.get_max_diff(self.ref_model.parameters(), self.tst_model.parameters())
            self.assertLessEqual(max_abs_diff, self.max_abs_diff)
            self.assertLessEqual(max_rel_diff, self.max_rel_diff)

    def test_grad_clip(self):
        ref_optim = apex.optimizers.FusedLamb(self.ref_model.parameters())
        ref_optim = apex.fp16_utils.FP16_Optimizer(ref_optim, verbose=False)

        tst_optim = apex.optimizers.FusedLamb(self.tst_model.parameters(), max_grad_norm=0.01)
        tst_optim = apex.optimizers.FP16_UnfusedOptimizer(tst_optim)

        for i in range(self.iters):
            ref_loss = self.ref_model(self.x).sum()
            ref_optim.backward(ref_loss)
            ref_optim.clip_master_grads(0.01)
            ref_optim.step()

            tst_loss = self.tst_model(self.x).sum()
            tst_optim.backward(tst_loss)
            tst_optim.step()

            max_abs_diff, max_rel_diff = self.get_max_diff(self.ref_model.parameters(), self.tst_model.parameters())
            self.assertLessEqual(max_abs_diff, self.max_abs_diff)
            self.assertLessEqual(max_rel_diff, self.max_rel_diff)

    @unittest.skip('Not support grad being None')
    def test_grad_None(self):
        self.fail()

    @unittest.skip('Not support same weight decay as pytorch')
    def test_weight_decay(self):
        self.fail()

    @unittest.skip('Not support empty parameter groups')
    def test_group_empty(self):
        self.fail()

if __name__ == '__main__':
    script_path = os.path.dirname(os.path.realpath(__file__))
    unittest.main()