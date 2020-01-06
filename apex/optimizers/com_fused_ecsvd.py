import types
import torch
import importlib
from .compression import *
import torch.distributed as dist
import time


class ComFusedECSVD(torch.optim.Optimizer):
    """Implements Adam algorithm. Currently GPU-only.  Requires Apex to be installed via
    ``python setup.py install --cuda_ext --cpp_ext``.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params,
                 lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-8, eps_inside_sqrt=False,
                 weight_decay=0., max_grad_norm=0., amsgrad=False):
        global fused_adam_cuda
        fused_adam_cuda = importlib.import_module("fused_adam_cuda")
        global fused_mixed_cuda
        fused_mixed_cuda = importlib.import_module("fused_mixed_cuda")

        if amsgrad:
            raise RuntimeError('FusedAdam does not support the AMSGrad variant.')
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(ComFusedECSVD, self).__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1

    def _compute_grad_norm(self, fp16_grads_flat, norm_type=2):
        try:
            norm = float(torch.norm(fp16_grads_flat, 2.0, dtype=torch.float32))
        except TypeError as err:
            norm = float(torch.norm(fp16_grads_flat.float(), 2.0))
        if norm == float('inf') or norm == -float('inf') or norm != norm:
            # print(norm)
            return -1
        else:
            return norm


    def _orthogonalize(matrix):
        n, m = matrix.shape
        for i in range(m):
            # Normalize the i'th column
            col = matrix[:, i: i + 1]
            col /= torch.sqrt(torch.sum(col ** 2))
            # Project it on the rest and remove it
            if i + 1 < m:
                rest = matrix[:, i + 1:]
                # rest -= torch.matmul(col.t(), rest) * col
                rest -= torch.sum(col * rest, dim=0) * col

    def _warmup(self,matrix,rnk):
        u,s,v = torch.svd(matrix)

        m = matrix.shape[0]
        n = matrix.shape[1]
        rnk = min(m,n,rnk)
        p = torch.mm(u,torch.diag(s))[:,:rnk]
        q = v[:,:rnk]
        return p,q





    def step(self, closure=None, grads=None, output_params=None, scale=1., grad_norms=None, adam_freeze=False, rnk = 2,
             clip_key=True):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output params (list of tensors, optional): A reduced precision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients. (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        myskip = False
        if closure is not None:
            loss = closure()

        if grads is None:
            grads_group = [None] * len(self.param_groups)
        # backward compatibility
        # assuming a list/generator of parameter means single group
        elif isinstance(grads, types.GeneratorType):
            grads_group = [grads]
        elif type(grads[0]) != list:
            grads_group = [grads]
        else:
            grads_group = grads

        if output_params is None:
            output_params_group = [None] * len(self.param_groups)
        elif isinstance(output_params, types.GeneratorType):
            output_params_group = [output_params]
        elif type(output_params[0]) != list:
            output_params_group = [output_params]
        else:
            output_params_group = output_params

        if grad_norms is None:
            grad_norms = [None] * len(self.param_groups)

        start_time = time.time()

        for group, grads_this_group, output_params_this_group, grad_norm_group in zip(self.param_groups, grads_group,
                                                                                      output_params_group, grad_norms):
            if grads_this_group is None:
                grads_this_group = [None] * len(group['params'])
            if output_params_this_group is None:
                output_params_this_group = [None] * len(group['params'])

            if grad_norm_group is None:
                grad_norm_group = [None] * len(group['params'])
            elif not isinstance(grad_norm_group, list):
                grad_norm_group = [grad_norm_group]

            bias_correction = 1 if group['bias_correction'] else 0

            for p, grad, output_param, grad_norm in zip(group['params'], grads_this_group, output_params_this_group,
                                                        grad_norm_group):

                # compute combined scale factor for this group
                combined_scale = scale
                if group['max_grad_norm'] > 0:
                    # norm is in fact norm*scale
                    clip = ((grad_norm / scale) + 1e-6) / group['max_grad_norm']

                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'FusedAdam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['ecbuffer'] = torch.zeros_like(p.data)
                    state['temp_error'] = torch.zeros_like(p.data)
                    state['temp_exp'] = torch.zeros_like(p.data)
                    # state['temp_final_exp'] = torch.zeros_like(p.data)
                    state['temp_final_error'] = torch.zeros_like(p.data)
                    state['ComQ'] = None
                    state['temp_P'] = None
                    state['temp_Q'] = None

                exp_avg_sq = state['exp_avg_sq']
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                comQ = state['comQ']

                if not ('temp_final_error' in state):
                    state['temp_final_error'] = torch.zeros_like(p.data)

                # model_update = state['model_update']

                out_p = torch.tensor([], dtype=torch.float) if output_param is None else output_param
                if not adam_freeze or len(grad)<100:
                    dist.all_reduce(grad)
                    grad.mul_(1 / dist.get_world_size())

                    if self._compute_grad_norm(grad) == -1:
                        myskip = True
                        return True
                else:
                    if self._compute_grad_norm(grad) == -1:
                        print("Gradient Exploding")
                    # if dist.get_rank() == 0 or dist.get_rank()==16:
                    #   if (state['step']+1)%10 ==0:
                    #      print('Grad is:  ',grad[0:10])


                    scaled_grad = grad.float() / combined_scale
                    ecbuffer = state['ecbuffer']
                    buffer_exp = exp_avg * beta1 + (1 - beta1) * scaled_grad + ecbuffer.data
                    # buffer_avg.mul_(beta1).add_(1-beta1,scaled_grad)

                    if comQ is None:
                        P, Q = self._warmup(exp_avg, rnk)
                        comQ.set_(Q)
                        state['temp_P'] = P
                        state['temp_Q'] = Q


                    if self._compute_grad_norm(ecbuffer) == -1:
                        print('Getting error in ecbuffer')
                        return True

                    temp_P = state['temp_P']
                    temp_Q = state['temp_Q']

                    matlen = int(np.sqrt(len(buffer_exp)))
                    matrix = buffer_exp.view(matlen,-1)

                    torch.matmul(matrix, comQ, out=temp_P)
                    dist.all_reduce(temp_P)
                    temp_P /= dist.get_world_size()
                    self._orthogonalize(temp_P)
                    torch.matmul(matrix.t(), temp_P, out=temp_Q)
                    dist.all_reduce(temp_Q)
                    temp_Q /= dist.get_world_size()

                    temp_exp = torch.mm(temp_P,temp_Q.t())
                    # if dist.get_rank() == 0 or dist.get_rank()==16:
                    #   if (state['step']+1)%10 ==0:
                    #      print('Before is:  ',temp_exp[0:10])
                    temp_error = buffer_exp.data  - temp_exp

                    # dist.all_reduce(temp_exp)
                    # temp_exp.mul_(1 / dist.get_world_size())
                    # if dist.get_rank() == 0 or dist.get_rank()==16:
                    #   if (state['step']+1)%10 ==0:
                    #      print('After is:  ',temp_exp[0:10])

                    state['temp_exp'] = temp_exp
                    state['temp_error'] = temp_error
                    if self._compute_grad_norm(temp_exp) == -1:
                        myskip = True
                        print("Compressed Gradient Exploding")
                        return True

        end_time = time.time()
        if dist.get_rank() == 0:
            print("Communication time overall for one step is", start_time - end_time)

        for group, grads_this_group, output_params_this_group, grad_norm_group in zip(self.param_groups, grads_group,
                                                                                      output_params_group, grad_norms):

            bias_correction = 1 if group['bias_correction'] else 0
            if not isinstance(grad_norm_group, list):
                grad_norm_group = [grad_norm_group]
            for p, grad, output_param, grad_norm in zip(group['params'], grads_this_group, output_params_this_group,
                                                        grad_norm_group):
                combined_scale = scale
                state = self.state[p]

                exp_avg_sq = state['exp_avg_sq']
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                temp_error = state['temp_error']
                temp_exp = state['temp_exp']

                out_p = torch.tensor([], dtype=torch.float) if output_param is None else output_param
                if not adam_freeze:
                    # print(exp_avg)
                    if group['max_grad_norm'] > 0 and clip_key:
                        grad_norm = torch.norm(grad.float())
                        clip = ((grad_norm / scale) + 1e-6) / group['max_grad_norm']
                        if clip > 1:
                            if dist.get_rank() == 0:
                                print('Clip is:  ', clip)
                            combined_scale = scale * clip

                    fused_adam_cuda.adam(p.data,
                                         out_p,
                                         exp_avg,
                                         exp_avg_sq,
                                         grad,
                                         group['lr'],
                                         beta1,
                                         beta2,
                                         group['eps'],
                                         combined_scale,
                                         state['step'],
                                         self.eps_mode,
                                         bias_correction,
                                         group['weight_decay'])
                else:
                    state['ecbuffer'].data.set_(temp_error.data)
                    state['comQ'] = state['temp_Q']
                    exp_avg.set_(temp_exp)
                    # if dist.get_rank() == 0 or dist.get_rank() == 10:
                    #   if (state['step']+1)%1 ==0:
                    #      print('After2 is:  ',exp_avg[0:10])
                    fused_mixed_cuda.mixed(p.data,
                                           out_p,
                                           exp_avg,
                                           exp_avg_sq,
                                           grad,
                                           group['lr'],
                                           beta1,
                                           beta2,
                                           group['eps'],
                                           combined_scale,
                                           state['step'],
                                           self.eps_mode,
                                           bias_correction,
                                           group['weight_decay'])

                state['step'] += 1

        return myskip
