__author__ = 'haixu'

import torch
import torch.nn as nn


class Warp(nn.Module):
    def __init__(self, inc, outc, neighbour=3):
        super(Warp, self).__init__()
        self.neighbour = neighbour
        self.zero_padding = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(inc, outc, kernel_size=neighbour, stride=neighbour, bias=None)
        self.warp_gate = nn.Conv2d(inc, neighbour * neighbour, kernel_size=3, padding=1, stride=1)
        nn.init.constant_(self.warp_gate.weight, 0)
        self.warp_gate.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, info):
        x = info[0]
        offset = info[1]

        dtype = offset.data.type()
        N = self.neighbour * self.neighbour

        m = torch.sigmoid(self.warp_gate(x))
        x = self.zero_padding(x)
        ## Neighbourhood Warp Operation
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_warped = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        ## Warp Gate
        m = m.contiguous().permute(0, 2, 3, 1)
        m = m.unsqueeze(dim=1)
        m = torch.cat([m for _ in range(x_warped.size(1))], dim=1)
        x_warped *= m

        x_warped = self._reshape_x_warped(x_warped, self.neighbour)
        out = self.conv(x_warped)
        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.neighbour - 1) // 2, (self.neighbour - 1) // 2 + 1),
            torch.arange(-(self.neighbour - 1) // 2, (self.neighbour - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(torch.arange(1, h + 1, 1), torch.arange(1, w + 1, 1))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)
        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_warped = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_warped

    @staticmethod
    def _reshape_x_warped(x_warped, neighbour):
        b, c, h, w, N = x_warped.size()
        x_warped = torch.cat(
            [x_warped[..., s:s + neighbour].contiguous().view(b, c, h, w * neighbour) for s in range(0, N, neighbour)],
            dim=-1)
        x_warped = x_warped.contiguous().view(b, c, h * neighbour, w * neighbour)
        return x_warped


class MotionGRU(nn.Module):
    def __init__(self, in_channel, motion_hidden, neighbour):
        super(MotionGRU, self).__init__()
        self.update = nn.Conv2d(in_channel + motion_hidden, motion_hidden, kernel_size=3, stride=1, padding=1)
        nn.init.constant_(self.update.weight, 0)
        self.update.register_backward_hook(self._set_lr)

        self.reset = nn.Conv2d(in_channel + motion_hidden, motion_hidden, kernel_size=3, stride=1, padding=1)
        nn.init.constant_(self.reset.weight, 0)
        self.reset.register_backward_hook(self._set_lr)

        self.output = nn.Conv2d(in_channel + motion_hidden, motion_hidden, kernel_size=3, stride=1, padding=1)
        nn.init.constant_(self.output.weight, 0)
        self.output.register_backward_hook(self._set_lr)

        self.warp = Warp(in_channel, in_channel, neighbour)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x_t, pre_offset, mean):
        stacked_inputs = torch.cat([x_t, pre_offset], dim=1)
        update_gate = torch.sigmoid(self.update(stacked_inputs))
        reset_gate = torch.sigmoid(self.reset(stacked_inputs))
        offset = torch.tanh(self.output(torch.cat([x_t, pre_offset * reset_gate], dim=1)))
        offset = pre_offset * (1 - update_gate) + offset * update_gate
        mean = mean + 0.5 * (pre_offset - mean)
        offset = offset + mean

        x_t = self.warp([x_t, offset])
        return x_t, offset, mean
