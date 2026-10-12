def autopad(k, p=None, d=1):
    if p is not None:
        return p

    if isinstance(k, (tuple, list)):
        if isinstance(d, int):
            return tuple(((x - 1) * d) // 2 for x in k)
        elif isinstance(d, (tuple, list)):
            return tuple(((x - 1) * dd) // 2 for x, dd in zip(k, d))
        else:
            raise TypeError(f"Unsupported dilation type: {type(d)}")

    if isinstance(d, int):
        return ((k - 1) * d) // 2

    raise TypeError(f"Unsupported kernel/dilation combination: k={k}, d={d}")


class Conv(nn.Module):

    default_act = nn.SiLU(inplace=True)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()

        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)

        if act is True:
            self.act = self.default_act
        elif isinstance(act, nn.Module):
            self.act = copy.deepcopy(act)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CDF53Wavelet(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.channels = channels
        self.pad_size = 2

        h0 = torch.tensor([-0.125, 0.250, 0.750, 0.250, -0.125], dtype=torch.float32)
        h1 = torch.tensor([0.000, -0.250, 0.500, -0.250, 0.000], dtype=torch.float32)

        ll = torch.ger(h0, h0)
        lh = torch.ger(h1, h0)
        hl = torch.ger(h0, h1)
        hh = torch.ger(h1, h1)


        filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1)


        filters = filters.repeat(channels, 1, 1, 1)

        self.register_buffer("filters", filters)

    def _pad(self, x):
        h, w = x.shape[-2:]
        mode = "reflect" if h > self.pad_size and w > self.pad_size else "replicate"
        return F.pad(
            x,
            (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
            mode=mode,
        )

    def forward(self, x):
        x = self._pad(x)

        out = F.conv2d(
            x,
            self.filters,
            stride=2,
            padding=0,
            groups=self.channels,
        )

        b, _, h, w = out.shape


        out = out.view(b, self.channels, 4, h, w)

        LL = out[:, :, 0, :, :].contiguous()
        LH = out[:, :, 1, :, :].contiguous()
        HL = out[:, :, 2, :, :].contiguous()
        HH = out[:, :, 3, :, :].contiguous()

        return LL, LH, HL, HH


class WFE(nn.Module):

    def __init__(self, c1, embed_dim=32):
        super().__init__()

        self.c_reduce = min(max(16, c1 // 4), 64)

        self.reduce = nn.Sequential(
            nn.Conv2d(c1, self.c_reduce, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.c_reduce),
            nn.SiLU(inplace=True),
        )

        self.dwt = CDF53Wavelet(self.c_reduce)


        self.beta_raw = nn.Parameter(torch.tensor(-1.2587))


        feat_dim = self.c_reduce * 8
        hidden_dim = max(32, embed_dim * 2)

        self.fc = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    @staticmethod
    def rms_energy(x):
        return torch.sqrt(x.pow(2).mean(dim=(2, 3), keepdim=True) + 1e-6)

    @staticmethod
    def magnitude_shrink(x, thr):
        return F.relu(x.abs() - thr)

    @staticmethod
    def hf_pool(x):
        avg_feat = F.adaptive_avg_pool2d(x, 1).flatten(1)
        max_feat = F.adaptive_max_pool2d(x, 1).flatten(1)
        return avg_feat, max_feat

    def forward(self, x):
        x = self.reduce(x)

        LL, LH, HL, HH = self.dwt(x)


        ll_mean = LL.mean(dim=(2, 3))
        ll_std = torch.sqrt(LL.var(dim=(2, 3), unbiased=False) + 1e-6)


        sigma_lh = self.rms_energy(LH)
        sigma_hl = self.rms_energy(HL)
        sigma_hh = self.rms_energy(HH)

        sigma_hf = (sigma_lh + sigma_hl + sigma_hh) / 3.0

        beta = F.softplus(self.beta_raw) + 1e-6
        thr = beta * sigma_hf

        LH = self.magnitude_shrink(LH, thr)
        HL = self.magnitude_shrink(HL, thr)
        HH = self.magnitude_shrink(HH, thr)


        lh_avg, lh_max = self.hf_pool(LH)
        hl_avg, hl_max = self.hf_pool(HL)
        hh_avg, hh_max = self.hf_pool(HH)

        feat = torch.cat(
            [
                ll_mean,
                ll_std,
                lh_avg,
                lh_max,
                hl_avg,
                hl_max,
                hh_avg,
                hh_max,
            ],
            dim=1,
        )

        return self.fc(feat)


class WGADConv(nn.Module):

    def __init__(
        self,
        c1,
        c2,
        k=3,
        s=1,
        p=None,
        g=1,
        d=1,
        num_bases=3,
        latent_ratio=4,
        embed_dim=32,
        act=True,
    ):
        super().__init__()

        if isinstance(k, (tuple, list)):
            if len(k) != 2 or k[0] != k[1]:
                raise ValueError(f"WGADConv only supports square kernel, got k={k}")
            k = k[0]

        self.k = k
        self.s = s
        self.padding = autopad(k, p, d)
        self.d = d

        raw_cm = min(c1, c2) // latent_ratio
        self.cm = max(16, min(raw_cm, 128))


        if self.cm < 32:
            self.groups = 1
        elif self.cm < 64:
            self.groups = 2
        else:
            self.groups = 4

        self.cm = max(self.groups, (self.cm // self.groups) * self.groups)
        cin_g = self.cm // self.groups


        self.static = nn.Sequential(
            nn.Conv2d(
                c1,
                c2,
                kernel_size=k,
                stride=s,
                padding=self.padding,
                groups=g,
                dilation=d,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
        )


        self.encoder = WFE(c1, embed_dim=embed_dim)


        self.reduce = nn.Sequential(
            nn.Conv2d(c1, self.cm, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.cm),
            nn.SiLU(inplace=True),
        )


        self.basis = nn.Parameter(
            torch.randn(num_bases, self.cm, cin_g, k, k)
            * math.sqrt(2.0 / (cin_g * k * k))
        )


        self.alpha = nn.Sequential(
            nn.Linear(embed_dim, num_bases),
            nn.LayerNorm(num_bases),
        )


        self.tau_raw = nn.Parameter(torch.tensor(0.5413))
        self.lambda_raw = nn.Parameter(torch.tensor(-4.0))


        self.expand = nn.Sequential(
            nn.Conv2d(self.cm, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
        )

        if act is True:
            self.act = nn.SiLU(inplace=True)
        elif isinstance(act, nn.Module):
            self.act = copy.deepcopy(act)
        else:
            self.act = nn.Identity()

    def _group_conv(self, x, weight):
        b, c, h, w = x.shape
        cin_g = c // self.groups

        x = x.reshape(1, b * c, h, w)
        weight = weight.reshape(b * c, cin_g, self.k, self.k)

        y = F.conv2d(
            x,
            weight,
            stride=self.s,
            padding=self.padding,
            dilation=self.d,
            groups=b * self.groups,
        )

        return y.reshape(b, c, y.shape[-2], y.shape[-1])

    def forward(self, x):

        z = self.encoder(x)


        y_static = self.static(x)


        x_lat = self.reduce(x)

        tau = F.softplus(self.tau_raw) + 1e-6
        alpha = torch.softmax(self.alpha(z) / tau, dim=1)

        kernel = torch.einsum("bm,mcoij->bcoij", alpha, self.basis)

        y_dyn = self._group_conv(x_lat, kernel)
        y_dyn = self.expand(y_dyn)

        lambda_dyn = torch.sigmoid(self.lambda_raw)

        y = y_static + lambda_dyn * y_dyn

        return self.act(y)


class Bottleneck_WGAD(nn.Module):

    def __init__(
        self,
        c1,
        c2,
        shortcut=True,
        g=1,
        k=(3, 3),
        e=0.5,
        num_bases=3,
        embed_dim=32,
    ):
        super().__init__()

        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = WGADConv(
            c_,
            c2,
            k=k[1],
            s=1,
            g=g,
            num_bases=num_bases,
            embed_dim=embed_dim,
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k_WGAD(nn.Module):

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=True,
        g=1,
        e=0.5,
        k=3,
        num_bases=3,
        embed_dim=32,
    ):
        super().__init__()

        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)

        self.m = nn.Sequential(
            *[
                Bottleneck_WGAD(
                    c_,
                    c_,
                    shortcut=shortcut,
                    g=g,
                    k=(1, k),
                    e=1.0,
                    num_bases=num_bases,
                    embed_dim=embed_dim,
                )
                for _ in range(n)
            ]
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3k2_WGAD(nn.Module):

    def __init__(
        self,
        c1,
        c2,
        n=1,
        c3k=False,
        e=0.5,
        g=1,
        shortcut=True,
        num_bases=3,
        embed_dim=32,
    ):
        super().__init__()

        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)

        self.m = nn.ModuleList(
            [
                C3k_WGAD(
                    self.c,
                    self.c,
                    n=2,
                    shortcut=shortcut,
                    g=g,
                    e=0.5,
                    k=3,
                    num_bases=num_bases,
                    embed_dim=embed_dim,
                )
                if c3k
                else Bottleneck_WGAD(
                    self.c,
                    self.c,
                    shortcut=shortcut,
                    g=g,
                    k=(3, 3),
                    e=1.0,
                    num_bases=num_bases,
                    embed_dim=embed_dim,
                )
                for _ in range(n)
            ]
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))

        for block in self.m:
            y.append(block(y[-1]))

        return self.cv2(torch.cat(y, dim=1))


class GSDConv(nn.Module):

    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True):
        super().__init__()

        if s != 2:
            raise ValueError(f"GSDConv is designed for stride-2 Conv replacement, got s={s}")

        if isinstance(k, (tuple, list)):
            if len(k) != 2 or k[0] != k[1]:
                raise ValueError(f"GSDConv only supports square kernel, got k={k}")
            k = k[0]

        if k % 2 != 1:
            raise ValueError(f"GSDConv kernel size should be odd, got k={k}")


        self.pixel_unshuffle = nn.PixelUnshuffle(2)


        phase_mid = 2 * c1
        self.phase_mid = phase_mid


        self.phase_aggr = nn.Conv2d(
            in_channels=4 * c1,
            out_channels=phase_mid,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=c1,
            bias=False,
        )
        self.phase_bn = nn.BatchNorm2d(phase_mid)
        self.phase_act = self._make_act(act)


        self.pw = nn.Sequential(
            nn.Conv2d(phase_mid, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            self._make_act(act),
        )


        self.dw = nn.Sequential(
            nn.Conv2d(
                c2,
                c2,
                kernel_size=k,
                stride=1,
                padding=autopad(k, p, d),
                groups=c2,
                dilation=d,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            self._make_act(act),
        )

        self._init_weights(c1)

    @staticmethod
    def _make_act(act=True):
        if act is True:
            return nn.SiLU(inplace=True)
        if isinstance(act, nn.Module):
            return copy.deepcopy(act)
        return nn.Identity()

    def _init_weights(self, c1):
        with torch.no_grad():
            self.phase_aggr.weight.zero_()


            for ch in range(c1):
                avg_out = 2 * ch
                reserve_out = 2 * ch + 1


                self.phase_aggr.weight[avg_out, :, 0, 0] = 0.25


                self.phase_aggr.weight[reserve_out, :, 0, 0] = 0.0

            nn.init.constant_(self.phase_bn.weight, 1.0)
            nn.init.constant_(self.phase_bn.bias, 0.0)

    @staticmethod
    def _pad_to_even(x):
        h, w = x.shape[-2:]

        pad_h = h % 2
        pad_w = w % 2

        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        return x

    def forward(self, x):

        x = self._pad_to_even(x)


        x = self.pixel_unshuffle(x)


        x = self.phase_aggr(x)
        x = self.phase_bn(x)
        x = self.phase_act(x)


        x = self.pw(x)


        x = self.dw(x)

        return x

import torch
import torch.nn as nn


class GSPUp(nn.Module):

    def __init__(self, c1, scale_factor=2, act=True):
        super().__init__()

        if scale_factor != 2:
            raise ValueError(f"GSPUp only supports scale_factor=2, got {scale_factor}")

        self.c1 = c1

        self.phase_expand = nn.Conv2d(
            c1,
            c1 * 4,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=c1,
            bias=False,
        )

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.detail_extract = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True) if act else nn.Identity(),
        )

        self.detail_expand = nn.Conv2d(
            c1,
            c1 * 4,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=c1,
            bias=True,
        )

        self.gamma_raw = nn.Parameter(torch.full((1, c1, 1, 1), -4.0))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():

            self.phase_expand.weight.fill_(1.0)


            nn.init.normal_(self.detail_expand.weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.detail_expand.bias, 0.0)

    def forward(self, x):
        base = self.pixel_shuffle(self.phase_expand(x))

        feat = self.detail_extract(x)
        detail = self.pixel_shuffle(self.detail_expand(feat))

        residual = torch.sigmoid(detail) * torch.tanh(detail)
        gamma = torch.sigmoid(self.gamma_raw)

        return base + gamma * residual
