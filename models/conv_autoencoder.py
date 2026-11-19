"""Convolutional autoencoders shared by training and fixed-point recovery."""

import math
from operator import index as integer_index
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# Shared blocks preserve the parameter names used by released checkpoints.
def _make_norm(num_channels, norm_type="bn", gn_groups=8):
    """
    norm_type: "bn" | "gn" | "none"
    """
    if norm_type == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "gn":
        if integer_index(gn_groups) <= 0:
            raise ValueError("gn_groups must be a positive integer.")
        g = min(gn_groups, num_channels)
        while g > 1 and (num_channels % g != 0):
            g -= 1
        return nn.GroupNorm(g, num_channels)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")


def _normalize_spatial_size(input_size):
    if isinstance(input_size, int):
        result = (input_size, input_size)
    elif isinstance(input_size, (tuple, list)) and len(input_size) == 2:
        result = tuple(integer_index(value) for value in input_size)
    else:
        raise ValueError("input_size must be an int or a tuple/list of length 2.")
    if min(result) <= 0:
        raise ValueError("input_size must contain positive integers.")
    return result


def _conv_block(in_channels, out_channels, stride=1, kernel_size=3, use_sn=False, norm_type="bn", gn_groups=8):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
    if use_sn:
        conv = spectral_norm(conv)
    return nn.Sequential(
        conv,
        _make_norm(out_channels, norm_type, gn_groups),
        nn.ReLU(inplace=True)
    )

class UpSampleBlock(nn.Module):
    """
    Deconv or resize-conv upsampling block.
    """
    def __init__(self, in_channels, out_channels, stride=2, use_sn=False, upsample_mode="resize_conv", norm_type="bn", gn_groups=8):
        super().__init__()
        self.upsample_mode = upsample_mode
        self.stride = stride
        self.out_channels = out_channels
        if self.upsample_mode == "deconv":
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        elif self.upsample_mode in {"resize_conv", "nearest_conv"}:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError("upsample_mode must be 'deconv', 'resize_conv', or 'nearest_conv'")
        self.up = spectral_norm(conv) if use_sn else conv
        self.bn = _make_norm(out_channels, norm_type, gn_groups)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, target_size=None):
        if self.upsample_mode == "deconv":
            if target_size is not None:
                output_size = (x.shape[0], self.out_channels, target_size[0], target_size[1])
                x = self.up(x, output_size=output_size)
            else:
                x = self.up(x)
        else:
            if target_size is not None or self.stride > 1:
                mode = "nearest" if self.upsample_mode == "nearest_conv" else "bilinear"
                x = F.interpolate(x, size=target_size, scale_factor=self.stride if target_size is None else None,
                                  mode=mode, align_corners=False if mode == "bilinear" else None)
            x = self.up(x)
        return self.relu(self.bn(x))


class ResBlock(nn.Module):
    def __init__(self, channels, norm_type="bn", gn_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = _make_norm(channels, norm_type, gn_groups)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = _make_norm(channels, norm_type, gn_groups)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = x + residual
        return self.relu(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type="bn", gn_groups=8):
        super().__init__()
        self.block = _conv_block(
            in_ch,
            out_ch,
            stride=2,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsample_mode="deconv", norm_type="bn", gn_groups=8):
        super().__init__()
        if upsample_mode not in {"deconv", "nearest_conv"}:
            raise ValueError("upsample_mode must be 'deconv' or 'nearest_conv'")
        self.block = UpSampleBlock(
            in_ch,
            out_ch,
            stride=2,
            upsample_mode=upsample_mode,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )

    def forward(self, x, target_size=None):
        return self.block(x, target_size=target_size)


def _default_encoder_channels(base_channels):
    return [
        base_channels,
        base_channels,
        base_channels * 2,
        base_channels * 2,
        base_channels * 4,
        base_channels * 4,
        base_channels * 4,
        base_channels * 4,
        base_channels * 4,
    ]


# Both architectures share reconstruction, feature extraction and clean-image fitting.
class AutoencoderBase(nn.Module):
    def encode(self, x, return_aux=False):
        raise NotImplementedError

    def decode(self, z, aux=None):
        raise NotImplementedError

    def _run_dummy_in_eval(self, forward_fn, in_channels, input_size):
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                spatial_size = _normalize_spatial_size(input_size)
                dummy = torch.zeros(1, in_channels, *spatial_size)
                return forward_fn(dummy)
        finally:
            self.train(was_training)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                w = m.weight_orig if hasattr(m, "weight_orig") else m.weight
                nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        encoded = self.encode(x, return_aux=True)
        if isinstance(encoded, tuple):
            z, aux = encoded
        else:
            z, aux = encoded, None
        return self.decode(z, aux)

    def transform(self, x):
        with torch.no_grad():
            return self.encode(x, return_aux=False)

    def get_recon(self, x):
        with torch.no_grad():
            return torch.clamp(self.forward(x), 0.0, 1.0)

    def fit(self, train_loader, device=None, learning_rate=1e-3, epochs=100, verbose=True,
            shrinkage_weight=0.0, save_ckpt_path=None):
        """Fit one stage with MSE and optional Jacobian shrinkage."""
        epochs = integer_index(epochs)
        if epochs <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive and finite.")
        if not math.isfinite(shrinkage_weight) or shrinkage_weight < 0:
            raise ValueError("shrinkage_weight must be finite and non-negative.")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        criterion_mse = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        last_avg_loss = None
        for epoch in range(epochs):
            self.train()
            total_loss, count = 0.0, 0
            for data in train_loader:
                img = data[0].to(device)
                if img.ndim != 4 or img.numel() == 0 or not img.is_floating_point() or not torch.isfinite(img).all():
                    raise ValueError("Training images must be finite, nonempty floating-point BCHW tensors.")
                if img.min() < 0 or img.max() > 1:
                    raise ValueError("Clean training images must be in [0, 1].")
                img.requires_grad_(shrinkage_weight > 0)
                optimizer.zero_grad()

                recon = self.forward(img)
                if recon.shape != img.shape or not torch.isfinite(recon).all():
                    raise ValueError("AE training output must be finite and match the input shape.")
                loss = criterion_mse(recon, img)
                if shrinkage_weight > 0:
                    v = torch.randn_like(img).to(device)
                    recon_proj = torch.sum(recon * v)
                    grad_img = torch.autograd.grad(
                        outputs=recon_proj,
                        inputs=img,
                        create_graph=True,
                        only_inputs=True,
                    )[0]
                    loss_jacobian = torch.mean(grad_img.view(grad_img.size(0), -1).pow(2).sum(dim=1))
                    loss = loss + shrinkage_weight * loss_jacobian
                if not torch.isfinite(loss):
                    raise FloatingPointError("AE training loss is not finite.")
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * img.shape[0]
                count += img.shape[0]
            if count == 0:
                raise ValueError("The AE training loader is empty.")
            learning_rate_used = optimizer.param_groups[0]["lr"]
            scheduler.step()
            last_avg_loss = total_loss / count
            if verbose and ((epoch + 1) % 10 == 0 or epoch == epochs - 1 or epoch == 0):
                print(f"epoch [{epoch + 1}/{epochs}], lr:{learning_rate_used:.4e}, loss:{last_avg_loss:.4e}")

        if save_ckpt_path is not None:
            ckpt = {
                "epoch": epochs,
                "model_state": self.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "last_avg_loss": last_avg_loss,
            }
            target = Path(save_ckpt_path)
            temporary = target.with_suffix(target.suffix + ".tmp")
            torch.save(ckpt, temporary)
            temporary.replace(target)
            if verbose:
                print(f"Saved checkpoint to {save_ckpt_path}")

        if verbose:
            print(f"Training finished after {epochs} epochs, loss:{last_avg_loss:.4f}")
        return self


# Residual stages retain spatial shapes for exact decoder alignment.
class ResConvAE(AutoencoderBase):
    def __init__(
        self,
        in_channels=3,
        input_size=32,
        stage_channels=(32, 64, 128),
        blocks_per_stage=(1, 2, 2),
        bottleneck_channels=16,
        norm_type="bn",
        gn_groups=8,
        upsample_mode="deconv",
    ):
        super().__init__()
        if upsample_mode not in {"deconv", "nearest_conv"}:
            raise ValueError("upsample_mode must be 'deconv' or 'nearest_conv'")
        if norm_type not in {"bn", "gn", "none"}:
            raise ValueError("norm_type must be 'bn', 'gn', or 'none'")

        self.in_channels = int(in_channels)
        self.input_size = _normalize_spatial_size(input_size)
        self.bottleneck_channels = int(bottleneck_channels)
        self.norm_type = norm_type
        self.gn_groups = int(gn_groups)
        self.upsample_mode = upsample_mode

        self.stage_channels = [int(value) for value in stage_channels]
        self.blocks_per_stage = [int(value) for value in blocks_per_stage]
        if not self.stage_channels:
            raise ValueError("stage_channels must be a non-empty sequence.")
        if any(value <= 0 for value in self.stage_channels):
            raise ValueError("stage_channels must contain positive integers.")
        if len(self.blocks_per_stage) != len(self.stage_channels):
            raise ValueError("blocks_per_stage must have the same length as stage_channels.")
        if any(value < 0 for value in self.blocks_per_stage):
            raise ValueError("blocks_per_stage must contain non-negative integers.")

        encoder_layers = []
        prev_channels = self.in_channels
        for stage_idx, stage_channels in enumerate(self.stage_channels):
            if stage_idx == 0:
                encoder_layers.append(
                    _conv_block(
                        prev_channels,
                        stage_channels,
                        kernel_size=3,
                        norm_type=norm_type,
                        gn_groups=gn_groups,
                    )
                )
            else:
                encoder_layers.append(
                    DownBlock(
                        prev_channels,
                        stage_channels,
                        norm_type=norm_type,
                        gn_groups=gn_groups,
                    )
                )
            for _ in range(self.blocks_per_stage[stage_idx]):
                encoder_layers.append(ResBlock(stage_channels, norm_type=norm_type, gn_groups=gn_groups))
            prev_channels = stage_channels
        encoder_layers.append(nn.Conv2d(prev_channels, self.bottleneck_channels, kernel_size=3, stride=1, padding=1))
        self.encoder_conv = nn.ModuleList(encoder_layers)

        decoder_layers = [nn.Conv2d(self.bottleneck_channels, self.stage_channels[-1], kernel_size=3, stride=1, padding=1)]
        current_channels = self.stage_channels[-1]
        for _ in range(self.blocks_per_stage[-1]):
            decoder_layers.append(ResBlock(current_channels, norm_type=norm_type, gn_groups=gn_groups))
        for stage_idx in range(len(self.stage_channels) - 2, -1, -1):
            stage_channels = self.stage_channels[stage_idx]
            decoder_layers.append(
                UpBlock(
                    current_channels,
                    stage_channels,
                    upsample_mode=upsample_mode,
                    norm_type=norm_type,
                    gn_groups=gn_groups,
                )
            )
            for _ in range(self.blocks_per_stage[stage_idx]):
                decoder_layers.append(ResBlock(stage_channels, norm_type=norm_type, gn_groups=gn_groups))
            current_channels = stage_channels
        decoder_layers.extend(
            [
                nn.Conv2d(current_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),
            ]
        )
        self.decoder_conv = nn.ModuleList(decoder_layers)

        enc_out = self._run_dummy_in_eval(
            lambda x: self.encode(x, return_aux=False),
            self.in_channels,
            self.input_size,
        )
        self.enc_out_shape = enc_out.shape[1:]

        self._initialize_weights()

    def encode(self, x, return_aux=False):
        spatial_shapes = []
        feat = x
        for layer in self.encoder_conv:
            if return_aux and isinstance(layer, DownBlock):
                spatial_shapes.append(feat.shape[2:])
            feat = layer(feat)
        if return_aux:
            return feat, spatial_shapes
        return feat

    def decode(self, z, aux=None):
        out = z
        shapes = list(aux) if aux is not None else []
        for layer in self.decoder_conv:
            if isinstance(layer, UpBlock):
                out = layer(out, target_size=shapes.pop() if shapes else None)
            else:
                out = layer(out)
        return out


# The plain convolutional variant supports spatial or linear bottlenecks.
# The standard recovery presets load ResConvAE.
class PureConvAutoencoder(AutoencoderBase):
    """Build a convolutional encoder with strided layers at the selected indices."""

    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        bottleneck_channels=16,
        latent_dim=128,
        input_size=28,
        encoder_channels=None,
        downsample_indices=None,
        bottleneck_mode="conv",  # "linear" or "conv"
        use_sn=False, # whether to use spectral normalization
        upsample_mode="resize_conv", # "deconv" or "resize_conv"
        norm_type="bn", # "bn", "gn", or "none"
        gn_groups=8, # number of groups for GroupNorm if used
    ):
        super().__init__()
        assert bottleneck_mode in {"linear", "conv"}, "bottleneck_mode must be 'linear' or 'conv'"
        self.bottleneck_mode = bottleneck_mode
        assert upsample_mode in {"deconv", "resize_conv"}, "upsample_mode must be 'deconv' or 'resize_conv'"
        self.upsample_mode = upsample_mode
        assert norm_type in {"bn", "gn", "none"}, "norm_type must be 'bn', 'gn', or 'none'"
        encoder_channels = [
            int(value) for value in (
                encoder_channels if encoder_channels is not None else _default_encoder_channels(base_channels)
            )
        ]
        if not encoder_channels:
            raise ValueError("encoder_channels must be a non-empty list.")
        if any(value <= 0 for value in encoder_channels):
            raise ValueError("encoder_channels must contain positive integers.")
        bottleneck_channels = int(bottleneck_channels)
        if bottleneck_channels <= 0:
            raise ValueError("bottleneck_channels must be a positive integer.")
        self.downsample_indices = [
            int(value) for value in (downsample_indices if downsample_indices is not None else [1, 3, 5])
        ]
        if self.downsample_indices != sorted(self.downsample_indices) or len(self.downsample_indices) != len(set(self.downsample_indices)):
            raise ValueError("downsample_indices must be a sorted list of unique integers.")
        if any(idx < 0 or idx >= len(encoder_channels) for idx in self.downsample_indices):
            raise ValueError("downsample_indices must stay within [0, len(encoder_channels) - 1].")
        enc_channels = encoder_channels + [bottleneck_channels]
        enc_strides = [2 if idx in self.downsample_indices else 1 for idx in range(len(enc_channels))]

        enc_layers = []
        prev_channels = in_channels
        for out_channels, stride in zip(enc_channels, enc_strides):
            enc_layers.append(
                _conv_block(
                    prev_channels,
                    out_channels,
                    stride=stride,
                    use_sn=use_sn,
                    norm_type=norm_type,
                    gn_groups=gn_groups,
                )
            )
            prev_channels = out_channels
        self.encoder_conv = nn.Sequential(*enc_layers)
        self.encoder_channels = encoder_channels
        self.bottleneck_channels = bottleneck_channels

        enc_out = self._run_dummy_in_eval(
            self.encoder_conv,
            in_channels,
            input_size,
        )
        self.enc_out_shape = enc_out.shape[1:]  # (C, H, W)

        if self.bottleneck_mode == "linear":
            flat_dim = math.prod(self.enc_out_shape)
            self.bottleneck_enc = nn.Linear(flat_dim, latent_dim)
            self.bottleneck_dec = nn.Linear(latent_dim, flat_dim)
            # Do spectral normalization for linear layers if specified
            if use_sn:
                self.bottleneck_enc = spectral_norm(self.bottleneck_enc)
                self.bottleneck_dec = spectral_norm(self.bottleneck_dec)
        else:
            self.bottleneck_enc = None
            self.bottleneck_dec = None

        dec_layers = [
            _conv_block(
                enc_channels[-1],
                enc_channels[-2],
                stride=1,
                use_sn=use_sn,
                norm_type=norm_type,
                gn_groups=gn_groups,
            )
        ]
        prev_channels = enc_channels[-2]
        for idx in range(len(enc_channels) - 2, -1, -1):
            out_channels = enc_channels[idx]
            if idx in self.downsample_indices:
                dec_layers.append(
                    UpSampleBlock(
                        prev_channels,
                        out_channels,
                        stride=2,
                        use_sn=use_sn,
                        upsample_mode=upsample_mode,
                        norm_type=norm_type,
                        gn_groups=gn_groups,
                    )
                )
            else:
                dec_layers.append(
                    _conv_block(
                        prev_channels,
                        out_channels,
                        stride=1,
                        use_sn=use_sn,
                        norm_type=norm_type,
                        gn_groups=gn_groups,
                    )
                )
            prev_channels = out_channels
        dec_layers.extend(
            [
                nn.Conv2d(prev_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),
            ]
        )
        self.decoder_conv = nn.ModuleList(dec_layers)
        self._initialize_weights()

    def encode(self, x, return_aux=False):
        spatial_shapes = []
        feat = x
        for i, layer in enumerate(self.encoder_conv):
            if return_aux and i in self.downsample_indices:
                spatial_shapes.append(feat.shape[2:])
            feat = layer(feat)
        if self.bottleneck_mode == "linear":
            b, c, h, w = feat.shape
            z = self.bottleneck_enc(feat.view(b, -1))
            if return_aux:
                return z, (spatial_shapes, (c, h, w))
            return z
        if return_aux:
            return feat, spatial_shapes
        return feat

    def decode(self, z, aux=None):
        if self.bottleneck_mode == "linear":
            spatial_shapes, feat_shape = aux
            c, h, w = feat_shape
            out = self.bottleneck_dec(z).view(z.shape[0], c, h, w)
        else:
            spatial_shapes = aux if aux is not None else []
            out = z
        shapes = list(spatial_shapes)
        for layer in self.decoder_conv:
            if isinstance(layer, UpSampleBlock):
                target_s = shapes.pop() if shapes else None
                out = layer(out, target_size=target_s)
            else:
                out = layer(out)
        return out
