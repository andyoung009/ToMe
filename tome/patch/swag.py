# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# SWAG: https://github.com/facebookresearch/SWAG
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple

import torch

from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r

# Since we don't necessarily have the swag code available, this patch is a little bit more involved

# 这段代码定义了一个函数 `make_block_class`，它接受一个参数 `block_cls`，并返回一个新的类 `ToMeBlock`，该类继承自 `block_cls``。
# `ToMeBlock` 类在原有的 `block_cls` 类的基础上进行了修改，增加了一个 `ToMe` 操作，该操作 self-attention 和 MLP 之间进行。
# 同时，该类还计算并传播了 token 的大小和来源信息。
# 在 `forward` 方法中，首先对输入进行 Layer Normalization，然后进行 self-attention 操作，得到 `x_attn` 和 `metric`。
# 如果 `prop_attn` 为 True使用 `_tome_info["size"]` 作为 self-attention 的输出大小；否则，使用默认值 None。
# 然后对 `x_attn` 进行 dropout 操作，并将其与输入相加得到 `x`。
# 接下来，从 `_tome_info["r"]` 中弹出一个值 `r`，如果 `r` 大于 0进行 ToMe 操作。
# 具体来说，使用 `bipartite_soft_matching` 函数对 `metric` 进行匹配，得到 `merge`，然后使用 `merge_wavg` 函数对 `x` 进行加权平均
# 得到新的 `x` 和更新后的 `_tome_info["size"]`。如果 `trace_source` 为 True，则还会更新 `_tome_info["source"]`。
# 最后，对 `x` 进行 Layer Normalization，然后进行 MLP 操作，得到 `y`。将 `x` 和 `y` 相加并返回。
def make_block_class(block_cls):
    class ToMeBlock(block_cls):
        """
        Modifications:
        - Apply ToMe between the attention and mlp blocks
        - Compute and propogate token size and potentially the token sources.
        """

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            # Note: this is copied from swag.models.vision_transformer.EncoderBlock with modifications.
            x = self.ln_1(input)
            attn_size = (
                self._tome_info["size"] if self._tome_info["prop_attn"] else None
            )
            x_attn, metric = self.self_attention(x, size=attn_size)
            x = self.dropout(x_attn)
            x = x + input

            r = self._tome_info["r"].pop(0)
            if r > 0:
                # Apply ToMe here
                merge, _ = bipartite_soft_matching(
                    metric,
                    r,
                    self._tome_info["class_token"],
                    self._tome_info["distill_token"],
                )
                if self._tome_info["trace_source"]:
                    self._tome_info["source"] = merge_source(
                        merge, x, self._tome_info["source"]
                    )
                x, self._tome_info["size"] = merge_wavg(
                    merge, x, self._tome_info["size"]
                )

            y = self.ln_2(x)
            y = self.mlp(y)
            return x + y

    return ToMeBlock

# 这是一个Python函数，它接受一个名为"transformer_class"的参数，并返回一个新的类"ToMeVisionTransformer"。
# 这个新类是通过继承"transformer_class"类来创建的，因此它继承了"transformer_class"的所有属性和方法。
# 这个新类有一个名为"forward"的方法，它覆盖了基类的"forward"方法。在新的"forward"方法中，首先调用了基类的"forward"方法，然后对一些属性进行了修改：
# - "r"属性：这是一个整数，它表示在Tome中使用的"r"值。"r"值是一个超参数，用于设置要合并的连续的token的数量。在这个方法中，首先使用"parse_r"函数计算"r"值，
# 然后将其存储在"_tome_info"字典中的"r"键下。
# - "size"属性：这是一个整数，它表示在Tome中使用的token大小。在这个方法中，将其设置为None，并将其存储在"_tome_info"字典中的"size"键下。
# - "source"属性：这是一个字符串，它表示在Tome中使用的token来源。在这个方法中，将其设置为None，并将其存储在"_tome_info"字典中的"source"键下。
# 最后，这个新类被返回并可以被实例化和使用。这个新类在基类的基础上进行了修改，添加了Tome需要的一些属性和方法。
# 这样，当使用这个新的类创建模型时，# 就可以获得Tome的功能，例如合并连续的token，从而减少模型的计算量和参数量。
class ToMeAttention(torch.nn.MultiheadAttention):
    """
    Modifications:
    - Apply proportional attention
    - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        # To add sizes to the pre (or post) softmax attn matrix, there may be a hacky way to add what we want to 
        # the pre-softmax matrix by modifying the biases (i.e., in_proj_bias or k_bias, idk which) on the fly.
        qkv = torch.nn.functional.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        scale = self.head_dim**-0.5
        attn = (q * scale) @ k.transpose(-2, -1)

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)

        # Return k as well here
        return x, k.mean(1)


def make_transformer_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.encoder.layers), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def make_encoder_class(encoder_class):
    class ToMeEncoder(encoder_class):
        """
        Modifications:
        - Permute encoder dims so it's (batch, tokens, channels).
        """

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pos_embedding

            x = x.transpose(0, 1)
            x = self.ln(self.layers(self.dropout(x)))
            x = x.transpose(0, 1)
            return x

    return ToMeEncoder


def apply_patch(model, trace_source: bool = False, prop_attn: bool = True):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """

    if model.__class__.__name__ == "ToMeVisionTransformer":
        # This model was already patched!
        return

    EncoderClass = None
    BlockClass = None
    TransformerClass = model.__class__

    # Collect class names
    for module in model.modules():
        if module.__class__.__name__ == "EncoderBlock":
            BlockClass = module.__class__
        elif module.__class__.__name__ == "Encoder":
            EncoderClass = module.__class__

    if BlockClass is None or EncoderClass is None:
        print(
            "Error patching model: this model isn't a SWAG transformer or the interface has been updated."
        )
        return

    ToMeBlock = make_block_class(BlockClass)
    ToMeEncoder = make_encoder_class(EncoderClass)
    ToMeVisionTransformer = make_transformer_class(TransformerClass)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.classifier == "token",
        "distill_token": False,
    }

    for module in model.modules():
        if isinstance(module, BlockClass):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, torch.nn.MultiheadAttention):
            module.__class__ = ToMeAttention
        elif isinstance(module, EncoderClass):
            module.__class__ = ToMeEncoder
