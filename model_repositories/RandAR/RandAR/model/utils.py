# drop_path cloned from https://github.com/FoundationVision/LlamaGen/blob/main/utils/drop_path.py
import torch
import math


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


def interleave_tokens(seq1, seq2):
    """ Interleave two sequences """
    result = torch.zeros_like(torch.cat((seq1, seq2), dim=1))
    result[:, ::2] = seq1
    result[:, 1::2] = seq2
    return result


def calculate_num_query_tokens_for_parallel_decoding(cur_step, total_step, block_size, 
                                                     query_token_idx_cur_step, num_query_token_cur_step):
    # how many tokens will be decoded at this step in total
    num_target_decoded_tokens = (
        1.0 - math.cos(math.pi / 2.0 * (cur_step + 1) / total_step)
    ) * block_size + 1
    num_target_decoded_tokens = min(
        int(num_target_decoded_tokens), block_size
    )

    # how many tokens will be decoded at the next step
    num_query_tokens_next_step = (
        num_target_decoded_tokens - query_token_idx_cur_step - num_query_token_cur_step
    )
    num_query_tokens_next_step = max(num_query_tokens_next_step, 1)
    num_query_tokens_next_step = min(
        num_query_tokens_next_step,
        block_size - query_token_idx_cur_step - num_query_token_cur_step,
    )

    return num_query_tokens_next_step