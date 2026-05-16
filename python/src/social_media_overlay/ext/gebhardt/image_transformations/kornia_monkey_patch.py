import kornia.enhance.adjust as kadj

# Author Marcel Schubert

"""
Monkey-patch kornias blending: Before exporting, override the _blend_one function (or apply_sharpening) so it doesnt use a Python if.
For example, replace it with a tensor operation. 
One simple strategy is to just perform the linear blend formula directly (which mathematically handles all cases, since if factor=0 or 1 the formula still works).
This avoids any if factor == ... checks altogether. Alternatively, you could use a torch.where logic or Torchs symbolic guard utilities.
"""

def safe_blend(input1, input2, factor):
    factor = factor.to(dtype=input1.dtype, device=input1.device)
    while factor.ndim < input1.ndim:
        factor = factor.unsqueeze(-1)
    return input1 * (1.0 - factor) + input2 * factor

# Override kornia's internal function
kadj._blend_one = safe_blend