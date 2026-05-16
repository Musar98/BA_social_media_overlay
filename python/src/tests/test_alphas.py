from social_media_overlay.alphas import sample_alphas
import torch

def test_sample_alphas_shape_and_device():
    batch_size = 10
    device = "cpu"

    alphas = sample_alphas(batch_size, device)

    assert isinstance(alphas, torch.Tensor)
    assert alphas.shape == (batch_size, 1)
    assert alphas.device.type == device
    
def test_sample_alphas_range_and_zero():
    batch_size = 100
    low, high = -0.5, 0.5

    alphas = sample_alphas(batch_size, "cpu", zero_alpha_prob=0.3, low=low, high=high)

    # mask for non-zero values
    non_zero = alphas != 0.0

    if non_zero.any():
        assert torch.all(alphas[non_zero] >= low)
        assert torch.all(alphas[non_zero] <= high)
        
def test_sample_alphas_no_zeros_when_prob_zero():
    alphas = sample_alphas(100, "cpu", zero_alpha_prob=0.0)

    assert torch.all(alphas != 0.0)
    
def test_sample_alphas_all_zeros_when_prob_one():
    alphas = sample_alphas(20, "cpu", zero_alpha_prob=1.0)

    assert torch.all(alphas == 0.0)
    
def test_sample_alphas_zero_probability_approximate():
    batch_size = 1000
    prob = 0.3

    alphas = sample_alphas(batch_size, "cpu", zero_alpha_prob=prob)

    zero_ratio = (alphas == 0.0).float().mean().item()

    # allow tolerance because randomness
    assert abs(zero_ratio - prob) < 0.1
    
def test_sample_alphas_custom_range():
    low, high = 2.0, 5.0

    alphas = sample_alphas(100, "cpu", zero_alpha_prob=0.0, low=low, high=high)

    assert torch.all(alphas >= low)
    assert torch.all(alphas <= high)