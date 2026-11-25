import pytest
import torch

from fedgraph.gnn_models import AggreGCN


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gcn_skip_preserves_initial_predictions_and_learns(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        baseline = AggreGCN(3, 4, 2, 0.5, 2, first_layer_norm=True)
        baseline_rng = torch.get_rng_state()
        torch.random.default_generator.manual_seed(42)
        skip = AggreGCN(3, 4, 2, 0.5, 2, first_layer_norm=True, model_head="gcn-skip")
        torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)

    baseline = baseline.to(device).eval()
    skip = skip.to(device).eval()
    features = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]], device=device)
    edges = torch.tensor([[0, 1], [1, 0]], device=device)
    for name, weight in baseline.state_dict().items():
        torch.testing.assert_close(skip.state_dict()[name], weight, rtol=0, atol=0)
    torch.testing.assert_close(skip(features, edges), baseline(features, edges))

    (-skip(features, edges)[0, 0]).backward()
    assert torch.count_nonzero(skip.skip_head.weight.grad) > 0
    with torch.no_grad():
        skip.skip_head.weight[0].fill_(0.25)
        hidden = skip.input_norm(skip.convs[0](features)).relu()
        logits = skip.convs[-1](hidden, edges) + skip.skip_head(hidden)
        torch.testing.assert_close(skip(features, edges), logits.log_softmax(-1))

    skip.reset_parameters()
    assert torch.count_nonzero(skip.skip_head.weight) == 0


def test_gcn_skip_rejects_unsupported_depth():
    with pytest.raises(ValueError, match="two-layer"):
        AggreGCN(3, 4, 2, 0.5, 3, model_head="gcn-skip")
