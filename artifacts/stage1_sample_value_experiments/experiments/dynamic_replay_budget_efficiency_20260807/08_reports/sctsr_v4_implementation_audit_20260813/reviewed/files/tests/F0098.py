import torch
from stage1_sctsr_v4.bn_isolation import capture_batchnorm_buffers,preserve_batchnorm_buffers
from stage1_sctsr_v4.synthetic_fixture import TinyClassifier

def test_bn_buffers_restored_but_gradients_allowed():
    m=TinyClassifier();m.train();before=capture_batchnorm_buffers(m).digest()
    with preserve_batchnorm_buffers(m):
        out=m(torch.randn(4,1,2,2));out.sum().backward()
    assert capture_batchnorm_buffers(m).digest()==before
    assert any(p.grad is not None and torch.any(p.grad!=0) for p in m.parameters())
