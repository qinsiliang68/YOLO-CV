import random
import numpy as np
import torch
from stage1_sctsr_v4.rng_isolation import capture_global_rng,derive_counter_seed,replay_rng_domain

def test_counter_seed_is_deterministic_and_domain_separated():
    assert derive_counter_seed('a',1,2,'x')==derive_counter_seed('a',1,2,'x');assert derive_counter_seed('a',1,2,'x')!=derive_counter_seed('b',1,2,'x')

def test_replay_domain_restores_all_global_rng():
    random.seed(3);np.random.seed(3);torch.manual_seed(3);before=capture_global_rng().digest()
    with replay_rng_domain('replay_augmentation',1,121,'x'):
        random.random();np.random.rand();torch.rand(3)
    assert capture_global_rng().digest()==before
