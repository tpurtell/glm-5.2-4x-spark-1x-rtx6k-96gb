from glmrt_reference.glm52_config import glm_shape_config, tiny_config


def test_glm_shape_facts_match_phase0_requirements():
    cfg = glm_shape_config()
    assert cfg.hidden_size == 6144
    assert cfg.num_hidden_layers == 78
    assert cfg.first_k_dense_replace == 3
    assert cfg.routed_experts == 256
    assert cfg.experts_per_token == 8


def test_tiny_config_is_small_but_moe_shaped():
    cfg = tiny_config()
    assert cfg.hidden_size < 6144
    assert cfg.routed_experts >= cfg.experts_per_token
