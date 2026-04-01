"""Unit tests for MTFC model components.

Tests cover:
- Spatio-temporal mixing (STAddition, STAdditionProjection, ST_SharedProjection)
- Spectrogram estimation methods (st_addition, st_addition_projection,
  stf_attention_temporal_values, filter_banks)
- Attention mechanisms (SpatioTemporal_Temporal_AttentionHead, N_CrossAttentionHeads)
- Patch embedding with filter banks
"""

import torch
import pytest
from types import SimpleNamespace
from models.MTFC import (
    MultiScaleTemporalPatchEmbedding_FilterBanks,
    STAddition,
    STAdditionProjection,
    ST_SharedProjection,
    SpectrogramEstimator,
    SpatioTemporal_Temporal_AttentionHead,
    N_CrossAttentionHeads,
)


def test_MultiScaleTemporalPatchEmbedding_FilterBanks():
    """Test MultiScaleTemporalPatchEmbedding_FilterBanks forward pass.

    Verifies that the multi-branch filter bank patch embedding produces
    correct output shape (B, F, P, D) where F is the number of filter banks,
    P is the number of patches, and D is the embedding dimension.
    """
    args = SimpleNamespace(
        data_name="BCI-IV-2a", chn=22, patch_size=125, time_sample_num=1001, class_num=4
    )

    n_filter_banks = 4
    emb_size = 40
    fs = 250

    model = MultiScaleTemporalPatchEmbedding_FilterBanks(
        args, n_filter_banks=n_filter_banks, emb_size=emb_size, fs=fs
    )

    batch_size = 2
    channels = args.chn
    time_points = args.time_sample_num
    x = torch.randn(batch_size, channels, time_points)

    output = model(x)

    expected_P = 8

    assert output.shape == (batch_size, n_filter_banks, expected_P, emb_size)
    assert isinstance(output, torch.Tensor)


def test_STAddition():
    """Test STAddition combines temporal and spatial embeddings correctly.

    Verifies that:
    1. Output shape is (B, C, P, D)
    2. Each output element correctly computes: output[b,c,p,d] = spatial[b,c,d] + temporal[b,p,d]
    """
    B, P, C, D = 2, 8, 22, 40
    st_addition = STAddition(num_channels=C, num_patches=P)

    x_temporal = torch.randn(B, P, D)
    x_spatial = torch.randn(B, C, D)

    output = st_addition(x_temporal, x_spatial)

    assert output.shape == (B, C, P, D)

    expected_val = x_spatial[0, 5, 10] + x_temporal[0, 3, 10]
    assert torch.allclose(output[0, 5, 3, 10], expected_val)


def test_ST_SharedProjection():
    """Test ST_SharedProjection preserves shape and applies transformation.

    Verifies that:
    1. Output shape matches input shape for both temporal and spatial inputs
    2. The projection is applied (output differs from input)
    """
    D = 40
    B, P, C = 2, 8, 22

    proj = ST_SharedProjection(emb_size=D)

    x_temporal = torch.randn(B, P, D)
    x_spatial = torch.randn(B, C, D)

    out_temporal = proj(x_temporal)
    out_spatial = proj(x_spatial)

    assert out_temporal.shape == (B, P, D)
    assert out_spatial.shape == (B, C, D)
    assert out_temporal.shape == x_temporal.shape
    assert out_spatial.shape == x_spatial.shape


def test_STAdditionProjection():
    """Test STAdditionProjection applies shared projection before addition.

    Verifies that:
    1. Output shape is (B, C, P, D)
    2. The shared_projection attribute exists and is of correct type
    """
    B, P, C, D = 2, 8, 22, 40
    st_addition_proj = STAdditionProjection(num_channels=C, num_patches=P, emb_size=D)

    x_temporal = torch.randn(B, P, D)
    x_spatial = torch.randn(B, C, D)

    output = st_addition_proj(x_temporal, x_spatial)

    assert output.shape == (B, C, P, D)

    assert hasattr(st_addition_proj, "shared_projection")
    assert isinstance(st_addition_proj.shared_projection, ST_SharedProjection)


def test_SpectrogramEstimator_STAddition():
    """Test SpectrogramEstimator with st_addition method.

    Verifies that the MLP-based spectrogram estimator produces correct
    output shape (B, C, P, F) for the st_addition method.
    """
    B, P, C, D, F = 2, 8, 22, 40, 4
    st_mixer = STAddition(C, P)

    estimator = SpectrogramEstimator(
        method="st_addition",
        emb_size=D,
        n_freqs=F,
        num_channels=C,
        num_patches=P,
    )

    x_temporal = torch.randn(B, P, D)
    x_spatial = torch.randn(B, C, D)
    z_st = st_mixer(x_temporal, x_spatial)

    stft = estimator(z_st, x_temporal, x_spatial)

    assert stft.shape == (B, C, P, F)


def test_SpectrogramEstimator_STAdditionProjection():
    """Test SpectrogramEstimator with st_addition_projection method.

    Verifies that the MLP-based spectrogram estimator produces correct
    output shape (B, C, P, F) for the st_addition_projection method.
    """
    B, P, C, D, F = 2, 8, 22, 40, 4
    st_mixer = STAdditionProjection(C, P, D)

    estimator = SpectrogramEstimator(
        method="st_addition_projection",
        emb_size=D,
        n_freqs=F,
        num_channels=C,
        num_patches=P,
    )

    x_temporal = torch.randn(B, P, D)
    x_spatial = torch.randn(B, C, D)
    z_st = st_mixer(x_temporal, x_spatial)

    stft = estimator(z_st, x_temporal, x_spatial)

    assert stft.shape == (B, C, P, F)


def test_SpectrogramEstimator_FilterBanks():
    """Test SpectrogramEstimator with filter_banks method (no shared projection).

    Verifies that the filter banks method produces correct output shape
    (B, C, P, F) using element-wise multiplication without shared projection.
    """
    B, F, P, C, D = 2, 4, 8, 22, 40

    estimator_no_proj = SpectrogramEstimator(
        method="filter_banks",
        emb_size=D,
        n_freqs=F,
        num_channels=C,
        num_patches=P,
        use_ct_shared_projection=False,
    )

    x_embed_frequency = torch.randn(B, F, D)
    x_embed_temporal = torch.randn(B, P, D)
    x_embed_spatial = torch.randn(B, C, D)

    stft = estimator_no_proj(x_embed_frequency, x_embed_temporal, x_embed_spatial)

    assert stft.shape == (B, C, P, F)


def test_SpectrogramEstimator_FilterBanksWithSharedProjection():
    """Test SpectrogramEstimator with filter_banks method (with shared projection).

    Verifies that:
    1. The ct_shared_projection attribute is correctly created when enabled
    2. Output shape is (B, C, P, F)
    """
    B, F, P, C, D = 2, 4, 8, 22, 40

    estimator_with_proj = SpectrogramEstimator(
        method="filter_banks",
        emb_size=D,
        n_freqs=F,
        num_channels=C,
        num_patches=P,
        use_ct_shared_projection=True,
    )

    assert hasattr(estimator_with_proj, "ct_shared_projection")
    assert isinstance(estimator_with_proj.ct_shared_projection, ST_SharedProjection)

    x_embed_frequency = torch.randn(B, F, D)
    x_embed_temporal = torch.randn(B, P, D)
    x_embed_spatial = torch.randn(B, C, D)

    stft = estimator_with_proj(x_embed_frequency, x_embed_temporal, x_embed_spatial)

    assert stft.shape == (B, C, P, F)


def test_SpatioTemporal_Temporal_AttentionHead():
    """Test SpatioTemporal_Temporal_AttentionHead cross-attention mechanism.

    Verifies that the cross-attention head:
    1. Accepts query (B, C, P, D) and key_value (B, P, D)
    2. Produces output (B, C, P, D)
    3. Correctly divides embedding dimension by num_heads
    """
    B, D, P, C = 2, 40, 8, 22
    num_heads = 4

    attn_head = SpatioTemporal_Temporal_AttentionHead(emb_size=D, num_heads=num_heads)

    query = torch.randn(B, C, P, D)
    key_value = torch.randn(B, P, D)

    output = attn_head(query, key_value)

    assert output.shape == (B, C, P, D)
    assert D % num_heads == 0


def test_N_CrossAttentionHeads():
    """Test N_CrossAttentionHeads parallel attention mechanism.

    Verifies that:
    1. Output shape is (B, F, C, P, D) for F parallel attention heads
    2. The correct number of component attention heads are instantiated
    """
    B, D, P, C, F = 2, 40, 8, 22, 4
    num_heads = 4

    n_attn_heads = N_CrossAttentionHeads(emb_size=D, num_heads=num_heads, n_comps=F)

    query = torch.randn(B, C, P, D)
    key_value = torch.randn(B, P, D)

    output = n_attn_heads(query, key_value)

    assert output.shape == (B, F, C, P, D)
    assert len(n_attn_heads.component_attn) == F


def test_SpectrogramEstimator_STF_AttentionTemporalValues():
    """Test SpectrogramEstimator with stf_attention_temporal_values method.

    Verifies the complete pipeline for stf_attention_temporal_values:
    1. STAddition produces (B, C, P, D)
    2. N_CrossAttentionHeads produces (B, F, C, P, D)
    3. SpectrogramEstimator produces (B, C, P, F)
    """
    B, P, C, D, F = 2, 8, 22, 40, 4
    num_heads = 4

    st_mixer = STAddition(C, P)
    stf_attention_head = N_CrossAttentionHeads(
        emb_size=D, num_heads=num_heads, n_comps=F
    )
    spectrogram_estimator = SpectrogramEstimator(
        method="stf_attention_temporal_values",
        emb_size=D,
        n_freqs=F,
        num_channels=C,
        num_patches=P,
    )

    x_temporal = torch.randn(B, P, D)
    x_spatial = torch.randn(B, C, D)

    z_st = st_mixer(x_temporal, x_spatial)
    z_st_attended = stf_attention_head(z_st, x_temporal)
    stft = spectrogram_estimator(z_st_attended, x_temporal, x_spatial)

    assert z_st.shape == (B, C, P, D)
    assert z_st_attended.shape == (B, F, C, P, D)
    assert stft.shape == (B, C, P, F)


def test_mtfc_with_dummy_dataset():
    """Test MTFC model with dummy_dataset using cross_validation script.

    Verifies that:
    1. The cross_validation main function runs successfully with dummy_dataset
    2. MTFC model can process random data without errors
    3. Returns valid accuracy, kappa, and stft_reconstruction_loss values
    """
    import yaml
    from argparse import Namespace
    from cross_validation import main as cv_main

    quick_config = {
        "training": {
            "val_size": 0.0,
            "n_iter": 2,
            "eval_inter": 1,
            "folds": 2,
            "n_repeats": 1,
            "lr": 1.0e-3,
            "batch_size": 8,
        },
        "mtf_c": {
            "patch_size": 6,
            "filter_banks": 7,
            "freq_downsample": 1,
            "wsize_divisor": 2,
            "spa_dim": 16,
            "gate_flag": False,
            "posemb_flag": True,
            "branch": "all",
            "chn_attn_flag": False,
            "fts_attn_flag": True,
            "sst_method": "stf_attention_temporal_values",
            "stft_reconstruction": True,
            "patch_emb_size": 80,
            "n_heads_patch": 4,
            "sst_emb_size": 40,
            "tem_depth": 1,
            "chn_depth": 1,
        },
    }

    args = Namespace(
        model_name="mtf_c",
        dataset="dummy_dataset",
        subject=1,
        device="cpu",
        verbose=False,
    )

    with open("./configs/test_cv_config.yaml", "w") as f:
        yaml.dump(quick_config, f)

    all_accuracies, all_kappas, all_stft_loss, _, _, _ = cv_main(
        args=args, config="test_cv_config", model_configs=quick_config["mtf_c"]
    )

    assert isinstance(all_accuracies, list)
    assert len(all_accuracies) == 1
    assert 0.0 <= all_accuracies[0] <= 1.0

    assert isinstance(all_kappas, list)
    assert len(all_kappas) == 1
    assert -1.0 <= all_kappas[0] <= 1.0

    assert isinstance(all_stft_loss, list)
    assert len(all_stft_loss) == 1


def test_parse_branch_config():
    """Test parse_branch_config utility function.

    Verifies that branch configuration strings are parsed correctly into
    transformer branch specifications.
    """
    from models.MTFC import parse_branch_config

    assert parse_branch_config("all") == ["fts"]

    assert parse_branch_config("ft_s") == [("f", "t"), ("s",)]

    assert parse_branch_config("f_t_s") == [("f",), ("t",), ("s",)]

    assert parse_branch_config("f_ts") == [("f",), ("t", "s")]

    assert parse_branch_config("fs_t") == [("f", "s"), ("t",)]


def test_mtfc_branch_all():
    """Test MTFC with branch='all' (single transformer for all embeddings).

    Verifies that:
    1. Model creates single transformer branch
    2. Classifier input dimension is FTS
    3. Forward pass produces correct output shapes
    """
    from models.MTFC import MTFC
    from types import SimpleNamespace

    args = SimpleNamespace(
        data_name="BCI-IV-2a",
        chn=22,
        patch_size=125,
        time_sample_num=1001,
        class_num=4,
        gate_flag=False,
        posemb_flag=True,
        branch="all",
        chn_attn_flag=False,
        fts_attn_flag=True,
        sst_method="stf_attention_temporal_values",
        stft_reconstruction=True,
        spa_dim=16,
    )

    model = MTFC(
        args,
        n_filter_banks=4,
        freq_downsample=1,
        patch_emb_size=40,
        n_heads_patch=4,
        sst_emb_size=40,
        depth=1,
        n_classes=4,
        fs=250,
    )

    assert model.num_branches == 1
    assert model.classifier.fc[0].in_features == 40
    assert "branch_0" in model.transformers

    x = torch.randn(2, 1, 22, 1001)
    stft, embed, out = model(x)

    assert stft.shape == (2, 22, 4, 8)
    assert embed.shape == (2, 40)
    assert out.shape == (2, 4)


def test_mtfc_frequency_reconstruction_sst_shared_projection_true():
    """Test MTFC with frequency reconstruction and sst_shared_projection=True.

    Verifies that:
    1. freq_to_bandpowers MLP is created with correct input dimension (FTS)
    2. Model outputs band powers of shape (B, F)
    """
    from models.MTFC import MTFC
    from types import SimpleNamespace

    args = SimpleNamespace(
        data_name="BCI-IV-2a",
        chn=22,
        patch_size=125,
        time_sample_num=1001,
        class_num=4,
        gate_flag=False,
        posemb_flag=True,
        branch="f_t_s",
        chn_attn_flag=False,
        fts_attn_flag=True,
        sst_method="filter_banks",
        stft_reconstruction="frequency",
        spa_dim=16,
        ct_shared_projection=True,
        sst_shared_projection=True,
    )

    model = MTFC(
        args,
        n_filter_banks=4,
        freq_downsample=1,
        patch_emb_size=20,
        n_heads_patch=4,
        sst_emb_size=40,
        depth=1,
        n_classes=4,
        fs=250,
    )

    assert hasattr(model, "freq_to_bandpowers")
    assert isinstance(model.freq_to_bandpowers, torch.nn.Sequential)

    x = torch.randn(2, 1, 22, 1001)
    band_powers, embed, out = model(x)

    assert band_powers.shape == (2, 4)  # (B, F)
    assert embed.shape == (2, 120)  # FTS * 3 branches
    assert out.shape == (2, 4)


def test_mtfc_frequency_reconstruction_sst_shared_projection_false():
    """Test MTFC with frequency reconstruction and sst_shared_projection=False.

    Verifies that:
    1. freq_to_bandpowers MLP is created with correct input dimension (D)
    2. Model outputs band powers of shape (B, F)
    """
    from models.MTFC import MTFC
    from types import SimpleNamespace

    args = SimpleNamespace(
        data_name="BCI-IV-2a",
        chn=22,
        patch_size=125,
        time_sample_num=1001,
        class_num=4,
        gate_flag=False,
        posemb_flag=True,
        branch="f_t_s",
        chn_attn_flag=False,
        fts_attn_flag=True,
        sst_method="filter_banks",
        stft_reconstruction="frequency",
        spa_dim=16,
        ct_shared_projection=False,
        sst_shared_projection=False,
    )

    model = MTFC(
        args,
        n_filter_banks=4,
        freq_downsample=1,
        patch_emb_size=20,
        n_heads_patch=4,
        sst_emb_size=40,
        depth=1,
        n_classes=4,
        fs=250,
    )

    assert hasattr(model, "freq_to_bandpowers")
    assert isinstance(model.freq_to_bandpowers, torch.nn.Sequential)

    x = torch.randn(2, 1, 22, 1001)
    band_powers, embed, out = model(x)

    assert band_powers.shape == (2, 4)  # (B, F)
    assert embed.shape == (2, 60)  # D * 3 branches
    assert out.shape == (2, 4)


def test_mtfc_sst_shared_projection_true():
    """Test MTFC with sst_shared_projection=True.

    Verifies that:
    1. sst_shared_projection_layer is created
    2. transformer_dim equals FTS
    3. Classifier input dimension is FTS * num_branches
    """
    from models.MTFC import MTFC
    from types import SimpleNamespace

    args = SimpleNamespace(
        data_name="BCI-IV-2a",
        chn=22,
        patch_size=125,
        time_sample_num=1001,
        class_num=4,
        gate_flag=False,
        posemb_flag=True,
        branch="f_t_s",
        chn_attn_flag=False,
        fts_attn_flag=True,
        sst_method="stf_attention_temporal_values",
        stft_reconstruction=True,
        spa_dim=16,
        ct_shared_projection=True,
        sst_shared_projection=True,
    )

    model = MTFC(
        args,
        n_filter_banks=4,
        freq_downsample=1,
        patch_emb_size=20,
        n_heads_patch=4,
        sst_emb_size=40,
        depth=1,
        n_classes=4,
        fs=250,
    )

    assert hasattr(model, "sst_shared_projection_layer")
    assert model.transformer_dim == 40  # FTS
    assert model.classifier.fc[0].in_features == 120  # FTS * 3

    x = torch.randn(2, 1, 22, 1001)
    stft, embed, out = model(x)

    assert embed.shape == (2, 120)


def test_mtfc_sst_shared_projection_false():
    """Test MTFC with sst_shared_projection=False.

    Verifies that:
    1. sst_shared_projection_layer is NOT created
    2. transformer_dim equals D (patch_emb_size)
    3. Classifier input dimension is D * num_branches
    """
    from models.MTFC import MTFC
    from types import SimpleNamespace

    args = SimpleNamespace(
        data_name="BCI-IV-2a",
        chn=22,
        patch_size=125,
        time_sample_num=1001,
        class_num=4,
        gate_flag=False,
        posemb_flag=True,
        branch="f_t_s",
        chn_attn_flag=False,
        fts_attn_flag=True,
        sst_method="stf_attention_temporal_values",
        stft_reconstruction=True,
        spa_dim=16,
        ct_shared_projection=True,
        sst_shared_projection=False,
    )

    model = MTFC(
        args,
        n_filter_banks=4,
        freq_downsample=1,
        patch_emb_size=20,
        n_heads_patch=4,
        sst_emb_size=40,
        depth=1,
        n_classes=4,
        fs=250,
    )

    assert not hasattr(model, "sst_shared_projection_layer")
    assert model.transformer_dim == 20  # D (patch_emb_size)
    assert model.classifier.fc[0].in_features == 60  # D * 3

    x = torch.randn(2, 1, 22, 1001)
    stft, embed, out = model(x)

    assert embed.shape == (2, 60)


def test_mtfc_ct_shared_projection_true():
    """Test MTFC with ct_shared_projection=True for filter_banks.

    Verifies that:
    1. spectrogram_estimator has ct_shared_projection
    2. Model runs correctly
    """
    from models.MTFC import MTFC
    from types import SimpleNamespace

    args = SimpleNamespace(
        data_name="BCI-IV-2a",
        chn=22,
        patch_size=125,
        time_sample_num=1001,
        class_num=4,
        gate_flag=False,
        posemb_flag=True,
        branch="f_t_s",
        chn_attn_flag=False,
        fts_attn_flag=True,
        sst_method="filter_banks",
        stft_reconstruction=True,
        spa_dim=16,
        ct_shared_projection=True,
        sst_shared_projection=True,
    )

    model = MTFC(
        args,
        n_filter_banks=4,
        freq_downsample=1,
        patch_emb_size=40,
        n_heads_patch=4,
        sst_emb_size=40,
        depth=1,
        n_classes=4,
        fs=250,
    )

    assert hasattr(model.spectrogram_estimator, "ct_shared_projection")

    x = torch.randn(2, 1, 22, 1001)
    stft, embed, out = model(x)

    assert stft.shape == (2, 22, 4, 8)
    assert embed.shape == (2, 120)


def test_mtfc_ct_shared_projection_false():
    """Test MTFC with ct_shared_projection=False for filter_banks.

    Verifies that:
    1. spectrogram_estimator does NOT have ct_shared_projection
    2. Model runs correctly
    """
    from models.MTFC import MTFC
    from types import SimpleNamespace

    args = SimpleNamespace(
        data_name="BCI-IV-2a",
        chn=22,
        patch_size=125,
        time_sample_num=1001,
        class_num=4,
        gate_flag=False,
        posemb_flag=True,
        branch="f_t_s",
        chn_attn_flag=False,
        fts_attn_flag=True,
        sst_method="filter_banks",
        stft_reconstruction=True,
        spa_dim=16,
        ct_shared_projection=False,
        sst_shared_projection=True,
    )

    model = MTFC(
        args,
        n_filter_banks=4,
        freq_downsample=1,
        patch_emb_size=40,
        n_heads_patch=4,
        sst_emb_size=40,
        depth=1,
        n_classes=4,
        fs=250,
    )

    assert not hasattr(model.spectrogram_estimator, "ct_shared_projection")

    x = torch.randn(2, 1, 22, 1001)
    stft, embed, out = model(x)

    assert stft.shape == (2, 22, 4, 8)
    assert embed.shape == (2, 120)


def test_mtfc_branch_f_ts():
    """Test MTFC with branch='f_ts' (freq + temporal-spatial transformers).

    Verifies that:
    1. Model creates 2 transformer branches
    2. Classifier input dimension is FTS * 2
    3. Forward pass produces correct output shapes
    """
    from models.MTFC import MTFC
    from types import SimpleNamespace

    args = SimpleNamespace(
        data_name="BCI-IV-2a",
        chn=22,
        patch_size=125,
        time_sample_num=1001,
        class_num=4,
        gate_flag=False,
        posemb_flag=True,
        branch="f_ts",
        chn_attn_flag=False,
        fts_attn_flag=True,
        sst_method="stf_attention_temporal_values",
        stft_reconstruction=True,
        spa_dim=16,
    )

    model = MTFC(
        args,
        n_filter_banks=4,
        freq_downsample=1,
        patch_emb_size=40,
        n_heads_patch=4,
        sst_emb_size=40,
        depth=1,
        n_classes=4,
        fs=250,
    )

    assert model.num_branches == 2
    assert model.classifier.fc[0].in_features == 80
    assert "branch_0" in model.transformers
    assert "branch_1" in model.transformers

    x = torch.randn(2, 1, 22, 1001)
    stft, embed, out = model(x)

    assert stft.shape == (2, 22, 4, 8)
    assert embed.shape == (2, 80)
    assert out.shape == (2, 4)


def test_mtfc_branch_fs_t():
    """Test MTFC with branch='fs_t' (freq-spatial + temporal transformers).

    Verifies that:
    1. Model creates 2 transformer branches
    2. Classifier input dimension is FTS * 2
    3. Forward pass produces correct output shapes
    """
    from models.MTFC import MTFC
    from types import SimpleNamespace

    args = SimpleNamespace(
        data_name="BCI-IV-2a",
        chn=22,
        patch_size=125,
        time_sample_num=1001,
        class_num=4,
        gate_flag=False,
        posemb_flag=True,
        branch="fs_t",
        chn_attn_flag=False,
        fts_attn_flag=True,
        sst_method="stf_attention_temporal_values",
        stft_reconstruction=True,
        spa_dim=16,
    )

    model = MTFC(
        args,
        n_filter_banks=4,
        freq_downsample=1,
        patch_emb_size=40,
        n_heads_patch=4,
        sst_emb_size=40,
        depth=1,
        n_classes=4,
        fs=250,
    )

    assert model.num_branches == 2
    assert model.classifier.fc[0].in_features == 80
    assert "branch_0" in model.transformers
    assert "branch_1" in model.transformers

    x = torch.randn(2, 1, 22, 1001)
    stft, embed, out = model(x)

    assert stft.shape == (2, 22, 4, 8)
    assert embed.shape == (2, 80)
    assert out.shape == (2, 4)


def test_mtfc_branch_f_ts_filter_banks():
    """Test MTFC with branch='f_ts' and filter_banks method.

    Verifies that:
    1. Model works with filter_banks SST method
    2. Produces correct output shapes
    """
    from models.MTFC import MTFC
    from types import SimpleNamespace

    args = SimpleNamespace(
        data_name="BCI-IV-2a",
        chn=22,
        patch_size=125,
        time_sample_num=1001,
        class_num=4,
        gate_flag=False,
        posemb_flag=True,
        branch="f_ts",
        chn_attn_flag=False,
        fts_attn_flag=True,
        sst_method="filter_banks",
        stft_reconstruction=True,
        spa_dim=16,
    )

    model = MTFC(
        args,
        n_filter_banks=4,
        freq_downsample=1,
        patch_emb_size=40,
        n_heads_patch=4,
        sst_emb_size=40,
        depth=1,
        n_classes=4,
        fs=250,
    )

    assert model.num_branches == 2
    assert model.classifier.fc[0].in_features == 80

    x = torch.randn(2, 1, 22, 1001)
    stft, embed, out = model(x)

    assert stft.shape == (2, 22, 4, 8)
    assert embed.shape == (2, 80)
    assert out.shape == (2, 4)


def test_mtfc_branch_fs_t_filter_banks():
    """Test MTFC with branch='fs_t' and filter_banks method.

    Verifies that:
    1. Model works with filter_banks SST method
    2. Produces correct output shapes
    """
    from models.MTFC import MTFC
    from types import SimpleNamespace

    args = SimpleNamespace(
        data_name="BCI-IV-2a",
        chn=22,
        patch_size=125,
        time_sample_num=1001,
        class_num=4,
        gate_flag=False,
        posemb_flag=True,
        branch="fs_t",
        chn_attn_flag=False,
        fts_attn_flag=True,
        sst_method="filter_banks",
        stft_reconstruction=True,
        spa_dim=16,
    )

    model = MTFC(
        args,
        n_filter_banks=4,
        freq_downsample=1,
        patch_emb_size=40,
        n_heads_patch=4,
        sst_emb_size=40,
        depth=1,
        n_classes=4,
        fs=250,
    )

    assert model.num_branches == 2
    assert model.classifier.fc[0].in_features == 80

    x = torch.randn(2, 1, 22, 1001)
    stft, embed, out = model(x)

    assert stft.shape == (2, 22, 4, 8)
    assert embed.shape == (2, 80)
    assert out.shape == (2, 4)


def test_mtfc_branch_ft_s():
    """Test MTFC with branch='ft_s' (freq-temporal + spatial transformers).

    Verifies that:
    1. Model creates 2 transformer branches
    2. Classifier input dimension is FTS * 2
    3. Forward pass produces correct output shapes
    """
    from models.MTFC import MTFC
    from types import SimpleNamespace

    args = SimpleNamespace(
        data_name="BCI-IV-2a",
        chn=22,
        patch_size=125,
        time_sample_num=1001,
        class_num=4,
        gate_flag=False,
        posemb_flag=True,
        branch="ft_s",
        chn_attn_flag=False,
        fts_attn_flag=True,
        sst_method="stf_attention_temporal_values",
        stft_reconstruction=True,
        spa_dim=16,
    )

    model = MTFC(
        args,
        n_filter_banks=4,
        freq_downsample=1,
        patch_emb_size=40,
        n_heads_patch=4,
        sst_emb_size=40,
        depth=1,
        n_classes=4,
        fs=250,
    )

    assert model.num_branches == 2
    assert model.classifier.fc[0].in_features == 80
    assert "branch_0" in model.transformers
    assert "branch_1" in model.transformers

    x = torch.randn(2, 1, 22, 1001)
    stft, embed, out = model(x)

    assert stft.shape == (2, 22, 4, 8)
    assert embed.shape == (2, 80)
    assert out.shape == (2, 4)


def test_mtfc_branch_f_t_s():
    """Test MTFC with branch='f_t_s' (separate freq, temporal, spatial transformers).

    Verifies that:
    1. Model creates 3 transformer branches
    2. Classifier input dimension is FTS * 3
    3. Forward pass produces correct output shapes
    """
    from models.MTFC import MTFC
    from types import SimpleNamespace

    args = SimpleNamespace(
        data_name="BCI-IV-2a",
        chn=22,
        patch_size=125,
        time_sample_num=1001,
        class_num=4,
        gate_flag=False,
        posemb_flag=True,
        branch="f_t_s",
        chn_attn_flag=False,
        fts_attn_flag=True,
        sst_method="stf_attention_temporal_values",
        stft_reconstruction=True,
        spa_dim=16,
    )

    model = MTFC(
        args,
        n_filter_banks=4,
        freq_downsample=1,
        patch_emb_size=40,
        n_heads_patch=4,
        sst_emb_size=40,
        depth=1,
        n_classes=4,
        fs=250,
    )

    assert model.num_branches == 3
    assert model.classifier.fc[0].in_features == 120
    assert "branch_0" in model.transformers
    assert "branch_1" in model.transformers
    assert "branch_2" in model.transformers

    x = torch.randn(2, 1, 22, 1001)
    stft, embed, out = model(x)

    assert stft.shape == (2, 22, 4, 8)
    assert embed.shape == (2, 120)
    assert out.shape == (2, 4)


def test_mtfc_branch_ft_s_filter_banks():
    """Test MTFC with branch='ft_s' and filter_banks method.

    Verifies that:
    1. Model works with filter_banks SST method
    2. Produces correct output shapes
    """
    from models.MTFC import MTFC
    from types import SimpleNamespace

    args = SimpleNamespace(
        data_name="BCI-IV-2a",
        chn=22,
        patch_size=125,
        time_sample_num=1001,
        class_num=4,
        gate_flag=False,
        posemb_flag=True,
        branch="ft_s",
        chn_attn_flag=False,
        fts_attn_flag=True,
        sst_method="filter_banks",
        stft_reconstruction=True,
        spa_dim=16,
    )

    model = MTFC(
        args,
        n_filter_banks=4,
        freq_downsample=1,
        patch_emb_size=40,
        n_heads_patch=4,
        sst_emb_size=40,
        depth=1,
        n_classes=4,
        fs=250,
    )

    assert model.num_branches == 2
    assert model.classifier.fc[0].in_features == 80

    x = torch.randn(2, 1, 22, 1001)
    stft, embed, out = model(x)

    assert stft.shape == (2, 22, 4, 8)
    assert embed.shape == (2, 80)
    assert out.shape == (2, 4)


def test_mtfc_filter_banks_temporal_kernel():
    """Test MTFC with filter_banks method and different temporal_kernel values.

    Verifies that:
    1. temporal_kernel parameter is correctly passed to FilterBanksEmbedding_v5
    2. Different temporal_kernel values work correctly
    3. Forward pass produces correct output shapes
    """
    from models.MTFC import MTFC
    from types import SimpleNamespace

    args = SimpleNamespace(
        data_name="SSVEP",
        chn=8,
        patch_size=100,
        time_sample_num=1000,
        class_num=4,
        gate_flag=False,
        posemb_flag=True,
        branch="f_t_s",
        chn_attn_flag=False,
        fts_attn_flag=False,
        sst_method="filter_banks",
        stft_reconstruction="frequency",
        spa_dim=16,
    )

    for tk in [25, 43, 63]:
        model = MTFC(
            args,
            n_filter_banks=11,
            freq_downsample=1,
            patch_emb_size=40,
            n_heads_patch=4,
            sst_emb_size=40,
            depth=1,
            n_classes=4,
            fs=250,
            temporal_kernel=tk,
        )

        assert model.temporal_kernel == tk

        x = torch.randn(2, 1, 8, 1000)
        stft, embed, out = model(x)

        assert embed.shape[0] == 2
        assert embed.shape[1] == model.classifier.fc[0].in_features
        assert out.shape == (2, 4)


def test_mtfc_filter_banks_variant():
    """Test MTFC with different filter_banks_variant values.

    Verifies that:
    1. filter_banks_variant parameter correctly selects the embedding class
    2. All variant names work correctly
    3. Forward pass produces correct output shapes
    """
    from models.MTFC import MTFC, FILTER_BANKS_VARIANTS
    from types import SimpleNamespace

    args = SimpleNamespace(
        data_name="SSVEP",
        chn=8,
        patch_size=100,
        time_sample_num=1000,
        class_num=4,
        gate_flag=False,
        posemb_flag=True,
        branch="f_t_s",
        chn_attn_flag=False,
        fts_attn_flag=False,
        sst_method="filter_banks",
        stft_reconstruction="frequency",
        spa_dim=16,
    )

    for variant_name in FILTER_BANKS_VARIANTS.keys():
        model = MTFC(
            args,
            n_filter_banks=11,
            freq_downsample=1,
            patch_emb_size=40,
            n_heads_patch=4,
            sst_emb_size=40,
            depth=1,
            n_classes=4,
            fs=250,
            temporal_kernel=43,
            filter_banks_variant=variant_name,
        )

        assert model.filter_banks_variant == variant_name
        assert isinstance(
            model.frequency_embedding, FILTER_BANKS_VARIANTS[variant_name]
        )

        x = torch.randn(2, 1, 8, 1000)
        stft, embed, out = model(x)

        assert embed.shape[0] == 2
        assert embed.shape[1] == model.classifier.fc[0].in_features
        assert out.shape == (2, 4)
