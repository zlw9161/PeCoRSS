"""Model registry and builders with lazy imports."""

import inspect
from importlib import import_module

MODEL_REGISTRY = {
    "tfoc_sv": ("svrss.models.tfoc_ultra_sv_3D", "TFocNet"),
    "tfoc_sv_v1": ("svrss.models.tfoc_vanilla_maskdraw_light_clutter_n_v1", "TFocNet"),
    "tfoc_sv_v1_revise": (
        "svrss.models.tfoc_vanilla_maskdraw_light_clutter_n_v1_revise",
        "TFocNet",
    ),
    "tfoc_sv_att3_v2": (
        "svrss.models.tfoc_vanilla_maskdraw_light_block_clutter_att3_v2",
        "TFocNet",
    ),
    "tfoc_sv_att4_v2": (
        "svrss.models.tfoc_vanilla_maskdraw_light_block_clutter_att4_v2",
        "TFocNet",
    ),
    "att3_v2": (
        "svrss.models.tfoc_vanilla_maskdraw_light_block_clutter_att3_v2",
        "TFocNet",
    ),
    "att4_v2": (
        "svrss.models.tfoc_vanilla_maskdraw_light_block_clutter_att4_v2",
        "TFocNet",
    ),
}


def get_model_class(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            "Unsupported model '{}'. Available models: {}.".format(
                model_name, ", ".join(sorted(MODEL_REGISTRY.keys()))
            )
        )
    module_name, class_name = MODEL_REGISTRY[model_name]
    module = import_module(module_name)
    return getattr(module, class_name)


def build_model(
    model_name,
    n_classes,
    n_frames,
    signal_type,
    zero_doppler_filter=False,
    search_size=None,
):
    model_cls = get_model_class(model_name)
    kwargs = dict(
        n_classes=n_classes,
        n_frames=n_frames,
        signal_type=signal_type,
        zero_doppler_filter=zero_doppler_filter,
    )
    if search_size is not None:
        init_sig = inspect.signature(model_cls.__init__)
        if "search_size" in init_sig.parameters:
            kwargs["search_size"] = search_size
    return model_cls(**kwargs)


def TFocNet(n_classes, n_frames, signal_type, zero_doppler_filter=False):
    """Compatibility factory for legacy train scripts (`model=tfoc_sv`)."""
    return build_model(
        model_name="tfoc_sv",
        n_classes=n_classes,
        n_frames=n_frames,
        signal_type=signal_type,
        zero_doppler_filter=zero_doppler_filter,
    )
