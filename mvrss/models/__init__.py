from .tmvanet import TMVANet
from .mvnet import MVNet
from .mvanet import MVANet
from .mva_dcn import MVA_DCN
from .tmva_dcn import TMVA_DCN
from .tmva_tdc import TMVA_TDC
from .pkcin import PKCIn
from .adapkc import AdaPKC
from .pkcon import PKCOn
import os

# TFocNet variant:
# - v2: hard top-k + ST bridge
# - v1_revise: revised v1 interaction (pair-interleaved metric over candidate-n)
_tfoc_variant = os.environ.get("TFOC_MODEL_VARIANT", "").strip().lower()
if _tfoc_variant == "v2":
    from .tfoc_vanilla_maskdraw_light_clutter_n_v2 import TFocNet
elif _tfoc_variant in ("att4_v2", "att4-v2", "block", "block_att4_v2", "tfoc_block"):
    from .tfoc_vanilla_maskdraw_light_block_clutter_att4_v2 import TFocNet
elif _tfoc_variant in ("v2_revise", "v2-revise", "rev2"):
    from .tfoc_vanilla_maskdraw_light_clutter_n_v2_revise_standalone import TFocNet
elif _tfoc_variant in ("v1_revise", "v1-revise", "revise"):
    from .tfoc_vanilla_maskdraw_light_clutter_n_v1_revise import TFocNet
else:
    from .tfoc_vanilla_maskdraw_light_clutter_n_v1 import TFocNet

# from .tfoc_topk import TFocNet
# from .tfocnet_multiframe import TFocNet
# from .tfoc_testsync import TFocNet
# from .tfocnet_tfocfirst_2dconv_dyna import TFocNet
# from .tfocnet_tfocfirst_2dconv import TFocNet
# from .tfoc_ultra_hook import TFocNet
# from .tfoc_vanilla_maskdraw_light_clutter_n_release import TFocNet
# from .tfoc_vanilla_maskdraw_group_dilationconv import TFocNet
#from .tfoc_vanilla_maskdraw_light_block_clutter_att3_v2 import TFocNet
#from .vit_rd import vit_base_patch16_224_rd
try:
    from .vit import VitNet
except ModuleNotFoundError as _vit_import_error:
    class VitNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "VitNet dependencies are missing. Install optional packages "
                "(for example: matplotlib) before using model='vit'."
            ) from _vit_import_error