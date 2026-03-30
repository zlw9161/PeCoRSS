# `echmo_debug.py` Comparison Report

Generated on: 2026-03-20

## Compared Files
- `svrss/models/echmo_debug.py`
- `svrss-echmo/models/echmo_debug.py`

## Quick Summary
- The two files are **not identical**.
- Line counts differ:
  - `svrss/models/echmo_debug.py`: **655** lines
  - `svrss-echmo/models/echmo_debug.py`: **672** lines
- SHA-256 differs:
  - `svrss/models/echmo_debug.py`: `b82d805a5120813dc5b67d54a9f2379f9f4266299efbf5d9371d7399806af58a`
  - `svrss-echmo/models/echmo_debug.py`: `7c993af3d353c378c1e049b6bba9995d41d4be4263b79d3d6b3db29ff01cb0b1`

## Similarities
- Top-level structure is the same in both files:
  - `_init_translator_weights`
  - `Leanable_Match_FilterV2`
  - `Leanable_Match_FilterV5`
  - `LMF_Chain`
  - `Multi_Order_LDC`
  - `Group_Purturble_DF`
  - `Leanable_Coherent_Integration`
  - `EchMo`
- Both files implement the same model pipeline stages:
  - LMF -> MLDC -> GDF -> RDTranslator

## Key Differences

### 1. Imports
- `svrss-echmo` adds:
  - `complex_minmax`
  - `PKCIn4ANY`

### 2. `Multi_Order_LDC` implementation
- `svrss` version uses `self.ldc_list` with per-order `nn.Conv2d` modules.
- `svrss-echmo` version switches to parameterized kernels + `F.conv2d`:
  - uses `nn.ParameterList` (`hp_kernels`)
  - adds optional `hp_biases`
  - removes `self.ldc_list`
- Device handling is improved in `svrss-echmo`:
  - `torch.zeros(...).to(x.device)` instead of hardcoded `.cuda()`.

### 3. `Group_Purturble_DF` device handling
- `svrss` uses `x.real.cuda(), x.imag.cuda()`.
- `svrss-echmo` uses `x.real.to(x.device), x.imag.to(x.device)`.
- This makes `svrss-echmo` more device-agnostic.

### 4. RDTranslator input channels and output feature choice
- In `EchMo.__init__`:
  - `svrss`: `in_ch=self.df_groups*4`
  - `svrss-echmo`: `in_ch=self.df_groups` (amp-only path)
- In `EchMo.forward`:
  - `svrss-echmo` uses `x_d = torch.abs(x_d).unsqueeze(dim=2)`.
  - `svrss` uses concatenated `amp+phase+I+Q` (4-channel feature composition).

### 5. Debug/demo path defaults in `__main__`
- Dataset/coefs default paths differ:
  - `svrss`: Windows-style (`E:/DataSets/...`)
  - `svrss-echmo`: Linux-style (`/data/PeCoLab/...`)
- Example frame index differs:
  - `svrss`: `000025.npy`
  - `svrss-echmo`: `000000.npy`

## Practical Impact
- `svrss-echmo/models/echmo_debug.py` is more aligned with Linux execution and device portability.
- `svrss/models/echmo_debug.py` currently keeps richer translator input features (`amp+phase+I+Q`), while `svrss-echmo` simplifies to amp-only.
- MLDC in `svrss-echmo` is refactored toward direct trainable-kernel convolution, which changes parameter registration/forward style.

## Reference Diff Snippet
```diff
- from svrss.models.model_utils.echo_tools import complex_std, complex_double2float, filter_coef_loader
+ from svrss.models.model_utils.echo_tools import complex_std, complex_minmax, complex_double2float, filter_coef_loader
+ from svrss.models import PKCIn4ANY

- y_r = torch.zeros(B, self.orders, self.out_ch, P, R).cuda()
+ y_r = torch.zeros(B, self.orders, self.out_ch, P, R).to(x.device)

- x_real, x_imag = x.real.cuda(), x.imag.cuda()
+ x_real, x_imag = x.real.to(x.device), x.imag.to(x.device)

- in_ch=self.df_groups*4,
+ in_ch=self.df_groups, # for amp only

- x_d = torch.cat((torch.abs(x_d), torch.angle(x_d), x_d.real, x_d.imag), dim=1).unsqueeze(dim=2)
+ x_d = torch.abs(x_d).unsqueeze(dim=2) # amp-only
```
