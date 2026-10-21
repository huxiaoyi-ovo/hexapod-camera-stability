# Hexapod asset provenance

The standalone robot description and referenced geometry were copied without edits from the local checkout at `/home/artrc/Pcrnet_hexapod/resources/robots/hex_v4` (source commit `826fdac8bb49403af3fa53f4b967738257ebf0bb`; source checkout was clean at import time). The URDF is byte-identical to the source. Runtime code, configurations, control, simulation, and logging in this repository are original standalone project files and do not load the upstream repository.

The source repository's `LICENSE` is retained as [`UPSTREAM_LICENSE`](UPSTREAM_LICENSE). That license itself points to separate asset licenses. No hex_v4-specific CAD asset license was found in the source checkout, so this notice does not make a separate claim about licensing of those CAD assets.

The URDF's active references resolve to the copied meshes below. Its active toe collisions are spheres defined directly in the URDF. No external texture/material files are referenced by these selected DAE files.

SHA-256 hashes:

| File | SHA-256 |
|---|---|
| `urdf/hex_ground.urdf` | `187f8281e1976e050c7a960c288e033368c5d5ded3932725e34942a5085215e6` |
| `two_set_meshes/visual/body.dae` | `cea614c114aa6afaa709cf34edba325ebb40b5c6ca6354178ca43ad22ad06b09` |
| `two_set_meshes/visual/l_l_ankle.dae` | `d80f7763ae1b47fc3301b464464e449a60e90157b696db27b1037a1b554a59a9` |
| `two_set_meshes/visual/l_l_knee.dae` | `6068d5829876b08e486369ef48f911dc2f00dcdab63980009e77802face98d3a` |
| `two_set_meshes/visual/l_l_thigh.dae` | `06fca88848a4b6872e788854baea74ded44669abb45d172a255436171b1e3e35` |
| `two_set_meshes/visual/l_r_ankle.dae` | `d80edc49a6a96372e53faef82961fc2b95943007537e0a0f7f638ca6c1d19908` |
| `two_set_meshes/visual/l_r_knee.dae` | `b1d7cb55bf64f5af1d876f9b0f624013026c2c296090d80a54e335a1bdccde33` |
| `two_set_meshes/visual/l_r_thigh.dae` | `9a4cfc3c37a51aa02957cb09cd29f2dc86b2ed2135215cb7630e7c5860191b69` |
| `two_set_meshes/collision/l_l_ankle.obj` | `afefb7295b7c86ba0c878a7ec86db1671e0b3b7beb47c20cd2c9587783eba747` |
| `two_set_meshes/collision/l_l_knee.obj` | `074b0040aaeadb047c9fdc4e354ac38fc59091d35655a5ac0186d71640a87435` |
| `two_set_meshes/collision/l_l_thigh.obj` | `5addffd24edc247b54c52b158e90e7b9a0ffeee55d82bd945b441fbb111e3d26` |
| `two_set_meshes/collision/l_r_ankle.obj` | `03812b14d4aa4054efa4df5f489c0757498459e6bbcb80975d972d95ca821f20` |
| `two_set_meshes/collision/l_r_knee.obj` | `88c42b900f0d64edfd2ef49144648bc0559a88199b7a8e681ae03529b3c8e71e` |
| `two_set_meshes/collision/l_r_thigh.obj` | `44943f5efb40eaf97b01812530ce255bc6ec652e0cbe76fe072e0534a7959964` |
