# AgileX NERO model source

- Product: https://global.agilex.ai/products/nero
- Official repository: https://github.com/agilexrobotics/agx_arm_urdf
- Upstream commit: `f6642ce0d7872c686f29c99e9e10cd23d1d49313`
- Retrieved: 2026-07-14

The `meshes/` and `urdf/` directories are unmodified copies of the upstream
`nero/` directory. `nero_with_gripper.urdf` was expanded from the official
`nero_with_gripper_description.xacro` include chain with xacro 2.1.1. Its
`package://agx_arm_description/agx_arm_urdf/nero/` mesh prefixes were replaced
with paths relative to this directory so that standalone URDF loaders can find
the included meshes.

See `LICENSE` for the upstream MIT license and `UPSTREAM_README.md` for the
upstream usage notes.
