
## Examples on nuScenes testing augmentations

This is an example for testing our model with double fliping and rotation (angle = 6.25).

You can change the angle in the [config](https://github.com/dvlab-research/FocalsConv/blob/master/CenterPoint/test_aug_examples/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_focal_multimodal_doubleflip_rot6d25.py) to get the results tested on other angles, and then use the `nms_better2.py` to combine them together.

For example, you can use the below codes to combine your results on three angles.

```shell
python3 nms_better2.py --paths ${your_path_doubleflip}/infos_test_10sweeps_withvelo.json,${your_path_doubleflip_angle6d25}/infos_test_10sweeps_withvelo.json,${your_path_doubleflip_angle-6d25}/infos_test_10sweeps_withvelo.json --work_dir ${output_dir}
```
