
## Examples on nuScenes testing augmentations

This config yaml file is an example for testing our model with double fliping and rotation (angle = 6.25).

You can change the angle in the config file to get the results tested on other angles. An then use the nms_better2.py to combine them together.

For example, you can use the below codes to combine your results on three angles.

```shell
python3 ./tools/nms_better2.py --paths ${your_path_doubleflip}/infos_test_10sweeps_withvelo.json,${your_path_doubleflip_angle6d25}/infos_test_10sweeps_withvelo.json,${your_path_doubleflip_angle-6d25}/infos_test_10sweeps_withvelo.json
```
