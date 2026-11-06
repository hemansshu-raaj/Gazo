"""Build the app's bundled examples: 1024x1024 crops of *test* SIDD scenes (with ground truth) as NPZ.

  python -m eval.make_examples
"""
import json
import os

import numpy as np

from isp.data import SceneArrays

EXAMPLES = {
    "sidd_lowlight_iphone_iso100": ("0159_007_IP_00100_00100_3200_L", 900, 1400),
    "sidd_iso10000_pixel": ("0065_003_GP_10000_08460_4400_N", 1200, 1500),
    "sidd_normal_nexus6": ("0142_007_N6_00100_00100_4400_N", 1000, 1600),
}


def main(root="data/scenes", out="examples", size=1024):
    os.makedirs(out, exist_ok=True)
    for name, (scene, y, x) in EXAMPLES.items():
        sc = SceneArrays(os.path.join(root, scene))
        bayer = sc.noisy_crop(0, y, x, size)
        gt = (sc.gt_crop(y, x, size) * 255 + 0.5).astype(np.uint8)
        meta = {"dng_tags": sc.meta["dng_tags"], "cfa": [["R", "G"], ["G", "B"]], "scene": scene,
                "camera": sc.meta.get("camera"), "iso": sc.meta.get("iso"), "brightness": sc.meta.get("brightness")}
        np.savez_compressed(os.path.join(out, name + ".npz"), bayer=bayer.astype(np.float32), gt=gt,
                            meta=json.dumps(meta))
        print(name, os.path.getsize(os.path.join(out, name + ".npz")) / 1e6, "MB")


if __name__ == "__main__":
    main()
