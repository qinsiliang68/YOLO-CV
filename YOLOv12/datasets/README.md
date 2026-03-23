# Dataset Layout

Put your custom dataset inside this folder.

Recommended detection structure:

```text
datasets/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Then update `configs/datasets/custom_detect.yaml` to match your classes and paths.
