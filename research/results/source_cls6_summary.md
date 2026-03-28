# Source CLS6 Summary

## 2026-03-28 train7200 run

- Dataset: `sewerml_cls6_train7200` (`train=6480`, `val=720`)
- Run name: `yolo11m_cls6_train7200`
- Model: `yolo11m-cls.pt`
- Epochs recorded: `62`
- Best epoch: `42`
- Best top-1: `70.83%`
- Best top-5: `99.44%`
- Best val loss: `1.3639`
- Last epoch top-1: `68.61%`
- Last epoch top-5: `99.03%`
- Last epoch val loss: `1.3601`
- Curves: `research/results/source_train7200_curves.png`
- Normalized confusion matrix: `research/results/source_train7200_confusion_matrix_normalized.png`

## Reference runs

| Run | Dataset | Model | Best epoch | Best top-1 | Best top-5 |
| --- | --- | --- | ---: | ---: | ---: |
| `yolo11n_cls6_focus` | `train3000` | `yolo11n-cls.pt` | 9 | 61.33% | 98.67% |
| `yolo11m_cls6_focus` | `train3000` | `yolo11m-cls.pt` | 10 | 59.33% | 99.00% |
| `yolo11m_cls6_train7200` | `train7200` | `yolo11m-cls.pt` | 42 | 70.83% | 99.44% |
