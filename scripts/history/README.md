本目录用于保存历史训练入口脚本，避免后续重构 `main.py` 后丢失旧流程。

当前保留：

- `main_pipeline_full_174f329.py`
  说明：
  2026-03-29 之前的全流程入口，覆盖 source 分类、target 微调、CAM 导出和 pseudo-box 生成。
  如果后面需要回看旧链路，可直接参考该文件。
