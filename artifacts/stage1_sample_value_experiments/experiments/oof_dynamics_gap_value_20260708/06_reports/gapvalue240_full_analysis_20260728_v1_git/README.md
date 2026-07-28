# Stage1 GapValue 深度分析静态报告

本目录是可再生成的 experiment output，不是训练、selection 或人工真相源。

- `index.html`：主阅读入口。
- `FINAL_REPORT_CN.md`：中文文本摘要与结论边界。
- `tables/`：报告使用的完整 CSV，共 39 份。
- `charts/`：由表格生成的 PNG，共 11 张。
- `audit/`：上游提供的只读审计副本，共 7 份。
- `analysis_contract.yaml`：本报告使用的分析边界和判定合同。
- `manifest.json`：除自身外所有文件的大小和 SHA-256。

生成规则：

1. 生成器先写同级 `.inprogress` 目录；
2. 所有文件和清单完成后才将目录原子改名；
3. 已存在的正式目录或 `.inprogress` 目录都不会被覆盖；
4. 图中 zoomed axis 均明确标注，精确值以 CSV 为准。

生命周期：报告可由冻结表格重新生成；不得反向修改训练产物或 selection CSV。
