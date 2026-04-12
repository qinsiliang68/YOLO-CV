# Bucket Pilot Verdict

- machine: `A`

- G0 Spec@R99.5: `0.547619`

| Signal | Verdict | Q1 Spec@R99.5 | Q5 Spec@R99.5 | Q1-Q5 | Q1-G0 | Q1-Q3 Monotonic | Rationale |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| R | mixed | 0.523810 | 0.583333 | -0.059523 | -0.023809 | False | 存在非单调或局部增益，需结合样本分析进一步判断。 |
