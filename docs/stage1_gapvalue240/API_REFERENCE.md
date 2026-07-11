# Run API

Every `run_NNN.py` exports:

```python
prepare(machine_config, attempt_id=None, allow_new_attempt_after_validated=False)
train(machine_config, attempt_id)
evaluate(machine_config, attempt_id)
validate(machine_config, attempt_id)
run(machine_config, attempt_id=None, allow_new_attempt_after_validated=False)
```

The module constant `RUN_SLOT` is the only run-specific code value. All scientific parameters are looked up from the signed frozen matrix. This permits single-run deployment, independent recovery, and future maintenance without a monolithic 240-run function.
