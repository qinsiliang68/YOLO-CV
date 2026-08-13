# Commit 04 TDD evidence

This rollback unit implements the fixed-base-step replay gradient injection and
the narrow frozen-Ultralytics integration. `RED.junit.xml` was produced against
the untouched expert package. `GREEN.junit.xml` covers the completed unit,
including a real `yolo11l-cls.pt`/`ClassificationTrainer` engineering canary.
The canary is explicitly non-scientific and does not use formal data.
