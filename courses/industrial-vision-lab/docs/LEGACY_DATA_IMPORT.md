# Legacy Dataset Import

The former `power_battery_pack_welding_inspection` working tree has been consolidated into this Jupyter project. Its separate Git metadata, Android application, and temporary backup runs are not part of the Jupyter course. The following data and training materials are preserved in the ignored, private import directory:

```text
courses/industrial-vision-lab/data/legacy_import/
  02_image_data/          original and annotated photos, including phone captures
  dataset/                source, derived, augmented, and split training samples
  models/                 locally produced model artifacts
  training/               training, evaluation, tests, reports, and run outputs
  01_project_overview/
  03_inspection_plan/
  04_validation_records/
  docs/
  database/               read-only SQLite migration sources
```

This directory is intentionally excluded from Git. The canonical copy is stored on the LAN machine at `/ds1/workspace/CV/cv-beginner-course/courses/industrial-vision-lab/data/legacy_import/`; a SHA-256 manifest is used to verify synchronization. Public sample subsets and lesson assets remain separately tracked under `data/public` and `sources/`.

The import is retained for traceability and future dataset curation. Generated augmentations are correlated with their source photographs and must not be counted as independent validation examples. The imported data has not been promoted to public training data or released under the code license.
