# Dataset Layout

The complete local/LAN-only dataset lives at `courses/industrial-vision-lab/data/dataset/`. It is intentionally ignored by Git and excluded from student packages.

```text
dataset/
  raw/                 Original images, red-box images, phone captures, inventory
  annotations/         Red-box coordinates and training manifests
  augmentation_inputs/ Foreground images, masks, and source manifest
  derived/             Derived inspection samples
  generated/           Augmented samples
  weak_labels/         Weak labels, review images, and manifests
  weld_appearance_v2/  Prepared image-classification dataset
  weld_appearance_v2_nopost/
  weld_window_crops/   Prepared weld-window crops
  yolo/                YOLO images, labels, and split manifest
  validation/          Mobile validation evidence and records
  records/legacy_sqlite/ Historical SQLite sources for MySQL migration audit
```

The unified dataset currently contains 743 files: 24 raw image/inventory files, 4 annotation manifests, 702 prepared dataset files, 11 validation files, and 2 SQLite migration sources. Regenerable Python bytecode and YOLO `.cache` files are excluded. The historical SQLite files are read-only migration sources; live records are stored in the LAN MySQL database. Publicly licensed teaching samples remain under `data/public` and are separate from this complete private dataset.
