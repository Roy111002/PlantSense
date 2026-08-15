---
license: cc0-1.0
pipeline_tag: image-classification
---

# PlantSense plant-health classifier

MobileNetV3-Small classifier trained for the PlantSense educational project.
Its default task is a generic visual screen: `healthy` versus `abnormal`.

## Evaluation

- Test accuracy: 0.9331
- Test macro F1: 0.9208
- Test samples: 10948
- Dataset revision: `f808f2706d3ce50ce14652d2b6863d3cc30cf9c4`
- Model SHA-256: `2a042e9d991bca794ecb56a9ed2dc43ef6a8ed97f7b8920367a49f156d6b2749`

## Intended use

The model provides conservative screening evidence for PlantSense. An
`abnormal` result does not identify a disease or cause. It must not trigger
pesticide or disease treatment without expert confirmation.

## Limitations

- PlantVillage images largely use controlled backgrounds.
- Plants unlike the PlantVillage training crops may be outside its domain.
- `abnormal` combines many diseases and is not a diagnosis.
- Real ESP32-CAM validation and target-domain fine-tuning are still required.
