# CXR Consistency Streamlit Demo

Демо для защиты: мультимодальная проверка согласованности рентгеновского снимка грудной клетки и медицинского заключения.

## Пути

- Checkpoint: `/root/cxr-consistency/experiments/checkpoint_full/convnext_tiny_cxrbert_baseline_full_checkpoint_bs96/best_model.pt`
- Config: `/root/cxr-consistency/experiments/checkpoint_full/convnext_tiny_cxrbert_baseline_full_checkpoint_bs96/config_snapshot.yaml`
- CSV: `/root/cxr-consistency/data/pairs/cxr_consistency_pairs_hard.csv`
- Реализация модели: `/root/cxr-consistency/src/cxr_consistency/model.py`

## Запуск

```bash
cd /root/cxr-consistency
streamlit run demo_streamlit/app.py --server.address 0.0.0.0 --server.port 8501
```

Если `streamlit` доступен только внутри виртуального окружения, сначала активируйте его.

## Что показывает приложение

- случайный/следующий пример без загрузки всех изображений в память;
- поиск по `study_id` и `patient_id`;
- инференс истинной пары с порогом по умолчанию `0.34`;
- интерактивные hard negative pairs: negation, laterality, temporal, partial mismatch, pathology swap, random report;
- diff между исходным и измененным отчетом;
- опциональный Grad-CAM, который не останавливает приложение при ошибке построения;
- Text attribution для отчета, который не останавливает приложение при ошибке расчета.

## Интерпретация решения

- Image Grad-CAM показывает визуальные области снимка, которые сильнее всего влияют на решение модели.
- Text attribution показывает слова отчета, изменение которых сильнее всего влияет на probability. Метод использует perturbation-based occlusion: значимые слова временно заменяются на `[MASK]`, после чего probability пересчитывается.
