# AGENTS.md

Операционное руководство для работы агентов (Codex/LLM) в данном репозитории.

## О проекте

Проект посвящён мультимодальной задаче проверки согласованности рентгеновского снимка грудной клетки (CXR) и медицинского заключения, а не генерации отчётов.

- Вход: `(изображение, текст заключения)`.
- Выход: бинарная метка согласованности.
- `label = 1` — изображение и заключение согласованы.
- `label = 0` — изображение и заключение противоречат друг другу.

## Что нельзя ломать

Обязательно сохранять:

- Разделение данных на уровне пациентов (patient-level split).
- Основной датасет пар: `data/pairs/cxr_consistency_pairs_hard.csv`.
- Настройку токенизатора и текстового энкодера CXR-BERT.
- Рабочую установку CUDA и PyTorch.
- Уже полученные результаты в:
  - `experiments/`
  - `results/coursework_export/`

Не запускать длительное обучение, большие sweep-эксперименты и генерацию данных без явного указания пользователя.

---

## Основной датасет

Основные файлы:

- Очищенные заключения: `data/processed/cxr_reports_clean.csv`
- Набор сложных пар: `data/pairs/cxr_consistency_pairs_hard.csv`

В финальных экспериментах исключён тип отрицательных примеров `random_report`.

### Основные типы сложных отрицательных пар

Используются в финальной версии работы:

- `pathology_matched_report`
- `distorted_negation`
- `laterality_conflict`
- `temporal_mismatch`
- `pathology_semantic_swap`
- `partial_mismatch`

### Устаревшие типы

Не использовать без специальной необходимости:

- `distorted_pathology`
- старые варианты `distorted_location`
- `distorted_severity`
- `view_matched_report`

---

## Финальные модели

Основные модели курсовой работы:

- `configs/final_full/convnext_tiny_cxrbert_full.yaml`
- `configs/final_full/deit_base_cxrbert_full.yaml`
- `configs/final_full/vit_base_cxrbert_full.yaml`

### Итоговые результаты

| Модель | ROC-AUC | F1 | AUC для pathology_matched_report |
|----------|---------:|---------:|---------:|
| ConvNeXt Tiny + CXR-BERT (unfrozen) | 0.9228 | 0.7378 | 0.8305 |
| DeiT Base + CXR-BERT (unfrozen) | 0.9156 | 0.7244 | 0.8159 |
| ViT Base + CXR-BERT (unfrozen) | 0.9072 | 0.7142 | 0.8077 |

### Итоговые выводы

- Основная модель: **ConvNeXt Tiny + CXR-BERT**.
- Лучший трансформерный энкодер изображений: **DeiT Base + CXR-BERT**.
- Базовый трансформерный ориентир: **ViT Base + CXR-BERT**.
- Наиболее сложный тип отрицательных примеров: **pathology_matched_report**.

---

## Основные скрипты

### Подготовка данных

- `scripts/01_download_and_merge.py`
- `scripts/02_prepare_task_dataset.py`
- `scripts/03_make_hard_pairs.py`

### Обучение

- `scripts/04_train.py`

### Финальные эксперименты

- `scripts/run_final_full.py`
- `scripts/audit_final_full.py`

### Серии экспериментов (sweeps)

- `scripts/run_sweep_a6000.py`
- `scripts/audit_sweep_results.py`
- `scripts/run_text_sweep_compact.py`
- `scripts/audit_text_sweep_compact.py`

### Агрегация и визуализация результатов

- `scripts/collect_experiment_results.py`
- `scripts/plot_experiment_results.py`

---

## Исторические и вспомогательные конфигурации

Эти конфигурации сохраняются для воспроизводимости, но не считаются финальными:

- `configs/best_found.yaml` — упрощённый baseline с использованием `random_report`.
- `configs/hard_pairs_convnext.yaml` — ранний сильный baseline на сложных парах.
- `configs/vit_base_nocollapse.yaml` — эксперимент по устранению коллапса ViT.
- `configs/finalists/` — промежуточные модели-финалисты.
- `configs/text_sweeps/` — расширенный перебор текстовых энкодеров; для курсовой предпочтителен компактный sweep.

Архив старых диагностических конфигураций:

- `results/coursework_export/archived_legacy_configs/`

---

## Результаты и отчёты

Основные артефакты проекта:

- Финальное ревью: `results/final_full_review.md`
- Анализ визуальных энкодеров: `results/final_visual_encoder_selection.md`
- Экспорт результатов для переноса: `results/coursework_export/`
- Финальные результаты обучения:
  - `results/coursework_export/final_full/`
- Результаты сравнения визуальных энкодеров:
  - `results/coursework_export/visual_encoder_sweeps/`

---

## Порядок отладки

При возникновении проблем проверять компоненты в следующем порядке:

1. Целостность данных и корректность путей к изображениям.
2. Отсутствие утечки между обучающей и тестовой выборками.
3. Качество сформированных пар и распределение типов отрицательных примеров.
4. Баланс классов и корректность меток.
5. Усечение последовательностей токенизатором.
6. Коллапс логитов или вероятностей.
7. Гиперпараметры модели и планировщик скорости обучения (scheduler).