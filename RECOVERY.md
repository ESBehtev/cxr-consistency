# RECOVERY.md

# Восстановление проекта cxr-consistency

Документ содержит инструкции по развёртыванию проекта, восстановлению данных и воспроизведению основных результатов курсовой работы.

## 1. Клонирование репозитория

bash git clone <REPO_URL> cxr-consistency cd cxr-consistency 

Используйте ветку репозитория, содержащую:

- configs/final_full/
- configs/sweeps/
- configs/text_sweeps_compact/
- scripts/04_train.py
- src/cxr_consistency/

---

## 2. Создание окружения

Создайте и активируйте виртуальное окружение:

bash python3 -m venv .venv source .venv/bin/activate  python -m pip install --upgrade pip setuptools wheel  pip install -r requirements.txt pip install -e . 

После установки рекомендуется проверить доступность CUDA:

bash python - <<'PY' import torch  print("Torch:", torch.__version__) print("CUDA available:", torch.cuda.is_available())  if torch.cuda.is_available():     print("GPU:", torch.cuda.get_device_name(0)) PY 

Переустанавливать PyTorch или CUDA следует только при обнаружении ошибок на данном этапе.

---

## 3. Внешние зависимости

Для загрузки данных необходим доступ к следующим источникам:

### Kaggle

Используется для загрузки изображений MIMIC-CXR.

bash mkdir -p ~/.kaggle chmod 600 ~/.kaggle/kaggle.json 

### Hugging Face

Используется для:

- загрузки набора заключений;
- загрузки модели CXR-BERT;
- кэширования токенизатора и весов модели.

Используемая языковая модель:

text microsoft/BiomedVLP-CXR-BERT-specialized 

При первом запуске может потребоваться доступ в Интернет для загрузки весов и токенизатора.

---

## 4. Подготовка данных

Данный этап необходим только при полном восстановлении проекта с нуля.

### Загрузка и объединение данных

bash python scripts/01_download_and_merge.py 

### Подготовка очищенных записей

bash python scripts/02_prepare_task_dataset.py 

### Генерация сложных негативных пар

bash python scripts/03_make_hard_pairs.py 

После завершения должны существовать следующие файлы:

text data/raw/kaggle_mimic/ data/processed/hf_findings.csv data/processed/cxr_reports_clean.csv data/pairs/cxr_consistency_pairs_hard.csv 

---

## 5. Типы негативных примеров

Финальная версия проекта использует следующие типы сложных негативных пар:

- pathology_matched_report
- distorted_negation
- laterality_conflict
- temporal_mismatch
- pathology_semantic_swap
- partial_mismatch

Тип random_report присутствует в исходном наборе пар, однако исключается из финальных конфигураций обучения.

---

## 6. Воспроизведение финальных экспериментов

Основные конфигурации курсовой работы:

text configs/final_full/convnext_tiny_cxrbert_full.yaml configs/final_full/deit_base_cxrbert_full.yaml configs/final_full/vit_base_cxrbert_full.yaml 

### Запуск обучения

Рекомендуемый вариант:

bash tmux new -s final_full  source .venv/bin/activate  python scripts/run_final_full.py \     --max-parallel-jobs 3 \     --poll-seconds 30 

Консервативный вариант:

bash python scripts/run_final_full.py \     --max-parallel-jobs 2 \     --poll-seconds 30 

### Анализ результатов

bash python scripts/audit_final_full.py 

---

## 7. Эталонные результаты

| Модель | Лучшая эпоха | ROC-AUC | F1 | AUC для pathology_matched_report |
|----------|----------:|----------:|----------:|----------:|
| ConvNeXt Tiny + CXR-BERT | 8 | 0.9228 | 0.7378 | 0.8305 |
| DeiT Base + CXR-BERT | 5 | 0.9156 | 0.7244 | 0.8159 |
| ViT Base + CXR-BERT | 7 | 0.9072 | 0.7142 | 0.8077 |

Лучшая модель исследования:

text ConvNeXt Tiny + CXR-BERT 

В финальных конфигурациях сохранение чекпоинтов отключено. Для анализа используются файлы:

text metrics.jsonl summary.json status.json 

---

## 8. Дополнительные эксперименты

### Сравнение визуальных энкодеров

bash python scripts/run_sweep_a6000.py python scripts/audit_sweep_results.py 

Конфигурации:

text configs/sweeps/ 

### Сравнение текстовых энкодеров

bash python scripts/run_text_sweep_compact.py python scripts/audit_text_sweep_compact.py 

Конфигурации:

text configs/text_sweeps_compact/ 

---

## 9. Экспорт результатов

Основной архив результатов расположен в:

text results/coursework_export/ 

Ключевые файлы:

text results/coursework_export/FINAL_SUMMARY.md results/coursework_export/ALL_EXPERIMENTS_TABLE.csv results/coursework_export/COURSEWORK_NOTES.md 

### Финальные модели

text results/coursework_export/final_full/ 

Содержит:

- итоговые таблицы метрик;
- сравнение моделей;
- результаты по типам негативных примеров;
- конфигурации экспериментов;
- журналы метрик.

### Сравнение визуальных энкодеров

text results/coursework_export/visual_encoder_sweeps/ 

---

## 10. Проверка целостности проекта

Для быстрой проверки корректности установки:

bash python -m py_compile scripts/*.py src/cxr_consistency/*.py 

Проверка наличия ключевых файлов:

bash python - <<'PY' from pathlib import Path  paths = [     "data/pairs/cxr_consistency_pairs_hard.csv",     "configs/final_full/convnext_tiny_cxrbert_full.yaml",     "configs/final_full/deit_base_cxrbert_full.yaml",     "configs/final_full/vit_base_cxrbert_full.yaml", ]  for path in paths:     print(path, Path(path).exists()) PY 

---

## 11. Основные материалы для курсовой работы

При подготовке отчёта рекомендуется использовать:

text results/final_full_review.md results/final_visual_encoder_selection.md results/coursework_export/ 

Эти материалы содержат итоговые таблицы, сравнение моделей, результаты по типам негативных примеров и основные выводы исследования.