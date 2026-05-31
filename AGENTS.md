# AGENTS.md

## О проекте

Этот репозиторий содержит курсовую работу по теме:

«Автоматическая проверка согласованности рентгенограмм грудной клетки и медицинских заключений на основе мультимодальных представлений».

Основная задача проекта — НЕ генерация отчета.

Задача проекта:
определить, соответствует ли текстовое медицинское заключение рентгеновскому снимку.

На вход:
- рентгенограмма грудной клетки;
- текст заключения.

На выход:
- вероятность согласованности пары «изображение–текст».

Метка:
- 1 — согласованная пара;
- 0 — несогласованная пара.

---

# Исследовательская постановка

Проект исследует задачу multimodal consistency checking.

Ключевая идея:
клинически корректный текст ≠ семантически согласованный текст.

Модель должна обнаруживать:
- ложные патологии;
- пропущенные патологии;
- ошибки локализации;
- ошибки стороны;
- инверсию отрицаний;
- ошибки степени выраженности.

Проект ближе к:
- multimodal verification;
- semantic alignment;

а НЕ к:
- report generation;
- image-text retrieval;
- pathology classification.

---

# Научная новизна проекта

Основной вклад проекта:

1. Постановка consistency checking как отдельной задачи.
2. Использование сложных negative pairs.
3. Проверка alignment между image/text embeddings.
4. Сравнение разных комбинаций image/text encoder.
5. Анализ клинических ошибок.

Особенно важно:
модель должна различать
клинически близкие тексты,
а не только случайные mismatched reports.

---

# Структура репозитория

## configs/

Конфиги экспериментов.

Все основные параметры должны задаваться через config.

Не хардкодить гиперпараметры в train script.

---

## scripts/

Основные этапы pipeline:

### prepare_dataset.py
Подготовка и очистка датасета.

### make_pairs.py
Формирование positive/negative pairs.

Это критически важный скрипт проекта.

### train.py
Основное обучение модели.

### sweep.py
Перебор конфигураций и hyperparameter sweep.

---

## src/

Основная логика проекта.

### model.py
Архитектуры моделей:
- image encoder
- text encoder
- fusion
- classifier

### negatives.py
Логика формирования negative pairs.

Одна из самых важных частей проекта.

### metrics.py
F1, ROC-AUC, precision/recall и другие метрики.

### train_utils.py
Обучение, validation и вспомогательные функции.

---

# Датасеты

Основной датасет:
- MIMIC-CXR

Также могут использоваться:
- CheXpert
- OpenI
- IU X-Ray
- Kaggle MIMIC subsets

Типичные поля:
- subject_id
- study_id
- image_id
- report
- image_path
- split

---

# Критически важные правила

## 1. Split только по пациентам

Сначала:
- train/valid/test split

Потом:
- генерация negative pairs.

Никогда не смешивать пациентов между split.

Иначе:
- leakage
- ложные метрики
- невалидные результаты

---

## 2. Negative pairs важнее архитектуры

Слабые negative pairs
→ искусственно высокие метрики.

Проект должен использовать:
- random negatives
- clinically similar negatives
- distorted negatives

---

## 3. Проверять данные раньше модели

Если качество плохое:
сначала проверять:
- pair generation
- leakage
- labels
- empty reports
- preprocessing
- class balance

Только потом:
- architecture
- hyperparameters

---

# Positive pairs

Положительная пара:
- оригинальное изображение
- оригинальный отчет

Пример:

```python
(image_i, report_i, 1)
```

---

# Negative pairs

## Random negative

Случайный чужой report.

```python
(image_i, report_j, 0)
```

---

## Clinically similar negative

Подмена:
- похожей патологией;
- похожей лексикой;
- похожей тематикой;
- похожим report.

Это значительно более сложные negative pairs.

---

## Distorted negative

Синтетическое искажение report.

Допустимые операции:
- inversion of negation;
- pathology swap;
- localization swap;
- severity change;
- removal of important finding.

Примеры:

```text
"Признаков плеврального выпота не выявлено"
→
"Выявлены признаки плеврального выпота"
```

```text
"right lower lobe opacity"
→
"left upper lobe opacity"
```

Это один из ключевых компонентов проекта.

---

# Архитектура модели

Текущая логика:

1. Image encoder
2. Text encoder
3. Fusion block
4. Binary classifier

Типичное объединение признаков:

```python
z = [v, u, |v-u|, v*u]
```

где:
- `v` — image embedding;
- `u` — text embedding.

Classifier:
- MLP
- sigmoid output

---

# Поддерживаемые image encoder

CNN:
- resnet18
- resnet50
- densenet121
- efficientnet_b0
- mobilenet_v3_small

Transformer:
- convnext_tiny
- vit_tiny
- vit_small
- vit_base
- swin_tiny

Medical:
- biomedclip_vit

---

# Text encoder

Возможные варианты:
- simple encoder
- BERT
- ClinicalBERT
- PubMedBERT
- BioClinicalBERT
- BiomedVLP-CXR-BERT

Важно:
более сильный text encoder
не гарантирует лучший результат.

Основная проблема может быть:
- embedding mismatch;
- плохой alignment;
- слабый visual encoder.

---

# Метрики

Основные:
- F1
- ROC-AUC

Дополнительные:
- accuracy
- precision
- recall
- confusion matrix

Главная метрика:
- F1

Почему:
пропуск несогласованного отчета
клинически опаснее.

BLEU/ROUGE —
НЕ основные метрики проекта.

---

# Анализ ошибок

Особенно важно анализировать:
- negation errors;
- left/right confusion;
- localization mismatch;
- weak findings;
- severity mismatch;
- clinically similar reports.

Проект должен поддерживать:
не только numeric metrics,
но и qualitative error analysis.

---

# Ограничения вычислений

Эксперименты предполагают:
- single GPU;
- RTX 3060 / 4060 уровень;
- ограниченную VRAM.

Не предлагать:
- extremely heavy training;
- giant multimodal stacks;
- distributed pipelines;

если это не требуется явно.

---

# Предпочитаемый стек

Использовать:
- PyTorch
- torchvision
- timm
- transformers
- pandas
- numpy
- scikit-learn
- tqdm

---

# Требования к коду

Код должен быть:
- понятным;
- исследовательским;
- воспроизводимым;
- легко дебажиться.

Избегать:
- overengineering;
- hidden abstractions;
- unnecessary complexity.

Предпочтительно:
- явные tensor shapes;
- прозрачный training loop;
- простые dataset class;
- читаемые configs.

---

# Как помогать

Если нужно исправить код:
- возвращать готовый рабочий фрагмент;
- кратко объяснять проблему;
- не переписывать проект полностью.

Если нужно улучшить качество:
проверять по порядку:

1. data quality
2. pair generation
3. leakage
4. preprocessing
5. class balance
6. architecture
7. hyperparameters

---

# Типичные причины плохих метрик

Всегда проверять:
- перепутанные labels;
- leakage;
- плохие negative pairs;
- пустые reports;
- tokenizer truncation;
- corrupted image paths;
- frozen encoder;
- embedding dimension mismatch;
- split contamination.

Во многих случаях проблема в данных,
а не в архитектуре.

---

# Возможные дальнейшие улучшения

Потенциальные направления:
- cross-attention;
- patch-level ViT;
- contrastive learning;
- Grad-CAM;
- explainability;
- token-region alignment;
- real clinical inconsistency annotations.

Но текущий приоритет:
получить стабильный reproducible pipeline.

---

# Главная цель репозитория

Финальная система должна:
- стабильно обучаться;
- различать consistent/inconsistent pairs;
- работать со сложными negative pairs;
- показывать адекватные validation metrics;
- поддерживать сравнение encoder;
- поддерживать error analysis.

Проект предназначен для:
- курсовой;
- исследований;
- экспериментов;
- демонстрации мультимодального подхода.

Это НЕ production medical system.