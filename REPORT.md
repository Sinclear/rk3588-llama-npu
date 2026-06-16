================================================================================
ОТЧЁТ: Оптимизация NPU инференса LLM на RK3588
Дата: 2026-06-16
Платформа: Orange Pi 5 Max (RK3588, 16 GB RAM, 3 NPU cores @ 1 GHz)
================================================================================

РЕЗЮМЕ
================================================================================

Исследована полная цепочка оптимизации llama.cpp с ggml-rknpu2 backend на RK3588.
Результат: RSS снижен с 13.5 GB до 3.0 GB, decode ускорен с 3.5 до 6.9 tok/s
(за счёт выбора оптимальной модели). NPU compute = 80% bottleneck для dense 8B,
дальнейшая оптимизация кода невозможна — ограничение аппаратное.

ЛУЧШИЕ РЕЗУЛЬТАТЫ:
  - Gemma-4 E2B Q8_0: 6.9 tok/s, качество 5/5, RSS 4.7 GB  ← ЛУЧШИЙ DENSE
  - Qwen2.5-Coder-3B Q8_0: 6.1 tok/s, качество 4/5, RSS 3.0 GB  ← ЛУЧШИЙ ПО RSS
  - LFM2.5-8B-A1B Q8_0: 10 tok/s, MoE, качество OK, RSS 13.3 GB  ← ЛУЧШИЙ ПО СКОРОСТИ
  - T-lite 8B Q8_0: 3.5 tok/s, качество 5/5, RSS 7.5 GB  ← BASELINE


1. ПЛАТФОРМА И ИСХОДНОЕ СОСТОЯНИЕ
================================================================================

1.1 Аппаратное обеспечение
  - Orange Pi 5 Max (RK3588)
  - 16 GB LPDDR4x RAM
  - 3 NPU cores @ 1 GHz (topology0/1/2)
  - NPU: matmul до 256×256 (K, N), M=1 для decode
  - CMA memory: 3 GB зарезервировано для NPU

1.2 Исходные модели
  - T-lite-it-2.1 Q4_K_M (4.7 GB) — НЕ использует NPU (тип не поддерживается)
  - T-lite-it-1.0 Q8_0 (8.1 GB) — использует NPU (W8A8 pipeline)

1.3 Исходная производительность
  - Decode Q8_0: ~3.5 tok/s (NPU W8A8)
  - Decode Q4_K_M: ~3.5-3.8 tok/s (CPU fallback, быстрее т.к. меньше данных)
  - RSS: 13.5 GB (GGUF mmap + RKNPU buffer дублирование)
  - rkllm W4A4 для сравнения: ~13 tok/s, но качество 53%


2. ОПТИМИЗАЦИИ КОДА (ЭТАП 1)
================================================================================

2.1 Шаг 1: Tiling Cache (v1→v5)
  Цель: устранить repeated computation segments/offsets (168 matmul/токен)

  v1 (ПОЛНЫЙ КЭШ) — СЛОМАН:
    - Кэшировал segments, pipeline, B-matrix offsets
    - Баг: segment_size_bytes != io_attr.B.size (RKNN padding)
    - Баг: offset accumulation через all_n_segments теряет неактивные позиции

  v2 (SEGMENTS ONLY) — ВАЛИДИРОВАН:
    - Кэширует: segments (K/N), pipeline config, matmul_type, use_hadamard
    - НЕ кэширует: B-matrix offsets
    - Decode: ~3.8 tok/s, вывод корректный

  v3 (B OFFSETS) — ВАЛИДИРОВАН:
    - b_offsets[k_idx][n_idx_all] — предвычисленные B-matrix offsets
    - Воспроизводит оригинальный offset accumulation logic
    - Decode: ~3.77 tok/s

  v4 (BMatrixInfo + ACTIVE_IDX) — ВАЛИДИРОВАН:
    - BMatrixInfo { offset_in_dma, active_idx }
    - O(active) прямой доступ вместо O(active x all) matching loop
    - Decode: ~3.79 tok/s

  v5 (C + c_bound) — ВАЛИДИРОВАН (ФИНАЛЬНЫЙ):
    - c_bound: skip set_io_mem C после первого вызова (аналог b_bound)
    - a_bound: КРАШИТ — RKNN требует set_io_mem A перед каждым mem_sync
    - Decode: ~3.78 tok/s, качество 5/5

  Вывод: tiling cache не даёт прироста скорости (bottleneck = NPU compute 80%),
  но устраняет nested matching loop и даёт стабильность.

2.2 Шаг 2: GGUF Page Cache Release (madvise)
  Цель: устранить дублирование GGUF mmap (7.2 GB) + RKNPU buffer (6.2 GB) = 13.5 GB

  Ключевое открытие:
    - data в set_tensor = GGUF mmap pointer (MAP_SHARED, PROT_READ)
    - tensor->data = RKNPU virtual buffer (MAP_PRIVATE | MAP_ANONYMOUS)
    - Предыдущая попытка madvise(tensor->data) была ОШИБКОЙ — освобождала нужные NPU страницы

  Реализация (~10 строк):
    После rknn_mem_sync(TO_DEVICE) для NPU-тензоров:
      madvise(data, aligned_size, MADV_DONTNEED) — освобождает GGUF страницы
    GGUF открыт PROT_READ — страницы clean, ядро освобождает без потери данных

  Результат:
    | Метрика       | До      | После   | Разница     |
    |---------------|---------|---------|-------------|
    | RSS           | 13.5 GB | 7.5 GB  | -6.0 GB (44%) |
    | Decode        | 3.5 tok/s | 3.5 tok/s | 0       |
    | Качество      | 5/5     | 5/5     | OK          |
    | GGUF released | 0       | 6.46 GB | +196 тензоров |

2.3 Шаг 2b: NPU Weight Cache (УЖЕ РЕАЛИЗОВАН в оригинале)
  - b_bound: skip rknn_create_mem_from_fd и set_io_mem B после первого вызова
  - mem_B shared_ptr кэшируется в rknpu_matmul_context
  - B_ctx = 0.3% от decode времени — НЕ bottleneck

2.4 Шаг 2c: C_sync оптимизация — НЕ ДАЁТ ПРИРОСТА
  Тест 1: skip mem_sync(FROM_DEVICE):
    - Качество OK (ARM cache-coherent)
    - Скорость УПАЛА: 3.07 vs 3.5 tok/s
    - Без cache invalidate CPU читает stale из L2, что медленнее

  Тест 2: параллельный C_sync (#pragma omp):
    - Без значимого улучшения (~3.5 tok/s)
    - 3 сегмента, overhead от thread creation > выигрыш

  Вывод: C_sync = необходимый cache invalidation, оптимизация невозможна

2.5 Профилирование decode (168 matmul/токен, T-lite 8B Q8_0)
  | Фаза         | Время     | Доля  |
  |--------------|-----------|-------|
  | B_ctx        | 0.5 ms    | 0.3%  |
  | A_prep       | 5-7 ms    | 3%    |
  | NPU_run      | 153-166 ms | 80%  |
  | C_sync       | 23-38 ms  | 17%   |

  NPU compute = 80% — аппаратное ограничение, не устранимо программно.


3. ИССЛЕДОВАНИЯ АРХИТЕКТУРНЫХ ОГРАНИЧЕНИЙ
================================================================================

3.1 Batch Matmul — НЕВОЗМОЖЕН для decode
  - RKNN matmul: C(M,N) = A(M,K) × B(K,N), M = batch dimension
  - Decode: M=1 (один токен за раз), невозможно увеличить
  - rknn_matmul_run — блокирующий вызов, нет async API
  - Software pipelining (overlap CPU/NPU) невозможен
  - Prefill уже работает с M>1 через next_power_of_two (49 tok/s)

3.2 W4A4 Matmul — ВЫЧЕРКНУТ
  - W4A4 = сжатие модели, не аппаратное ускорение
  - rkllm W4A4: 13 tok/s, но качество 53% — непригодно
  - Качественное W4A4 требует отдельного обучения (research на месяцы)
  - Альтернатива: модель 4B Q8_0 даёт тот же размер с полным качеством

3.3 Q4_K_M → W8A8 — ВЫЧЕРКНУТ
  - Не даст ускорения: W8A8 при M=1 = тот же bottleneck
  - NPU offload для Q4_K_M бессмысленен — CPU decode быстрее


4. СРАВНЕНИЕ МОДЕЛЕЙ НА RK3588
================================================================================

4.1 Методология
  Устройство: Orange Pi 5 Max (RK3588, 16 GB RAM)
  Backend: llama.cpp + ggml-rknpu2 (с madvise патчем)
  Контекст: 512 токенов
  Квантование: Q8_0 (кроме отмеченных)
  Качество: 5 вопросов (столица, скорость света, автор, элемент, гора)

4.2 Dense модели на NPU

  | Модель                | Параметры | Квант  | RSS   | NPU buf | Decode  | Prefill | Качество |
  |-----------------------|-----------|--------|-------|---------|---------|---------|----------|
  | T-lite-it-1.0 8B      | 8B dense  | Q8_0   | 7.5 GB| 6.2 GB  | 3.5 tok/s | 49 tok/s | 5/5  |
  | Qwen3.5-4B            | 4B dense  | Q8_0   | 4.3 GB| 4.0 GB  | 4.5 tok/s | 27 tok/s | 5/5  |
  | Qwen2.5-Coder-3B     | 3B dense  | Q8_0   | 3.0 GB| 2.9 GB  | 6.1 tok/s | 51 tok/s | 4/5  |
  | Gemma-4 E2B           | 4.6B dense| Q8_0   | 4.7 GB| 2.2 GB  | 6.9 tok/s | 11 tok/s | 5/5  |
  | Gemma-4 E4B           | 7.5B dense| Q8_0   | 7.6 GB| 4.4 GB  | 4.1 tok/s | —     | 5/5      |
  | Gemma-4 E4B           | 7.5B dense| Q4_0   | 5.8 GB| 2.8 GB  | 4.3 tok/s | —     | 2/5!!!   |

4.3 MoE модель (CPU-only)

  | Модель                | Параметры | Квант  | RSS    | NPU buf | Decode  | Качество  |
  |-----------------------|-----------|--------|--------|---------|---------|-----------|
  | LFM2.5-8B-A1B         | 8B MoE (1B active) | Q8_0 | 13.3 GB | 0.7 GB | 10 tok/s | OK+CoT |

4.4 Ключевые наблюдения

  1. Корреляция размера модели и скорости:
     - Меньше параметров → быстрее decode (обратная корреляция)
     - 3B: 6.1 tok/s, 4.6B: 6.9 tok/s, 7.5B: 4.1 tok/s, 8B: 3.5 tok/s
     - Gemma-4 E2B (4.6B) быстрее Qwen-Coder-3B (3B) — архитектурная разница

  2. NPU utilization:
     - Dense Q8_0: 100% весов на NPU (W8A8 pipeline)
     - MoE: только 8% весов на NPU (shared layers), эксперты на CPU
     - Q4_0: поддерживается через W4A4_HADAMARD, но качество ПЛОХОЕ

  3. Q4_0 деградация качества:
     - Gemma-4 E4B Q4_0: Element 79 = "Mougn" (???), скорость света = 770 m/s (???)
     - Q8_0 той же модели: Element 79 = Gold (Au), скорость света = 299,792,458 m/s
     - Вывод: Q4_0 на RK3588 непригоден для задач требующих точности

  4. Gemma-4 E2B/E4B — DENSE модели (НЕ MoE):
     - "E2B" = реально 4.6B параметров (size_label=4.6B, n_expert=0)
     - "E4B" = реально 7.5B параметров (size_label=7.5B, n_expert=0)
     - Названия E2B/E4B — маркетинговые Google ("E" = "efficient")

  5. Tool calling через system prompt:
     - Qwen3.5-4B и Qwen2.5-Coder-3B: корректный JSON при явном system prompt
     - OpenAI tools API не работает — chat template не обрабатывает tools параметр
     - Параллельный вызов функций работает (JSON массив)
     - Multi-turn (call → response → answer) работает

  6. Чат с reasoning моделями:
     - Qwen3.5-4B: требует enable_thinking=false, иначе reasoning_content съедает output
     - LFM2.5-8B-A1B: CoT в reasoning_content, content пустой без настройки
     - Gemma-4: требует enable_thinking=false


5. ТЕХНИЧЕСКИЕ ОТКРЫТИЯ
================================================================================

5.1 RKNN API семантика
  - rknn_matmul_set_io_mem A: ОБЯЗАТЕЛЕН перед каждым mem_sync TO_DEVICE
    (a_bound крашит сервер — RKNN использует set_io_mem как триггер DMA)
  - rknn_matmul_set_io_mem B: можно skip после первого вызова (b_bound)
  - rknn_matmul_set_io_mem C: можно skip после первого вызова (c_bound)
  - rknn_mem_sync(TO_DEVICE): DMA transfer данных в NPU
  - rknn_mem_sync(FROM_DEVICE): cache invalidate для CPU чтения результата
  - rknn_matmul_run: БЛОКИРУЮЩИЙ вызов, нет async API

5.2 RKNN matmul tiling
  - RK3588 NPU: max K=256, max N=256 per core, M=1 для decode
  - Большие матрицы разбиваются на тайлы (segments)
  - segment_size из io_attr.B.size (RKNN ground truth), не математический расчёт
  - Активные N-сегменты: size_n > 0, неактивные: size_n = 0 (padding)

5.3 Память NPU
  - RKNPU buffer: MAP_PRIVATE | MAP_ANONYMOUS (virtual, 6.2 GB для 8B)
  - RKNPU DMA: через IOMMU domains (физическая CMA память, 3 GB)
  - rknn_create_mem_from_fd: DMA buffer для B-matrix весов (кэшируется через b_bound)
  - C-matrix: c_buffer_cache (shared_ptr, переиспользуется между matmul)
  - A-matrix: a_buffer_cache (shared_ptr, переиспользуется между matmul)

5.4 GGUF page release (madvise)
  - data в set_tensor = GGUF mmap pointer (MAP_SHARED, PROT_READ)
  - tensor->data = RKNPU virtual buffer (MAP_PRIVATE | MAP_ANONYMOUS)
  - madvise(data, MADV_DONTNEED) после mem_sync — освобождает GGUF страницы
  - madvise(tensor->data) — ОШИБКА, освобождает нужные NPU страницы
  - Результат: RSS 13.5 → 7.5 GB (экономия 6 GB)

5.5 Q4_K_M vs Q8_0 на RK3588
  | Параметр       | Q4_K_M       | Q8_0           |
  |----------------|--------------|----------------|
  | NPU support    | НЕТ          | ДА (W8A8)      |
  | NPU load       | 0%           | 100%           |
  | Decode         | 3.5-3.8 tok/s| 3.3-3.6 tok/s  |
  | RSS (madvise)  | ~5 GB        | ~7.5 GB        |
  | GGUF размер    | 4.7 GB       | 8.1 GB         |

  Q4_K_M быстрее потому что CPU fallback быстрее CPU Q8_0 decode (меньше данных)


6. ФАЙЛЫ И РЕПОЗИТОРИЙ
================================================================================

Репозиторий: /home/sa/projects/rk3588-npu-research/

tiling-cache/
  +-- tiling_cache.h           -- v2 (валидированная, segments only)
  +-- tiling_cache.cpp          -- v2 implementation
  +-- ggml-rknpu2_patched.cpp  -- v2 patched
  +-- RESULTS.md               -- бенчмарки и выводы
  +-- variant-b/               -- v3 (B-offset flat table)
  +-- variant-c/               -- v4 (BMatrixInfo + active_idx)
  +-- variant-c-cbound/        -- v5 (C + c_bound skip) — ФИНАЛЬНЫЙ
  +-- variant-c-cbound-abound-crash/  -- v6 (a_bound краш)

weight-cache/
  +-- ggml-rknpu2_patched.cpp  -- madvise патч (ФИНАЛЬНЫЙ, без debug логов)

OPi5Max /opt/rk-llama.cpp/:
  - ggml/src/ggml-rknpu2/ggml-rknpu2.cpp — v6 (C + c_bound + madvise на data)
  - ggml/src/ggml-rknpu2/tiling_cache.h/cpp — v5 (C + c_bound)
  - build/bin/llama-server — собран с v6

Модели на OPi5Max /root/models/:
  - t-lite-it-1.0-q8_0.gguf (8.1 GB) — baseline
  - LFM2.5-8B-A1B-Q8_0.gguf (8.4 GB) — MoE
  - Qwen3.5-4B-Q8_0.gguf (4.2 GB)
  - Qwen2.5-Coder-3B-Instruct-Q8_0.gguf (3.1 GB)
  - gemma-4-E2B-it-Q8_0.gguf (4.8 GB)
  - gemma-4-E4B-it-Q8_0.gguf (7.7 GB)
  - gemma-4-E4B-it-Q4_0.gguf (4.6 GB)


7. ВЫВОДЫ И РЕКОМЕНДАЦИИ
================================================================================

7.1 Оптимизация кода — исчерпана
  Все программные оптимизации для dense 8B Q8_0 на RK3588 исчерпаны:
  - NPU compute = 80% времени (аппаратное ограничение)
  - C_sync = 17% (cache invalidation, убрать нельзя)
  - A_prep = 3% (минимален)
  - B_ctx = 0.3% (уже кэшируется)
  - Batch matmul невозможен (M=1 для decode, rknn_matmul_run блокирующий)
  - Software pipelining невозможен (нет async API)

7.2 Лучший выбор модели для RK3588

  По скорости (decode):
    1. LFM2.5-8B-A1B Q8_0: 10 tok/s (MoE, CPU-only, RSS 13.3 GB)
    2. Gemma-4 E2B Q8_0: 6.9 tok/s (dense, NPU, RSS 4.7 GB, качество 5/5)
    3. Qwen2.5-Coder-3B Q8_0: 6.1 tok/s (dense, NPU, RSS 3.0 GB, качество 4/5)

  По качеству:
    1. Gemma-4 E2B Q8_0: 5/5, 6.9 tok/s
    2. Qwen3.5-4B Q8_0: 5/5, 4.5 tok/s
    3. T-lite 8B Q8_0: 5/5, 3.5 tok/s

  По RSS:
    1. Qwen2.5-Coder-3B Q8_0: 3.0 GB, 6.1 tok/s
    2. Qwen3.5-4B Q8_0: 4.3 GB, 4.5 tok/s
    3. Gemma-4 E2B Q8_0: 4.7 GB, 6.9 tok/s

  Рекомендация: Gemma-4 E2B Q8_0 — лучший баланс скорость/качество/RSS

7.3 Дальнейшие направления
  - Меньшая модель (1.5B-2B): ещё быстрее, но хуже качество
  - Custom fused ops на NPU: LayerNorm+matmul (требует RKNN SDK доступа)
  - Speculative decoding: маленькая модель + верификация большой
  - MoE архитектуры: LFM2.5-8B-A1B даёт 10 tok/s на CPU
  - Q4_0: поддерживается NPU, но сильная деградация качества

7.4 Не рекомендуется
  - Q4_K_M: не поддерживается NPU (0% offload)
  - Q4_0: сильная деградация качества (галлюцинации)
  - W4A4 matmul: качество 53%, непригодно для использования
  - Batch matmul: невозможно для decode (M=1, блокирующий API)