---
sidebar_label: Поиск по ядрам
---

# Поиск оптимизации ядер

После алгебраического упрощения каждому ядру нужны *решения по планированию*: как тайлить циклы, где параллелизовать, использовать ли tensor cores. Svod предлагает две стратегии: быстрые эвристики и тщательный beam search.

Выполняется на Стадии 7 [пайплайна кодогенерации](../codegen/overview.md).

Исходники Tinygrad: `tinygrad/codegen/opt/`. Исходники Svod: `schedule/src/optimizer/`.

---

## Пространство действий

Оптимизационные преобразования модифицируют структуру циклов, меняя типы осей. Каждое действие изменяет один диапазон:

| Действие | Эффект | Целевое оборудование |
|----------|--------|---------------------|
| UPCAST(axis, amount) | Векторизация размерности (SIMD) | Все |
| UNROLL(axis, amount) | Развёртка размерности цикла | Все |
| LOCAL(axis, amount) | Использование GPU shared memory | GPU (LDS) / CPU (L1) |
| GROUP(axis, amount) | Двухстадийная редукция | Все |
| GROUPTOP(axis, amount) | Grouped reduction для tensor cores | GPU |
| THREAD(axis, amount) | Параллелизация на основе потоков CPU | CPU |
| SWAP(axis1, axis2) | Перестановка глобальных размерностей | Все |
| PADTO(axis, amount) | Паддинг для выравнивания | Все |
| NOLOCALS | Отключение локальной памяти | Все (ограничение) |
| TC | Включение использования tensor cores | GPU NVIDIA, AMD, Metal, Intel (WMMA/MFMA) |

В `BEAM_ACTIONS` 193 базовых действия (200 при `BEAM_PADTO=1`); сколько из них переживёт фильтрацию для конкретного ядра, зависит от его структуры и доступного параллелизма. NOLOCALS в этот список не входит — `generate_actions` добавляет его, только когда задан `NOLOCALS`/`SVOD_NOLOCALS`.

---

## Эвристики (по умолчанию)

Эвристический оптимизатор применяет оптимизации в фиксированном порядке (упрощённый псевдокод):

```rust
// Pseudocode — simplified from optimizer/heuristics.rs
fn hand_coded_optimizations(scheduler: &mut Scheduler) {
    // 1. Tensor cores (if matmul pattern detected)
    if let Some(tc) = detect_tensor_core_pattern(scheduler) {
        apply_tensor_core(scheduler, tc);
        return;  // TC handles everything
    }

    // 2. Grouped reductions (two-stage for large reductions)
    apply_grouped_reduction_if_needed(scheduler);

    // 3. Vectorization (UPCAST output dimensions)
    apply_upcast(scheduler, 4);

    // 4. GPU local memory (workgroup dimensions)
    apply_local_dims(scheduler);

    // 5. CPU threading
    apply_threading(scheduler);
}
```

**Плюсы**: Быстро (~50ms на ядро), предсказуемо, не требует аппаратных замеров.

**Минусы**: Может упустить возможности оптимизации, фиксированные эвристики не адаптируются к нагрузке.

---

## Beam search (опционально)

Для продакшн-нагрузок beam search находит лучшие расписания, компилируя и замеряя кандидатов (упрощённый псевдокод):

```rust
// Pseudocode — simplified from optimizer/beam.rs
// Actual API: beam_search_cached_remote(scheduler, config, compiler_identity,
//                                       behavior_fingerprint, compile_wave, benchmark)
fn beam_search(scheduler: Scheduler, config: &BeamConfig) -> Scheduler {
    let mut beam = vec![(scheduler, Duration::MAX)];

    loop {
        // EXPAND: every applicable action on every beam member
        let candidates: Vec<Scheduler> = beam.iter()
            .flat_map(|(state, _)| generate_actions(state))
            .collect();

        // COMPILE in helper worker processes, then time config.num_runs each
        let mut timed = vec![];
        for (candidate, compiled) in compile_wave(&candidates) {
            if !seen_binary.insert(compiled.binary_key) { continue; }  // identical code
            if bloated(&mut least_compute_ops, compiled.compute_ops) { continue; }
            timed.push((candidate, benchmark(&compiled)));
        }

        // Keep top K by execution time
        timed.sort_by_key(|(_, time)| *time);
        timed.truncate(config.beam_width);

        // Stop when the best candidate no longer improves by min_progress_ns
        if best(&beam) - best(&timed) < config.min_progress_ns { break; }
        beam = timed;
    }

    beam.into_iter().next().unwrap().0
}
```

**Плюсы**: Находит близкие к оптимальным расписания, адаптируется к железу.

**Минусы**: Минуты на ядро (но результаты кэшируются по хэшу AST).

---

## Конфигурация

```bash
# Disable optimization (debugging)
SVOD_NOOPT=1 cargo run

# Enable beam search with width 8
BEAM=8 cargo run
```

Или программно:

```rust
let config = PrepareConfig::from(
    OptimizerConfig::builder()
        .strategy(OptStrategy::Beam { width: 8 })
        .build()
);

tensor.realize_with(&config)?;
```

---

## Сравнение: как оптимизируют другие компиляторы

| Аспект | XLA | TVM/Ansor | Triton | **Svod** |
|--------|-----|-----------|--------|-----------|
| **Философия** | Фиксированные эвристики | Поиск | Управление программистом | На основе паттернов |
| **Фьюзинг** | Консервативные правила | Tile-and-fuse | Block-level | Перезапись графа |
| **Автотюнинг** | Нет | Эволюционный + cost model | Grid search | Beam search |
| **Стоимость тюнинга** | 0 | Часы | Минуты | Минуты (кэшируется) |
| **Гибкость** | Низкая | Высокая | Средняя | Высокая |
| **Прозрачность** | Низкая (C++-проходы) | Средняя (Python) | Средняя (DSL) | Высокая (декларативные паттерны) |

**XLA** использует фиксированные эвристики для решений по фьюзингу. Безопасно и предсказуемо, но оставляет производительность на столе. Правила фьюзинга захардкожены в C++.

**TVM/Ansor** разделяет *что* вычислять и *как* вычислять. Ansor использует эволюционный поиск с обучаемой cost model. Лучшая в классе производительность, но тюнинг занимает часы на модель.

**Triton** предоставляет Python-подобный DSL для блочных алгоритмов. Хороший баланс контроля и автоматизации, но требует экспертизы в GPU-программировании.

**Svod** выражает оптимизации как компонуемые паттерны. Beam search добавляет автотюнинг при необходимости, с кэшированием результатов по хэшу AST для повторного использования.
