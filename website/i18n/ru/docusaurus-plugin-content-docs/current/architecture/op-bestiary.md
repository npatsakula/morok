---
sidebar_label: Бестиарий операций
---

# Бестиарий операций: справочник по UOp

При отладке IR-дампов Morok вы встретите операции, назначение которых не очевидно из названия. Эта глава документирует нетривиальные операции с сигнатурами, описанием полей и примерами.

**Что покрыто:** Операции, требующие пояснений — управление циклами, редукции, работа с памятью, структура ядер, векторизация, тензорные ядра.

**Что НЕ покрыто:** Тривиальные ALU-операции (`Add`, `Mul`, `Sqrt` и т.д.), которые работают ровно так, как вы ожидаете.

---

## Управление циклами: RANGE и END

### RANGE — открытие скоупа цикла

```rust
Range {
    end: Arc<UOp>,           // loop bound (exclusive)
    axis_id: AxisId,         // identifier for deduplication
    axis_type: AxisType,     // scheduling behavior
    deps: SmallVec<[Arc<UOp>; 2]>,  // range dependencies
}
```

**Поля:**

| Поле | Тип | Назначение |
|------|-----|------------|
| `end` | `Arc<UOp>` | Верхняя граница (исключительно), обычно `CONST` |
| `axis_id` | `AxisId` | `Unrenumbered(n)` до разделения ядер, `Renumbered(n)` после |
| `axis_type` | `AxisType` | Определяет способ планирования цикла (см. ниже) |
| `deps` | `SmallVec<[Arc<UOp>; 2]>` | Другие RANGE, от которых зависит этот |

**Иерархия AxisType:**

| Тип | Приоритет | GPU-маппинг | Назначение |
|-----|-----------|-------------|------------|
| `Outer` | -2 | — | Маркер границы ядра |
| `Loop` | -1 | цикл `for` | Последовательная итерация |
| `Global` | 0 | `blockIdx` | Параллелизм по гриду |
| `Thread` | 0 | пул потоков | CPU-параллелизм |
| `Warp` | 1 | warp/wavefront | Субгрупповой параллелизм |
| `Local` | 2 | `threadIdx` | Параллелизм внутри воркгруппы |
| `GroupReduce` | 2 | shared memory | Двухэтапная редукция |
| `Upcast` | 3 | SIMD | Векторизация |
| `Reduce` | 4 | аккумулятор | Ось редукции |
| `Unroll` | 5 | развёрнутый | Развёртка цикла |

Приоритет определяет порядок вложенности — меньшие значения соответствуют внешним циклам.

**Пример:**
```text
RANGE(end=128, axis_id=R0, type=Global)
└── CONST(128) : Index
```

### END — закрытие скоупа цикла

```rust
End {
    computation: Arc<UOp>,              // value computed inside loop
    ranges: SmallVec<[Arc<UOp>; 4]>,    // ranges being closed
}
```

END закрывает один или несколько скоупов RANGE и убирает их из активного набора. Можно закрыть несколько RANGE одновременно.

**Пример:**
```text
END
├── STORE(...)           — computation
├── RANGE(R0, Global)    — first range closed
└── RANGE(R1, Local)     — second range closed
```

---

## Редукция: REDUCE vs REDUCE_AXIS

Две операции с похожими названиями, но с разным назначением.

### REDUCE_AXIS — редукция по измерению тензора (высокоуровневая)

```rust
ReduceAxis {
    src: Arc<UOp>,           // input tensor
    reduce_op: ReduceOp,     // Add, Mul, Max, Min
    axes: Vec<usize>,        // axes to reduce
}
```

Используется **до** rangeify. Работает по измерениям тензора, как `.sum(axis=0)` в NumPy.

**Пример:**
```text
REDUCE_AXIS(Add, axes=[1])
└── BUFFER[10, 20] : Float32
```

Редуцирует тензор `[10, 20]` до `[10]`, суммируя по оси 1.

### REDUCE — редукция по итерациям RANGE (низкоуровневая)

```rust
Reduce {
    src: Arc<UOp>,                      // value to accumulate
    ranges: SmallVec<[Arc<UOp>; 4]>,    // ranges being reduced
    reduce_op: ReduceOp,                // Add, Mul, Max, Min
}
```

Используется **после** rangeify. Аккумулирует значения по итерациям RANGE и закрывает указанные RANGE.

**Варианты ReduceOp:**

| Op | Нейтральный элемент | Операция | Tinygrad |
|----|---------------------|----------|----------|
| `Add` | 0 | `acc + value` | ✓ |
| `Mul` | 1 | `acc * value` | ✓ |
| `Max` | -∞ | `max(acc, value)` | ✓ |
| `Min` | +∞ | `min(acc, value)` | Только Morok |

> **Совместимость:** Спецификация Tinygrad ограничивает REDUCE_AXIS до `{Add, Mul, Max}`. Morok расширяет это добавлением `Min`.

**Пример:**
```text
REDUCE(Add)
├── MUL                      — value to accumulate
│   ├── LOAD(A, ...)
│   └── LOAD(B, ...)
└── RANGE(R2, Reduce)        — range being reduced
    └── CONST(64)
```

### ALLREDUCE — редукция между устройствами

```rust
AllReduce {
    src: Arc<UOp>,           // local partial result
    device: Arc<UOp>,        // device specification
    reduce_op: ReduceOp,     // reduction operation
}
```

Выполняет распределённую редукцию между несколькими устройствами. Используется для мульти-GPU обучения.

---

## Операции с буферами

### BUFFER — объявление буфера

```rust
Buffer {
    unique: Arc<UOp>,        // UNIQUE op for identity
    device: Arc<UOp>,        // DEVICE op
    size: usize,             // total element count
}
```

Объявляет буфер для хранения данных тензора. Поле `unique` гарантирует различимость буферов даже при одинаковых размере и устройстве.

### BUFFERIZE — маркер материализации

```rust
Bufferize {
    compute: Arc<UOp>,                  // computation to materialize
    ranges: SmallVec<[Arc<UOp>; 4]>,    // output dimensions
    opts: BufferizeOpts,                // address space, device
}
```

Отмечает место, где вычисление должно материализоваться в память. Триггерит разделение на ядра.

**BufferizeOpts:**

| Поле | Тип | Назначение |
|------|-----|------------|
| `device` | `Option<DeviceSpec>` | Целевое устройство, `None` для локального |
| `addrspace` | `AddrSpace` | `Global` (устройство) или `Local` (shared) |

**Пример:**
```text
BUFFERIZE(opts={addrspace=Global})
├── REDUCE(Add, ...)         — computation
├── RANGE(R0, Global)        — output dim 0
└── RANGE(R1, Global)        — output dim 1
```

### INDEX — многомерный доступ к буферу

```rust
Index {
    buffer: Arc<UOp>,                   // BUFFER or PARAM
    indices: SmallVec<[Arc<UOp>; 4]>,   // index per dimension
    gate: Option<Arc<UOp>>,             // optional predicate
}
```

Вычисляет адрес в памяти из многомерных индексов. Возвращает DType элемента (не указатель).

**Пример:**
```text
INDEX : Float32
├── PARAM(0)
├── RANGE(R0, Global)        — index for dim 0
├── RANGE(R1, Loop)          — index for dim 1
└── MUL(...)                 — index for dim 2
```

### POINTER_INDEX — низкоуровневая арифметика указателей

```rust
PointerIndex {
    ptr: Arc<UOp>,           // base pointer
    offset: Arc<UOp>,        // byte offset
}
```

Прямая арифметика указателей. Используется после линеаризации, когда индексы уже свёрнуты в линейные.

> **Совместимость:** Tinygrad использует `INDEX` с флагом `ptr=True` вместо отдельной операции.

### LOAD — чтение из памяти

```rust
Load {
    buffer: Arc<UOp>,        // buffer or pointer
    index: Arc<UOp>,         // INDEX op
    alt: Option<Arc<UOp>>,   // alternative value for gated loads
}
```

Читает значение из буфера по индексу. Для gated loads поле `alt` задаёт значение при ложном условии INDEX-а (позволяет полностью избежать обращения к памяти).

**Пример:**
```text
LOAD : Float32
├── PARAM(1)
└── INDEX
    ├── PARAM(1)
    ├── RANGE(R0)
    └── RANGE(R2)
```

### STORE — запись в память

```rust
Store {
    index: Arc<UOp>,                    // INDEX op (buffer accessed via index.src[0])
    value: Arc<UOp>,                    // value to write
    ranges: SmallVec<[Arc<UOp>; 4]>,    // ranges being closed
}
```

Записывает значение в буфер. Буфер доступен через INDEX-узел (через `index.src[0]`), а не через отдельное поле. STORE закрывает указанные RANGE, которые представляют выходные измерения итерации. Поле ranges используется для output upcasting: когда включён `Range(Upcast)`, он становится `UNROLL` при расширении, а затем сжимается через `CONTRACT`.

Для условной записи используйте INDEX с gate (у INDEX есть опциональное поле `gate`).

> **Совместимость:** У STORE в Morok нет отдельного поля `buffer` — источники: index=0, value=1, ranges=2+ (range_start=2). Устройство Tinygrad аналогично.

**Пример:**
```text
STORE
├── INDEX[R0, R1]            — write address (buffer via index.src[0])
├── REDUCE(Add, ...)         — value
├── RANGE(R0, Global)        — output dim 0 (closed)
└── RANGE(R1, Global)        — output dim 1 (closed)
```

---

## Структура ядра

### KERNEL — обёртка ядра

```rust
Kernel {
    sources: SmallVec<[Arc<UOp>; 4]>,   // arguments
    ast: Arc<UOp>,                       // computation (usually SINK)
}
```

Оборачивает готовое ядро для кодогенерации. Источники — аргументы ядра (`Param`, `DefineLocal`, `DefineVar`). Замечание: `Param` заменил `DefineGlobal` в PR batching_support для дедупликации ядер путём стирания идентичности буферов.

**Пример:**
```text
KERNEL
├── PARAM(slot=0, size=1024) — output buffer arg
├── PARAM(slot=1, size=1024) — input A arg
├── PARAM(slot=2, size=1024) — input B arg
└── SINK                     — computation
    └── STORE(...)
```

### SINK — коллектор нескольких корней

```rust
Sink {
    sources: SmallVec<[Arc<UOp>; 4]>,
}
```

Собирает несколько выходов в один корень. `ast` каждого ядра — как правило, SINK, содержащий операции STORE.

**Пример:**
```text
SINK
├── STORE(output_0, ...)
├── STORE(output_1, ...)
└── STORE(output_2, ...)
```

### AFTER — маркер зависимости

```rust
After {
    passthrough: Arc<UOp>,              // value that flows through
    deps: SmallVec<[Arc<UOp>; 4]>,      // operations that must complete
}
```

Выражает зависимости выполнения между ядрами без data dependency. Значение `passthrough` возвращается без изменений, но только после завершения всех `deps`.

**Пример:**
```text
SINK
├── AFTER
│   ├── PARAM(0)     — passthrough (buffer reference)
│   └── KERNEL(...)          — must complete first
└── KERNEL(...)              — can use buffer after AFTER
```

### BARRIER — барьер синхронизации

```rust
Barrier {
    src: Arc<UOp>,                      // value passing through
    deps: SmallVec<[Arc<UOp>; 4]>,      // operations to wait for
}
```

Синхронизация GPU-воркгруппы. Гарантирует, что все потоки в воркгруппе достигли барьера, прежде чем продолжить.

---

## Векторные операции

### VECTORIZE — создание вектора из скаляров

```rust
Vectorize {
    elements: SmallVec<[Arc<UOp>; 4]>,
}
```

Комбинирует N скалярных значений в вектор размера N. Все элементы должны иметь одинаковый базовый DType.

**Пример:**
```text
VECTORIZE : <4 x Float32>
├── CONST(1.0)
├── CONST(2.0)
├── CONST(3.0)
└── CONST(4.0)
```

### GEP — Get Element Pointer (извлечение из вектора)

```rust
Gep {
    vector: Arc<UOp>,        // source vector
    indices: Vec<usize>,     // positions to extract
}
```

Извлекает элементы из вектора:
- Один индекс — скаляр
- Несколько индексов — вектор меньшего размера

**Пример:**
```text
GEP([0, 2]) : <2 x Float32>
└── VECTORIZE : <4 x Float32>
    └── ...
```

### VConst — векторная константа

```rust
VConst {
    values: Vec<ConstValue>,
}
```

Вектор констант, известных на этапе компиляции. Эффективнее, чем `VECTORIZE` из `CONST`-узлов.

### CAT — конкатенация векторов

```rust
Cat {
    sources: SmallVec<[Arc<UOp>; 4]>,
}
```

Конкатенирует векторы в вектор большего размера. `vcount` на выходе = сумма `vcount` входов.

**Пример:**
```text
CAT : <8 x Float32>
├── VECTORIZE : <4 x Float32>
└── VECTORIZE : <4 x Float32>
```

### PtrCat — конкатенация указателей

```rust
PtrCat {
    sources: SmallVec<[Arc<UOp>; 4]>,
}
```

Группирует обращения к памяти для векторизованного load/store. Используется проходом devectorizer.

---

## Расширение: UNROLL и CONTRACT

### UNROLL — расширение вычисления по итерациям

```rust
Unroll {
    src: Arc<UOp>,                       // computation to expand
    unroll_axes: Vec<(usize, usize)>,    // (axis_index, factor) pairs
}
```

Создаёт несколько версий вычисления для разных значений итерации. Используется для оптимизации развёртки циклов.

**Пример:** `UNROLL(unroll_axes=[(0, 4)])` расширяет вычисление 4 раза с разными значениями индекса.

### CONTRACT — свёртка развёрнутых значений в вектор

```rust
Contract {
    src: Arc<UOp>,                       // unrolled computation
    upcast_ranges: Vec<(usize, usize)>,  // (axis_index, factor) pairs
}
```

Обратная операция к UNROLL — собирает расширенные скалярные значения в вектор. Размер выходного вектора = произведение множителей.

**Пример:**
```text
CONTRACT(upcast_ranges=[(0, 4)]) : <4 x Float32>
└── UNROLL(unroll_axes=[(0, 4)])
    └── LOAD(...)
```

Этот паттерн векторизует загрузку: расширить 4 итерации, затем упаковать результаты в 4-элементный вектор.

---

## Тензорные ядра: WMMA

### WMMA — Warp Matrix Multiply-Accumulate

```rust
Wmma {
    a: Arc<UOp>,             // matrix A fragment
    b: Arc<UOp>,             // matrix B fragment
    c: Arc<UOp>,             // accumulator C fragment
    metadata: WmmaMetadata,  // hardware configuration
}
```

Аппаратная операция тензорного ядра: `D = A * B + C`. Требует конкретных форм матриц и раскладок данных.

**Поля WmmaMetadata:**

| Поле | Тип | Назначение |
|------|-----|------------|
| `name` | `String` | Имя инструкции (например, `"__hmma..."`) |
| `dims` | `(N, M, K)` | Размерности матриц (например, `(16, 16, 16)`) |
| `dtype_in` | `DType` | Точность входных матриц (например, `Float16`) |
| `dtype_out` | `DType` | Точность выхода (например, `Float32`) |
| `device` | `String` | Строка целевого устройства |
| `threads` | `usize` | Потоков на warp (обычно 32) |
| `upcast_axes` | `WmmaUpcastAxes` | Векторизация для каждого операнда (поля: `a`, `b`, `c`) |
| `reduce_axes` | `Vec<(usize, usize)>` | Оси свёртки |
| `tile_grid` | `(usize, usize)` | Грид для мульти-FMA батчинга (по умолчанию (1,1)) |

**Пример:**
```text
WMMA(dims=(16, 16, 16), dtype_in=Float16, dtype_out=Float32)
├── A fragment : <8 x Float16>
├── B fragment : <8 x Float16>
└── C accumulator : <8 x Float32>
```

---

## Управление потоком выполнения

### IF / ENDIF — условное выполнение

```rust
If {
    condition: Arc<UOp>,                // boolean predicate
    body: SmallVec<[Arc<UOp>; 4]>,      // operations to execute
}

EndIf {
    if_op: Arc<UOp>,         // corresponding IF op
}
```

Выполнить тело, только если условие истинно. Используется для проверок границ и разреженных операций.

**Пример:**
```text
IF
├── LT(idx, bound)           — condition (src[0])
├── STORE(...)               — body[0]
└── STORE(...)               — body[1]

ENDIF
└── IF(...)                  — references IF op
```

---

## Операции определения

### PARAM — параметр буфера

```rust
Param { slot: usize, size: usize, device: Option<Arc<UOp>> }
```

Нормализованный параметр буфера — позиционная ссылка на входной/выходной буфер.
Создаётся при предварительной нормализации расписания (BUFFER->PARAM) для стирания идентичности буфера,
что позволяет структурную дедупликацию идентичных вычислений на разных буферах.
`slot` — позиция в списке аргументов ядра, `size` — количество элементов.

### DEFINE_LOCAL — аллокация shared-памяти

```rust
DefineLocal(usize)           // local memory index
```

Аллокация GPU shared memory (LDS). Видна внутри воркгруппы.

### DEFINE_VAR — символическая рантайм-переменная

```rust
DefineVar {
    name: String,            // variable name
    min_val: i64,            // minimum bound
    max_val: i64,            // maximum bound
}
```

Рантайм-переменная с известными границами. Используется для динамических форм, когда границы известны.

**Пример:**
```text
DEFINE_VAR(name="batch_size", min=1, max=128) : Index
```

### DEFINE_REG — аллокация регистра

```rust
DefineReg {
    size: usize,             // register size
    id: usize,               // unique accumulator ID
}
```

Аллоцирует регистр для промежуточного хранения. Поле `id` различает регистры одного DType — без него два reduce с одинаковым DType разделили бы один DEFINE_REG через hash consing. Используется в кодогенерации.

### BIND — привязка переменной

```rust
Bind {
    var: Arc<UOp>,           // DEFINE_VAR
    value: Arc<UOp>,         // concrete value
}
```

Привязывает символическую переменную к конкретному значению в рантайме.

---

## Специальные операции

### SPECIAL — аппаратные значения

```rust
Special {
    end: Arc<UOp>,           // upper bound for this dimension
    name: String,            // e.g., "blockIdx.x", "threadIdx.y"
}
```

Доступ к значениям, предоставляемым аппаратурой (индексы потоков/блоков). Это не цикл — значение даётся напрямую оборудованием.

**Пример:**
```text
SPECIAL(name="blockIdx.x", end=128) : Index
└── CONST(128)
```

### UNIQUE — маркер идентичности

```rust
Unique(usize)                // unique identifier
```

Создаёт уникальную идентичность для различения буферов. Два буфера с разными UNIQUE-значениями различимы, даже если в остальном идентичны.

### DEVICE — спецификация устройства

```rust
Device(DeviceSpec)           // device specification
```

Указывает целевое устройство для вычисления.

---

## Операции перемещения (Movement)

Высокоуровневые трансформации формы тензора. Преобразуются в явные INDEX-операции во время rangeify.

| Операция | Сигнатура | Назначение |
|----------|-----------|------------|
| `Reshape` | `{ src, new_shape }` | Изменить форму, те же элементы |
| `Permute` | `{ src, axes: Vec<usize> }` | Транспонирование / перестановка осей |
| `Expand` | `{ src, new_shape }` | Бродкаст до большей формы |
| `Pad` | `{ src, begin_pads, end_pads }` | Добавить паддинг |
| `Shrink` | `{ src, begins, ends }` | Извлечь подобласть |
| `Flip` | `{ src, axes: Vec<bool> }` | Развернуть по осям |

**Пример:** RESHAPE
```text
RESHAPE(new_shape=[6, 4]) : Shape[6, 4]
├── BUFFER[2, 3, 4] : Float32
└── CONST([6, 4]) : Shape
```

---

## Дополнительные операции

Следующие операции существуют в enum `Op`, но являются внутренними или редко встречаются при отладке:

| Операция | Назначение |
|----------|------------|
| `Copy` | Явное копирование значения |
| `BufferView` | View в существующий буфер со смещением/страйдом |
| `MStack` | Аллокация стека в памяти |
| `MSelect` | Выбор в памяти (условный доступ) |
| `Multi` | Операция с множественными выходами |
| `Assign` | Присваивание переменной |
| `Group` | Группировка операций для планирования |
| `Detach` | Отсоединение от графа (запрет оптимизации через узел) |
| `Contiguous` | Хинт, что данные непрерывны |
| `ContiguousBackward` | Обратный проход для хинта contiguous |
| `Precast` | Предварительное приведение типа |
| `Custom` / `CustomI` | Расширяемость пользовательскими операциями |

---

## Краткий справочник

### По категориям

| Категория | Операции |
|-----------|----------|
| **Управление циклами** | `RANGE`, `END` |
| **Редукция** | `REDUCE_AXIS`, `REDUCE`, `ALLREDUCE` |
| **Память** | `BUFFER`, `BUFFERIZE`, `INDEX`, `POINTER_INDEX`, `LOAD`, `STORE` |
| **Ядро** | `KERNEL`, `SINK`, `AFTER`, `BARRIER` |
| **Векторные** | `VECTORIZE`, `GEP`, `VCONST`, `CAT`, `PTRCAT` |
| **Расширение** | `UNROLL`, `CONTRACT` |
| **Аппаратные** | `WMMA`, `SPECIAL` |
| **Управление** | `IF`, `ENDIF` |
| **Определение** | `PARAM`, `DEFINE_LOCAL`, `DEFINE_VAR`, `DEFINE_REG`, `BIND`, `UNIQUE`, `DEVICE` |
| **Перемещение** | `RESHAPE`, `PERMUTE`, `EXPAND`, `PAD`, `SHRINK`, `FLIP` |
| **ALU** | `Unary(...)`, `Binary(...)`, `Ternary(...)`, `Cast`, `BitCast` |

### Операции, закрывающие RANGE

Операции, которые закрывают скоупы RANGE (убирают RANGE из активного набора):

| Операция | Начальный индекс RANGE |
|----------|------------------------|
| `BUFFERIZE` | 1 (compute=0, ranges=1+) |
| `REDUCE` | 1 (src=0, ranges=1+) |
| `STORE` | 2 (index=0, value=1, ranges=2+) |
| `WMMA` | 3 (a=0, b=1, c=2) |
| `END` | 1 (computation=0, ranges=1+) |

### Расширяемые операции

Операции, через которые UNROLL пропагируется по графу вычислений:

- ALU: `Unary`, `Binary`, `Ternary`
- Типы: `Cast`, `BitCast`
- Векторные: `Gep`, `Vectorize`
- Память: `Load`, `Store`, `Index`, `PointerIndex`
- Управление: `Reduce`, `End`, `After`
- Буферы: `Bufferize`
- Аппаратные: `Wmma`
