---
sidebar_label: ONNX-инференс
---

# Инференс ONNX-моделей

ONNX-импортёр Morok — рекомендуемый способ инференса моделей. Он загружает стандартные `.onnx`-файлы, раскладывает операторы на ленивые тензорные операции Morok и компилирует их через полный пайплайн оптимизаций — без C++ рантайма.

**Текущий статус:**

| Возможность | Статус |
|-------------|--------|
| Прямой инференс | Поддерживается |
| 162 / 200 операторов ONNX | [Таблица паритета](https://github.com/npatsakula/morok/blob/main/onnx/PARITY.md) |
| CNN-архитектуры (ResNet, DenseNet, VGG, ...) | Проверено 9 моделей |
| Расширения Microsoft (Attention, RotaryEmbedding) | Поддерживается |
| Динамический размер батча | Поддерживается (Variable API) |
| Обучение / обратный проход | Не поддерживается |

**Сравнение с другими фреймворками**

Среди чистых Rust-фреймворков у Morok самое широкое покрытие операторов ONNX — 162 оператора, 1361 пройденный conformance-тест на двух бэкендах (Clang + LLVM). У `candle` и `burn` операторов меньше, а тестовых наборов сопоставимого масштаба нет. Если же нужна максимальная совместимость с продакшн-моделями ONNX — используйте `ort`, Rust-обёртку вокруг C++ ONNX Runtime, которая покрывает полную спецификацию.

---

## Быстрый старт

Добавьте `morok-onnx` и `morok-tensor` в `Cargo.toml`:

```toml
[dependencies]
morok-onnx = { git = "https://github.com/npatsakula/morok" }
morok-tensor = { git = "https://github.com/npatsakula/morok" }
```

### Простой вариант: модели со встроенными весами

Для моделей, у которых все входы уже вшиты в файл (без рантайм-входов):

```rust
use morok_onnx::{OnnxImporter, OnnxModel};
use morok_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut importer = OnnxImporter::new();
    let OnnxModel { mut outputs, .. } = importer.import("model.onnx", &[])?;

    // Подготавливаем все выходы вместе, выполняем за один проход
    let mut outs: Vec<&mut Tensor> = outputs.values_mut().collect();
    Tensor::realize_batch(&mut outs)?;

    for (name, tensor) in &outputs {
        println!("{name}: {:?}", tensor.as_ndarray::<f32>()?);
    }
    Ok(())
}
```

### Модели с рантайм-входами

Большинству моделей нужны данные на этапе выполнения (изображения, токены, аудио). Деструктурируйте `OnnxModel` и используйте `remove()`, чтобы взять владение входными тензорами:

```rust
use morok_onnx::{OnnxImporter, OnnxModel};
use morok_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut importer = OnnxImporter::new();
    let OnnxModel { mut inputs, mut outputs, .. } = importer.import("model.onnx", &[])?;

    // Назначаем входные данные (лениво — без аллокации)
    let input = inputs.remove("input").unwrap();
    input.assign(&Tensor::from_slice(&my_data));

    // Подготавливаем все выходы вместе, выполняем за один проход
    // (внутренне резолвит assign входов — отдельный realize не нужен)
    let mut outs: Vec<&mut Tensor> = outputs.values_mut().collect();
    Tensor::realize_batch(&mut outs)?;
    Ok(())
}
```

---

## Архитектура

### Двухфазный дизайн

Импортёр обрабатывает ONNX-модели в два этапа:

**`import(path, dim_bindings)`** выполняет обе фазы одним вызовом: парсит protobuf, извлекает инициализаторы и спецификации входов, обходит граф в топологическом порядке, диспатчит каждый ONNX-узел в соответствующую реализацию Tensor и возвращает `OnnxModel { inputs, outputs, variables }`. Никаких вычислений — результат представляет собой набор ленивых хэндлов `Tensor`, которые компилируются и выполняются при вызове `realize()`.

```text
model.onnx → import(path, dims) → OnnxModel { inputs, outputs, variables } → realize() → results
```

Для продвинутых сценариев (изучение структуры графа до импорта) метод `import_model()` принимает предварительно распарсенный `ModelProto`.

### Декомпозиция операторов

Каждый оператор ONNX раскладывается на операции Morok Tensor. Степень сложности разная:

**Прямые отображения** — около 60 операторов напрямую соответствуют одному методу тензора:

```rust
// In the registry:
"Add" => x.try_add(y)?
"Relu" => x.relu()?
"Sigmoid" => x.sigmoid()?
"Equal" => x.try_eq(y)?
```

**Паттерны-билдеры** — сложные операторы с множеством необязательных параметров используют fluent API:

```rust
// Conv with optional bias, padding, dilation, groups
x.conv()
    .weight(w)
    .maybe_bias(bias)
    .auto_pad(AutoPad::SameLower)
    .group(32)
    .maybe_dilations(Some(&[2, 2]))
    .call()?
```

**Многошаговые декомпозиции** — операторы вроде BatchNormalization, Attention и Mod требуют промежуточных вычислений. Например, целочисленный `Mod` в стиле Python раскладывается на truncation mod + поправку знака:

```rust
let trunc_mod = x.try_mod(y)?;
let signs_differ = trunc_mod.bitwise_xor(y)?.try_lt(&zero)?;
let needs_adj = mod_ne_zero.bitwise_and(&signs_differ)?;
trunc_mod.try_add(&y.where_(&needs_adj, &zero)?)?
```

### Валидация атрибутов

Хелпер `Attrs` работает по принципу pop — каждый вызов `attrs.int("axis", -1)` или `attrs.float("epsilon", 1e-5)` забирает атрибут из словаря. После обработки оператора `attrs.done()` проверяет, что словарь пуст. Оставшиеся атрибуты вызывают ошибку — так неполные реализации операторов ловятся на этапе трассировки, а не приводят к молчаливо неверным результатам.

### Версионирование opset

ONNX-модели объявляют импорты opset для каждого домена. Импортёр отслеживает их и передаёт версию каждому обработчику. Операторы переключают поведение в зависимости от версии — например, ось по умолчанию у Softmax сменилась с `1` (opset < 13) на `-1` (opset >= 13), а `ReduceSum` перенёс оси из атрибута во входной тензор в opset 13.

---

## Работа с моделями

### Динамические размерности

Входы ONNX могут содержать символические размерности вроде `"batch_size"` или `"sequence_length"`. Привяжите их при импорте через параметр `dim_bindings`:

```rust
let model = importer.import("model.onnx", &[
    ("batch_size", 1),
    ("sequence_length", 512),
])?;

// Variables are auto-extracted from dim_param annotations
for (name, var) in &model.variables {
    println!("{name}: bounds {:?}", var.bounds());
}
```

Непривязанные динамические размерности дают понятную ошибку при импорте. Какие размерности динамические, можно узнать через `InputSpec::shape`:

```rust
for (name, spec) in &graph.inputs {
    for dim in &spec.shape {
        match dim {
            DimValue::Static(n) => print!("{n} "),
            DimValue::Dynamic(name) => print!("{name}? "),
        }
    }
}
```

### Внешние веса

Некоторые ONNX-модели хранят веса в отдельных файлах. Чтобы передать их, используйте `import_model_with_inputs()`:

```rust
let model = importer.import_model_with_inputs(
    "model.onnx",
    &[],
    external_weights,  // HashMap<String, Tensor>
)?;
```

### Расширения Microsoft

Импортёр поддерживает несколько contrib-операторов `com.microsoft`, которые часто встречаются в трансформерных моделях, экспортированных из ONNX Runtime:

| Расширение | Назначение |
|------------|-----------|
| `Attention` | Упакованная QKV-проекция с маскированием, past KV cache |
| `RotaryEmbedding` | Ротационные позиционные эмбеддинги (interleaved/non-interleaved) |
| `SkipLayerNormalization` | Fused residual + LayerNorm + масштабирование |
| `EmbedLayerNormalization` | Эмбеддинги токенов + позиций + сегментов → LayerNorm |

Стандартные трансформерные операторы ONNX (`Attention` из домена ai.onnx) тоже поддерживаются — с grouped query attention (GQA), каузальным маскированием, past KV cache и softcap.

---

## Control flow и ограничения

### Семантика If: обе ветки всегда выполняются

Оператор `If` в ONNX — это data-dependent control flow: условие определяет, какая ветка выполняется. Ленивые вычисления Morok принципиально несовместимы с этим: на этапе трассировки ничего не выполняется, и значение условия неизвестно.

**Решение Morok:** Трассировать *обе* ветки, а потом объединить результаты через `Tensor::where_()`:

```text
ONNX:    if condition { then_branch } else { else_branch }
Morok:   then_result.where_(&condition, &else_result)
```

Это даёт подход **«трассируй один раз — запускай многократно»** — скомпилированный граф обрабатывает любое значение условия в рантайме. Но есть жёсткое ограничение: **обе ветки должны возвращать одинаковые формы и типы данных.** Модели с shape-полиморфными ветками (then-ветка возвращает `[3, 4]`, а else-ветка — `[5, 6]`) трассировать нельзя.

На практике большинство ONNX-моделей с узлами `If` укладываются в это ограничение — условная логика в них выбирает значения, а не меняет форму данных.

### Нет Loop и Scan

Итеративный control flow (`Loop`, `Scan`) не реализован. Эти операторы требуют многократной трассировки или развёртки, что не ложится на архитектуру однократной трассировки. Модели с рекуррентными паттернами обычно работают через развёрнутые операторы (LSTM, GRU, RNN реализованы как нативные ops).

### Батч-выполнение

Несколько тензоров можно реализовать одновременно, разделяя вычисления между выходами
(тестируется в `tensor/src/test/unit/batch.rs`):

```rust
// Realize all outputs at once (shares compilation and execution)
let mut outputs: Vec<&mut Tensor> = model.outputs.values_mut().collect();
Tensor::realize_batch(&mut outputs)?;
```

Для повторного инференса используйте паттерн prepare/execute (тестируется в
`tensor/src/test/unit/variable.rs::test_prepare_execute_loop`):

```rust
let OnnxModel { mut inputs, mut outputs, variables } =
    importer.import("model.onnx", &[("batch", 1)])?;

// 1. Assign initial data (lazy — no allocation yet)
let input = inputs.remove("audio").unwrap();
input.assign(&Tensor::from_slice(&first_frame));

// 2. Compile the execution plan (resolves assigns, allocates buffers)
let mut outs: Vec<&mut Tensor> = outputs.values_mut().collect();
let mut plan = Tensor::prepare_batch(&mut outs)?;
plan.execute()?;  // first run

// 3. Fast loop: zero-copy writes via array_view_mut, no recompilation
for frame in audio_frames {
    input.array_view_mut::<f32>()?[..frame.len()].copy_from_slice(&frame);
    plan.execute()?;
}

// Re-execute with different variable bindings
let bound = variables["batch"].bind(8)?;
plan.execute_with_vars(&[bound.as_var_val()])?;
```

### Нет обучения

Импортёр только для инференса. Обратного прохода, вычисления градиентов и оптимизаторов нет.

### Нереализованные категории операторов

| Категория | Примеры | Причина |
|-----------|---------|---------|
| Квантизация | DequantizeLinear, QuantizeLinear | Нужна поддержка квантизованных типов в IR |
| Операции с последовательностями | SequenceConstruct, SequenceAt | Нетензорные типы не входят в систему типов Morok |
| Случайные числа | RandomNormal, RandomUniform | Stateful RNG пока не реализован |
| Обработка сигналов | DFT, STFT, MelWeightMatrix | Низкий приоритет; узкоспециализированные задачи |
| Текст | StringNormalizer, TfIdfVectorizer | Строковые типы не поддерживаются |

Для моделей с такими операторами используйте `ort` (обёртку над ONNX Runtime) — она покрывает полную спецификацию.

---

## Отладка

### Поузловая трассировка выходов

Установите уровень логирования trace, чтобы выводить промежуточные результаты:

```bash
RUST_LOG=morok_onnx::importer=trace cargo run
```

Это вызывает `realize()` для выхода каждого узла отдельно и печатает первые 5 значений — помогает при числовой бисекции, когда модель выдаёт неверные результаты. Учтите, что это ломает фьюзинг ядер (каждый узел выполняется отдельно), так что это чисто отладочный инструмент.

### Исследование графа

Чтобы понять, что нужно модели, используйте структуру `OnnxModel`:

```rust
let model = importer.import("model.onnx", &[])?;

println!("Inputs:");
for (name, tensor) in &model.inputs {
    println!("  {name}: {:?}", tensor.shape());
}

println!("Outputs: {:?}", model.outputs.keys().collect::<Vec<_>>());
println!("Variables: {:?}", model.variables.keys().collect::<Vec<_>>());
```

---

## Итого

| Аспект | Детали |
|--------|--------|
| **Точка входа** | `OnnxImporter::new()` |
| **Простой импорт** | `importer.import("model.onnx", &[])?` |
| **Динамические размерности** | `importer.import(path, &[("batch", 4)])?` |
| **Операторы** | 162 / 200 ([полная таблица паритета](https://github.com/npatsakula/morok/blob/main/onnx/PARITY.md)) |
| **Проверенные модели** | ResNet50, DenseNet121, VGG19, Inception, AlexNet, ShuffleNet, SqueezeNet, ZFNet |
| **Бэкенды** | Clang + LLVM (идентичные результаты) |
| **Расширения** | com.microsoft Attention, RotaryEmbedding, SkipLayerNorm, EmbedLayerNorm |
| **Ограничения** | Нет обучения, нет Loop/Scan, shape-полиморфный If |

**Далее:** [Практические примеры](./examples) — основы работы с тензорами, или [Пайплайн выполнения](./architecture/pipeline) — чтобы разобраться, как устроена компиляция.
