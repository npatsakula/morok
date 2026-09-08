---
sidebar_label: Практические примеры
---

# Практика: от тензоров до моделей

Эта глава обучает Svod через последовательные примеры. Начнём с базовых тензорных операций и дойдём до рабочего нейросетевого классификатора.

**Чему вы научитесь:**
- Создание и манипуляции с тензорами
- Операции с формами (reshape, transpose, broadcast)
- Матричное умножение
- Построение переиспользуемых слоёв
- Сборка полноценной модели

**Предварительные требования:**
- Базовое знание Rust
- Добавить `svod_tensor` в `Cargo.toml`

**Ключевой паттерн:** Svod использует *ленивые вычисления*. Операции строят граф вычислений без выполнения. Вызов `realize()` компилирует и запускает всё разом.

---

## Пример 1: Hello Tensor

Создадим тензоры, выполним операции и получим результаты.

```rust
use svod_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create tensors from slices
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([10.0f32, 20.0, 30.0, 40.0]);

    // Lazy operations (no execution yet); a scalar is a valid right-hand side
    let sum = (&a + &b)?;
    let scaled = (&sum * 0.1)?;

    // Execute and get results
    scaled.realize()?;
    let data = scaled.as_ndarray::<f32>()?;
    println!("Result: {:?}", data);
    // Output: [1.1, 2.2, 3.3, 4.4]

    Ok(())
}
```

**Что здесь происходит:**

1. `Tensor::from_slice()` создаёт одномерный тензор из массива. Суффикс `f32` указывает Rust тип элемента.

2. `&a + &b` ничего не вычисляет. Возвращается `Result<Tensor>` — несовпадение формы или dtype является восстановимой ошибкой, отсюда и `?` — оборачивающий тензор, который *описывает* сложение. `&` заимствует тензоры, чтобы их можно было использовать повторно. `2.0 * &a` тоже работает: скаляры принимаются с любой стороны и материализуются в dtype тензора.

3. `realize()` — здесь происходит магия. Метод принимает `&self`, поэтому реализованный тензор может оставаться за общим заимствованием. Svod:
   - Анализирует граф вычислений
   - Фьюзит операции, где это возможно
   - Генерирует оптимизированный код
   - Выполняет на целевом устройстве

4. `as_ndarray()` извлекает уже вычисленный результат в виде `ndarray::ArrayD` для просмотра.

**Попробуйте:** Уберите вызов `realize()`. Тогда `as_ndarray()` вернёт ошибку «нет буфера» — ничего не было вычислено, а значит и читать нечего. `to_ndarray()`, `to_vec()` и `item()` реализуют тензор по требованию, вместо того чтобы падать; `as_ndarray()` / `as_vec()` не реализуют его никогда, поэтому остаются пригодны там, где реализация была бы ошибкой.

---

## Пример 2: Гимнастика с формами

Нейросети постоянно меняют форму данных. Освоим базовые операции.

```rust
use svod_tensor::Tensor;
use ndarray::array;

fn shape_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 1D tensor with 6 elements
    let data = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    println!("Original shape: {:?}", data.dims()?);  // [6]

    // Reshape to a 2x3 matrix (or create directly with from_ndarray)
    let matrix = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    println!("Matrix shape: {:?}", matrix.dims()?);  // [2, 3]
    // [[1, 2, 3],
    //  [4, 5, 6]]

    // Transpose to 3x2
    let transposed = matrix.try_transpose(0, 1)?;
    println!("Transposed shape: {:?}", transposed.dims()?);  // [3, 2]
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]

    // Broadcasting: add a row vector to every row
    // [3, 2] + [1, 2] → [3, 2]
    let bias = Tensor::from_ndarray(&array![[100.0f32, 200.0]]);
    let biased = (&transposed + &bias)?;

    biased.realize()?;
    println!("{:?}", biased.as_ndarray::<f32>()?);
    // [[101, 204],
    //  [102, 205],
    //  [103, 206]]

    Ok(())
}
```

**Ключевые операции:**

| Операция | Что делает |
|----------|------------|
| `try_reshape(&[2, 3])` | Изменить форму (то же количество элементов) |
| `try_reshape(&[-1, 3])` | Вывести размерность из общего числа элементов |
| `try_transpose(0, 1)` | Поменять местами размерности 0 и 1 |
| `try_squeeze(dim)` | Убрать размерность длины 1 |
| `try_unsqueeze(dim)` | Добавить размерность длины 1 |

**Чтение формы:** `dims()` возвращает `Vec<usize>` и падает с ошибкой, если хотя бы одна ось символьная; `dim(axis)` возвращает эту ось как `SInt` (символьную или константную), а `dim_const(axis)` — как `usize`, с ошибкой `NonConstDim`, когда она не константна; `shape()` возвращает целиком `Shape` из `SInt`. `dtype()` не может завершиться ошибкой, а `Tensor` реализует `Debug` — форма, dtype, устройство и признак реализованности, но никогда сами данные, ведь это потребовало бы чтения с устройства. Отрицательные оси всюду отсчитываются с конца.

**Правила broadcasting** (такие же, как в NumPy/PyTorch):
- Формы выравниваются справа
- Каждая размерность должна совпадать или быть равна 1
- Размерности равные 1 «растягиваются» до нужного значения

```text
[3, 2] + [1, 2] → [3, 2]  ✓ (1 broadcasts to 3)
[3, 2] + [2]    → [3, 2]  ✓ (implicit [1, 2])
[3, 2] + [3]    → error   ✗ (2 ≠ 3)
```

---

## Пример 3: Матричное умножение

Матричное умножение — рабочая лошадка нейросетей. Каждый слой его использует.

```rust
use svod_tensor::Tensor;
use ndarray::array;

fn matmul_example() -> Result<(), Box<dyn std::error::Error>> {
    // Input: 4 samples, 3 features each → shape [4, 3]
    let input = Tensor::from_ndarray(&array![
        [1.0f32, 2.0, 3.0],    // sample 0
        [4.0, 5.0, 6.0],       // sample 1
        [7.0, 8.0, 9.0],       // sample 2
        [10.0, 11.0, 12.0],    // sample 3
    ]);

    // Weights: 3 inputs → 2 outputs → shape [3, 2]
    let weights = Tensor::from_ndarray(&array![
        [0.1f32, 0.2],  // feature 0 → outputs
        [0.3, 0.4],     // feature 1 → outputs
        [0.5, 0.6],     // feature 2 → outputs
    ]);

    // Matrix multiply: [4, 3] @ [3, 2] → [4, 2]
    let output = input.dot(&weights)?;

    output.realize()?;
    println!("Output shape: {:?}", output.dims()?);  // [4, 2]
    println!("{:?}", output.as_ndarray::<f32>()?);
    // Each row: weighted sum of that sample's features

    Ok(())
}
```

**Правила форм для `dot()`:**

| Левый | Правый | Результат |
|-------|--------|-----------|
| `[M, K]` | `[K, N]` | `[M, N]` |
| `[K]` | `[K, N]` | `[N]` (вектор-матрица) |
| `[M, K]` | `[K]` | `[M]` (матрица-вектор) |
| `[B, M, K]` | `[B, K, N]` | `[B, M, N]` (батч) |

Внутренние размерности должны совпадать (`K`). Суть: «для каждой строки левого — скалярное произведение с каждым столбцом правого».

---

## Пример 4: Линейный слой

Линейный слой вычисляет `y = x @ W.T + b`. Svod предоставляет `nn::Linear` из коробки.

```rust
use svod_tensor::{Tensor, nn::{Linear, Layer}};

fn linear_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a layer: 4 inputs → 2 outputs, with a bias
    let layer = Linear::with_dims(4, 2, true, svod_dtype::DType::Float32);

    // Single sample with 4 features
    let input = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);

    // Forward pass
    let output = layer.forward(&input)?;

    output.realize()?;
    println!("Output: {:?}", output.as_ndarray::<f32>()?);

    Ok(())
}
```

**Зачем транспонировать веса?**

В PyTorch принято хранить веса как `[out_features, in_features]`. Для слоя 4 → 2:
- Форма весов: `[2, 4]`
- Форма входа: `[4]` или `[batch, 4]`
- Нужно: `input @ weight.T` = `[batch, 4] @ [4, 2]` = `[batch, 2]`

Такое соглашение удобно для чтения матрицы весов: строка `i` содержит все веса, ведущие в выход `i`.

---

## Пример 5: Классификатор MNIST

Построим полноценную нейросеть, используя `sequential()` для цепочки слоёв.

```rust
use svod_tensor::{Tensor, nn::{Linear, Relu, Layer}};

fn mnist_example() -> Result<(), Box<dyn std::error::Error>> {
    // Architecture: 784 (28×28 pixels) → 128 (hidden) → 10 (digits)
    let fc1 = Linear::with_dims(784, 128, true, svod_dtype::DType::Float32);
    let fc2 = Linear::with_dims(128, 10, true, svod_dtype::DType::Float32);

    // Simulate a 28×28 grayscale image (flattened to 784)
    let fake_image: Vec<f32> = (0..784)
        .map(|i| (i as f32) / 784.0)
        .collect();
    let input = Tensor::from_slice(fake_image)
        .try_reshape(&[1, 784])?;  // batch size 1

    // Forward pass: linear → ReLU → linear
    let logits = input.sequential(&[&fc1, &Relu, &fc2])?;
    let probs = logits.softmax(-1)?;

    // Get predicted class; realize both results in one compilation
    let prediction = logits.argmax(Some(-1))?;
    Tensor::realize_batch([&probs, &prediction])?;

    println!("Probabilities: {:?}", probs.as_ndarray::<f32>()?);
    println!("Predicted digit: {:?}", prediction.as_ndarray::<i32>()?);

    Ok(())
}
```

**Ключевые концепции:**

1. **`sequential()`** соединяет слои в цепочку: выход каждого слоя подаётся на вход следующему. Ручная прокладка не нужна.

2. **Активация ReLU:** `Relu` — zero-size слой, который применяет `max(0, x)`. Вносит нелинейность — без неё стек линейных слоёв оставался бы одним большим линейным слоем.

3. **Logits и вероятности:** Сырой выход последнего слоя (logits) может быть любым вещественным числом. `softmax()` превращает их в вероятности с суммой 1.

4. **argmax:** Возвращает индекс максимального значения — предсказанный класс.

5. **Размерность батча:** Форма `[1, 784]` для одного изображения. Для 32 изображений — `[32, 784]`. Модель обрабатывает батчи автоматически.

6. **`realize_batch`:** Два результата, у которых общий подграф (здесь — logits), компилируются и выполняются вместе, так что общая часть вычисляется один раз. Метод принимает общие ссылки — `[&a, &b]` — потому что реализация фиксируется в реестре тензоров, а не в самом хендле.

---

## Пример 6: Под капотом

Хотите увидеть, что генерирует Svod? Вот как заглянуть в IR и в скомпилированные ядра.

```rust
use svod_tensor::Tensor;

fn inspect_compilation() -> Result<(), Box<dyn std::error::Error>> {
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
    let c = (&a + &b)?;

    // Print the computation graph (before compilation)
    println!("=== IR Graph ===");
    println!("{}", c.uop().tree());

    // Compile and inspect the execution plan
    let plan = c.prepare()?;  // prepare() takes &self
    println!("\nKernels: {}", plan.kernels().count());

    // Execute
    plan.execute()?;

    Ok(())
}
```

**Что вы увидите:**

1. **IR-граф:** UOp-дерево показывает операции вроде `BUFFER`, `LOAD`, `ADD`, `STORE`. Это промежуточное представление Svod до оптимизаций.

2. **План выполнения:** `prepare()` возвращает скомпилированные ядра. Обратите внимание, как Svod фьюзит обе загрузки и сложение в одно ядро — промежуточные буферы не нужны.

**Совет по отладке:** Если что-то кажется медленным или неправильным, напечатайте IR-дерево. Ищите:
- Неожиданные операции (лишние reshape, дополнительные копии)
- Отсутствие фьюзинга (отдельные ядра там, где хватило бы одного)
- Несовпадения форм (часто коренная причина ошибок)

---

## Пример 7: Слои, модули и state dict

Структура слоя владеет своими параметрами и гиперпараметрами, нужными её
forward-проходу. `#[derive(Module)]` превращает эти поля в плоский `StateDict`
(`HashMap<String, Tensor>`) с ключами ровно такими, как их называет PyTorch, —
чекпоинт загружается без написанного вручную отображения имён:

```rust
use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{LayerNorm, Module, StateDict};

#[derive(Clone, Module)]
struct Block {
    intermediate: usize,            // primitives are skipped automatically
    #[module(skip)]                 // a non-primitive that carries no weights
    dtype: DType,
    norm: LayerNorm,                // child module: "norm.weight", "norm.bias"
    #[module(key = "Wi.weight")]    // checkpoint name, dots allowed
    wi: Tensor,
    #[module(key = "Wo.weight")]
    wo: Tensor,
    #[module(optional)]             // written when Some, absent-tolerant on load
    out_bias: Option<Tensor>,
}

fn load(checkpoint: &StateDict) -> Result<Block, Box<dyn std::error::Error>> {
    let mut block = Block {
        intermediate: 3072,
        norm: LayerNorm::with_dims(768, true, 1e-5, DType::Float32),
        wi: Tensor::zeros(&[3072, 768], DType::Float32),
        wo: Tensor::zeros(&[768, 3072], DType::Float32),
        out_bias: None,
    };
    // Reads "layers.0.norm.weight", "layers.0.Wi.weight", ...
    block.load_state_dict(checkpoint, "layers.0")?;
    // ...and writes them back out under any prefix
    let _round_trip: StateDict = block.state_dict("layers.0");
    Ok(block)
}
```

| Атрибут | Действие |
|---------|----------|
| `#[module(key = "Wi.weight")]` | Заменяет сегмент ключа, взятый из имени поля (может содержать точки и цифры) |
| `#[module(key = "")]` | Уплощение: ключи поля используют родительский префикс без изменений |
| `#[module(skip)]` | Игнорировать непримитивное поле (конфиг, dtype, режим) |
| `#[module(optional)]` | Обязателен для `Option<Tensor>`: сохраняется при `Some`, при загрузке допускает отсутствие ключа |
| `#[module(optional = "self.has_bias")]` | Ключ обязателен, когда предикат истинен, и пропускается иначе |

Дочерние модули собираются через blanket-реализации: `Vec<Block>` даёт своим
элементам ключи `0.`, `1.`, …, а массивы, `Option`, кортежи и `Box`
делегируют так же. Для перечислений derive тоже работает. Forward-проход
остаётся вне `Module`: он живёт в трейте `Layer`
(`fn forward(&self, x: &Tensor) -> Result<Tensor>`), когда это позволяет
сигнатура, и в собственных методах в остальных случаях.

Встроенные слои реализуют оба трейта: `new` — для уже загруженных тензоров, а
`with_dims` — для свежей инициализации Kaiming-uniform (свёртки) или
единичной аффинной (нормализации):

| Слой | `with_dims` | Ключи state dict |
|------|-------------|------------------|
| `Linear` | `(in, out, bias, dtype)` | `weight`, `bias` (если есть) |
| `Conv1d` | `(in_c, out_c, kernel, bias, dtype)` | `weight`, `bias` |
| `Conv2d` / `ConvTranspose2d` | `(in_c, out_c, (kh, kw), bias, dtype)` | `weight`, `bias` |
| `BatchNorm2d` | `(channels, eps, dtype)` | `weight`, `bias`, `running_mean`, `running_var` |
| `LayerNorm` | `(size, bias, eps, dtype)` | `weight`, `bias` (если есть) |
| `RmsNorm` | `(size, eps, dtype)` | `weight` |
| `Embedding` | `(vocab_size, embed_dim, dtype)` | `weight` |

Гиперпараметры задаются методами `with_*` в стиле билдера прямо на структуре —
`Conv1d::new(w, bias).with_stride(2).with_padding((1, 1)).with_groups(4)`,
`LayerNorm::with_dims(..).with_axis(-2)`.

---

## Пример 8: Рекуррентные слои

`rnn()`, `gru()` и `lstm()` — это билдеры на `Tensor`. Они принимают либо имена
весов из PyTorch (`weight_ih`, `weight_hh`, `bias_ih`, `bias_hh`, `h0`, `c0`),
либо имена из ONNX (`w`, `r`/`r_weights`, `bias`, `initial_h`, `initial_c`), и
сами переставляют блоки гейтов:

```rust
use svod_tensor::Tensor;
use ndarray::Array3;

// seq=2, batch=1, input=3, hidden=4
let x = Tensor::from_ndarray(&Array3::from_elem((2, 1, 3), 0.1f32));
let w = Tensor::from_ndarray(&Array3::from_elem((1, 12, 3), 0.1f32));
let r = Tensor::from_ndarray(&Array3::from_elem((1, 12, 4), 0.1f32));

let out = x.gru().w(&w).r_weights(&r).hidden_size(4).call()?;
// ONNX-shaped: y [seq, num_directions, batch, hidden], y_h [num_directions, batch, hidden]
// PyTorch-shaped: output [seq, batch, D*hidden], h_n [num_directions, batch, hidden]
assert_eq!(out.y.dims()?, vec![2, 1, 1, 4]);
assert_eq!(out.output.dims()?, vec![2, 1, 4]);
```

`layout` выбирает `RnnLayout::SeqFirst` (`[seq, batch, input]`, по умолчанию)
или `BatchFirst`; `direction` принимает `RnnDirection::{Forward, Backward,
Bidirectional}`, и двунаправленный проход конкатенирует оба направления по оси
признаков. У GRU параметр `linear_before_reset` по умолчанию соответствует
размещению PyTorch с весами PyTorch и размещению ONNX с весами ONNX.
`LstmOutput` добавляет `y_c` / `c_n` для состояния ячейки.

Ось времени должна быть конкретной, а вот ось батча может быть символьной
`Variable`. Для написанного вручную цикла — например, декодера, шагающего по
одному токену, — используйте ячейки напрямую: `GruCell`/`LstmCell`/`RnnCell`
предоставляют `step(&x, &h) -> Result<..>`, а `RnnStack::new(cells)` шагает
сразу по всему стеку.

---

## Пример 9: Спектрограммы

`stft()` — это одна `conv1d` с оконным ядром ДПФ, поэтому всё преобразование
остаётся в графе (и ось батча может оставаться символьной). Результат —
`[B, F, T, 2]` (или `[F, T, 2]` для небатчированного сигнала `[L]`) с
`(real, imag)` на последней оси, что совпадает с
`torch.stft(..., return_complex=false)`:

```rust
use svod_tensor::Tensor;
use svod_tensor::nn::Window;

let x = Tensor::from_slice(vec![0.25f32; 64]);
let spec = x.stft().n_fft(16).hop(4).window(Window::Hann).call()?;
assert_eq!(spec.dims()?, vec![9, 17, 2]);   // [F, T, (re, im)]

let mag = spec.magnitude(0.0)?;             // sqrt(re² + im² + eps)
let signal = spec.istft().n_fft(16).hop(4).window(Window::Hann).length(64).call()?;
```

Значения по умолчанию повторяют torch: `hop = n_fft / 4`,
`win_length = n_fft`, периодическое окно Ханна, `center`, `onesided`, без
нормализации — и те же самые нужно передать в `istft`. `Window` — это `Hann`,
`Hamming`, `Rectangular` или `Custom(tensor)`, а
`Tensor::window(&Window::Hann, n, periodic, dtype)` материализует окно. Кроме
`magnitude`, для последней оси длины 2 есть `power`, `complex_abs`,
`complex_mul` и `Tensor::complex_from_polar(&mag, &phase)`.

---

## Ошибки

Каждый способный завершиться ошибкой метод тензора возвращает
`svod_tensor::error::Result<T>`, чья ошибка — это `Error(Box<ErrorKind>)`
размером с указатель; сопоставляйте причину через `err.kind()` (или
`into_kind()`, чтобы забрать её по значению). Крейты-потребители конвертируют
её через snafu-шный `context(false)`, так что собственный enum ошибок модели
поглощает её обычным `?` — никакого `.context(TensorSnafu)` на каждом вызове.

Не всё может завершиться ошибкой. `cast`, `neg`, `abs`, `floor`, `ceil`,
`round`, `trunc`, `square`, `sign` и конструкторы `Tensor::full` / `zeros` /
`ones` не могут упасть и возвращают обычный `Tensor`; `-&a` тоже обычный, а
бинарные операторы возвращают `Result<Tensor>`.

---

## Итого

Вы освоили основные паттерны работы с Svod:

| Задача | Код |
|--------|-----|
| Создать тензор | `Tensor::from_slice([1.0f32, 2.0])` |
| Арифметика | `(&a + &b)?`, `(&a * 2.0)?`, `(2.0 * &a)?`, `-&a` |
| Изменить форму | `t.try_reshape(&[2, 3])?` |
| Транспонирование | `t.try_transpose(0, 1)?` |
| Матричное умножение | `a.dot(&b)?` |
| Посмотреть метаданные | `t.dims()?`, `t.dim_const(-1)?`, `t.dtype()` |
| Линейный слой | `Linear::with_dims(in, out, bias, dtype)` |
| Цепочка слоёв | `x.sequential(&[&fc1, &Relu, &fc2])?` |
| Активация | `t.relu()?`, `t.softmax(-1)?` |
| Загрузить веса | `model.load_state_dict(&sd, "")?` |
| Спектрограмма | `x.stft().n_fft(512).hop(160).call()?` |
| Рекуррентный слой | `x.lstm().weight_ih(&w).weight_hh(&r).hidden_size(h).call()?` |
| Выполнить | `t.realize()?` |
| Батч-реализация | `Tensor::realize_batch([&a, &b])?` |
| Извлечь данные | `t.to_vec::<f32>()?`, `t.to_ndarray::<f32>()?`, `t.item::<f32>()?` |

**Паттерн ленивых вычислений:**

1. Постройте граф вычислений с помощью операций
2. Вызовите `realize()` один раз в конце
3. Svod оптимизирует и выполняет всё вместе

**Дальше:**

- [Op Bestiary](./architecture/op-bestiary) — справочник по IR-операциям
- [Пайплайн выполнения](./architecture/pipeline) — как устроена компиляция
- [Движок паттернов](./architecture/optimizations/pattern-system) — перезапись на основе паттернов
