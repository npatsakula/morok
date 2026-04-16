---
sidebar_label: Практические примеры
---

# Практика: от тензоров до моделей

Эта глава обучает Morok через последовательные примеры. Начнём с базовых тензорных операций и дойдём до рабочего нейросетевого классификатора.

**Чему вы научитесь:**
- Создание и манипуляции с тензорами
- Операции с формами (reshape, transpose, broadcast)
- Матричное умножение
- Построение переиспользуемых слоёв
- Сборка полноценной модели

**Предварительные требования:**
- Базовое знание Rust
- Добавить `morok_tensor` в `Cargo.toml`

**Ключевой паттерн:** Morok использует *ленивые вычисления*. Операции строят граф вычислений без выполнения. Вызов `realize()` компилирует и запускает всё разом.

---

## Пример 1: Hello Tensor

Создадим тензоры, выполним операции и получим результаты.

```rust
use morok_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create tensors from slices
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([10.0f32, 20.0, 30.0, 40.0]);

    // Lazy operations (no execution yet)
    let sum = &a + &b;
    let mut scaled = &sum * &Tensor::from_slice([0.1f32]);

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

2. `&a + &b` ничего не вычисляет. Возвращается новый `Tensor`, который *описывает* сложение. `&` заимствует тензоры, чтобы их можно было использовать повторно.

3. `realize()` — здесь происходит магия. Morok:
   - Анализирует граф вычислений
   - Фьюзит операции, где это возможно
   - Генерирует оптимизированный код
   - Выполняет на целевом устройстве

4. `as_ndarray()` извлекает результат в виде `ndarray::ArrayD` для просмотра.

**Попробуйте:** Уберите вызов `realize()`. Код всё ещё запустится, но `data` будет пустым — ничего не было вычислено.

---

## Пример 2: Гимнастика с формами

Нейросети постоянно меняют форму данных. Освоим базовые операции.

```rust
use ndarray::array;

fn shape_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 1D tensor with 6 elements
    let data = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    println!("Original shape: {:?}", data.shape()?);  // [6]

    // Reshape to a 2x3 matrix (or create directly with from_ndarray)
    let matrix = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    println!("Matrix shape: {:?}", matrix.shape());  // [2, 3]
    // [[1, 2, 3],
    //  [4, 5, 6]]

    // Transpose to 3x2
    let transposed = matrix.try_transpose(0, 1)?;
    println!("Transposed shape: {:?}", transposed.shape()?);  // [3, 2]
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]

    // Broadcasting: add a row vector to every row
    // [3, 2] + [1, 2] → [3, 2]
    let bias = Tensor::from_ndarray(&array![[100.0f32, 200.0]]);
    let mut biased = &transposed + &bias;

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
    let mut output = input.dot(&weights)?;

    output.realize()?;
    println!("Output shape: {:?}", output.shape()?);  // [4, 2]
    println!("{:?}", biased.as_ndarray::<f32>()?);
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

Линейный слой вычисляет `y = x @ W.T + b`. Morok предоставляет `nn::Linear` из коробки.

```rust
use morok_tensor::{Tensor, nn::{Linear, Layer}};

fn linear_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a layer: 4 inputs → 2 outputs
    let layer = Linear::with_dims(4, 2, morok_dtype::DType::Float32);

    // Single sample with 4 features
    let input = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);

    // Forward pass
    let mut output = layer.forward(&input)?;

    output.realize()?;
    println!("Output: {:?}", biased.as_ndarray::<f32>()?);

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
use morok_tensor::{Tensor, nn::{Linear, Relu, Layer}};

fn mnist_example() -> Result<(), Box<dyn std::error::Error>> {
    // Architecture: 784 (28×28 pixels) → 128 (hidden) → 10 (digits)
    let fc1 = Linear::with_dims(784, 128, morok_dtype::DType::Float32);
    let fc2 = Linear::with_dims(128, 10, morok_dtype::DType::Float32);

    // Simulate a 28×28 grayscale image (flattened to 784)
    let fake_image: Vec<f32> = (0..784)
        .map(|i| (i as f32) / 784.0)
        .collect();
    let input = Tensor::from_slice(fake_image)
        .try_reshape(&[1, 784])?;  // batch size 1

    // Forward pass: linear → ReLU → linear
    let logits = input.sequential(&[&fc1, &Relu, &fc2])?;
    let mut probs = logits.softmax(-1)?;

    // Get results
    probs.realize()?;
    println!("Probabilities: {:?}", probs_biased.as_ndarray::<f32>()?);

    // Get predicted class
    let mut prediction = logits.argmax(Some(-1))?;
    prediction.realize()?;
    println!("Predicted digit: {:?}", pred_output.as_ndarray::<i32>()?);

    Ok(())
}
```

**Ключевые концепции:**

1. **`sequential()`** соединяет слои в цепочку: выход каждого слоя подаётся на вход следующему. Ручная прокладка не нужна.

2. **Активация ReLU:** `Relu` — zero-size слой, который применяет `max(0, x)`. Вносит нелинейность — без неё стек линейных слоёв оставался бы одним большим линейным слоем.

3. **Logits и вероятности:** Сырой выход последнего слоя (logits) может быть любым вещественным числом. `softmax()` превращает их в вероятности с суммой 1.

4. **argmax:** Возвращает индекс максимального значения — предсказанный класс.

5. **Размерность батча:** Форма `[1, 784]` для одного изображения. Для 32 изображений — `[32, 784]`. Модель обрабатывает батчи автоматически.

---

## Пример 6: Под капотом

Хотите увидеть, что генерирует Morok? Вот как заглянуть в IR и сгенерированный код.

```rust
fn inspect_compilation() -> Result<(), Box<dyn std::error::Error>> {
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
    let mut c = &a + &b;

    // Print the computation graph (before compilation)
    println!("=== IR Graph ===");
    println!("{}", c.uop().tree());

    // Compile and inspect the execution plan
    let plan = c.prepare()?;
    println!("\nKernels: {}", plan.kernels().count());

    // Execute
    plan.execute()?;

    Ok(())
}
```

**Что вы увидите:**

1. **IR-граф:** UOp-дерево показывает операции вроде `BUFFER`, `LOAD`, `ADD`, `STORE`. Это промежуточное представление Morok до оптимизаций.

2. **Сгенерированный код:** Реальный LLVM IR или GPU-код, который выполняется. Обратите внимание, как Morok фьюзит загрузки и сложение в одно ядро — промежуточные буферы не нужны.

**Совет по отладке:** Если что-то кажется медленным или неправильным, напечатайте IR-дерево. Ищите:
- Неожиданные операции (лишние reshape, дополнительные копии)
- Отсутствие фьюзинга (отдельные ядра там, где хватило бы одного)
- Несовпадения форм (часто коренная причина ошибок)

---

## Итого

Вы освоили основные паттерны работы с Morok:

| Задача | Код |
|--------|-----|
| Создать тензор | `Tensor::from_slice([1.0f32, 2.0])` |
| Арифметика | `&a + &b`, `&a * &b`, `-&a` |
| Изменить форму | `t.try_reshape(&[2, 3])?` |
| Транспонирование | `t.try_transpose(0, 1)?` |
| Матричное умножение | `a.dot(&b)?` |
| Линейный слой | `Linear::with_dims(in, out, dtype)` |
| Цепочка слоёв | `x.sequential(&[&fc1, &Relu, &fc2])?` |
| Активация | `t.relu()?`, `t.softmax(-1)?` |
| Выполнить | `t.realize()?` |
| Батч-реализация | `Tensor::realize_batch(&mut [&mut a, &mut b])?` |
| Извлечь данные | `biased.as_ndarray::<f32>()?` |

**Паттерн ленивых вычислений:**

1. Постройте граф вычислений с помощью операций
2. Вызовите `realize()` один раз в конце
3. Morok оптимизирует и выполняет всё вместе

**Дальше:**

- [Op Bestiary](./architecture/op-bestiary) — справочник по IR-операциям
- [Пайплайн выполнения](./architecture/pipeline) — как устроена компиляция
- [Движок паттернов](./architecture/optimizations/pattern-system) — перезапись на основе паттернов
