---
sidebar_label: प्रैक्टिकल उदाहरण
---

# प्रैक्टिकल: Tensor से मॉडल तक

यह चैप्टर प्रोग्रेसिव उदाहरणों के ज़रिए Morok सिखाता है। आप बेसिक tensor ऑपरेशनों से शुरू करेंगे और एक काम करने वाले न्यूरल नेटवर्क क्लासिफ़ायर तक पहुँचेंगे।

**आप क्या सीखेंगे:**
- Tensor बनाना और मैनिपुलेट करना
- Shape ऑपरेशन (reshape, transpose, broadcast)
- मैट्रिक्स मल्टिप्लिकेशन
- रीयूज़ेबल लेयर बनाना
- एक पूरा मॉडल कम्पोज़ करना

**पूर्व-आवश्यकताएँ:**
- बेसिक Rust ज्ञान
- अपनी `Cargo.toml` में `morok_tensor` जोड़ें

**मुख्य पैटर्न:** Morok *lazy evaluation* इस्तेमाल करता है। ऑपरेशन एक कम्प्यूटेशन ग्राफ़ बनाते हैं बिना एक्ज़ीक्यूट किए। `realize()` कॉल करें तो सब कुछ एक साथ कम्पाइल और रन होता है।

---

## उदाहरण 1: Hello Tensor

चलिए tensor बनाते हैं, ऑपरेशन करते हैं, और रिज़ल्ट लेते हैं।

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

**क्या हो रहा है:**

1. `Tensor::from_slice()` array डेटा से 1D tensor बनाता है। `f32` सफ़िक्स Rust को एलिमेंट टाइप बताता है।

2. `&a + &b` अभी कुछ कम्प्यूट नहीं करता। यह एक नया `Tensor` रिटर्न करता है जो एडिशन को *रिप्रेज़ेंट* करता है। `&` tensor को बॉरो करता है ताकि हम उन्हें फिर से इस्तेमाल कर सकें।

3. `realize()` वो जगह है जहाँ जादू होता है। Morok:
   - कम्प्यूटेशन ग्राफ़ एनालाइज़ करता है
   - जहाँ मुमकिन हो ऑपरेशन फ़्यूज़ करता है
   - ऑप्टिमाइज़्ड कोड जनरेट करता है
   - टारगेट डिवाइस पर एक्ज़ीक्यूट करता है

4. `as_ndarray()` रिज़ल्ट को `ndarray::ArrayD` के रूप में निकालता है ताकि आप देख सकें।

**यह करके देखें:** `realize()` कॉल हटा दें। कोड तब भी चलेगा, लेकिन `data` खाली होगा — कुछ भी कम्प्यूट नहीं हुआ।

---

## उदाहरण 2: Shape ट्रिक्स

न्यूरल नेटवर्क लगातार डेटा को reshape करते हैं। चलिए बेसिक्स सीखते हैं।

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

**मुख्य ऑपरेशन:**

| ऑपरेशन | क्या करता है |
|---------|-------------|
| `try_reshape(&[2, 3])` | Shape बदलें (कुल एलिमेंट समान रहें) |
| `try_reshape(&[-1, 3])` | कुल साइज़ से डायमेंशन इन्फ़र करें |
| `try_transpose(0, 1)` | डायमेंशन 0 और 1 को स्वैप करें |
| `try_squeeze(dim)` | साइज़ 1 का डायमेंशन हटाएँ |
| `try_unsqueeze(dim)` | साइज़ 1 का डायमेंशन जोड़ें |

**Broadcasting नियम** (NumPy/PyTorch जैसे ही):
- Shape दाईं ओर से अलाइन होती हैं
- हर डायमेंशन मैच होना चाहिए या 1 होना चाहिए
- साइज़ 1 वाले डायमेंशन मैच करने के लिए "स्ट्रेच" होते हैं

```text
[3, 2] + [1, 2] → [3, 2]  ✓ (1 broadcasts to 3)
[3, 2] + [2]    → [3, 2]  ✓ (implicit [1, 2])
[3, 2] + [3]    → error   ✗ (2 ≠ 3)
```

---

## उदाहरण 3: मैट्रिक्स मल्टिप्लाई

मैट्रिक्स मल्टिप्लिकेशन न्यूरल नेटवर्क का वर्कहॉर्स है। हर लेयर इसे इस्तेमाल करती है।

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

**`dot()` के Shape नियम:**

| Left | Right | Result |
|------|-------|--------|
| `[M, K]` | `[K, N]` | `[M, N]` |
| `[K]` | `[K, N]` | `[N]` (vector-matrix) |
| `[M, K]` | `[K]` | `[M]` (matrix-vector) |
| `[B, M, K]` | `[B, K, N]` | `[B, M, N]` (batched) |

इनर डायमेंशन मैच होना चाहिए (`K`)। इसे ऐसे सोचें: "left की हर रो का right के हर कॉलम के साथ dot product।"

---

## उदाहरण 4: Linear लेयर बनाना

एक linear लेयर `y = x @ W.T + b` कम्प्यूट करती है। Morok में `nn::Linear` बिल्ट-इन है।

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

**Weights को transpose क्यों करते हैं?**

PyTorch कन्वेंशन weights को `[out_features, in_features]` के रूप में स्टोर करता है। 4 → 2 मैपिंग वाली लेयर के लिए:
- Weight shape: `[2, 4]`
- Input shape: `[4]` या `[batch, 4]`
- हमें चाहिए: `input @ weight.T` = `[batch, 4] @ [4, 2]` = `[batch, 2]`

यह कन्वेंशन weight मैट्रिक्स को पढ़ना आसान बनाता है: रो `i` में वो सभी weights होते हैं जो आउटपुट `i` में फ़ीड होते हैं।

---

## उदाहरण 5: MNIST क्लासिफ़ायर

चलिए `sequential()` से लेयर चेन करके एक पूरा न्यूरल नेटवर्क बनाते हैं।

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

**मुख्य कॉन्सेप्ट:**

1. **`sequential()`** लेयर्स को चेन करता है: हर लेयर का आउटपुट अगली लेयर का इनपुट बनता है। मैन्युअल वायरिंग की ज़रूरत नहीं।

2. **ReLU एक्टिवेशन:** `Relu` एक ज़ीरो-साइज़ लेयर है जो `max(0, x)` अप्लाई करती है। यह नॉन-लीनियरिटी लाती है — इसके बिना, linear लेयर स्टैक करना बस एक बड़ी linear लेयर होगी।

3. **Logits बनाम probabilities:** लास्ट लेयर का रॉ आउटपुट (logits) कोई भी रियल नंबर हो सकता है। `softmax()` उन्हें probabilities में बदलता है जिनका योग 1 होता है।

4. **argmax:** मैक्सिमम वैल्यू का इंडेक्स रिटर्न करता है — यानी प्रेडिक्टेड क्लास।

5. **Batch डायमेंशन:** हम सिंगल इमेज के लिए shape `[1, 784]` इस्तेमाल करते हैं। 32 इमेज के लिए `[32, 784]` इस्तेमाल करें। मॉडल बैच ऑटोमैटिकली हैंडल करता है।

---

## उदाहरण 6: अंदर की बात

जानना चाहते हैं कि Morok क्या जनरेट करता है? IR और जनरेटेड कोड कैसे देखें, यह रहा।

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

**आपको क्या दिखेगा:**

1. **IR Graph:** UOp tree `BUFFER`, `LOAD`, `ADD`, `STORE` जैसे ऑपरेशन दिखाता है। यह ऑप्टिमाइज़ेशन से पहले Morok का इंटरमीडिएट रिप्रेज़ेंटेशन है।

2. **जनरेटेड कोड:** वास्तविक LLVM IR या GPU कोड जो रन होता है। ध्यान दें कि Morok loads और add को एक सिंगल कर्नेल में फ़्यूज़ करता है — कोई इंटरमीडिएट बफ़र नहीं चाहिए।

**डीबगिंग टिप:** अगर कुछ स्लो या गलत लगे, तो IR tree प्रिंट करें। देखें:
- अनएक्सपेक्टेड ऑपरेशन (रिडंडेंट reshapes, एक्स्ट्रा कॉपीज़)
- मिसिंग फ़्यूज़न (जहाँ एक कर्नेल काफ़ी हो वहाँ अलग-अलग कर्नेल)
- Shape मिसमैच (अक्सर एरर की असली वजह)

---

## सारांश

आपने Morok इस्तेमाल करने के कोर पैटर्न सीख लिए:

| टास्क | कोड |
|-------|------|
| Tensor बनाएँ | `Tensor::from_slice([1.0f32, 2.0])` |
| अरिथमेटिक | `&a + &b`, `&a * &b`, `-&a` |
| Reshape | `t.try_reshape(&[2, 3])?` |
| Transpose | `t.try_transpose(0, 1)?` |
| मैट्रिक्स मल्टिप्लाई | `a.dot(&b)?` |
| Linear लेयर | `Linear::with_dims(in, out, dtype)` |
| लेयर चेन करें | `x.sequential(&[&fc1, &Relu, &fc2])?` |
| एक्टिवेशन | `t.relu()?`, `t.softmax(-1)?` |
| एक्ज़ीक्यूट करें | `t.realize()?` |
| बैच realize | `Tensor::realize_batch(&mut [&mut a, &mut b])?` |
| डेटा निकालें | `biased.as_ndarray::<f32>()?` |

**Lazy evaluation पैटर्न:**

1. ऑपरेशन से अपना कम्प्यूटेशन ग्राफ़ बनाएँ
2. अंत में एक बार `realize()` कॉल करें
3. Morok सब कुछ ऑप्टिमाइज़ और एक साथ एक्ज़ीक्यूट करता है

**आगे:**

- [Op Bestiary](./architecture/op-bestiary) — IR ऑपरेशन रेफ़रेंस
- [एक्ज़ीक्यूशन पाइपलाइन](./architecture/pipeline) — कम्पाइलेशन कैसे काम करता है
- [पैटर्न इंजन](./architecture/optimizations/pattern-system) — पैटर्न-आधारित रीराइट्स
