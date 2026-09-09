---
sidebar_label: प्रैक्टिकल उदाहरण
---

# प्रैक्टिकल: Tensor से मॉडल तक

यह चैप्टर प्रोग्रेसिव उदाहरणों के ज़रिए Svod सिखाता है। आप बेसिक tensor ऑपरेशनों से शुरू करेंगे और एक काम करने वाले न्यूरल नेटवर्क क्लासिफ़ायर तक पहुँचेंगे।

**आप क्या सीखेंगे:**
- Tensor बनाना और मैनिपुलेट करना
- Shape ऑपरेशन (reshape, transpose, broadcast)
- मैट्रिक्स मल्टिप्लिकेशन
- रीयूज़ेबल लेयर बनाना
- एक पूरा मॉडल कम्पोज़ करना

**पूर्व-आवश्यकताएँ:**
- बेसिक Rust ज्ञान
- अपनी `Cargo.toml` में `svod_tensor` जोड़ें

**मुख्य पैटर्न:** Svod *lazy evaluation* इस्तेमाल करता है। ऑपरेशन एक कम्प्यूटेशन ग्राफ़ बनाते हैं बिना एक्ज़ीक्यूट किए। `realize()` कॉल करें तो सब कुछ एक साथ कम्पाइल और रन होता है।

---

## उदाहरण 1: Hello Tensor

चलिए tensor बनाते हैं, ऑपरेशन करते हैं, और रिज़ल्ट लेते हैं।

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

**क्या हो रहा है:**

1. `Tensor::from_slice()` array डेटा से 1D tensor बनाता है। `f32` सफ़िक्स Rust को एलिमेंट टाइप बताता है।

2. `&a + &b` अभी कुछ कम्प्यूट नहीं करता। यह `Result<Tensor>` रिटर्न करता है — shape या dtype मिसमैच एक रिकवरेबल एरर है, इसीलिए `?` — जो एक ऐसे tensor को रैप करता है जो एडिशन को *रिप्रेज़ेंट* करता है। `&` tensor को बॉरो करता है ताकि हम उन्हें फिर से इस्तेमाल कर सकें। `2.0 * &a` भी चलता है: scalar दोनों तरफ़ स्वीकार होते हैं और tensor के dtype में मटीरियलाइज़ किए जाते हैं।

3. `realize()` वो जगह है जहाँ जादू होता है। यह `&self` लेता है, इसलिए realized tensor शेयर्ड बॉरो के पीछे भी रह सकता है। Svod:
   - कम्प्यूटेशन ग्राफ़ एनालाइज़ करता है
   - जहाँ मुमकिन हो ऑपरेशन फ़्यूज़ करता है
   - ऑप्टिमाइज़्ड कोड जनरेट करता है
   - टारगेट डिवाइस पर एक्ज़ीक्यूट करता है

4. `as_ndarray()` पहले से कम्प्यूट हो चुके रिज़ल्ट को `ndarray::ArrayD` के रूप में निकालता है ताकि आप देख सकें।

**यह करके देखें:** `realize()` कॉल हटा दें। तब `as_ndarray()` "no buffer" एरर के साथ फ़ेल होगा — कुछ भी कम्प्यूट नहीं हुआ, इसलिए पढ़ने के लिए कोई रिज़ल्ट ही नहीं है। `to_ndarray()`, `to_vec()` और `item()` फ़ेल होने के बजाय ज़रूरत पड़ने पर ख़ुद realize कर लेते हैं; `as_ndarray()` / `as_vec()` कभी realize नहीं करते, इसलिए वे वहाँ भी इस्तेमाल लायक रहते हैं जहाँ realization अपने आप में एक बग होगा।

---

## उदाहरण 2: Shape ट्रिक्स

न्यूरल नेटवर्क लगातार डेटा को reshape करते हैं। चलिए बेसिक्स सीखते हैं।

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

**मुख्य ऑपरेशन:**

| ऑपरेशन | क्या करता है |
|---------|-------------|
| `try_reshape(&[2, 3])` | Shape बदलें (कुल एलिमेंट समान रहें) |
| `try_reshape(&[-1, 3])` | कुल साइज़ से डायमेंशन इन्फ़र करें |
| `try_transpose(0, 1)` | डायमेंशन 0 और 1 को स्वैप करें |
| `try_squeeze(dim)` | साइज़ 1 का डायमेंशन हटाएँ |
| `try_unsqueeze(dim)` | साइज़ 1 का डायमेंशन जोड़ें |

**Shape पढ़ना:** `dims()` एक `Vec<usize>` देता है और अगर कोई axis symbolic हो तो एरर देता है; `dim(axis)` उस axis को `SInt` (symbolic या constant) के रूप में लौटाता है और `dim_const(axis)` `usize` के रूप में, जो constant न होने पर `NonConstDim` के साथ फ़ेल होता है; `shape()` पूरा `Shape` लौटाता है, जो `SInt` से बना है। `dtype()` कभी फ़ेल नहीं होता, और `Tensor` `Debug` इम्प्लीमेंट करता है — shape, dtype, डिवाइस और realized है या नहीं, कभी डेटा नहीं, क्योंकि उसके लिए डिवाइस से पढ़ना पड़ेगा। Negative axes हर जगह अंत से गिने जाते हैं।

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

एक linear लेयर `y = x @ W.T + b` कम्प्यूट करती है। Svod में `nn::Linear` बिल्ट-इन है।

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

**मुख्य कॉन्सेप्ट:**

1. **`sequential()`** लेयर्स को चेन करता है: हर लेयर का आउटपुट अगली लेयर का इनपुट बनता है। मैन्युअल वायरिंग की ज़रूरत नहीं।

2. **ReLU एक्टिवेशन:** `Relu` एक ज़ीरो-साइज़ लेयर है जो `max(0, x)` अप्लाई करती है। यह नॉन-लीनियरिटी लाती है — इसके बिना, linear लेयर स्टैक करना बस एक बड़ी linear लेयर होगी।

3. **Logits बनाम probabilities:** लास्ट लेयर का रॉ आउटपुट (logits) कोई भी रियल नंबर हो सकता है। `softmax()` उन्हें probabilities में बदलता है जिनका योग 1 होता है।

4. **argmax:** मैक्सिमम वैल्यू का इंडेक्स रिटर्न करता है — यानी प्रेडिक्टेड क्लास।

5. **Batch डायमेंशन:** हम सिंगल इमेज के लिए shape `[1, 784]` इस्तेमाल करते हैं। 32 इमेज के लिए `[32, 784]` इस्तेमाल करें। मॉडल बैच ऑटोमैटिकली हैंडल करता है।

6. **`realize_batch`:** जो दो रिज़ल्ट एक ही सबग्राफ़ शेयर करते हैं (यहाँ logits) वे साथ में कम्पाइल और रन होते हैं, इसलिए शेयर्ड हिस्सा एक ही बार कम्प्यूट होता है। यह शेयर्ड रेफ़रेंस लेता है — `[&a, &b]` — क्योंकि realization tensor रजिस्ट्री में दर्ज होता है, हैंडल में नहीं।

---

## उदाहरण 6: अंदर की बात

जानना चाहते हैं कि Svod क्या जनरेट करता है? IR और कम्पाइल किए गए कर्नेल कैसे देखें, यह रहा।

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

**आपको क्या दिखेगा:**

1. **IR Graph:** UOp tree `BUFFER`, `LOAD`, `ADD`, `STORE` जैसे ऑपरेशन दिखाता है। यह ऑप्टिमाइज़ेशन से पहले Svod का इंटरमीडिएट रिप्रेज़ेंटेशन है।

2. **एक्ज़ीक्यूशन प्लान:** `prepare()` कम्पाइल किए गए कर्नेल लौटाता है। ध्यान दें कि Svod दोनों loads और add को एक ही कर्नेल में फ़्यूज़ करता है — कोई इंटरमीडिएट बफ़र नहीं चाहिए।

**डीबगिंग टिप:** अगर कुछ स्लो या गलत लगे, तो IR tree प्रिंट करें। देखें:
- अनएक्सपेक्टेड ऑपरेशन (रिडंडेंट reshapes, एक्स्ट्रा कॉपीज़)
- मिसिंग फ़्यूज़न (जहाँ एक कर्नेल काफ़ी हो वहाँ अलग-अलग कर्नेल)
- Shape मिसमैच (अक्सर एरर की असली वजह)

---

## उदाहरण 7: लेयर, मॉड्यूल और state dict

एक लेयर struct अपने पैरामीटर के साथ-साथ वे हाइपर-पैरामीटर भी रखता है जो उसके
forward को चाहिए। `#[derive(Module)]` उन फ़ील्ड्स को एक फ़्लैट `StateDict`
(`HashMap<String, Tensor>`) में बदल देता है, जिसकी keys ठीक वैसी होती हैं जैसे
PyTorch उन्हें नाम देता है, ताकि चेकपॉइंट बिना हाथ से लिखी किसी मैपिंग के लोड हो जाए:

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

| एट्रिब्यूट | असर |
|-----------|------|
| `#[module(key = "Wi.weight")]` | फ़ील्ड-नाम वाले key सेगमेंट को बदलें (इसमें डॉट और अंक हो सकते हैं) |
| `#[module(key = "")]` | फ़्लैटन: फ़ील्ड की keys पैरेंट प्रिफ़िक्स को बिना बदले इस्तेमाल करती हैं |
| `#[module(skip)]` | किसी नॉन-प्रिमिटिव फ़ील्ड को नज़रअंदाज़ करें (config, dtype, mode) |
| `#[module(optional)]` | `Option<Tensor>` पर ज़रूरी: `Some` होने पर सेव होता है, लोड करते समय key का न होना चल जाता है |
| `#[module(optional = "self.has_bias")]` | प्रेडिकेट सही होने पर key ज़रूरी है, वरना छोड़ दी जाती है |

Children ब्लैंकेट impl के ज़रिए कम्पोज़ होते हैं: `Vec<Block>` अपने एलिमेंट को `0.`,
`1.`, … से key करता है, और array, `Option`, tuple तथा `Box` भी उसी तरह डेलिगेट
करते हैं। Enum पर भी derive चलता है। Forward पास `Module` से बाहर रहता है: जहाँ
सिग्नेचर इजाज़त दे वहाँ वह `Layer` ट्रेट (`fn forward(&self, x: &Tensor) ->
Result<Tensor>`) में रहता है, और बाक़ी जगह inherent मेथड में।

बिल्ट-इन लेयर दोनों ट्रेट इम्प्लीमेंट करती हैं, जिनमें लोड किए गए tensor के लिए
`new` है और ताज़ा Kaiming-uniform (convolution) या identity-affine
(normalization) इनिशियलाइज़ेशन के लिए `with_dims`:

| लेयर | `with_dims` | State-dict keys |
|-------|-------------|-----------------|
| `Linear` | `(in, out, bias, dtype)` | `weight`, `bias` (जब मौजूद हो) |
| `Conv1d` | `(in_c, out_c, kernel, bias, dtype)` | `weight`, `bias` |
| `Conv2d` / `ConvTranspose2d` | `(in_c, out_c, (kh, kw), bias, dtype)` | `weight`, `bias` |
| `BatchNorm2d` | `(channels, eps, dtype)` | `weight`, `bias`, `running_mean`, `running_var` |
| `LayerNorm` | `(size, bias, eps, dtype)` | `weight`, `bias` (जब मौजूद हो) |
| `RmsNorm` | `(size, eps, dtype)` | `weight` |
| `Embedding` | `(vocab_size, embed_dim, dtype)` | `weight` |

हाइपर-पैरामीटर struct पर builder-स्टाइल `with_*` मेथड से सेट होते हैं —
`Conv1d::new(w, bias).with_stride(2).with_padding((1, 1)).with_groups(4)`,
`LayerNorm::with_dims(..).with_axis(-2)`।

---

## उदाहरण 8: Recurrent लेयर

`rnn()`, `gru()` और `lstm()` `Tensor` पर बिल्डर हैं। ये या तो PyTorch वाले weight
नाम (`weight_ih`, `weight_hh`, `bias_ih`, `bias_hh`, `h0`, `c0`) लेते हैं या ONNX
वाले (`w`, `r`/`r_weights`, `bias`, `initial_h`, `initial_c`), और gate ब्लॉक का
क्रम आपके लिए ठीक कर देते हैं:

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

`layout` `RnnLayout::SeqFirst` (`[seq, batch, input]`, डिफ़ॉल्ट) या `BatchFirst`
चुनता है; `direction` `RnnDirection::{Forward, Backward, Bidirectional}` लेता है,
और एक bidirectional पास दोनों दिशाओं को feature axis पर concatenate करता है। GRU
का `linear_before_reset` PyTorch weights के साथ PyTorch वाली placement पर और ONNX
weights के साथ ONNX वाली पर डिफ़ॉल्ट होता है। `LstmOutput` cell state के लिए
`y_c` / `c_n` भी जोड़ता है।

Time axis कंक्रीट होना चाहिए, लेकिन batch axis एक symbolic `Variable` हो सकता है।
हाथ से लिखे लूप के लिए — जैसे एक बार में एक टोकन आगे बढ़ाने वाला decoder — सीधे
cells इस्तेमाल करें: `GruCell`/`LstmCell`/`RnnCell` `step(&x, &h) -> Result<..>`
एक्सपोज़ करते हैं, और `RnnStack::new(cells)` पूरे स्टैक को एक साथ स्टेप करता है।

---

## उदाहरण 9: Spectrogram

`stft()` एक windowed DFT कर्नेल के विरुद्ध एक ही `conv1d` है, इसलिए पूरा ट्रांसफ़ॉर्म
ग्राफ़ के अंदर रहता है (और batch axis symbolic बना रह सकता है)। नतीजा
`[B, F, T, 2]` होता है — या unbatched `[L]` सिग्नल के लिए `[F, T, 2]` — जिसके
आख़िरी axis पर `(real, imag)` होते हैं, ठीक
`torch.stft(..., return_complex=false)` की तरह:

```rust
use svod_tensor::Tensor;
use svod_tensor::nn::Window;

let x = Tensor::from_slice(vec![0.25f32; 64]);
let spec = x.stft().n_fft(16).hop(4).window(Window::Hann).call()?;
assert_eq!(spec.dims()?, vec![9, 17, 2]);   // [F, T, (re, im)]

let mag = spec.magnitude(0.0)?;             // sqrt(re² + im² + eps)
let signal = spec.istft().n_fft(16).hop(4).window(Window::Hann).length(64).call()?;
```

डिफ़ॉल्ट torch जैसे ही हैं: `hop = n_fft / 4`, `win_length = n_fft`, एक periodic
Hann window, `center`, `onesided`, कोई normalization नहीं — और `istft` को भी वही
देने होते हैं। `Window` `Hann`, `Hamming`, `Rectangular` या `Custom(tensor)` होता
है, और `Tensor::window(&Window::Hann, n, periodic, dtype)` एक window मटीरियलाइज़
करता है। `magnitude` के अलावा, आख़िरी 2-वाले axis पर `power`, `complex_abs`,
`complex_mul` और `Tensor::complex_from_polar(&mag, &phase)` भी हैं।

Mel front-end वही ग्राफ़ है, बस अंत में एक filterbank contraction और एक log जुड़
जाता है। `mel_spectrogram()` `stft` के पैरामीटर और साथ में mel वाले पैरामीटर लेता है
और `[B, n_mels, T]` लौटाता है:

```rust
use svod_tensor::nn::{MelLog, MelNorm, MelScale};

let x = Tensor::from_slice(vec![0.25f32; 16000]);
let mel = x
    .mel_spectrogram()
    .sample_rate(16000)
    .n_fft(400)
    .hop(160)
    .n_mels(80)
    .mel_scale(MelScale::Slaney)
    .norm(MelNorm::Slaney)
    .log(MelLog::Whisper)
    .call()?;
assert_eq!(mel.dims()?, vec![80, 101]);
```

डिफ़ॉल्ट torchaudio के `MelSpectrogram` वाले हैं (HTK स्केल, कोई normalization
नहीं, `power = 2`, `f_min = 0`, `f_max = sample_rate / 2`, कोई log नहीं);
`MelScale::Slaney` के साथ `MelNorm::Slaney` `librosa.filters.mel` है, यानी Whisper
के पीछे का filterbank। `MelLog::Ln { min, max }` `ln(clamp(x))` है और
`MelLog::Whisper` `log_mel_spectrogram` की `log10` / `max - 8` पर floor /
`(x + 4) / 4` वाली पूँछ; `mel_log` इनमें से किसी को भी अकेले लगाता है, और
`Tensor::mel_filterbank(...)` `[n_mels, F]` टेबल को materialize करता है।

---

## एरर

हर fallible tensor मेथड `svod_tensor::error::Result<T>` लौटाता है, जिसका एरर एक
पॉइंटर-साइज़ `Error(Box<ErrorKind>)` है; कारण पर `err.kind()` के ज़रिए मैच करें (या
उसे वैल्यू से लेने के लिए `into_kind()`)। डाउनस्ट्रीम crate इसे snafu के
`context(false)` से कन्वर्ट करते हैं, इसलिए किसी मॉडल का अपना एरर enum इसे सादे
`?` से सोख लेता है — हर कॉल साइट पर `.context(TensorSnafu)` की ज़रूरत नहीं।

सब कुछ fallible नहीं है। `cast`, `neg`, `abs`, `floor`, `ceil`, `round`,
`trunc`, `square`, `sign` और `Tensor::full` / `zeros` / `ones` कंस्ट्रक्टर फ़ेल
नहीं हो सकते और सादा `Tensor` लौटाते हैं; `-&a` भी वैसे ही सादा है, जबकि binary
ऑपरेटर `Result<Tensor>` लौटाते हैं।

---

## सारांश

आपने Svod इस्तेमाल करने के कोर पैटर्न सीख लिए:

| टास्क | कोड |
|-------|------|
| Tensor बनाएँ | `Tensor::from_slice([1.0f32, 2.0])` |
| अरिथमेटिक | `(&a + &b)?`, `(&a * 2.0)?`, `(2.0 * &a)?`, `-&a` |
| Reshape | `t.try_reshape(&[2, 3])?` |
| Transpose | `t.try_transpose(0, 1)?` |
| मैट्रिक्स मल्टिप्लाई | `a.dot(&b)?` |
| निरीक्षण | `t.dims()?`, `t.dim_const(-1)?`, `t.dtype()` |
| Linear लेयर | `Linear::with_dims(in, out, bias, dtype)` |
| लेयर चेन करें | `x.sequential(&[&fc1, &Relu, &fc2])?` |
| एक्टिवेशन | `t.relu()?`, `t.softmax(-1)?` |
| Weights लोड करें | `model.load_state_dict(&sd, "")?` |
| Spectrogram | `x.stft().n_fft(512).hop(160).call()?` |
| Mel spectrogram | `x.mel_spectrogram().sample_rate(16000).n_fft(400).n_mels(80).call()?` |
| Recurrent लेयर | `x.lstm().weight_ih(&w).weight_hh(&r).hidden_size(h).call()?` |
| एक्ज़ीक्यूट करें | `t.realize()?` |
| बैच realize | `Tensor::realize_batch([&a, &b])?` |
| डेटा निकालें | `t.to_vec::<f32>()?`, `t.to_ndarray::<f32>()?`, `t.item::<f32>()?` |

**Lazy evaluation पैटर्न:**

1. ऑपरेशन से अपना कम्प्यूटेशन ग्राफ़ बनाएँ
2. अंत में एक बार `realize()` कॉल करें
3. Svod सब कुछ ऑप्टिमाइज़ और एक साथ एक्ज़ीक्यूट करता है

**आगे:**

- [Op Bestiary](./architecture/op-bestiary) — IR ऑपरेशन रेफ़रेंस
- [एक्ज़ीक्यूशन पाइपलाइन](./architecture/pipeline) — कम्पाइलेशन कैसे काम करता है
- [पैटर्न इंजन](./architecture/optimizations/pattern-system) — पैटर्न-आधारित रीराइट्स
