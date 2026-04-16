---
sidebar_label: ONNX इन्फ़रेंस
---

# ONNX मॉडल इन्फ़रेंस

Morok का ONNX इम्पोर्टर मॉडल इन्फ़रेंस का सबसे अच्छा तरीका है। यह स्टैंडर्ड `.onnx` फ़ाइलें लोड करता है, ऑपरेटरों को Morok के lazy tensor ऑपरेशनों में तोड़ता है, और पूरी ऑप्टिमाइज़ेशन पाइपलाइन से कम्पाइल करता है — कोई C++ रनटाइम नहीं चाहिए।

**वर्तमान स्थिति:**

| क्षमता | स्थिति |
|--------|--------|
| फ़ॉरवर्ड इन्फ़रेंस | समर्थित |
| 162 / 200 ONNX ऑपरेटर | [पैरिटी विवरण](https://github.com/patsak/morok/blob/main/onnx/PARITY.md) |
| CNN आर्किटेक्चर (ResNet, DenseNet, VGG, ...) | 9 मॉडल सत्यापित |
| Microsoft एक्सटेंशन (Attention, RotaryEmbedding) | समर्थित |
| डायनामिक बैच साइज़ | समर्थित (Variable API) |
| ट्रेनिंग / बैकवर्ड पास | समर्थित नहीं |

**दूसरे फ़्रेमवर्क से तुलना**

Pure-Rust फ़्रेमवर्कों में Morok का ONNX ऑपरेटर कवरेज सबसे ज़्यादा है — 162 ऑपरेटर, दोनों बैकएंड (Clang + LLVM) पर 1361 पासिंग conformance टेस्ट। `candle` और `burn` में ऑपरेटर कम हैं और इतने बड़े टेस्ट सूट नहीं हैं। अगर प्रोडक्शन ONNX मॉडलों के साथ पूरी कम्पैटिबिलिटी चाहिए, तो `ort` इस्तेमाल करें — C++ ONNX Runtime का Rust रैपर, जो पूरा ONNX स्पेक कवर करता है।

---

## त्वरित शुरुआत

अपनी `Cargo.toml` में `morok-onnx` और `morok-tensor` जोड़ें:

```toml
[dependencies]
morok-onnx = { git = "https://github.com/patsak/morok" }
morok-tensor = { git = "https://github.com/patsak/morok" }
```

### सरल: ऑल-इनिशियलाइज़र मॉडल

उन मॉडलों के लिए जहाँ सभी इनपुट फ़ाइल में अंतर्निहित हैं (कोई रनटाइम इनपुट नहीं):

```rust
use morok_onnx::{OnnxImporter, OnnxModel};
use morok_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut importer = OnnxImporter::new();
    let OnnxModel { mut outputs, .. } = importer.import("model.onnx", &[])?;

    // सभी आउटपुट एक साथ शेड्यूल करें, एक ही पास में एक्ज़ीक्यूट करें
    let mut outs: Vec<&mut Tensor> = outputs.values_mut().collect();
    Tensor::realize_batch(&mut outs)?;

    for (name, tensor) in &outputs {
        println!("{name}: {:?}", tensor.as_ndarray::<f32>()?);
    }
    Ok(())
}
```

### रनटाइम इनपुट वाले मॉडल

अधिकांश मॉडलों को रनटाइम डेटा (इमेज, टोकन, ऑडियो) की आवश्यकता होती है। `OnnxModel` को destructure करें और इनपुट tensor की ownership लेने के लिए `remove()` इस्तेमाल करें:

```rust
use morok_onnx::{OnnxImporter, OnnxModel};
use morok_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut importer = OnnxImporter::new();
    let OnnxModel { mut inputs, mut outputs, .. } = importer.import("model.onnx", &[])?;

    // इनपुट डेटा असाइन करें (lazy — अभी कोई एलोकेशन नहीं)
    let input = inputs.remove("input").unwrap();
    input.assign(&Tensor::from_slice(&my_data));

    // सभी आउटपुट एक साथ शेड्यूल करें, एक ही पास में एक्ज़ीक्यूट करें
    // (इनपुट के assign को इंटरनली रिज़ॉल्व करता है — अलग से realize की ज़रूरत नहीं)
    let mut outs: Vec<&mut Tensor> = outputs.values_mut().collect();
    Tensor::realize_batch(&mut outs)?;
    Ok(())
}
```

---

## आर्किटेक्चर

### दो-चरणीय डिज़ाइन

इम्पोर्टर ONNX मॉडलों को दो अलग-अलग चरणों में प्रोसेस करता है:

**`import(path, dim_bindings)`** दोनों चरणों को एक ही कॉल में करता है: protobuf को पार्स करता है, इनिशियलाइज़र और इनपुट स्पेक्स निकालता है, ग्राफ़ को टोपोलॉजिकल क्रम में ट्रैवर्स करते हुए प्रत्येक ONNX नोड को उसके Tensor इम्प्लीमेंटेशन पर डिस्पैच करता है, और एक `OnnxModel { inputs, outputs, variables }` लौटाता है। कोई एक्ज़ीक्यूशन नहीं होता — रिज़ल्ट lazy `Tensor` हैंडल का सेट है जो `realize()` करने पर कम्पाइल और एक्ज़ीक्यूट होता है।

```text
model.onnx → import(path, dims) → OnnxModel { inputs, outputs, variables } → realize() → results
```

एडवांस्ड यूज़ केस के लिए (इम्पोर्ट से पहले ग्राफ़ स्ट्रक्चर जाँचना), `import_model()` एक पहले से पार्स्ड `ModelProto` स्वीकार करता है।

### ऑपरेटर विघटन

हर ONNX ऑपरेटर Morok Tensor ऑपरेशनों में टूटता है। कॉम्प्लेक्सिटी अलग-अलग होती है:

**प्रत्यक्ष मैपिंग** — लगभग 60 ऑपरेटर एक tensor मेथड पर 1:1 मैप होते हैं:

```rust
// In the registry:
"Add" => x.try_add(y)?
"Relu" => x.relu()?
"Sigmoid" => x.sigmoid()?
"Equal" => x.try_eq(y)?
```

**बिल्डर पैटर्न** — कई वैकल्पिक पैरामीटर वाले जटिल ऑपरेटर fluent API का उपयोग करते हैं:

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

**मल्टी-स्टेप डीकम्पोज़िशन** — BatchNormalization, Attention, और Mod जैसे ऑपरेटरों को बीच में कई कैलकुलेशन करनी पड़ती हैं। जैसे, Python-स्टाइल integer `Mod` ट्रंकेशन mod + साइन एडजस्टमेंट में टूटता है:

```rust
let trunc_mod = x.try_mod(y)?;
let signs_differ = trunc_mod.bitwise_xor(y)?.try_lt(&zero)?;
let needs_adj = mod_ne_zero.bitwise_and(&signs_differ)?;
trunc_mod.try_add(&y.where_(&needs_adj, &zero)?)?
```

### एट्रिब्यूट वैलिडेशन

`Attrs` हेल्पर पॉप-आधारित एक्सट्रैक्शन का उपयोग करता है — `attrs.int("axis", -1)` या `attrs.float("epsilon", 1e-5)` के प्रत्येक कॉल पर एट्रिब्यूट मैप से हटा दिया जाता है। ऑपरेटर पूरा होने के बाद, `attrs.done()` सुनिश्चित करता है कि मैप खाली है। कोई भी बचा हुआ एट्रिब्यूट एक एरर ट्रिगर करता है, जो अधूरे ऑपरेटर इम्प्लीमेंटेशन को चुपचाप गलत परिणाम देने के बजाय ट्रेस टाइम पर ही पकड़ लेता है।

### Opset वर्शनिंग

ONNX मॉडल प्रति डोमेन opset इम्पोर्ट घोषित करते हैं। इम्पोर्टर इन्हें ट्रैक करता है और प्रत्येक ऑपरेटर हैंडलर को वर्शन पास करता है। ऑपरेटर वर्शन के आधार पर व्यवहार बदलते हैं — उदाहरण के लिए, `Softmax` का डिफ़ॉल्ट axis `1` (opset < 13) से `-1` (opset >= 13) में बदल गया, और `ReduceSum` ने अपने axes को opset 13 पर एक एट्रिब्यूट से इनपुट tensor में स्थानांतरित किया।

---

## मॉडलों के साथ काम करना

### डायनामिक डायमेंशन

ONNX इनपुट में `"batch_size"` या `"sequence_length"` जैसे सिम्बॉलिक डायमेंशन हो सकते हैं। इन्हें इम्पोर्ट टाइम पर `dim_bindings` पैरामीटर के ज़रिए बाइंड करें:

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

अनबाउंड डायनामिक डायमेंशन इम्पोर्ट टाइम पर एक स्पष्ट एरर देते हैं। आप `InputSpec::shape` के ज़रिए जाँच सकते हैं कि कौन से डायमेंशन डायनामिक हैं:

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

### एक्सटर्नल वेट्स

कुछ ONNX मॉडल वेट्स अलग फ़ाइलों में स्टोर करते हैं। इन्हें प्रदान करने के लिए `import_model_with_inputs()` का उपयोग करें:

```rust
let model = importer.import_model_with_inputs(
    "model.onnx",
    &[],
    external_weights,  // HashMap<String, Tensor>
)?;
```

### Microsoft एक्सटेंशन

इम्पोर्टर कई `com.microsoft` contrib ऑपरेटरों को सपोर्ट करता है जो आमतौर पर ONNX Runtime से एक्सपोर्ट किए गए ट्रांसफ़ॉर्मर मॉडलों में पाए जाते हैं:

| एक्सटेंशन | विवरण |
|-----------|-------|
| `Attention` | पैक्ड QKV प्रोजेक्शन, मास्किंग, पास्ट KV cache के साथ |
| `RotaryEmbedding` | रोटरी पोज़िशनल एम्बेडिंग (इंटरलीव्ड/नॉन-इंटरलीव्ड) |
| `SkipLayerNormalization` | फ़्यूज़्ड रेसिड्यूअल + LayerNorm + स्केल |
| `EmbedLayerNormalization` | टोकन + पोज़िशन + सेगमेंट एम्बेडिंग → LayerNorm |

मानक ONNX ट्रांसफ़ॉर्मर ऑपरेटर (ai.onnx डोमेन से `Attention`) भी GQA, कॉज़ल मास्किंग, पास्ट KV cache, और softcap के साथ समर्थित हैं।

---

## कंट्रोल फ़्लो और सीमाएँ

### सिमैंटिक If: दोनों ब्रांच हमेशा एक्ज़ीक्यूट होती हैं

ONNX के `If` ऑपरेटर में डेटा-डिपेंडेंट कंट्रोल फ़्लो होता है — कंडीशन तय करती है कि कौन सी ब्रांच चलेगी। Morok का lazy इवैल्यूएशन मॉडल इसके साथ मौलिक रूप से असंगत है: चूँकि ट्रेस टाइम पर कुछ भी एक्ज़ीक्यूट नहीं होता, कंडीशन का मान अज्ञात होता है।

**Morok का समाधान:** *दोनों* ब्रांचों को ट्रेस करें, फिर `Tensor::where_()` से परिणामों को मर्ज करें:

```text
ONNX:    if condition { then_branch } else { else_branch }
Morok:   then_result.where_(&condition, &else_result)
```

यह **एक बार ट्रेस करो, कई बार चलाओ** सक्षम करता है — कम्पाइल्ड ग्राफ़ रनटाइम पर किसी भी कंडीशन वैल्यू को हैंडल करता है। लेकिन इसकी एक कठोर बाधा है: **दोनों ब्रांचों को समान आउटपुट शेप और DType प्रोड्यूस करना चाहिए।** शेप-पॉलीमॉर्फ़िक ब्रांचों वाले मॉडल (जहाँ then-ब्रांच `[3, 4]` और else-ब्रांच `[5, 6]` प्रोड्यूस करती है) को ट्रेस नहीं किया जा सकता।

व्यवहार में, `If` नोड वाले अधिकांश ONNX मॉडल इस बाधा को पूरा करते हैं क्योंकि वे कंडीशनल लॉजिक का उपयोग वैल्यू सिलेक्शन के लिए करते हैं, शेप-बदलने वाले कंट्रोल फ़्लो के लिए नहीं।

### कोई Loop या Scan नहीं

इटरेटिव कंट्रोल फ़्लो (`Loop`, `Scan`) इम्प्लीमेंट नहीं है। इन ऑपरेटरों को बार-बार ट्रेसिंग या अनरोलिंग की आवश्यकता होती है, जो सिंगल-ट्रेस आर्किटेक्चर से टकराता है। रिकरेंट पैटर्न उपयोग करने वाले मॉडल आमतौर पर अनरोल्ड ऑपरेटरों के ज़रिए काम करते हैं (LSTM, GRU, RNN नेटिव ops के रूप में इम्प्लीमेंट हैं)।

### बैच एक्ज़ीक्यूशन

कई tensor को एक साथ realize किया जा सकता है, जिससे आउटपुट के बीच कम्प्यूटेशन शेयर होता है
(`tensor/src/test/unit/batch.rs` में टेस्ट किया गया):

```rust
// Realize all outputs at once (shares compilation and execution)
let mut outputs: Vec<&mut Tensor> = model.outputs.values_mut().collect();
Tensor::realize_batch(&mut outputs)?;
```

बार-बार इन्फ़रेंस के लिए, prepare/execute पैटर्न इस्तेमाल करें
(`tensor/src/test/unit/variable.rs::test_prepare_execute_loop` में टेस्ट किया गया):

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

### कोई ट्रेनिंग नहीं

इम्पोर्टर केवल इन्फ़रेंस के लिए है। कोई बैकवर्ड पास, ग्रेडिएंट कम्प्यूटेशन, या ऑप्टिमाइज़र सपोर्ट नहीं है।

### अनुपलब्ध ऑपरेटर श्रेणियाँ

| श्रेणी | उदाहरण | कारण |
|--------|--------|------|
| क्वांटाइज़ेशन | DequantizeLinear, QuantizeLinear | IR में क्वांटाइज़्ड DType सपोर्ट आवश्यक |
| सीक्वेंस ऑप्स | SequenceConstruct, SequenceAt | नॉन-tensor टाइप Morok के टाइप सिस्टम में नहीं हैं |
| रैंडम | RandomNormal, RandomUniform | स्टेटफ़ुल RNG अभी तक इम्प्लीमेंट नहीं |
| सिग्नल प्रोसेसिंग | DFT, STFT, MelWeightMatrix | कम प्राथमिकता; विशिष्ट उपयोग |
| टेक्स्ट | StringNormalizer, TfIdfVectorizer | स्ट्रिंग टाइप समर्थित नहीं |

इन ऑपरेटरों वाले मॉडलों के लिए `ort` (ONNX Runtime रैपर) इस्तेमाल करें, जो पूरा स्पेक कवर करता है।

---

## डीबगिंग

### प्रति-नोड आउटपुट ट्रेसिंग

मध्यवर्ती आउटपुट डंप करने के लिए ट्रेस लॉग लेवल सेट करें:

```bash
RUST_LOG=morok_onnx::importer=trace cargo run
```

यह प्रत्येक नोड के आउटपुट को अलग-अलग realize करता है और पहले 5 मान प्रिंट करता है — जब कोई मॉडल गलत परिणाम देता है तो न्यूमेरिकल बाइसेक्शन के लिए उपयोगी है। ध्यान दें कि यह कर्नेल फ़्यूज़न को तोड़ता है (प्रत्येक नोड अलग से चलता है), इसलिए यह पूरी तरह से एक डीबगिंग टूल है।

### ग्राफ़ का निरीक्षण

मॉडल को क्या चाहिए, यह जानने के लिए `OnnxModel` स्ट्रक्चर इस्तेमाल करें:

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

## सारांश

| पहलू | विवरण |
|------|-------|
| **एंट्री पॉइंट** | `OnnxImporter::new()` |
| **सरल इम्पोर्ट** | `importer.import("model.onnx", &[])?` |
| **डायनामिक dims** | `importer.import(path, &[("batch", 4)])?` |
| **ऑपरेटर** | 162 / 200 ([पूर्ण पैरिटी तालिका](https://github.com/patsak/morok/blob/main/onnx/PARITY.md)) |
| **सत्यापित मॉडल** | ResNet50, DenseNet121, VGG19, Inception, AlexNet, ShuffleNet, SqueezeNet, ZFNet |
| **बैकएंड** | Clang + LLVM (समान परिणाम) |
| **एक्सटेंशन** | com.microsoft Attention, RotaryEmbedding, SkipLayerNorm, EmbedLayerNorm |
| **सीमाएँ** | कोई ट्रेनिंग नहीं, कोई Loop/Scan नहीं, शेप-पॉलीमॉर्फ़िक If |

**आगे:** [प्रैक्टिकल उदाहरण](./examples) — tensor बेसिक्स, या [एक्ज़ीक्यूशन पाइपलाइन](./architecture/pipeline) — कम्पाइलेशन कैसे काम करता है।
