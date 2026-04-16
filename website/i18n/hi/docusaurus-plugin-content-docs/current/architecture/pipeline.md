---
sidebar_label: एक्ज़ीक्यूशन पाइपलाइन
---

# Tensor से मशीन कोड तक

ज़्यादातर ML फ़्रेमवर्क में कम्प्यूटेशन तुरंत होता है। PyTorch में `a + b` लिखें और यह *अभी* चलता है — GPU नंबर क्रंच करता है इससे पहले कि आप रिज़ल्ट देख सकें। यह eager execution समझने में आसान है, लेकिन ऑप्टिमाइज़ेशन के मौके छूट जाते हैं। कम्पाइलर उस कम्प्यूटेशन को कैसे ऑप्टिमाइज़ करे जो उसने अभी देखा ही नहीं?

Morok उलटा तरीका अपनाता है: **lazy evaluation**। जब आप `a.try_add(&b)?` लिखते हैं, कुछ भी कम्प्यूट नहीं होता। Morok एक ग्राफ़ बनाता है जो बताता है *क्या* कम्प्यूट करना है, *कब* नहीं। जादू तब होता है जब आप `realize()` कॉल करते हैं — वो एक मेथड पूरी कम्पाइलेशन पाइपलाइन ट्रिगर करता है, हाई-लेवल tensor ऑपरेशनों से लेकर JIT-कम्पाइल्ड मशीन कोड तक।

यह चैप्टर उस पूरी यात्रा को ट्रेस करता है।

```text
tensor.realize()
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  LAZY GRAPH                                             │
│  Tensor ops build UOp DAG (no computation yet)          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  RANGEIFY                                               │
│  Movement ops → explicit RANGE loops                    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  KERNEL SPLITTING                                       │
│  Split at STORE boundaries → multiple KERNELs          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  OPTIMIZATION & CODEGEN                                 │
│  Heuristics/beam → LLVM IR → JIT compile               │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  EXECUTION                                              │
│  Parallel kernel launch → result buffer                │
└─────────────────────────────────────────────────────────┘
```

हर बॉक्स एक अलग फ़ेज़ है। चलिए इन्हें एक-एक करके देखते हैं।

---

## Lazy Evaluation: ग्राफ़ बनाना

Morok में `Tensor` काफ़ी हल्का होता है:

```rust
pub struct Tensor {
    entry: Arc<TensorEntry>,      // Computation graph
    buffer: Option<Arc<Buffer>>,  // Materialized data (if any)
}
```

`entry` में `TensorEntry` होता है जिसमें UOp ग्राफ़ है — वो कम्प्यूटेशन जो यह tensor रिप्रेज़ेंट करता है। `buffer` ऑप्शनल है: lazy tensor के पास नहीं होता, सिर्फ़ realized tensor के पास होता है।

### Tensor बनाने के तीन तरीके

**1. इनपुट tensor** — बफ़र तुरंत एलोकेट होता है:

```rust
let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
// `a.buffer` = Some(Arc<Buffer>) with actual data
```

जब आप डेटा से tensor बनाते हैं, Morok डिवाइस मेमोरी एलोकेट करता है और आपके bytes कॉपी करता है। UOp ग्राफ़ में एक `BUFFER` नोड होता है जो इस एलोकेशन को पॉइंट करता है।

**2. Lazy ऑपरेशन** — कोई बफ़र नहीं, सिर्फ़ ग्राफ़:

```rust
let b = a.try_add(&a)?;   // b.buffer = None
let c = b.try_mul(&a)?;   // c.buffer = None
```

अरिथमेटिक ऑपरेशन कुछ भी कम्प्यूट नहीं करते। ये एक UOp ग्राफ़ बनाते हैं: `Binary(Add, a.uop, a.uop)`। Tensor पूरी तरह से भविष्य के काम की डिस्क्रिप्शन के रूप में मौजूद होता है।

**3. Movement ऑपरेशन** — ओरिजिनल बफ़र शेयर करता है:

```rust
let d = a.try_reshape(&[1, 3])?;  // d.buffer = same as a.buffer
```

Reshape, permute, और इसी तरह के ऑपरेशन मौजूदा डेटा के नए *views* बनाते हैं। बफ़र शेयर होता है; सिर्फ़ UOp ग्राफ़ बदलता है नई indexing को डिस्क्राइब करने के लिए।

### ग्लोबल रजिस्ट्री

Morok तीन ग्लोबल maps मेंटेन करता है (lock-free, thread-safe):

| Map | Key → Value | उद्देश्य |
|-----|-------------|---------|
| `TENSORS` | tensor_id → `Weak<TensorEntry>` | ग्राफ़ सब्स्टिट्यूशन के लिए सभी tensor ट्रैक करें |
| `BUFFERS` | uop_id → `Arc<Buffer>` | शेड्यूलिंग के दौरान बफ़र खोजें |

यह रजिस्ट्री एक ज़रूरी फ़ीचर देती है: **ग्लोबल ग्राफ़ सब्स्टिट्यूशन**। जब ऑप्टिमाइज़ेशन कोई UOp ट्रांसफ़ॉर्म करता है, उस UOp को रेफ़रेंस करने वाले सभी tensor ऑटोमैटिकली अपडेटेड वर्शन देखते हैं। कोई stale रेफ़रेंस नहीं, कोई मैन्युअल अपडेट नहीं।

### Hash Consing इन एक्शन

क्योंकि UOps hash consing (content-based deduplication) इस्तेमाल करते हैं, आइडेंटिकल कम्प्यूटेशन मेमोरी शेयर करते हैं:

```rust
let x = a.try_add(&b)?;
let y = a.try_add(&b)?;
// x.uop() and y.uop() point to the SAME Arc<UOp>
```

यह कैशिंग के लिए ज़रूरी है: जब हम कर्नेल कम्पाइल करते हैं, UOp ID से कैश करते हैं। Hash consing से आइडेंटिकल कम्प्यूटेशन ऑटोमैटिकली कैश हिट करते हैं, भले ही अलग-अलग बनाए गए हों।

---

## Rangeify: लूप्स को एक्सप्लिसिट बनाना

जब आप `tensor.reshape([2, 3]).expand([4, 2, 3]).sum(axis=0)` लिखते हैं, ये movement ऑपरेशन (reshape, expand) हाई-लेवल डिस्क्रिप्शन हैं। असल लूप जनरेट करने के लिए, हमें एक्सप्लिसिट इटरेशन स्ट्रक्चर चाहिए।

**Rangeify** movement ऑपरेशन को `RANGE` लूप्स और `INDEX` अरिथमेटिक में ट्रांसफ़ॉर्म करता है। एंट्री पॉइंट `schedule/src/rangeify/transforms.rs` में `rangeify()` है।

### Rangeify पाइपलाइन

Rangeify सिंगल ट्रांसफ़ॉर्मेशन नहीं है — यह एक मल्टी-स्टेज पाइपलाइन है:

| स्टेज | उद्देश्य |
|-------|---------|
| **0. Range असाइनमेंट** | हर tensor डायमेंशन के लिए RANGE UOps बनाएँ |
| **1. Early Movement Ops** | Range असाइनमेंट से पहले movement ऑपरेशन साफ़ करें |
| **2. Load Collapse** | Range-इंडिपेंडेंट डिटेक्शन से REDUCE ऑपरेशन एलिमिनेट करें |
| **3. Split Ranges** | Modulo वाली ranges को स्प्लिट करें, ranges को flatten करें |
| **4. Initial Symbolic** | एल्जेब्रिक सिम्प्लिफ़िकेशन, कॉन्स्टेंट फ़ोल्डिंग |
| **5. Simplify Ranges** | कॉस्ट एनालिसिस के साथ एडजेसेंट ranges को मर्ज करें |
| **6. Split Store** | STORE बाउंड्रीज़ पर ग्राफ़ स्प्लिट करें |
| **7. Apply Opts** | ऑप्टिमाइज़ेशन सर्च (beam या heuristic) |
| **Mega-pass** | Symbolic + reduce + बफ़र फ़ोल्डिंग + बफ़र रिमूवल + रिडक्शन सिम्प्लिफ़िकेशन |

Mega-pass कई symbolic और स्ट्रक्चरल ऑप्टिमाइज़ेशन को एक सिंगल fixpoint लूप में कम्बाइन करता है। Per-kernel पासेज़ फिर `apply_pre_optimization()` में चलते हैं।

हर पास पैटर्न-आधारित रीराइटिंग इस्तेमाल करता है (देखें [पैटर्न इंजन](./optimizations/pattern-system) चैप्टर)। पैटर्न तब तक फ़ायर होते हैं जब तक कोई और मैच न हो, फिर अगला पास शुरू होता है।

### पहले और बाद

यह tensor एक्सप्रेशन देखें:

```text
Before: BUFFER.reshape([2, 3]).expand([4, 2, 3]).sum(axis=0)
```

Rangeify के बाद, movement ops एक्सप्लिसिट index कम्प्यूटेशन बन जाते हैं:

```text
After:
STORE
├── INDEX[RANGE(0..2), RANGE(0..3)]          — index (src[0])
├── REDUCE(Add)                              — value (src[1])
│   ├── LOAD
│   │   └── INDEX[RANGE(0..4), RANGE(0..2), RANGE(0..3)]
│   └── RANGE(0..4, Reduce)
├── RANGE(0..2, Global)                      — output dim 0 (range)
└── RANGE(0..3, Global)                      — output dim 1 (range)
```

`EXPAND` एक `RANGE(0..4)` बन गया जो बफ़र index को अफ़ेक्ट नहीं करता — broadcasting। `RESHAPE` अलग index अरिथमेटिक बन गया। `SUM` `REDUCE(Add)` बन गया जिसमें पहली range `Reduce` टाइप की मार्क है।

### Movement → Index अरिथमेटिक

हर movement ऑपरेशन का एक स्पेसिफ़िक ट्रांसफ़ॉर्मेशन है:

| ऑपरेशन | ट्रांसफ़ॉर्मेशन |
|---------|----------------|
| **RESHAPE** | Index एक्सप्रेशन को Flatten/unflatten करें |
| **PERMUTE** | INDEX में डायमेंशन रीऑर्डर करें |
| **EXPAND** | Index 0 बन जाता है (या range index को अफ़ेक्ट नहीं करती) |
| **PAD** | WHERE(in_bounds, LOAD, pad_value) |
| **SHRINK** | INDEX में ऑफ़सेट एडजस्टमेंट |
| **FLIP** | `size - 1 - index` |

Rangeify के बाद, कोई movement ops नहीं बचते — सिर्फ़ indices पर अरिथमेटिक ऑपरेशन।

---

## कर्नेल स्प्लिटिंग: बाउंड्रीज़ ढूँढना

एक कम्प्यूटेशन ग्राफ़ में कई आउटपुट हो सकते हैं, या इंटरमीडिएट वैल्यूज़ जिन्हें materialize करना पड़े। **कर्नेल स्प्लिटिंग** इन बाउंड्रीज़ को पहचानती है और अलग-अलग कर्नेल बनाती है।

एंट्री पॉइंट `schedule/src/rangeify/kernel.rs` में `run_kernel_split_pipeline()` है।

### कर्नेल स्प्लिटिंग पाइपलाइन

स्प्लिटिंग कई कोऑर्डिनेटेड स्टेप्स से गुज़रती है:

**स्टेप 1: BUFFERIZE → STORE**

`BUFFERIZE` नोड मार्क करते हैं कि वैल्यूज़ को कहाँ materialize होना चाहिए। `pm_add_buffers_patterns()` उन्हें एक्सप्लिसिट `STORE` ऑपरेशन में बदलता है:

```text
Before: BUFFERIZE(computation, ranges)
After:  END(STORE(INDEX(...), computation), ranges)
```

`END` रैपर कैप्चर करता है कि कौन सी ranges इस store को scope करती हैं। इस फ़ेज़ में बफ़र एलोकेट और ID असाइन होते हैं।

**स्टेप 2: Stores को कर्नेल में स्प्लिट करें**

`split_all_stores()` और `split_store()` ग्राफ़ को STORE बाउंड्रीज़ पर स्प्लिट करते हैं, अलग-अलग कर्नेल बनाते हैं। बफ़र नंबरिंग स्प्लिटिंग के दौरान `LocalAddBufferContext.dg` काउंटर से असाइन होती है।

```text
Before: END(STORE(...), ranges)
After:  KERNEL(SINK(STORE(...)), ranges, buffer_list)
```

`KERNEL` नोड सब कुछ रैप करता है: कम्प्यूटेशन (`SINK` के रूप में), इटरेशन ranges, और उन बफ़र्स की लिस्ट जो यह कर्नेल रीड और राइट करता है।

**स्टेप 3: असाइनमेंट ठीक करें**

`fix_assign()` हर buffer_id को उस कर्नेल से मैप करता है जो उसे लिखता है और डिपेंडेंसी ग्राफ़ बनाता है।

### डिपेंडेंसी ट्रैकिंग

जब एक कर्नेल का आउटपुट दूसरे कर्नेल का इनपुट बनता है, हमें डिपेंडेंसी ट्रैकिंग चाहिए:

1. `fix_assign()` हर buffer_id को लिखने वाले कर्नेल से मैप करता है और डिपेंडेंसी ग्राफ़ बनाता है
2. जब कर्नेल B कोई बफ़र रीड करता है जो कर्नेल A ने लिखा है, तो B, A पर निर्भर है
3. डिपेंडेंसीज़ IR में `AFTER` नोड के रूप में दिखती हैं

डिपेंडेंसीज़ IR में `AFTER` नोड के रूप में दिखती हैं, जो सुनिश्चित करती हैं कि कर्नेल सही क्रम में एक्ज़ीक्यूट हों।

### बफ़र नंबरिंग

बफ़र नंबरिंग `split_store()` में `LocalAddBufferContext.dg` काउंटर हैंडल करता है। बफ़र indices स्प्लिट प्रोसेस में पैटर्न-मैच ऑर्डर में असाइन होते हैं — अलग रीनंबरिंग पास की ज़रूरत नहीं।

---

## शेड्यूल बनाना: एक्ज़ीक्यूशन की तैयारी

कर्नेल स्प्लिट होने के बाद, हमें उन्हें **शेड्यूल** करना है: एक्ज़ीक्यूशन ऑर्डर तय करें, बफ़र एलोकेट करें, और कम्पाइलेशन के लिए तैयार करें।

`tensor/src/schedule.rs` में `create_schedule()` एक `Vec<ScheduleItem>` प्रोड्यूस करता है:

```rust
pub struct ScheduleItem {
    pub kernel: Arc<UOp>,              // KERNEL wrapper
    pub ast: Arc<UOp>,                 // Inner computation (for codegen)
    pub buffers: Vec<Buffer>,          // Device buffers
    pub buffer_uop_ids: Vec<u64>,      // UOp IDs for registry cleanup
    pub fixedvars: HashMap<String, i64>,  // Bound iteration variables
    pub bound_ranges: Vec<BoundRange>,    // Ranges needing expansion
    pub source_buffers: HashMap<u64, Arc<UOp>>,  // DEFINE_GLOBAL -> BUFFER mapping
    pub dependencies: Vec<u64>,        // Producer kernel IDs
    pub alias_registered_ids: Vec<u64>, // Alias UOp IDs for cleanup
}
```

### बफ़र एलोकेशन स्ट्रैटेजी

- **इनपुट बफ़र**: पहले से एलोकेटेड (`Tensor::from_slice` से)
- **इंटरमीडिएट बफ़र**: शेड्यूलिंग के दौरान एलोकेटेड (कर्नेल आउटपुट के लिए जो दूसरे कर्नेल को फ़ीड करते हैं)
- **आउटपुट बफ़र**: एलोकेटेड और फ़ाइनल tensor के साथ रजिस्टर्ड

### पैरेलल ग्रुप एनालिसिस

सभी कर्नेल को सीक्वेंशियल एक्ज़ीक्यूशन की ज़रूरत नहीं। इंडिपेंडेंट कर्नेल पैरेलल चल सकते हैं:

```text
Kernel A (writes buf0)
Kernel B (writes buf1)  ─── no dependency ─── can run in parallel
Kernel C (reads buf0, buf1)  ─── depends on A and B
```

शेड्यूलर पैरेलल ग्रुप ढूँढने के लिए **Kahn's algorithm** इस्तेमाल करता है:

1. कर्नेल डिपेंडेंसी DAG बनाएँ
2. बिना incoming edges वाले सभी कर्नेल ढूँढें → ग्रुप 1
3. ग्रुप 1 हटाएँ, दोहराएँ → ग्रुप 2, वगैरह

हर ग्रुप के कर्नेल पैरेलल एक्ज़ीक्यूट होते हैं, फिर अगला ग्रुप शुरू होता है।

---

## कोड जनरेशन: UOp से LLVM IR तक

कर्नेल शेड्यूल होने के बाद, हम असल कोड जनरेट करते हैं। Morok वर्तमान में LLVM बैकएंड सपोर्ट करता है:

| बैकएंड | कम्पाइल स्पीड | आउटपुट क्वालिटी | उपयोग |
|--------|---------------|-----------------|-------|
| **LLVM** | धीमा | हाइली ऑप्टिमाइज़्ड | प्रोडक्शन |

`Renderer` trait कोड जनरेशन को ऐब्स्ट्रैक्ट करता है:

```rust
pub trait Renderer {
    fn render(&self, uop: &Arc<UOp>, name: Option<&str>) -> Result<RenderedKernel>;
    fn backend_name(&self) -> &str;
    fn decompositor(&self) -> Option<TypedPatternMatcher<()>>;
}
```

### LLVM CPU Renderer

LLVM renderer (`codegen/src/llvm/cpu/`) UOp ग्राफ़ ट्रैवर्स करता है और LLVM IR एमिट करता है:

```llvm
define void @kernel_0(ptr noalias align 32 %buf0, ptr noalias align 32 %buf1) #0 {
entry:
  br label %loop_0

loop_0:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop_0 ]
  ; ... computation ...
  %i.next = add nsw i32 %i, 1
  %cond = icmp slt i32 %i.next, 128
  br i1 %cond, label %loop_0, label %exit

exit:
  ret void
}
```

हर buffer एक direct `ptr noalias align 32` parameter है — args array के through कोई indirection नहीं। Symbolic variables (dynamic shapes के लिए) और thread ID अतिरिक्त typed parameters के रूप में पास होते हैं (जैसे `i32 %N`)।

### पोस्ट-ऑप्टिमाइज़ेशन पासेज़

कोड जनरेशन से पहले, ~15 पैटर्न-आधारित पासेज़ IR को साफ़ करते हैं:

| पास | उद्देश्य |
|------|---------|
| `pm_add_loads` | INDEX ऑपरेशन को LOAD में रैप करें |
| `pre_expand` | UNROLL/UPCAST ranges को एक्सप्लिसिट ऑपरेशन में बदलें |
| `devectorize` | कॉन्टिग्युअस मेमोरी एक्सेसेज़ ग्रुप करें |
| `pm_reduce_devectorize` | वेक्टर रिडक्शन हैंडल करें (K-vec, bool, horizontal) |
| `pm_bool_devectorize` | Boolean वेक्टर पैटर्न हैंडल करें |
| `merge_sibling_ends` | एडजेसेंट END ऑपरेशन मर्ज करें |
| `pm_fma_decomposition` | `a*b+c` को fused multiply-add में बदलें (सपोर्ट करने वाले बैकएंड के लिए) |
| `pm_float_decomp` | फ़्लोटिंग-पॉइंट ऑपरेशन डीकम्पोज़ करें |
| `bool_storage_patterns` | मेमोरी ऑपरेशन के लिए bool ↔ uint8 कन्वर्ट करें |

ये पासेज़ ऑप्टिमाइज़्ड AST को कोड जनरेशन के लिए उपयुक्त रूप में ट्रांसफ़ॉर्म करते हैं। नतीजा क्लीन, वेक्टराइज़्ड कोड होता है जिसमें प्रॉपर मेमोरी एक्सेस पैटर्न हैं।

### बैकएंड सपोर्ट

Morok कई कोड जनरेशन बैकएंड सपोर्ट करता है:

| बैकएंड | आउटपुट | स्थिति |
|--------|--------|-------|
| **LLVM** | नेटिव मशीन कोड | प्राइमरी (CPU) |
| **C** | C सोर्स कोड | उपलब्ध |
| **MLIR** | MLIR dialect | उपलब्ध |

---

## एक्ज़ीक्यूशन: कर्नेल चलाना

कोड जनरेशन LLVM IR स्ट्रिंग प्रोड्यूस करता है। एक्ज़ीक्यूशन में JIT कम्पाइलेशन और कर्नेल लॉन्च शामिल है।

### ExecutionPlan

`prepare()` (सिंगल tensor) या `prepare_batch()` (मल्टीपल tensor) एक `ExecutionPlan` बनाता है:

```rust
pub struct ExecutionPlan {
    kernels: Vec<PreparedKernel>,       // Compiled kernels (topological order)
    buffers: Vec<Buffer>,
    ast_to_buffer: HashMap<u64, usize>, // AST id -> buffer index mapping
    output_buffer_indices: Vec<usize>,  // Indices of output buffers (multi-output)
}
```

Plans अब `realize_batch()` / `prepare_batch()` के ज़रिए **मल्टीपल आउटपुट** सपोर्ट करते हैं। जब कई tensor सबग्राफ़ शेयर करते हैं, बैच शेड्यूलिंग कम्पाइलर को आउटपुट के बीच कर्नेल शेयर करने देती है।

मुख्य मेथड:

| मेथड | उद्देश्य |
|------|---------|
| `output_buffer_at(i)` | i-वाँ आउटपुट बफ़र लें (SINK source ऑर्डर से मैच) |
| `num_outputs()` | इस प्लान में आउटपुट बफ़र की संख्या |
| `execute_with_vars(var_vals)` | अलग symbolic variable values के साथ फिर से एक्ज़ीक्यूट करें (कोई रीकम्पाइलेशन नहीं) |

प्लान **रीयूज़ेबल** है: एक बार कम्पाइल करें, अलग-अलग डेटा के साथ कई बार एक्ज़ीक्यूट करें।

### JIT कम्पाइलेशन

LLVM रनटाइम (`runtime/src/llvm.rs`) IR को मशीन कोड में कम्पाइल करता है:

1. LLVM IR स्ट्रिंग को module में **पार्स** करें
2. module well-formed है **वेरिफ़ाई** करें
3. LLVM की O3 पास पाइपलाइन से **ऑप्टिमाइज़** करें
4. नेटिव मशीन कोड में **JIT कम्पाइल** करें
5. (AST ID, device) से रीयूज़ के लिए **कैश** करें

```rust
// Simplified JIT flow
let module = Module::parse_ir(context, ir_string)?;
module.verify()?;
pass_manager.run(&module);  // O3 optimization
let function = execution_engine.get_function::<KernelFn>(&name)?;
// Cache: (ast_id, device) → function
```

### कर्नेल एक्ज़ीक्यूशन

कर्नेल कम्पाइल होने के बाद, एक्ज़ीक्यूशन टोपोलॉजिकल ऑर्डर में कर्नेल इटरेट करता है, डिपेंडेंसीज़ का ध्यान रखते हुए:

```rust
for kernel in &plan.kernels {
    // Dependencies tracked per-kernel via kernel.dependencies
    kernel.execute(buffers);
}
```

कर्नेल अपना डिवाइस स्पेसिफ़िकेशन रखते हैं, इसलिए एक प्लान मल्टीपल डिवाइसेज़ को स्पैन कर सकता है।

### कर्नेल कैशिंग

Hash consing कर्नेल कैशिंग को बहुत इफ़ेक्टिव बनाता है:

- **Key**: `(UOp ID, device string)`
- **Storage**: Lock-free HashMap (papaya crate)
- **Hit rate**: ज़्यादा, क्योंकि आइडेंटिकल कम्प्यूटेशन UOp IDs शेयर करते हैं

जब आप एक ही एक्सप्रेशन दो बार कम्प्यूट करते हैं, दूसरी बार कैश हिट होता है — कोई रीकम्पाइलेशन नहीं।

---

## वर्क्ड उदाहरण: मैट्रिक्स मल्टिप्लाई

चलिए `C = A @ B` को पूरी पाइपलाइन से ट्रेस करते हैं। मान लें 4×4 matrices हैं।

### स्टेज 1: Lazy ग्राफ़ कंस्ट्रक्शन

```rust
let a = Tensor::from_slice(a_data);  // Input buffer allocated
let b = Tensor::from_slice(b_data);  // Input buffer allocated
let c = a.matmul(&b);                 // Graph built, no computation
```

इस पॉइंट पर, `c` एक lazy tensor है जिसका UOp ग्राफ़ यह है:

```text
REDUCE_AXIS(Add, axis=2)
└── MUL
    ├── EXPAND(A, [4, 4, 4])    — A: [4, 4] → [4, 1, 4] → [4, 4, 4]
    └── EXPAND(B, [4, 4, 4])    — B: [4, 4] → [1, 4, 4] → [4, 4, 4]
```

### स्टेज 2: Rangeify

Movement ops एक्सप्लिसिट लूप्स बन जाते हैं:

```text
STORE
├── INDEX[BUFFER(C), RANGE(i, 0..4), RANGE(j, 0..4)]  — index
├── REDUCE(Add)                                          — value
│   ├── MUL
│   │   ├── LOAD(A)
│   │   │   └── INDEX[BUFFER(A), RANGE(i), RANGE(k, 0..4, Reduce)]
│   │   └── LOAD(B)
│   │       └── INDEX[BUFFER(B), RANGE(k), RANGE(j)]
│   └── RANGE(k, Reduce)
├── RANGE(i, Global)                                     — output dim 0
└── RANGE(j, Global)                                     — output dim 1
```

`i` और `j` ranges आउटपुट डायमेंशन हैं। `k` range रिडक्शन (contracted) डायमेंशन है।

### स्टेज 3: कर्नेल स्प्लिटिंग

सिंगल STORE → सिंगल KERNEL:

```text
KERNEL
├── SINK(STORE(...))
├── ranges: [i: 0..4, j: 0..4]
└── buffers: [C (output), A (input), B (input)]
```

### स्टेज 4: शेड्यूल

एक `ScheduleItem` जिसमें:
- `kernel`: KERNEL UOp
- `ast`: इनर SINK/STORE
- `buffers`: [C, A, B]
- `dependencies`: [] (कोई पिछला कर्नेल नहीं)

### स्टेज 5: ऑप्टिमाइज़ेशन

Heuristic ऑप्टिमाइज़र अप्लाई करता है:
- वेक्टराइज़ेशन: j डायमेंशन को 4 से UPCAST
- लूप ऑर्डरिंग: अच्छे cache बिहेवियर को सुनिश्चित करें

### स्टेज 6: कोड जनरेशन

जनरेटेड LLVM IR (सरलीकृत):

```llvm
define void @matmul(ptr noalias align 32 %C, ptr noalias align 32 %A, ptr noalias align 32 %B) #0 {
entry:
  br label %loop_i

loop_i:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop_i.end ]
  br label %loop_j

loop_j:
  %j = phi i64 [ 0, %loop_i ], [ %j.next, %loop_k.end ]
  %acc = ... ; initialize accumulator
  br label %loop_k

loop_k:
  %k = phi i64 [ 0, %loop_j ], [ %k.next, %loop_k ]
  %a_val = load float, ptr ...  ; A[i, k]
  %b_val = load float, ptr ...  ; B[k, j]
  %prod = fmul float %a_val, %b_val
  %acc.new = fadd float %acc, %prod
  %k.next = add i64 %k, 1
  %k.cond = icmp slt i64 %k.next, 4
  br i1 %k.cond, label %loop_k, label %loop_k.end

loop_k.end:
  store float %acc.new, ptr ...  ; C[i, j]
  ; ... continue j, i loops
}
```

### स्टेज 7: एक्ज़ीक्यूशन

1. LLVM IR को JIT कम्पाइल करें
2. एक्ज़ीक्यूट करें: `kernel([C_ptr, A_ptr, B_ptr], [])`
3. रिज़ल्ट C बफ़र में है

कुल: एक फ़ंक्शन कॉल, रिज़ल्ट तैयार।

---

## तुलना: दूसरे फ़्रेमवर्क कैसे एक्ज़ीक्यूट करते हैं

| पहलू | PyTorch | JAX | TVM | **Morok** |
|-------|---------|-----|-----|-----------|
| **इवैल्यूएशन** | Eager (तुरंत) | Traced (jit decorator) | Lazy (te.compute) | Lazy (realize) |
| **ग्राफ़ कैप्चर** | torch.compile | jax.jit trace | एक्सप्लिसिट schedule | Implicit via ops |
| **कम्पाइलेशन** | TorchInductor | XLA बैकएंड | Auto-scheduler | Pattern + beam |
| **कैशिंग** | प्रति-ग्राफ़ hash | प्रति-trace | प्रति-schedule | प्रति-AST (hash consing) |
| **पैरेलिज़्म** | DataParallel/DDP | pmap/pjit | Parallel schedule | Parallel groups |

**PyTorch**: डिफ़ॉल्ट रूप से eager, ऑप्टिमाइज़ेशन के लिए torch.compile। TorchInductor Triton या C++ कोड जनरेट करता है।

**JAX**: फ़ंक्शनल ट्रांसफ़ॉर्मेशन (jit, grad, vmap) कम्प्यूटेशन ट्रेस करते हैं। XLA ऑप्टिमाइज़्ड कर्नेल में कम्पाइल करता है।

**TVM**: कम्प्यूटेशन और schedule का एक्सप्लिसिट अलगाव। Auto-scheduler अच्छे schedules सर्च करता है।

**Morok**: पूरी तरह lazy — `realize()` तक कुछ एक्ज़ीक्यूट नहीं होता। Hash consing ऑटोमैटिक कैशिंग देता है। ऑप्शनल beam search के साथ पैटर्न-आधारित ऑप्टिमाइज़ेशन प्रोडक्शन क्वालिटी के लिए।

---

## गहरी समझ

पाइपलाइन कई डिज़ाइन सिद्धांतों को मूर्त रूप देती है:

**Lazy evaluation ग्लोबल ऑप्टिमाइज़ेशन सक्षम करता है।** कम्प्यूटेशन को डिफ़र करके, हम कोड जनरेट करने से पहले पूरा ग्राफ़ देखते हैं। कोई लोकल डिसीज़न ग्लोबल ऑप्टिमाइज़ेशन को सीमित नहीं करता।

**एक्सप्लिसिट लूप्स हार्डवेयर-स्पेसिफ़िक शेड्यूलिंग सक्षम करते हैं।** Movement ops सुविधाजनक ऐब्स्ट्रैक्शन हैं, लेकिन GPU को लूप्स चाहिए। Rangeify इस गैप को पाटता है।

**Hash consing कैशिंग को ऑटोमैटिक बनाता है।** आइडेंटिकल कम्प्यूटेशन पॉइंटर शेयर करते हैं, इसलिए cache keys ट्रिवियल हैं। कॉम्प्लेक्स ग्राफ़ हैशिंग की ज़रूरत नहीं।

**कंसर्न का अलगाव हर स्टेज को सिंपल रखता है।** Rangeify को LLVM की जानकारी नहीं। कोड जनरेशन को tensor सिमैंटिक्स की जानकारी नहीं। हर स्टेज एक काम अच्छे से करता है।

नतीजा: एक कम्पाइलेशन पाइपलाइन जो शक्तिशाली और मेंटेन करने योग्य दोनों है। `tensor.realize()` से मशीन कोड तक, हर स्टेप विज़िबल, डिबगेबल, और एक्सटेंसिबल है।
