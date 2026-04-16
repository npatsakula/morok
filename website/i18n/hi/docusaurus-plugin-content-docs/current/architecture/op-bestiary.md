---
sidebar_label: Op Bestiary
---

# Op Bestiary: UOp ऑपरेशनों की फ़ील्ड गाइड

Morok IR डंप डीबग करते समय आपको ऐसे ऑपरेशन मिलेंगे जो नाम से स्पष्ट नहीं होते। यह चैप्टर नॉन-ट्रिवियल ऑपरेशनों को सिग्नेचर, फ़ील्ड एक्सप्लेनेशन और उदाहरणों के साथ डॉक्यूमेंट करता है।

**क्या कवर है:** वे ऑपरेशन जिन्हें एक्सप्लेनेशन चाहिए — लूप कंट्रोल, रिडक्शन, मेमोरी ऑपरेशन, कर्नेल स्ट्रक्चर, वेक्टराइज़ेशन, tensor cores।

**क्या कवर नहीं है:** ट्रिवियल ALU ऑपरेशन (`Add`, `Mul`, `Sqrt`, आदि) जो बिल्कुल वैसे ही काम करते हैं जैसा आप सोचते हैं।

---

## लूप कंट्रोल: RANGE और END

### RANGE — लूप स्कोप ओपनर

```rust
Range {
    end: Arc<UOp>,           // loop bound (exclusive)
    axis_id: AxisId,         // identifier for deduplication
    axis_type: AxisType,     // scheduling behavior
    deps: SmallVec<[Arc<UOp>; 2]>,  // range dependencies
}
```

**फ़ील्ड्स:**

| फ़ील्ड | टाइप | उद्देश्य |
|--------|------|----------|
| `end` | `Arc<UOp>` | अपर बाउंड (exclusive), आमतौर पर एक `CONST` |
| `axis_id` | `AxisId` | कर्नेल स्प्लिटिंग से पहले `Unrenumbered(n)`, बाद में `Renumbered(n)` |
| `axis_type` | `AxisType` | लूप को कैसे शेड्यूल किया जाएगा यह तय करता है (नीचे देखें) |
| `deps` | `SmallVec<[Arc<UOp>; 2]>` | दूसरी ranges जिन पर यह range डिपेंड करती है |

**AxisType हायरार्की:**

| टाइप | प्रायोरिटी | GPU मैपिंग | उद्देश्य |
|------|-----------|------------|----------|
| `Outer` | -2 | — | कर्नेल बाउंड्री मार्कर |
| `Loop` | -1 | `for` लूप | सीक्वेंशियल इटरेशन |
| `Global` | 0 | `blockIdx` | ग्रिड पैरेललिज़्म |
| `Thread` | 0 | thread pool | CPU पैरेललिज़्म |
| `Warp` | 1 | warp/wavefront | सब-ग्रुप पैरेललिज़्म |
| `Local` | 2 | `threadIdx` | वर्कग्रुप पैरेललिज़्म |
| `GroupReduce` | 2 | shared memory | दो-स्टेज रिडक्शन |
| `Upcast` | 3 | SIMD | वेक्टराइज़ेशन |
| `Reduce` | 4 | accumulator | रिडक्शन डायमेंशन |
| `Unroll` | 5 | unrolled | लूप अनरोलिंग |

प्रायोरिटी लूप नेस्टिंग ऑर्डर तय करती है — कम वैल्यू वाले आउटर लूप होते हैं।

**उदाहरण:**
```text
RANGE(end=128, axis_id=R0, type=Global)
└── CONST(128) : Index
```

### END — लूप स्कोप क्लोज़र

```rust
End {
    computation: Arc<UOp>,              // value computed inside loop
    ranges: SmallVec<[Arc<UOp>; 4]>,    // ranges being closed
}
```

END एक या ज़्यादा RANGE स्कोप बंद करता है और उन्हें एक्टिव सेट से हटाता है। एक साथ कई ranges बंद की जा सकती हैं।

**उदाहरण:**
```text
END
├── STORE(...)           — computation
├── RANGE(R0, Global)    — पहली range बंद हुई
└── RANGE(R1, Local)     — दूसरी range बंद हुई
```

---

## रिडक्शन: REDUCE बनाम REDUCE_AXIS

दो ऑपरेशन जिनके नाम मिलते-जुलते हैं पर काम अलग-अलग है।

### REDUCE_AXIS — Tensor डायमेंशन रिडक्शन (हाई-लेवल)

```rust
ReduceAxis {
    src: Arc<UOp>,           // input tensor
    reduce_op: ReduceOp,     // Add, Mul, Max, Min
    axes: Vec<usize>,        // axes to reduce
}
```

**Rangeify से पहले** इस्तेमाल होता है। NumPy के `.sum(axis=0)` की तरह tensor डायमेंशन पर काम करता है।

**उदाहरण:**
```text
REDUCE_AXIS(Add, axes=[1])
└── BUFFER[10, 20] : Float32
```

यह `[10, 20]` tensor को axis 1 पर sum करके `[10]` में बदलता है।

### REDUCE — Range इटरेशन रिडक्शन (लो-लेवल)

```rust
Reduce {
    src: Arc<UOp>,                      // value to accumulate
    ranges: SmallVec<[Arc<UOp>; 4]>,    // ranges being reduced
    reduce_op: ReduceOp,                // Add, Mul, Max, Min
}
```

**Rangeify के बाद** इस्तेमाल होता है। RANGE इटरेशन के दौरान वैल्यूज़ accumulate करता है और स्पेसिफ़ाइड ranges बंद करता है।

**ReduceOp वैरिएंट:**

| Op | आइडेंटिटी | ऑपरेशन | Tinygrad |
|----|-----------|---------|----------|
| `Add` | 0 | `acc + value` | ✓ |
| `Mul` | 1 | `acc * value` | ✓ |
| `Max` | -∞ | `max(acc, value)` | ✓ |
| `Min` | +∞ | `min(acc, value)` | केवल Morok |

> **कम्पैटिबिलिटी:** Tinygrad का स्पेक REDUCE_AXIS को `{Add, Mul, Max}` तक सीमित रखता है। Morok इसे `Min` के साथ एक्सटेंड करता है।

**उदाहरण:**
```text
REDUCE(Add)
├── MUL                      — accumulate करने के लिए वैल्यू
│   ├── LOAD(A, ...)
│   └── LOAD(B, ...)
└── RANGE(R2, Reduce)        — reduce की जा रही range
    └── CONST(64)
```

### ALLREDUCE — क्रॉस-डिवाइस रिडक्शन

```rust
AllReduce {
    src: Arc<UOp>,           // local partial result
    device: Arc<UOp>,        // device specification
    reduce_op: ReduceOp,     // reduction operation
}
```

कई डिवाइसों में डिस्ट्रिब्यूटेड रिडक्शन करता है। मल्टी-GPU ट्रेनिंग के लिए इस्तेमाल होता है।

---

## बफ़र ऑपरेशन

### BUFFER — बफ़र डिक्लेरेशन

```rust
Buffer {
    unique: Arc<UOp>,        // UNIQUE op for identity
    device: Arc<UOp>,        // DEVICE op
    size: usize,             // total element count
}
```

Tensor स्टोरेज के लिए बफ़र डिक्लेयर करता है। `unique` फ़ील्ड यह सुनिश्चित करती है कि समान size/device होने पर भी बफ़र अलग रहें।

### BUFFERIZE — मटेरियलाइज़ेशन मार्कर

```rust
Bufferize {
    compute: Arc<UOp>,                  // computation to materialize
    ranges: SmallVec<[Arc<UOp>; 4]>,    // output dimensions
    opts: BufferizeOpts,                // address space, device
}
```

मार्क करता है कि कम्प्यूटेशन को मेमोरी में मटेरियलाइज़ होना चाहिए। कर्नेल स्प्लिटिंग ट्रिगर करता है।

**BufferizeOpts:**

| फ़ील्ड | टाइप | उद्देश्य |
|--------|------|----------|
| `device` | `Option<DeviceSpec>` | टारगेट डिवाइस, लोकल के लिए `None` |
| `addrspace` | `AddrSpace` | `Global` (डिवाइस) या `Local` (shared) |

**उदाहरण:**
```text
BUFFERIZE(opts={addrspace=Global})
├── REDUCE(Add, ...)         — computation
├── RANGE(R0, Global)        — आउटपुट dim 0
└── RANGE(R1, Global)        — आउटपुट dim 1
```

### INDEX — मल्टी-डायमेंशनल बफ़र एक्सेस

```rust
Index {
    buffer: Arc<UOp>,                   // BUFFER or PARAM
    indices: SmallVec<[Arc<UOp>; 4]>,   // index per dimension
    gate: Option<Arc<UOp>>,             // optional predicate
}
```

मल्टी-डायमेंशनल indices से मेमोरी एड्रेस कैलकुलेट करता है। एलिमेंट dtype रिटर्न करता है (पॉइंटर नहीं)।

**उदाहरण:**
```text
INDEX : Float32
├── PARAM(0)
├── RANGE(R0, Global)        — dim 0 के लिए index
├── RANGE(R1, Loop)          — dim 1 के लिए index
└── MUL(...)                 — dim 2 के लिए index
```

### POINTER_INDEX — लो-लेवल पॉइंटर अरिथमेटिक

```rust
PointerIndex {
    ptr: Arc<UOp>,           // base pointer
    offset: Arc<UOp>,        // byte offset
}
```

डायरेक्ट पॉइंटर अरिथमेटिक। लीनियराइज़ेशन के बाद जब indices फ़्लैटन हो जाते हैं तब इस्तेमाल होता है।

> **कम्पैटिबिलिटी:** Tinygrad अलग ऑपरेशन के बजाय `INDEX` में `ptr=True` फ़्लैग इस्तेमाल करता है।

### LOAD — मेमोरी रीड

```rust
Load {
    buffer: Arc<UOp>,        // buffer or pointer
    index: Arc<UOp>,         // INDEX op
    alt: Option<Arc<UOp>>,   // alternative value for gated loads
}
```

बफ़र से index पर वैल्यू रीड करता है। गेटेड loads के लिए, `alt` फ़ील्ड INDEX की `gate` false होने पर एक वैल्यू प्रदान करती है (मेमोरी एक्सेस पूरी तरह स्किप हो जाती है)।

**उदाहरण:**
```text
LOAD : Float32
├── PARAM(1)
└── INDEX
    ├── PARAM(1)
    ├── RANGE(R0)
    └── RANGE(R2)
```

### STORE — मेमोरी राइट

```rust
Store {
    index: Arc<UOp>,                    // INDEX op (buffer accessed via index.src[0])
    value: Arc<UOp>,                    // value to write
    ranges: SmallVec<[Arc<UOp>; 4]>,    // ranges being closed
}
```

बफ़र में वैल्यू लिखता है। बफ़र INDEX नोड के ज़रिए एक्सेस होता है (`index.src[0]` से), अलग फ़ील्ड से नहीं। STORE स्पेसिफ़ाइड ranges बंद करता है, जो आउटपुट इटरेशन डायमेंशन दर्शाती हैं। ranges फ़ील्ड आउटपुट अपकास्टिंग के लिए इस्तेमाल होती है: जब `Range(Upcast)` शामिल हो, तो expansion के दौरान यह `UNROLL` बनती है, फिर `CONTRACT` से कॉन्ट्रैक्ट होती है।

गेटेड stores के लिए, gate वाला INDEX इस्तेमाल करें (INDEX में एक ऑप्शनल `gate` फ़ील्ड होती है)।

> **कम्पैटिबिलिटी:** Morok के STORE में अलग `buffer` फ़ील्ड नहीं है — sources हैं: index=0, value=1, ranges=2+ (range_start=2)। Tinygrad का लेआउट भी ऐसा ही है।

**उदाहरण:**
```text
STORE
├── INDEX[R0, R1]            — राइट एड्रेस (index.src[0] से बफ़र)
├── REDUCE(Add, ...)         — वैल्यू
├── RANGE(R0, Global)        — आउटपुट dim 0 (बंद)
└── RANGE(R1, Global)        — आउटपुट dim 1 (बंद)
```

---

## कर्नेल स्ट्रक्चर

### KERNEL — कर्नेल रैपर

```rust
Kernel {
    sources: SmallVec<[Arc<UOp>; 4]>,   // arguments
    ast: Arc<UOp>,                       // computation (usually SINK)
}
```

कोड जनरेशन के लिए एक पूरे कर्नेल को रैप करता है। Sources कर्नेल आर्ग्युमेंट होते हैं (`Param`, `DefineLocal`, `DefineVar`)। नोट: `Param` ने batching_support PR में `DefineGlobal` को रिप्लेस किया ताकि बफ़र आइडेंटिटी हटाकर कर्नेल डीडुप्लिकेशन सक्षम हो सके।

**उदाहरण:**
```text
KERNEL
├── PARAM(slot=0, size=1024) — आउटपुट बफ़र arg
├── PARAM(slot=1, size=1024) — इनपुट A arg
├── PARAM(slot=2, size=1024) — इनपुट B arg
└── SINK                     — computation
    └── STORE(...)
```

### SINK — मल्टीपल रूट कलेक्टर

```rust
Sink {
    sources: SmallVec<[Arc<UOp>; 4]>,
}
```

कई आउटपुट को एक सिंगल रूट में कलेक्ट करता है। हर कर्नेल का `ast` आमतौर पर एक SINK होता है जिसमें STORE ऑपरेशन होते हैं।

**उदाहरण:**
```text
SINK
├── STORE(output_0, ...)
├── STORE(output_1, ...)
└── STORE(output_2, ...)
```

### AFTER — डिपेंडेंसी मार्कर

```rust
After {
    passthrough: Arc<UOp>,              // value that flows through
    deps: SmallVec<[Arc<UOp>; 4]>,      // operations that must complete
}
```

कर्नेल्स के बीच बिना डेटा डिपेंडेंसी के एक्ज़ीक्यूशन डिपेंडेंसी एक्सप्रेस करता है। `passthrough` वैल्यू बिना बदले रिटर्न होती है, लेकिन सभी `deps` पूरे होने के बाद ही।

**उदाहरण:**
```text
SINK
├── AFTER
│   ├── PARAM(0)     — passthrough (बफ़र रेफ़रेंस)
│   └── KERNEL(...)          — पहले पूरा होना ज़रूरी
└── KERNEL(...)              — AFTER के बाद बफ़र इस्तेमाल कर सकता है
```

### BARRIER — सिंक्रोनाइज़ेशन फ़ेंस

```rust
Barrier {
    src: Arc<UOp>,                      // value passing through
    deps: SmallVec<[Arc<UOp>; 4]>,      // operations to wait for
}
```

GPU वर्कग्रुप सिंक्रोनाइज़ेशन। यह सुनिश्चित करता है कि वर्कग्रुप के सभी threads आगे बढ़ने से पहले barrier तक पहुँचें।

---

## वेक्टर ऑपरेशन

### VECTORIZE — स्केलर्स से वेक्टर बनाएँ

```rust
Vectorize {
    elements: SmallVec<[Arc<UOp>; 4]>,
}
```

N स्केलर वैल्यूज़ को साइज़ N के वेक्टर में जोड़ता है। सभी एलिमेंट्स का बेस dtype एक ही होना चाहिए।

**उदाहरण:**
```text
VECTORIZE : <4 x Float32>
├── CONST(1.0)
├── CONST(2.0)
├── CONST(3.0)
└── CONST(4.0)
```

### GEP — Get Element Pointer (वेक्टर एक्सट्रैक्ट)

```rust
Gep {
    vector: Arc<UOp>,        // source vector
    indices: Vec<usize>,     // positions to extract
}
```

वेक्टर से एलिमेंट्स निकालता है:
- सिंगल index → स्केलर
- मल्टीपल indices → छोटा वेक्टर

**उदाहरण:**
```text
GEP([0, 2]) : <2 x Float32>
└── VECTORIZE : <4 x Float32>
    └── ...
```

### VConst — वेक्टर कॉन्स्टेंट

```rust
VConst {
    values: Vec<ConstValue>,
}
```

कम्पाइल-टाइम कॉन्स्टेंट्स का वेक्टर। `CONST` नोड्स के `VECTORIZE` से ज़्यादा एफ़िशिएंट।

### CAT — वेक्टर्स को कॉन्कैटनेट करें

```rust
Cat {
    sources: SmallVec<[Arc<UOp>; 4]>,
}
```

वेक्टर्स को एक बड़े वेक्टर में जोड़ता है। आउटपुट `vcount` = इनपुट `vcount` का योग।

**उदाहरण:**
```text
CAT : <8 x Float32>
├── VECTORIZE : <4 x Float32>
└── VECTORIZE : <4 x Float32>
```

### PtrCat — पॉइंटर्स को कॉन्कैटनेट करें

```rust
PtrCat {
    sources: SmallVec<[Arc<UOp>; 4]>,
}
```

वेक्टराइज़्ड load/store के लिए मेमोरी एक्सेस को ग्रुप करता है। Devectorizer पास द्वारा इस्तेमाल होता है।

---

## Expansion: UNROLL और CONTRACT

### UNROLL — इटरेशन के अनुसार कम्प्यूटेशन एक्सपैंड करें

```rust
Unroll {
    src: Arc<UOp>,                       // computation to expand
    unroll_axes: Vec<(usize, usize)>,    // (axis_index, factor) pairs
}
```

अलग-अलग इटरेशन वैल्यूज़ के लिए कम्प्यूटेशन के कई वर्शन बनाता है। लूप अनरोलिंग ऑप्टिमाइज़ेशन के लिए इस्तेमाल होता है।

**उदाहरण:** `UNROLL(unroll_axes=[(0, 4)])` कम्प्यूटेशन को अलग-अलग index वैल्यूज़ के साथ 4 बार एक्सपैंड करता है।

### CONTRACT — अनरोल्ड वैल्यूज़ को वेक्टर में कॉलैप्स करें

```rust
Contract {
    src: Arc<UOp>,                       // unrolled computation
    upcast_ranges: Vec<(usize, usize)>,  // (axis_index, factor) pairs
}
```

UNROLL का उल्टा — एक्सपैंडेड स्केलर वैल्यूज़ को वेक्टर में कलेक्ट करता है। आउटपुट वेक्टर साइज़ = factors का गुणनफल।

**उदाहरण:**
```text
CONTRACT(upcast_ranges=[(0, 4)]) : <4 x Float32>
└── UNROLL(unroll_axes=[(0, 4)])
    └── LOAD(...)
```

यह पैटर्न एक load को वेक्टराइज़ करता है: 4 इटरेशन एक्सपैंड करो, फिर रिज़ल्ट को 4-एलिमेंट वेक्टर में पैक करो।

---

## Tensor Cores: WMMA

### WMMA — Warp Matrix Multiply-Accumulate

```rust
Wmma {
    a: Arc<UOp>,             // matrix A fragment
    b: Arc<UOp>,             // matrix B fragment
    c: Arc<UOp>,             // accumulator C fragment
    metadata: WmmaMetadata,  // hardware configuration
}
```

हार्डवेयर tensor core ऑपरेशन: `D = A × B + C`। स्पेसिफ़िक मैट्रिक्स शेप और डेटा लेआउट की ज़रूरत होती है।

**WmmaMetadata फ़ील्ड्स:**

| फ़ील्ड | टाइप | उद्देश्य |
|--------|------|----------|
| `name` | `String` | इंस्ट्रक्शन नाम (जैसे, `"__hmma..."`) |
| `dims` | `(N, M, K)` | मैट्रिक्स डायमेंशन (जैसे, `(16, 16, 16)`) |
| `dtype_in` | `DType` | इनपुट मैट्रिक्स प्रिसिज़न (जैसे, `Float16`) |
| `dtype_out` | `DType` | आउटपुट प्रिसिज़न (जैसे, `Float32`) |
| `device` | `String` | टारगेट डिवाइस स्ट्रिंग |
| `threads` | `usize` | प्रति warp threads (आमतौर पर 32) |
| `upcast_axes` | `WmmaUpcastAxes` | प्रति-ऑपरेंड वेक्टराइज़ेशन (फ़ील्ड्स: `a`, `b`, `c`) |
| `reduce_axes` | `Vec<(usize, usize)>` | कॉन्ट्रैक्शन axes |
| `tile_grid` | `(usize, usize)` | मल्टी-FMA बैचिंग ग्रिड (डिफ़ॉल्ट (1,1)) |

**उदाहरण:**
```text
WMMA(dims=(16, 16, 16), dtype_in=Float16, dtype_out=Float32)
├── A fragment : <8 x Float16>
├── B fragment : <8 x Float16>
└── C accumulator : <8 x Float32>
```

---

## कंट्रोल फ़्लो

### IF / ENDIF — कंडीशनल एक्ज़ीक्यूशन

```rust
If {
    condition: Arc<UOp>,                // boolean predicate
    body: SmallVec<[Arc<UOp>; 4]>,      // operations to execute
}

EndIf {
    if_op: Arc<UOp>,         // corresponding IF op
}
```

कंडीशन true होने पर ही body एक्ज़ीक्यूट करता है। बाउंड्री चेक और sparse ऑपरेशन के लिए इस्तेमाल होता है।

**उदाहरण:**
```text
IF
├── LT(idx, bound)           — condition (src[0])
├── STORE(...)               — body[0]
└── STORE(...)               — body[1]

ENDIF
└── IF(...)                  — IF op को रेफ़रेंस करता है
```

---

## डेफ़िनिशन ऑपरेशन

### PARAM — बफ़र पैरामीटर

```rust
Param { slot: usize, size: usize, device: Option<Arc<UOp>> }
```

नॉर्मलाइज़्ड बफ़र पैरामीटर — इनपुट/आउटपुट बफ़र का पोज़िशनल रेफ़रेंस।
प्री-शेड्यूल नॉर्मलाइज़ेशन (BUFFER→PARAM) द्वारा बनाया जाता है ताकि बफ़र आइडेंटिटी हटाकर
आइडेंटिकल कम्प्यूटेशन का स्ट्रक्चरल डीडुप्लिकेशन हो सके।
`slot` कर्नेल आर्ग्युमेंट लिस्ट में पोज़िशन है, `size` एलिमेंट काउंट है।

### DEFINE_LOCAL — Shared मेमोरी एलोकेशन

```rust
DefineLocal(usize)           // local memory index
```

GPU shared memory (LDS) एलोकेशन। एक वर्कग्रुप के अंदर विज़िबल।

### DEFINE_VAR — सिम्बॉलिक रनटाइम वेरिएबल

```rust
DefineVar {
    name: String,            // variable name
    min_val: i64,            // minimum bound
    max_val: i64,            // maximum bound
}
```

ज्ञात bounds वाला रनटाइम वेरिएबल। डायनामिक shapes के लिए इस्तेमाल होता है जहाँ bounds पता हैं।

**उदाहरण:**
```text
DEFINE_VAR(name="batch_size", min=1, max=128) : Index
```

### DEFINE_REG — रजिस्टर एलोकेशन

```rust
DefineReg {
    size: usize,             // register size
    id: usize,               // unique accumulator ID
}
```

इंटरमीडिएट स्टोरेज के लिए रजिस्टर एलोकेट करता है। `id` फ़ील्ड एक ही dtype के रजिस्टर्स को अलग करती है — इसके बिना, दो same-dtype reduces hash consing से एक DEFINE_REG शेयर करेंगे। कोड जनरेशन में इस्तेमाल होता है।

### BIND — वेरिएबल बाइंडिंग

```rust
Bind {
    var: Arc<UOp>,           // DEFINE_VAR
    value: Arc<UOp>,         // concrete value
}
```

रनटाइम पर एक सिम्बॉलिक वेरिएबल को कॉन्क्रीट वैल्यू से बाइंड करता है।

---

## स्पेशल ऑपरेशन

### SPECIAL — हार्डवेयर-प्रदत्त वैल्यूज़

```rust
Special {
    end: Arc<UOp>,           // upper bound for this dimension
    name: String,            // e.g., "blockIdx.x", "threadIdx.y"
}
```

हार्डवेयर-प्रदत्त वैल्यूज़ (thread/block indices) एक्सेस करता है। यह लूप नहीं है — हार्डवेयर सीधे वैल्यू देता है।

**उदाहरण:**
```text
SPECIAL(name="blockIdx.x", end=128) : Index
└── CONST(128)
```

### UNIQUE — आइडेंटिटी मार्कर

```rust
Unique(usize)                // unique identifier
```

बफ़र disambiguation के लिए यूनीक आइडेंटिटी बनाता है। अलग UNIQUE वैल्यू वाले दो बफ़र अलग माने जाते हैं, भले ही बाकी सब समान हो।

### DEVICE — डिवाइस स्पेसिफ़िकेशन

```rust
Device(DeviceSpec)           // device specification
```

कम्प्यूटेशन के लिए टारगेट डिवाइस स्पेसिफ़ाई करता है।

---

## मूवमेंट ऑपरेशन

हाई-लेवल tensor शेप ट्रांसफ़ॉर्मेशन। Rangeify के दौरान ये एक्सप्लिसिट INDEX ऑपरेशन में बदल जाते हैं।

| ऑपरेशन | सिग्नेचर | उद्देश्य |
|---------|----------|----------|
| `Reshape` | `{ src, new_shape }` | शेप बदलें, एलिमेंट्स वही |
| `Permute` | `{ src, axes: Vec<usize> }` | ट्रांसपोज़/रीऑर्डर axes |
| `Expand` | `{ src, new_shape }` | बड़ी शेप में ब्रॉडकास्ट |
| `Pad` | `{ src, begin_pads, end_pads }` | पैडिंग जोड़ें |
| `Shrink` | `{ src, begins, ends }` | सब-रीजन निकालें |
| `Flip` | `{ src, axes: Vec<bool> }` | axes के अनुसार रिवर्स |

**उदाहरण:** RESHAPE
```text
RESHAPE(new_shape=[6, 4]) : Shape[6, 4]
├── BUFFER[2, 3, 4] : Float32
└── CONST([6, 4]) : Shape
```

---

## अतिरिक्त ऑपरेशन

ये ऑपरेशन `Op` enum में हैं लेकिन इंटरनल हैं या डीबगिंग में कम दिखते हैं:

| ऑपरेशन | उद्देश्य |
|---------|----------|
| `Copy` | एक वैल्यू की एक्सप्लिसिट कॉपी |
| `BufferView` | offset/stride के साथ मौजूदा बफ़र में व्यू |
| `MStack` | मेमोरी स्टैक एलोकेशन |
| `MSelect` | मेमोरी सिलेक्ट (कंडीशनल मेमोरी एक्सेस) |
| `Multi` | मल्टी-आउटपुट ऑपरेशन |
| `Assign` | वेरिएबल असाइनमेंट |
| `Group` | शेड्यूलिंग के लिए ऑपरेशन ग्रुप करें |
| `Detach` | ग्राफ़ से डिटैच (ऑप्टिमाइज़ेशन रोकें) |
| `Contiguous` | कॉन्टिग्यूअस डेटा का हिंट |
| `ContiguousBackward` | कॉन्टिग्यूअस हिंट का बैकवर्ड पास |
| `Precast` | टाइप कन्वर्शन के लिए प्री-कास्ट |
| `Custom` / `CustomI` | कस्टम ऑपरेशन एक्सटेंसिबिलिटी |

---

## क्विक रेफ़रेंस

### कैटेगरी अनुसार

| कैटेगरी | ऑपरेशन |
|---------|--------|
| **लूप कंट्रोल** | `RANGE`, `END` |
| **रिडक्शन** | `REDUCE_AXIS`, `REDUCE`, `ALLREDUCE` |
| **मेमोरी** | `BUFFER`, `BUFFERIZE`, `INDEX`, `POINTER_INDEX`, `LOAD`, `STORE` |
| **कर्नेल** | `KERNEL`, `SINK`, `AFTER`, `BARRIER` |
| **वेक्टर** | `VECTORIZE`, `GEP`, `VCONST`, `CAT`, `PTRCAT` |
| **Expansion** | `UNROLL`, `CONTRACT` |
| **हार्डवेयर** | `WMMA`, `SPECIAL` |
| **कंट्रोल** | `IF`, `ENDIF` |
| **डेफ़िनिशन** | `PARAM`, `DEFINE_LOCAL`, `DEFINE_VAR`, `DEFINE_REG`, `BIND`, `UNIQUE`, `DEVICE` |
| **मूवमेंट** | `RESHAPE`, `PERMUTE`, `EXPAND`, `PAD`, `SHRINK`, `FLIP` |
| **ALU** | `Unary(...)`, `Binary(...)`, `Ternary(...)`, `Cast`, `BitCast` |

### Range-Ending ऑपरेशन

वे ऑपरेशन जो RANGE स्कोप बंद करते हैं (ranges को एक्टिव सेट से हटाते हैं):

| ऑपरेशन | Range स्टार्ट Index |
|---------|-------------------|
| `BUFFERIZE` | 1 (compute=0, ranges=1+) |
| `REDUCE` | 1 (src=0, ranges=1+) |
| `STORE` | 2 (index=0, value=1, ranges=2+) |
| `WMMA` | 3 (a=0, b=1, c=2) |
| `END` | 1 (computation=0, ranges=1+) |

### Expandable ऑपरेशन

वे ऑपरेशन जो UNROLL को कम्प्यूटेशन ग्राफ़ में प्रोपेगेट करते हैं:

- ALU: `Unary`, `Binary`, `Ternary`
- Type: `Cast`, `BitCast`
- Vector: `Gep`, `Vectorize`
- Memory: `Load`, `Store`, `Index`, `PointerIndex`
- Control: `Reduce`, `End`, `After`
- Buffer: `Bufferize`
- Hardware: `Wmma`
