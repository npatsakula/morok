---
sidebar_label: Op Bestiary
---

# Op Bestiary: UOp ऑपरेशनों की फ़ील्ड गाइड

Svod IR डंप डीबग करते समय आपको ऐसे ऑपरेशन मिलेंगे जो नाम से स्पष्ट नहीं होते। यह चैप्टर नॉन-ट्रिवियल ऑपरेशनों को सिग्नेचर, फ़ील्ड एक्सप्लेनेशन और उदाहरणों के साथ डॉक्यूमेंट करता है।

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
| `Placeholder` | -3 | — | RESHAPE कैशिंग के दौरान इस्तेमाल होने वाला अस्थायी कैनोनिकल range |
| `Device` | -2 | — | डिवाइस-चयन डायमेंशन, launch पर हर डिवाइस के लिए bind होता है |
| `Weak` | -1 | `for` लूप | rangeify द्वारा उत्पन्न अनपैरेललाइज़्ड range |
| `Loop` | -1 | `for` लूप | एक्सप्लिसिट रेगुलर लूप; schedule-स्तर के रैपर `END(Call)` पेयर के ज़रिए स्ट्रक्चरली पहचाने जाते हैं |
| `Global` | 0 | `blockIdx` | ग्रिड पैरेललिज़्म |
| `Thread` | 0 | thread pool | CPU पैरेललिज़्म |
| `Warp` | 1 | warp/wavefront | सब-ग्रुप पैरेललिज़्म |
| `Local` | 2 | `threadIdx` | वर्कग्रुप पैरेललिज़्म |
| `GroupReduce` | 2 | shared memory | दो-स्टेज रिडक्शन |
| `Upcast` | 3 | SIMD | वेक्टराइज़ेशन |
| `Reduce` | 4 | accumulator | रिडक्शन डायमेंशन |
| `Unroll` | 5 | unrolled | लूप अनरोलिंग |

प्रायोरिटी लूप नेस्टिंग ऑर्डर तय करती है — कम वैल्यू वाले आउटर लूप होते हैं। कर्नेल बाउंड्री `Call`/`Function` के ज़रिए स्ट्रक्चरली व्यक्त होती है, इसके लिए कोई अलग ऐक्सिस टाइप नहीं है।

**उदाहरण:**
```mermaid
flowchart TD
  R["RANGE(end=128, axis_id=R0, type=Global)"] --> C["CONST(128) : Index"]
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
```mermaid
flowchart TD
  E["END"] -->|"computation"| S["STORE(...)"]
  E -->|"first range closed"| R0["RANGE(R0, Global)"]
  E -->|"second range closed"| R1["RANGE(R1, Local)"]
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
```mermaid
flowchart TD
  RA["REDUCE_AXIS(Add, axes=[1])"] --> B["BUFFER[10, 20] : Float32"]
```

यह `[10, 20]` tensor को axis 1 पर sum करके `[10]` में बदलता है।

### REDUCE — Range इटरेशन रिडक्शन (लो-लेवल)

```rust
Reduce {
    src: Arc<UOp>,                      // value to accumulate
    ranges: SmallVec<[Arc<UOp>; 4]>,    // ranges being reduced
    reduce_op: ReduceOp,                // Add, Mul, Max, Min
    num_axes: usize,                    // reduced axes of the shaped source
}
```

**Rangeify के बाद** इस्तेमाल होता है। RANGE इटरेशन के दौरान वैल्यूज़ accumulate करता है और स्पेसिफ़ाइड ranges बंद करता है।

**ReduceOp वैरिएंट:**

| Op | आइडेंटिटी | ऑपरेशन | Tinygrad |
|----|-----------|---------|----------|
| `Add` | 0 | `acc + value` | ✓ |
| `Mul` | 1 | `acc * value` | ✓ |
| `Max` | -∞ | `max(acc, value)` | ✓ |
| `Min` | +∞ | `min(acc, value)` | केवल Svod |

> **कम्पैटिबिलिटी:** Tinygrad का स्पेक REDUCE_AXIS को `{Add, Mul, Max}` तक सीमित रखता है। Svod इसे `Min` के साथ एक्सटेंड करता है।

**उदाहरण:**
```mermaid
flowchart TD
  RED["REDUCE(Add)"] -->|"value to accumulate"| MUL["MUL"]
  MUL --> LA["LOAD(A, ...)"]
  MUL --> LB["LOAD(B, ...)"]
  RED -->|"range being reduced"| R2["RANGE(R2, Reduce)"]
  R2 --> C["CONST(64)"]
```

### ALLREDUCE — क्रॉस-डिवाइस रिडक्शन

```rust
AllReduce {
    src: Arc<UOp>,           // local partial result
    device: DeviceSpec,      // device specification
    reduce_op: ReduceOp,     // reduction operation
}
```

कई डिवाइसों में डिस्ट्रिब्यूटेड रिडक्शन करता है। मल्टी-GPU ट्रेनिंग के लिए इस्तेमाल होता है।

---

## बफ़र ऑपरेशन

### BUFFER — बफ़र डिक्लेरेशन

```rust
Buffer {
    shape: Arc<UOp>,         // flat storage shape (one element count)
    arg: Box<ParamArg>,      // slot, dtype, address space, device
}
```

Tensor स्टोरेज के लिए बफ़र डिक्लेयर करता है। `arg.slot` फ़ील्ड यह सुनिश्चित करती है कि समान size/device होने पर भी बफ़र अलग रहें। `arg.addrspace` तय करती है कि बफ़र किस मेमोरी में रहेगा: डिवाइस मेमोरी के लिए `Global`, GPU shared memory (LDS) के लिए `Local`, और रजिस्टर/scratch एलोकेशन के लिए `Reg`।

### STAGE — मटेरियलाइज़ेशन मार्कर

```rust
Stage {
    compute: Arc<UOp>,                  // computation to materialize
    ranges: SmallVec<[Arc<UOp>; 4]>,    // output dimensions
    opts: Box<BufferizeOpts>,           // address space, device
}
```

मार्क करता है कि कम्प्यूटेशन को मेमोरी में मटेरियलाइज़ होना चाहिए। कर्नेल स्प्लिटिंग ट्रिगर करता है।

**BufferizeOpts:**

| फ़ील्ड | टाइप | उद्देश्य |
|--------|------|----------|
| `device` | `Option<DeviceSpec>` | टारगेट डिवाइस, लोकल के लिए `None` |
| `local_axis` | `Option<AxisId>` | वह `GroupReduce` ऐक्सिस जो एक LOCAL स्टेजिंग बफ़र का मालिक है |
| `addrspace` | `AddrSpace` | `Global` (डिवाइस) या `Local` (shared) |
| `removable` | `bool` | `false` होने पर `buffer_removal` को इस STAGE को इनलाइन करने की अनुमति नहीं — मल्टी-कंज़्यूमर realize बाउंड्री पर इस्तेमाल होता है ताकि बफ़र मेगा-pass फ़िक्सपॉइंट इटरेशन के बीच टिका रहे |

**उदाहरण:**
```mermaid
flowchart TD
  BZ["STAGE(opts=(addrspace=Global))"] -->|"computation"| RED["REDUCE(Add, ...)"]
  BZ -->|"output dim 0"| R0["RANGE(R0, Global)"]
  BZ -->|"output dim 1"| R1["RANGE(R1, Global)"]
```

### INDEX — मल्टी-डायमेंशनल बफ़र एक्सेस

```rust
Index {
    buffer: Arc<UOp>,                   // BUFFER, PARAM or STACK
    indices: SmallVec<[Arc<UOp>; 4]>,   // index per dimension
}
```

मल्टी-डायमेंशनल indices से मेमोरी एड्रेस कैलकुलेट करता है। एलिमेंट dtype रिटर्न करता है (पॉइंटर नहीं)। किसी index को `idx.valid(cond)` से कंडीशनल बनाया जा सकता है, जो उसे `WHERE(cond, idx, INVALID)` में रैप कर देता है। किसी `STACK` पर INDEX एड्रेस के बजाय एक lane चुनता है: कॉन्स्टेंट स्केलर index सीधे stacked source में फ़ोल्ड हो जाता है।

**उदाहरण:**
```mermaid
flowchart TD
  IDX["INDEX : Float32"] --> P["PARAM(0)"]
  IDX -->|"index for dim 0"| R0["RANGE(R0, Global)"]
  IDX -->|"index for dim 1"| R1["RANGE(R1, Loop)"]
  IDX -->|"index for dim 2"| M["MUL(...)"]
```

### LOAD — मेमोरी रीड

```rust
Load {
    index: Arc<UOp>,         // INDEX op (buffer accessed via the INDEX)
    alt: Option<Arc<UOp>>,   // alternative value for gated loads
    gate: Option<Arc<UOp>>,  // predicate for gated loads
}
```

बफ़र से index पर वैल्यू रीड करता है; अलग `buffer` फ़ील्ड नहीं है, बफ़र तक INDEX नोड के ज़रिए पहुँचा जाता है। गेटेड loads के लिए, `gate` false होने पर `alt` वह वैल्यू देती है (और मेमोरी एक्सेस पूरी तरह टल जाता है)। `alt` और `gate` हमेशा साथ सेट होते हैं: कोई load या तो दोनों रखता है या एक भी नहीं। रेंडरर सिंगल-ऐक्सिस `INDEX` की माँग करते हैं, इसलिए मल्टी-इंडेक्स एक्सेस को load के कोड जनरेशन तक पहुँचने से पहले rangeify के दौरान फ़्लैटन कर देना ज़रूरी है।

**उदाहरण:**
```mermaid
flowchart TD
  L["LOAD : Float32"] --> IDX["INDEX"]
  IDX --> P1["PARAM(1)"]
  IDX --> R0["RANGE(R0)"]
  IDX --> R2["RANGE(R2)"]
```

### STORE — मेमोरी राइट

```rust
Store {
    index: Arc<UOp>,                    // INDEX op (buffer accessed via index.src[0])
    value: Arc<UOp>,                    // value to write
    gate: Option<Arc<UOp>>,             // predicate for gated stores
}
```

बफ़र में वैल्यू लिखता है। बफ़र INDEX नोड के ज़रिए एक्सेस होता है (`index.src[0]` से), अलग फ़ील्ड से नहीं। expansion के दौरान `UPCAST` और `UNROLL` range ऐक्सिस वर्गीकरण ही बने रहते हैं।

गेटेड stores के लिए `store_gated` `gate` सेट करता है; `pm_move_gates_from_index` ही वह है जो gate को address expression से उठाकर LOAD/STORE पर रख देता है।

> **कम्पैटिबिलिटी:** Svod के STORE में अलग `buffer` फ़ील्ड नहीं है — sources हैं: index=0, value=1। किसी STAGE या REDUCE के उलट, STORE कोई ranges बंद नहीं करता।

**उदाहरण:**
```mermaid
flowchart TD
  ST["STORE"] -->|"write address (buffer via index.src[0])"| IDX["INDEX[R0, R1]"]
  ST -->|"value"| RED["REDUCE(Add, ...)"]
  IDX --> R0["RANGE(R0, Global)"]
  IDX --> R1["RANGE(R1, Global)"]
```

---

## कर्नेल स्ट्रक्चर और कॉलेबल IR

Schedule-स्तर का काम एक कॉलेबल IR के ज़रिए व्यक्त होता है जो tinygrad
के `CALL`/`FUNCTION`/`PROGRAM` मॉडल के अनुरूप है: `Function` एक बॉडी
(आमतौर पर stores का `Sink`) को परिभाषित करता है जिसे आर्ग्युमेंट से
पैरामीट्राइज़ किया जाता है, `Call` कंक्रीट आर्ग्युमेंट के साथ इसे invoke
करता है, और `Program` बॉडी को सख़्त `SINK → LINEAR → SOURCE → BINARY`
स्टेजिंग के ज़रिए कंपाइलेशन तक पहुँचाता है।

### CALL — फ़ंक्शन बॉडी invoke करना

```rust
Call {
    body: Arc<UOp>,                     // FUNCTION (या उसकी बॉडी)
    args: SmallVec<[Arc<UOp>; 4]>,      // कंक्रीट आर्ग्युमेंट वैल्यूज़
    info: Box<CallInfo>,                // ऐनोटेशन (name, origin, …)
}
```

आर्ग्युमेंट के साथ कॉलेबल बॉडी invoke करता है। Range-ending: `args` में
मौजूद किसी भी `Range` को क्लोज़ करता है (range_start_index = 1; `body=0`,
`args=1+`)।

`CallInfo` कैश-कुंजी के लिए सुरक्षित ऐनोटेशन कैरी करता है:

| फ़ील्ड | टाइप | उद्देश्य |
|--------|------|----------|
| `name` | `Option<String>` | इंसान के पढ़ने योग्य कॉलेबल नाम |
| `grad_tag` | `Option<String>` | फ़्यूचर ग्रेडिएंट-कॉलबैक आइडेंटिटी के लिए रिज़र्व |
| `origin` | `Option<OriginId>` | स्टोर की गई वैल्यू के रूट का origin — कर्नेल किसके खाते में जाता है |
| `origins` | `OriginSet` | स्ट्रिप होने से पहले बॉडी से जितने origin पहुँच में थे, वे सब |
| `precompile` / `precompile_backward` | `bool` | प्री-कंपाइल हिंट |

किसी dispatch की जो attribution profiler की rollups पढ़ती हैं, वह कर्नेल के CALL पर ही रहती है;
देखें [Profiling और Benchmarking](../tile-kernels/profiling.md)।

### FUNCTION — री-यूज़ेबल बॉडी

```rust
Function {
    body: Arc<UOp>,                     // कंप्यूटेशन
    args: SmallVec<[Arc<UOp>; 4]>,      // फ़ॉर्मल पैरामीटर
    info: Box<CallInfo>,
}
```

री-यूज़ेबल कॉलेबल। इसका dtype हमेशा `Void` होता है; जो बॉडी कई वैल्यू
रिटर्न करती है उसे `Tuple` में रैप किया जाता है ताकि फ़ंक्शन बाउंड्री
Void बनी रहे। Range-ending आकार `Call` जैसा ही है।

### TUPLE / GET_TUPLE — मल्टी-वैल्यू रिटर्न

```rust
Tuple { src: SmallVec<[Arc<UOp>; 4]> }
GetTuple { src: Arc<UOp>, index: usize }
```

`Tuple` विषम वैल्यूज़ को पैक करता है; इसका dtype हमेशा `Void` होता है।
`GetTuple` एक `Tuple` (या जिस `Function` की बॉडी `Tuple` है) से
`index` एलिमेंट निकालता है; इसका dtype अंदरूनी एलिमेंट से मेल खाता है।
Void फ़ंक्शन बाउंड्री से कई आउटपुट गुज़ारने के लिए इस्तेमाल होता है।

### PROGRAM — कंपाइल-पाइपलाइन कंटेनर

```rust
Program {
    sink: Arc<UOp>,                     // रूट SINK
    info: Box<ProgramInfo>,             // नाम, launch dims, ABI slots, target
    linear: Option<Arc<UOp>>,           // LINEAR (linearize के बाद)
    source: Option<Arc<UOp>>,           // SOURCE (render के बाद)
    binary: Option<Arc<UOp>>,           // PROGRAM_BINARY (compile के बाद)
}
```

`codegen/src/program_pipeline.rs` के ज़रिए लागू होने वाले `SINK → LINEAR
→ SOURCE → PROGRAM_BINARY` स्टेजिंग (`do_linearize`/`do_render`/
`do_compile`/`get_program`) से कर्नेल को गुज़ारता है। हर स्टेज अगला
फ़ील्ड भरती है। C/LLVM रेंडरर `Op::Linear` इनपुट की उम्मीद रखते हैं
और panic के बजाय per-context `pending_error` के ज़रिए
`Error::InvalidGraph` रिपोर्ट करते हैं; रेंडरर तक पहुँचने वाला
मल्टी-इंडेक्स `INDEX` भी इसी तरह अस्वीकार होता है, इसलिए indices को पहले
ही एक सिंगल ऐक्सिस तक फ़्लैटन कर देना चाहिए।

### LINEAR — लीनियराइज़्ड ऑप स्ट्रीम

```rust
Linear { ops: SmallVec<[Arc<UOp>; 8]> }
```

लीनियराइज़ेशन से उत्पन्न ऑप्स का फ़्लैट क्रम। उपभोक्ता ग्राफ़ को फिर से
ट्रैवर्स किए बिना सीधे `ops` पर इटरेट कर सकते हैं।

### SOURCE / PROGRAM_BINARY — कंपाइलेशन आर्टिफ़ैक्ट्स

```rust
Source { code: String, identity: Option<Box<SourceStageIdentity>> }
ProgramBinary { bytes: Vec<u8>, identity: Option<Box<BinaryStageIdentity>> }
```

प्रोग्राम पाइपलाइन की टर्मिनल स्टेजेज़। दोनों लीफ़ हैं (कोई चाइल्ड नहीं)।
ऑप्शनल `identity` वह सिमैंटिक प्रूफ़ है जो एक स्टेज को ठीक उससे पहले वाली
स्टेज से बाँधती है (`SourceStageIdentity` ABI, target, entry नाम और
LINEAR/SOURCE digests रखती है; `BinaryStageIdentity` उसे compiler key और
binary digest के साथ लपेटती है), ताकि कोई कैश्ड आर्टिफ़ैक्ट बदले हुए ग्राफ़
पर दोबारा इस्तेमाल न हो सके।

### SINK — मल्टीपल रूट कलेक्टर

```rust
Sink {
    sources: SmallVec<[Arc<UOp>; 4]>,
    info: Option<Box<KernelInfo>>,      // कर्नेल AST के लिए स्ट्रक्चरल मार्कर
}
```

कई आउटपुट को एक सिंगल रूट में कलेक्ट करता है। `Function` की बॉडी आमतौर
पर stores का `Sink` होती है। `info` फ़ील्ड एक हैश-कॉन्स्ड स्ट्रक्चरल
मार्कर है जो टाइप-इरेज़्ड साइड-चैनल मेटाडेटा पर निर्भर हुए बिना कर्नेल-
AST SINK को बाकी समान-source SINK से अलग करता है।

**उदाहरण:**
```mermaid
flowchart TD
  SINK["SINK"] --> S0["STORE(output_0, ...)"]
  SINK --> S1["STORE(output_1, ...)"]
  SINK --> S2["STORE(output_2, ...)"]
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
```mermaid
flowchart TD
  SINK["SINK"] --> AF["AFTER"]
  AF -->|"passthrough (buffer reference)"| P0["PARAM(0)"]
  AF -->|"must complete first"| K1["KERNEL(...)"]
  SINK -->|"can use buffer after AFTER"| K2["KERNEL(...)"]
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

### STACK — lanes से एक shaped वैल्यू बनाएँ

```rust
Stack {
    sources: SmallVec<[Arc<UOp>; 4]>,
}
```

N वैल्यूज़ को N lanes वाली एक shaped वैल्यू में जोड़ता है। एलिमेंट dtype
स्केलर ही रहता है — lane count dtype को चौड़ा करके नहीं, बल्कि ख़ुद STACK
द्वारा ढोया जाता है — और sources कंस्ट्रक्शन पर promoted dtype में कास्ट कर
दिए जाते हैं।

**उदाहरण:**
```mermaid
flowchart TD
  V["STACK(len=4) : Float32"] --> C1["CONST(1.0)"]
  V --> C2["CONST(2.0)"]
  V --> C3["CONST(3.0)"]
  V --> C4["CONST(4.0)"]
```

### Lane सिलेक्शन — STACK पर INDEX

कोई अलग extract ऑपरेशन नहीं है। `INDEX` किसी `STACK` से एक lane ठीक उसी
तरह चुनता है जैसे वह किसी बफ़र से एड्रेस चुनता है, और कॉन्स्टेंट index
कंस्ट्रक्शन के समय ही सीधे stacked source में फ़ोल्ड हो जाता है।

**उदाहरण:**
```mermaid
flowchart TD
  G["INDEX : Float32"] --> V["STACK(len=4) : Float32"]
  G --> C["CONST(2) : Index"]
  V --> E["..."]
```

### VConst — वेक्टर कॉन्स्टेंट

```rust
VConst {
    values: Vec<ConstValue>,
}
```

कम्पाइल-टाइम कॉन्स्टेंट्स का वेक्टर। `CONST` नोड्स के `STACK` से ज़्यादा एफ़िशिएंट।

Lane aggregation के लिए `STACK` इस्तेमाल होता है; lane और address selection
के लिए `INDEX`। लूप अनरोलिंग को `AxisType::Unroll` वाले `Range` से दर्शाया
जाता है, किसी अलग ऑपरेशन से नहीं। Tensor-core expansion axes `WmmaMetadata`
में रहते हैं।

---

## Tensor Cores: WMMA

### WMMA — Warp Matrix Multiply-Accumulate

```rust
Wmma {
    a: Arc<UOp>,             // matrix A fragment
    b: Arc<UOp>,             // matrix B fragment
    c: Arc<UOp>,                 // accumulator C fragment
    metadata: Box<WmmaMetadata>, // hardware configuration
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
| `device` | `RendererDevice` | इस WMMA को उत्पन्न करने वाला रेंडरर / TC बैकएंड |
| `threads` | `usize` | प्रति warp threads (आमतौर पर 32) |
| `upcast_axes` | `Option<WmmaUpcastAxes>` | प्रति-source expansion axes (फ़ील्ड्स: `a`, `b`, `c`); `expander2` द्वारा sources और आउटपुट को shape दे देने के बाद क्लियर कर दी जाती हैं |
| `reduce_axes` | `Vec<AxisId>` | TC reduce ऐक्सिस IDs, expansion के दौरान `exclude_args` के रूप में इस्तेमाल |

**उदाहरण:**
```mermaid
flowchart TD
  W["WMMA(dims=(16, 16, 16), dtype_in=Float16, dtype_out=Float32)"] --> A["A fragment : 8 x Float16"]
  W --> B["B fragment : 8 x Float16"]
  W --> C["C accumulator : 8 x Float32"]
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
```mermaid
flowchart TD
  IF["IF"] -->|"condition (src[0])"| LT["LT(idx, bound)"]
  IF -->|"body[0]"| S0["STORE(...)"]
  IF -->|"body[1]"| S1["STORE(...)"]
  ENDIF["ENDIF"] -->|"references IF op"| IF
```

---

## डेफ़िनिशन ऑपरेशन

### PARAM — बफ़र पैरामीटर

```rust
Param { shape: Arc<UOp>, arg: Box<ParamArg> }
```

नॉर्मलाइज़्ड बफ़र पैरामीटर — इनपुट/आउटपुट बफ़र का पोज़िशनल रेफ़रेंस।
प्री-शेड्यूल नॉर्मलाइज़ेशन (BUFFER→PARAM) द्वारा बनाया जाता है ताकि बफ़र आइडेंटिटी हटाकर
आइडेंटिकल कम्प्यूटेशन का स्ट्रक्चरल डीडुप्लिकेशन हो सके।
`arg.slot` कर्नेल आर्ग्युमेंट लिस्ट में पोज़िशन है, और `shape` एलिमेंट काउंट
ढोता है। `ParamArg` स्केलर पैरामीटर (`UOp::scalar_param`) भी कवर करता है, जो
एड्रेस स्पेस के बजाय एक नाम और वैल्यू bounds रखते हैं।

### Shared मेमोरी और रजिस्टर

कोई समर्पित `DefineLocal` या `DefineReg` ऑपरेशन नहीं है। GPU shared memory
(LDS) और रजिस्टर/scratch एलोकेशन ऐसे `Buffer` नोड हैं जिनकी `arg.addrspace`
`AddrSpace::Local` या `AddrSpace::Reg` होती है; वे कोई डिवाइस नहीं ढोते और
सिर्फ़ एक वर्कग्रुप (LOCAL) या एक thread (REG) के अंदर विज़िबल होते हैं।

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
```mermaid
flowchart TD
  SP["SPECIAL(name=blockIdx.x, end=128) : Index"] --> C["CONST(128)"]
```

### UNIQUE / LUNIQUE — आइडेंटिटी मार्कर

```rust
Unique(usize)                // ग्लोबल आइडेंटिटी काउंटर
LUnique(usize)               // लोकल-स्कोप आइडेंटिटी काउंटर
```

बफ़र disambiguation के लिए यूनीक आइडेंटिटी बनाता है। अलग `Unique` वैल्यू
वाले दो बफ़र अलग माने जाते हैं, भले ही बाकी सब समान हो। `LUnique` लोकल
स्कोप (जैसे `Function` बॉडी) के अंदर वही disambiguation देता है, बिना
ग्लोबल काउंटर से टकराए — इससे कॉलेबल बॉडीज़ को कॉल साइट से स्वतंत्र
रूप से हैश-कॉन्स किया जा सकता है।

डिवाइस अपना कोई अलग नोड नहीं हैं: टारगेट उन ऑपरेशनों पर एक `DeviceSpec`
फ़ील्ड है जिन्हें उसकी ज़रूरत होती है (`Copy`, `GetAddr`, `AllReduce`,
`ParamArg.device`, `BufferizeOpts.device`, `ProgramInfo.target`)।

---

## मूवमेंट ऑपरेशन

हाई-लेवल tensor शेप ट्रांसफ़ॉर्मेशन। Rangeify के दौरान ये एक्सप्लिसिट INDEX ऑपरेशन में बदल जाते हैं।

| ऑपरेशन | सिग्नेचर | उद्देश्य |
|---------|----------|----------|
| `Reshape` | `{ src, new_shape }` | शेप बदलें, एलिमेंट्स वही |
| `Permute` | `{ src, axes: Vec<usize> }` | ट्रांसपोज़/रीऑर्डर axes |
| `Expand` | `{ src, new_shape }` | बड़ी शेप में ब्रॉडकास्ट |
| `Pad` | `{ src, begin_pads, end_pads }` | पैडिंग जोड़ें |
| `Shrink` | `{ src, offsets, sizes }` | सब-रीजन निकालें |
| `Flip` | `{ src, axes: Vec<bool> }` | axes के अनुसार रिवर्स |

**उदाहरण:** RESHAPE
```mermaid
flowchart TD
  RS["RESHAPE(new_shape=[6, 4]) : Shape[6, 4]"] --> B["BUFFER[2, 3, 4] : Float32"]
  RS --> C["CONST([6, 4]) : Shape"]
```

---

## अतिरिक्त ऑपरेशन

ये ऑपरेशन `Op` enum में हैं लेकिन इंटरनल हैं या डीबगिंग में कम दिखते हैं:

| ऑपरेशन | उद्देश्य |
|---------|----------|
| `Copy` | `{ src, device }` — किसी वैल्यू की दूसरे डिवाइस पर एक्सप्लिसिट कॉपी |
| `Slice` | `{ buffer, offset, size }` — buffer पर contiguous typed slice metadata |
| `GetAddr` | `{ src, device }` — किसी buffer-जैसे source का `UInt64` एड्रेस |
| `MStack` | `{ buffers }` — किसी मल्टी-डिवाइस tensor के प्रति-डिवाइस बफ़र |
| `MSelect` | `{ buffer, device_index }` — मल्टी-डिवाइस tensor में से एक डिवाइस का बफ़र |
| `Multi` | `{ src, axis }` — shard मार्कर: वह ऐक्सिस जिस पर मल्टी-डिवाइस tensor बँटा हुआ है |
| `Group` | शेड्यूलिंग के लिए ऑपरेशन ग्रुप करें |
| `Noop` | बिना ऑपरेंड और बिना किसी असर वाला प्लेसहोल्डर |
| `Detach` | ग्राफ़ से डिटैच (ऑप्टिमाइज़ेशन रोकें) |
| `Contiguous` | कॉन्टिग्यूअस डेटा का हिंट |
| `ContiguousBackward` | कॉन्टिग्यूअस हिंट का बैकवर्ड पास |
| `Precast` | टाइप कन्वर्शन के लिए प्री-कास्ट |
| `Custom` / `CustomI` | इनलाइन कस्टम ऑपरेशन एक्सटेंसिबिलिटी (`Custom` केवल C बैकएंड पर) |
| `CustomFunction` | रनटाइम कस्टम-फ़ंक्शन हुक (kinds: `EncDec`, `Graph`, `AllReduce`) |
| `Ins` | `{ sources, arg }` — किसी ISA रेंडरर द्वारा चुना गया target instruction |

---

## क्विक रेफ़रेंस

### कैटेगरी अनुसार

| कैटेगरी | ऑपरेशन |
|---------|--------|
| **लूप कंट्रोल** | `RANGE`, `END` |
| **रिडक्शन** | `REDUCE_AXIS`, `REDUCE`, `ALLREDUCE` |
| **मेमोरी** | `BUFFER`, `SLICE`, `STAGE`, `INDEX`, `LOAD`, `STORE`, `GETADDR` |
| **कर्नेल और कॉलेबल** | `SINK`, `CALL`, `FUNCTION`, `TUPLE`, `GET_TUPLE`, `PROGRAM`, `LINEAR`, `SOURCE`, `PROGRAM_BINARY`, `AFTER`, `BARRIER` |
| **वेक्टर** | `STACK`, `INDEX`, `VCONST` |
| **Expansion** | `AxisType::Upcast` या `AxisType::Unroll` वाला `RANGE` |
| **हार्डवेयर** | `WMMA`, `SPECIAL` |
| **कंट्रोल** | `IF`, `ENDIF` |
| **डेफ़िनिशन** | `PARAM`, `DEFINE_VAR`, `BIND`, `UNIQUE`, `LUNIQUE` |
| **मूवमेंट** | `RESHAPE`, `PERMUTE`, `EXPAND`, `PAD`, `SHRINK`, `FLIP` |
| **ALU** | `Unary(...)`, `Binary(...)`, `Ternary(...)`, `Cast`, `BitCast` |

### Range-Ending ऑपरेशन

वे ऑपरेशन जो RANGE स्कोप बंद करते हैं (ranges को एक्टिव सेट से हटाते हैं):

| ऑपरेशन | Range स्टार्ट Index |
|---------|-------------------|
| `STAGE` | 1 (compute=0, ranges=1+) |
| `REDUCE` | 1 (src=0, ranges=1+) |
| `WMMA` | 3 (a=0, b=1, c=2) |
| `END` | 1 (computation=0, ranges=1+) |
| `CALL` / `FUNCTION` | 1 (body=0, args=1+) |

### Expandable ऑपरेशन

वे ऑपरेशन जो expanded lanes को कम्प्यूटेशन ग्राफ़ में प्रोपेगेट करते हैं:

- ALU: `Unary`, `Binary`, `Ternary`
- Type: `Cast`, `BitCast`
- Shaped values: `Stack`
- Memory: `Load`, `Store`, `Index`
- Control: `Reduce`, `End`, `After`
- Buffer: `Stage`
- Hardware: `Wmma`
