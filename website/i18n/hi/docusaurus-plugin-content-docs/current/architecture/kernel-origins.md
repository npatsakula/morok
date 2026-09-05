---
sidebar_label: कर्नेल Origins
---

# कर्नेल Origins

जो profile बताता है कि `r_128_3_32_4_2_2_2_4_4_192_2` ने 100 ms लिए, वह कर्नेल की shape बताता
है — यह नहीं कि कर्नेल किसका है। Origins इसी दूसरे सवाल का जवाब देते हैं: dispatch होने वाला हर
कर्नेल जानता है कि वह किस module path, call site या ONNX node के लिए बना था, और profiler समय को
उसी path के साथ rollup कर सकता है — per layer, per block, per stage।

यह पेज इस्तेमाल का guide है: इसे चालू कैसे करें, model को instrument कैसे करें और output कैसे
पढ़ें। mechanism — हर node पर एक hash-consed field, जिसे kernel cut फिर से हटा देता है — का सार
अंत में है; पूरा ब्यौरा [IR design](./ir-design) और [op bestiary](./op-bestiary) पेजों पर है।

---

## इसे चालू करना

Capture default में बंद रहता है, और बंद रहते हुए इसकी कोई क़ीमत नहीं: nodes कोई origin नहीं ढोते,
और hashes उस build के hashes से byte-identical रहते हैं जिसमें यह feature है ही नहीं। दो
switches:

| Switch | असर |
|--------|--------|
| `SVOD_ORIGIN=1` | पूरे process के हर thread पर capture चालू |
| `SVOD_ORIGIN_DEPTH=<n>` | rollups पहले `n` path segments रखते हैं (unset या `0` = पूरा path) |

```bash
SVOD_DEVICE=AMD:0 SVOD_ORIGIN=1 cargo run --release -p svod-model --example gigaam_infer -- \
    audio.wav --profile --origin-depth 3 --profile-json profile.json
```

Tests में capture सिर्फ़ मौजूदा thread पर चालू करें, ताकि साथ-साथ चलने वाले tests अपनी graph
identity बनाए रखें:

```rust
let _capture = svod_ir::origin::capture_for_thread(true); // restored on drop
```

---

## Origins आते कहाँ से हैं

origin frames का एक path है — root सबसे पहले। हर frame इनमें से कोई एक होता है:

| Frame | कैसे दिखता है | कौन खोलता है |
|-------|-------------|-----------|
| `Module` | `encoder.layers.3.ffn1` | model code, हर module पर एक segment |
| `Label` | `ctc_head`, `initializer` | pipeline stages, ONNX importer, embedders |
| `Onnx` | `/encoder/Conv` या `#12:MatMul` | ONNX importer, हर node और subgraph branch पर एक |
| `Call` | `@ matmul model/src/gigaam/encoder.rs:262` | हर public `Tensor` op, अपने आप |

`Call` frame module path के नीचे बैठी flat file:line layer है। हर public op इसे अपने entry पर
खोलता है और सबसे बाहरी frame जीतता है, इसलिए दूसरी ops के ऊपर लिखी op (`matmul` के ऊपर `linear`)
user की line एक ही बार दर्ज करती है, svod का अपना source कभी नहीं। उसके ऊपर की module layers वही
हैं जो model code जोड़ता है।

### एक Rust model को instrument करना

हर module के `forward` में scope ठीक उसी नाम से खोलें जिस नाम से आप उसका state-dict prefix
लिखते हैं। model crate में यही काम करने वाले helpers मौजूद हैं:

```rust
use svod_ir::origin::OriginScope;
use crate::state::{scoped, scoped_index};

fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let x = scoped("subsampling", || self.subsampling.forward(x))?;
    let mut x = x;
    for (i, layer) in self.layers.iter().enumerate() {
        x = scoped_index("layers", i, || layer.forward(&x))?;   // layers.0, layers.1, …
    }
    scoped("final_norm", || self.final_norm.forward(&x))
}
```

हर module सिर्फ़ अपना segment खोलता है; nesting से पूरा path अपने आप बन जाता है, इसलिए profile
में जो path छपता है वह उन weights के state-dict key prefix जैसा ही होता है जिन्हें कर्नेल ने छुआ।
GigaAM और Whisper इसी तरह instrument किए गए हैं, और एक test assert करता है कि paths के दोनों sets
मेल खाते हैं।

Pipeline stages root पर labels हैं:

```rust
let _stage = OriginScope::label("ctc_head");
let plan = model.prepare_with_config(&config)?;   // every kernel below is ctc_head.…
```

किसी भी scope के बाहर बनी हर चीज़ `<unattributed>` row में जा गिरती है।

### ONNX graphs

कुछ करने की ज़रूरत नहीं। Importer हर node के लिए एक `Onnx` frame खोलता है (index, name, op type,
domain, opset), और हर subgraph branch (`then_branch`, `else_branch`) के लिए एक `Label` — ठीक उसी
node के नीचे जिसका वह branch है, इसलिए किसी `If` की body `#7:If.then_branch.#0:Add` जैसी दिखती है।
Initializers और graph inputs `initializer` और `input` के नीचे बैठते हैं।

### हाथ से लिखे कर्नेल

एक `tk` कर्नेल उसी scope के खाते में जाता है जो उसे बनाते समय सक्रिय था — वही नियम जो graph
कर्नेल पर लागू है। Scheduler उसकी body कभी नहीं देखता, इसलिए kernel constructor ख़ुद ही origin
harvest करके हटा देता है; एक ही हाथ के कर्नेल को launch करने वाली दो layers अब भी एक ही compiled
program साझा करती हैं।

---

## Output पढ़ना

Capture चालू हो तो `--profile` वही आम per-kernel table print करता है और उसके नीचे दो
rollups। यह sample GigaAM v3 encoder का है — f16, gfx1151 पर एक 60 s window, depth 3 पर काटा हुआ:

```
519 dispatches (519 GPU-stamped), total 444.237 ms
  total ms  count    mean µs      %  name
   103.183     16     6448.9   23.2  r_128_3_32_4_2_2_2_4_4_192_2n1
   100.305     16     6269.1   22.6  r_128_3_32_4_2_2_2_4_4_192_2
    80.530     32     2516.6   18.1  r_128_12_32_4_2_2_2_4_4_48_2
    …
origin rollup (depth 3, exclusive; rows sum to the total):
  total ms  count    mean µs      %  origin path
    27.833     32      869.8    6.3  ctc_head.GigaAmCtcJit.layers.3
    27.678     32      864.9    6.2  ctc_head.GigaAmCtcJit.layers.9
    27.620     32      863.1    6.2  ctc_head.GigaAmCtcJit.layers.0
    …
    23.334      2    11666.8    5.3  ctc_head.GigaAmCtcJit.subsampling
     0.661      4      165.2    0.1  ctc_head.GigaAmCtcJit.head
     0.131      1      131.0    0.0  ctc_head.GigaAmCtcJit
     0.007      1        6.6    0.0  <unattributed>
origin rollup (depth 3, inclusive; parents contain children, rows overlap):
  total ms  count    mean µs      %  origin path
   444.237    519      855.9  100.0  ctc_head
   444.237    519      855.9  100.0  ctc_head.GigaAmCtcJit
    27.833     32      869.8    6.3  ctc_head.GigaAmCtcJit.layers.3
    …
```

इसे कैसे पढ़ें:

- **Exclusive** हर dispatch को एक ही बार, उसके *primary* origin के खाते में डालता है — यानी उस
  scope के, जिसने वह value बनाई जो कर्नेल store करता है। Rows मिलकर पूरा कुल बनाती हैं, इसलिए
  सोलह `layers.N` rows के साथ `subsampling`, `head` और बची हुई `GigaAmCtcJit` row जोड़ने पर
  444 ms आते हैं। सोलह layers, हर एक पर 32 dispatches — यही पूरा encoder है; per-layer फैलाव
  (25.3 से 27.8 ms) असली है, और सबसे पहले नज़र इसी पर जाएगी।
- **Inclusive** हर dispatch को उसमें fuse हुए हर origin के हर ancestor के खाते में डालता है।
  Parent row अपने children को समेट लेती है, इसलिए `ctc_head` 100 % है और rows overlap करती हैं।
  इससे यह दिखता है कि किसी block का कितना समय उन कर्नेल में छिपा है जो module boundaries के पार
  fuse हो गए।
- **Depth** तय करती है कि path के कितने segments रखे जाएँ। यहाँ depth 3 per-layer rows देती है;
  depth 4 एक layer को `ffn1`, `mhsa`, `conv`, `ffn2`, `final_norm` में बाँट देती है; leaf पूरा
  path रखता है। `Call` frames कभी rollup key नहीं बनते — वे कर्नेल rows में और JSON में सिर्फ़
  detail हैं।
- जो कर्नेल दो modules को fuse कर देता है, वह exclusive rollup में सिर्फ़ उसी के खाते में जाता है
  जिसकी value वह store करता है (residual add layer पर गिरता है, `ffn2` पर नहीं), और inclusive
  में दोनों के।

`Whisper` भी यही section `render_table()` से print करता है — यह कोई भी `RunProfile` करता है।

### JSON

`--profile-json out.json` (या `RunProfile::to_json()`) हर run के लिए एक document लिखता है:

```json
{
  "origin_depth": 3,
  "stages": [{
    "name": "ctc_head", "wall_ms": 463.8, "gpu_ms": 444.2, "dispatches": 519,
    "kernels": [{
      "name": "r_128_3_32_4_2_2_2_4_4_192_2", "count": 1, "total_ms": 6.3,
      "origin": "ctc_head.GigaAmCtcJit.layers.3 @ add model/src/gigaam/encoder.rs:746",
      "origin_id": 41, "origins": ["…"], "origin_ids": [41, 39]
    }],
    "origins_exclusive": [{ "path": "ctc_head.GigaAmCtcJit.layers.3", "count": 32, "total_ms": 27.8, "percent": 6.3, "kernels": [] }],
    "origins_inclusive": []
  }],
  "origins": [{ "id": 41, "parent": 40, "frame": { "Module": { "name": "layers.3" } } }]
}
```

कर्नेल rows की key entry point *और* primary origin, दोनों मिलकर बनाते हैं, इसलिए एक ही program
हर उस scope के लिए एक बार दिखता है जिसने उसे dispatch किया। `origins` में सिर्फ़ वे frames होते
हैं जिन्हें run ने छुआ, अपने सारे `parent` समेत — इसलिए ids उस process के बिना भी resolve हो जाती
हैं जिसने file लिखी थी।

---

## Threads

Capture state हर thread का अपना होता है: switch, मौजूदा scope, और यह कि वह scope call frame है
या नहीं। Scopes काम के पीछे-पीछे दूसरे threads पर नहीं जाते; scope guard `!Send` है और उसी thread
को restore करता है जिस पर वह खोला गया था। इससे निकलने वाले नियम:

- Graph उसी thread पर बनाएँ जिसने scopes खोले थे। GigaAM और Whisper यही करते हैं;
  `prepare_with_config` के इर्द-गिर्द खोला गया stage label उसके अंदर बनी हर चीज़ को समेट लेता है।
- Scheduling और compiling detached चलते हैं (`OriginScope::suspend`) — caller पर और rayon
  workers पर, दोनों जगह एक जैसे — ताकि कोई ambient scope कभी kernel body में न रिस जाए;
  attribution तब तक CALL पर harvest हो चुकी होती है।
- किसी ऐसे worker तक scope ले जाने के लिए जिसे आप ख़ुद spawn करते हैं, `origin::current()`
  capture करें और वहाँ `origin::install(id)` से उसे फिर से install करें। Workers अपना switch
  किसी भी दूसरे thread की तरह `SVOD_ORIGIN` से seed करते हैं।
- BEAM search एक child process में origin-free kernel bodies पर चलता है; वह कोई scope कभी नहीं
  देखता।
- **Async code:** scopes का nest होना ज़रूरी है, इसलिए किसी scope को `.await` के पार पकड़े न
  रखें। scope खोलें, graph synchronously बनाएँ, उसे drop करें, फिर await करें। Guard `!Send`
  है, इसलिए जो future किसी guard को await के पार ज़िंदा रखता है उसे multi-threaded executor पर
  spawn नहीं किया जा सकता, और जब दो tasks एक ही thread पर scopes को आपस में गूँथ देती हैं (एक
  guard drop हो गया जबकि उसके बाद वाला अब भी सक्रिय है) तो debug build panic कर देता है। svod में
  graph construction synchronous है, इसलिए code की स्वाभाविक बनावट यह शर्त पहले से पूरी करती है।

---

## लागत और trade-offs

- **बंद:** कुछ नहीं। हर node पर एक thread-local read, कोई allocation नहीं, hashes में कोई बदलाव
  नहीं।
- **चालू:** हर scope entry पर एक interning (arena पर एक mutex, हर forward में सैकड़ों बार), call
  frame के लिए हर public op पर एक thread-local write, और cut पर union harvest करने के लिए हर
  कर्नेल पर एक toposort। capture चालू हो या बंद, GigaAM के dispatch counts और GPU time एक जैसे
  ही रहते हैं।
- **Identity बदल जाती है।** Origin किसी node की identity का हिस्सा है, इसलिए अलग-अलग scopes में
  बने दो एक जैसे expressions तब तक दो nodes रहते हैं जब तक cut origins हटा नहीं देता। कर्नेल
  programs पर इसका असर नहीं पड़ता — strip के बाद dedup लौट आता है — पर जो helper हर call site पर
  वही expression दोबारा बनाता है (एक mask clamp, एक table cast, एक input copy), वह उसे हर scope
  पर एक बार materialise कर देगा। ऐसे helpers `OriginScope::suspend()` के अंदर चलाएँ, या copy को
  अपने producer का origin विरासत में लेने दें; `custom_kernel` अपने inputs के लिए यही पहले से
  करता है। इसी वजह से constants, buffers और params कभी origin नहीं ढोते।
- **जो tests structural identity पर टिके हैं** (हाथ से बने दो ऐसे graphs जिनसे उम्मीद है कि वे
  hash-cons होकर एक ही node बन जाएँ) उन्हें `capture_for_thread(false)` के साथ चलाएँ।

---

## यह काम कैसे करता है, एक पैराग्राफ़ में

किसी scope के सक्रिय रहते बना हर `UOp` उस scope का 4-byte `OriginId` रखता है और उसे अपने content
hash में मिला देता है, इसलिए अलग-अलग scopes के एक जैसे subgraphs rangeify तक अलग बने रहते हैं।
kernel cut पर `split_store` body को एक बार walk करता है, store की गई value का origin primary के
तौर पर और बाक़ी सबका union set के तौर पर लेता है, दोनों को कर्नेल के `CALL` की `CallInfo` पर stamp
करता है, और body को origins हटाकर फिर से बनाता है। cut के बाद की हर चीज़ — optimizer, BEAM,
codegen, हर kernel cache — origin-free ASTs ही देखती है। Plan CALL की attribution को हर prepared op पर कॉपी
करता है, profiler उसे हर `KernelProfile` पर, और rollups parent chain को माँगी गई depth तक काट
देते हैं।
