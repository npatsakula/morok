---
sidebar_label: Phase 4 — Linearizer
---

# Phase 4: Linearizer

**गोल**: DAG को लीनियर इंस्ट्रक्शन सीक्वेंस में बदलें।

---

## Stage 16: Post-Index Symbolic

> **स्टेज एक नज़र में**
>
> **गोल**: Index lowering के बाद पूर्ण symbolic सिम्प्लीफ़िकेशन
> **मुख्य Patterns**: सभी symbolic नियम (140+)
> **प्रभाव**: सीरियलाइज़ेशन से पहले फ़ाइनल क्लीनअप

**यह क्या करता है**: Index lowering के बाद पूर्ण symbolic सिम्प्लीफ़िकेशन।

**यह क्यों ज़रूरी है**: अब indices कॉन्क्रीट integers (i32/i64) हैं, अरिथमेटिक पूरी तरह सिम्प्लीफ़ाई हो सकता है। लीनियराइज़ेशन से पहले एक्सप्रेशन साफ़ करने का यह आखिरी मौका है।

**Pattern**: `symbolic`

Svod में कोई GEP op नहीं है — addressing `INDEX(STACK(...))` है — इसलिए Tinygrad के
`gep_pushing` का यहाँ कोई समकक्ष नहीं। सबसे नज़दीकी analogue `alu_vectorize_reorder_patterns` है:
```text
Before:  ADD(STACK(x, x, x, x), STACK(y, y, y, y))
              ↓ [Reorder ALU over STACK]
After:   STACK(ADD(x, y), ADD(x, y), ADD(x, y), ADD(x, y))
```
*क्यों?* Collapse हुए ऑपरेशन पर constant folding और scalar ऑप्टिमाइज़ेशन सक्षम करता है। वह नियम tier-3 `sym()` में रहता है, इसलिए वह पहले ही Stage 14 पर चल चुका होता है, यहाँ नहीं।

---

## Stage 17: Pre-Matcher (ऑप्शनल)

> **स्टेज एक नज़र में**
>
> **गोल**: Decomposition से पहले बैकएंड-स्पेसिफ़िक patterns
> **मुख्य Patterns**: Renderer-स्पेसिफ़िक
> **प्रभाव**: हार्डवेयर-स्पेसिफ़िक ऑप्टिमाइज़ेशन

**यह क्या करता है**: Decomposition से पहले renderer-स्पेसिफ़िक patterns अप्लाई करता है।

**यह क्यों ज़रूरी है**: हर बैकएंड अपने patterns जोड़ सकता है। उदाहरण के लिए, DSP बैकएंड इसे जेनेरिक patterns को DSP-स्पेसिफ़िक SIMD intrinsics से बदलने के लिए इस्तेमाल करता है। इससे जेनेरिक पाइपलाइन बदले बिना हार्डवेयर-स्पेसिफ़िक ऑप्टिमाइज़ेशन मिलते हैं।

**Pattern**: `renderer.pre_matcher`

ज़्यादातर बैकएंड (CPU, GPU) को इसकी ज़रूरत नहीं। सिर्फ़ स्पेशलाइज़्ड हार्डवेयर इस्तेमाल करता है।

**नोट**: Svod में `pre_matcher` नहीं है। बैकएंड hooks `svod_device::device::Renderer` trait (`device/src/device.rs`) पर रहते हैं: `decompositor()`, `extra_matcher()`, `pre_isel_matcher()` और `isel_matcher()`। आखिरी दोनों PROGRAM बाउंड्री पर, Stage 20 और 21 के बीच चलते हैं, decomposition से पहले नहीं। (`svod_codegen::traits::Renderer` एक अलग, संकरा trait है जिसमें `render()`, `backend_name()` और `decompositor()` हैं।)

---

## Stage 18: Decompositions

> **स्टेज एक नज़र में**
>
> **गोल**: जो ऑपरेशन टारगेट सपोर्ट नहीं करता उन्हें रीराइट करें
> **मुख्य Patterns**: Power-of-2, transcendental approximations
> **प्रभाव**: हाई-लेवल ops को हार्डवेयर इंस्ट्रक्शन से मैप करता है

**यह क्या करता है**: जो ऑपरेशन टारगेट सपोर्ट नहीं करता, उनके लिए late rewrites।

**यह क्यों ज़रूरी है**: हार्डवेयर में हर ऑपरेशन नहीं होता। उदाहरण के लिए, ज़्यादातर CPUs में डायरेक्ट `sin` इंस्ट्रक्शन नहीं है। हम इसे उन ऑपरेशनों से approximate करते हैं जो मौजूद हैं (addition, multiplication, आदि)।

**Pattern**: `early_decomposition_patterns() + get_late_rewrite_patterns() + get_transcendental_patterns()` (साथ में `renderer.decompositor()`, जब बैकएंड दे)। `early_decomposition_patterns()` ख़ुद `symbolic_simple()` से शुरू होता है।

नोट: `pm_split_ends()` इस पास का हिस्सा नहीं है — वह Stage 19 के matcher में जुड़ता है और Stage 20 की शुरुआत में दोबारा चलता है।

| Pattern | उदाहरण | कब इस्तेमाल |
|---------|--------|-------------|
| `MOD → AND` | `x % 8 → x & 7` | Power-of-2 divisor |
| `MUL → SHL` | `x * 16 → x << 4` | Power-of-2 multiplier |
| `DIV → SHR` | `x / 8 → x >> 3` | Power-of-2 divisor (C-स्टाइल CDIV) |
| `FDIV → MUL` | `x / 2.0 → x * 0.5` | Float constant divisor (कॉन्स्टेंट भाजक) |
| `NEG` | `x * -1 → NEG(x)` | जब NEG सपोर्टेड हो |
| `MULACC` | `a * b + c → MULACC(a, b, c)` | जब FMA सपोर्टेड हो |
| Fast integer division | `x // 7 → (x * M) >> S` | Non-power-of-2 भाजक |
| De Morgan's law | `(!x) & (!y) → !(x \| y)` | Boolean सिम्प्लीफ़िकेशन (केवल AND-of-NOTs) |
| Comparison negations | `!(x < c) → (c-1) < x` | Integer comparisons |

Transcendental approximations (EXP2, LOG2, SIN, …) `get_transcendental_patterns()` से आते हैं (`ir/src/decompositions/mod.rs`, इम्प्लीमेंटेशन `ir/src/decompositions/transcendentals.rs` में)। ये हर ऑपरेशन के लिए तब चालू होते हैं जब renderer के पास वह इंस्ट्रक्शन न हो, या `TRANSCENDENTAL=2` पर हर ऑपरेशन के लिए। ऑप्शनल `Renderer::decompositor()` hook ऊपर से बैकएंड-स्पेसिफ़िक नियम जोड़ता है; in-tree कोई बैकएंड इसे इस्तेमाल नहीं करता।

**Svod**: `optimizer/mod.rs`

---

## Stage 19: Final Rewrite

> **स्टेज एक नज़र में**
>
> **गोल**: लीनियराइज़ेशन की तैयारी
> **मुख्य Patterns**: Weak-cast commit, renderer rewrites, END splitting
> **प्रभाव**: लीनियराइज़ेशन के लिए साफ़ representation

**यह क्या करता है**: लीनियराइज़ेशन की तैयारी।

**यह क्यों ज़रूरी है**: कुछ patterns decomposition के बाद अप्लाई करना आसान होता है। यह स्टेज लीनियर सीक्वेंस में कन्वर्ट करने से पहले फ़ाइनल क्लीनअप करता है।

**Pattern**: `pm_commit_weak() + pm_cast_weak() + pm_decomp` (Stage 18 के decompositions), साथ में `renderer.extra_matcher()` और `pm_split_ends()` — सब एक ही matcher में जुड़े। इसके बाद `pm_remove_invalid()` और `add_implicit_barriers()` अलग passes के रूप में चलते हैं।

नोट: `extra_matcher` और `pm_split_ends` इसी combined matcher का हिस्सा हैं, अलग passes नहीं। Svod में कोई CONST-वेक्टराइज़ेशन या GEP-resolution स्टेप नहीं है; Tinygrad के `pm_render` का यहाँ कोई समकक्ष नहीं।

**मल्टी-range ENDs स्प्लिट करें**:
```text
// Before: END closing multiple ranges
END(op, [range_a, range_b])

// After: nested single ENDs
END(END(op, range_a), range_b)
```

Ranges `(axis_id, axis_type.priority())` के अनुसार descending सॉर्ट होती हैं, इसलिए सबसे भीतरी END पहले बनता है। Void/Bool "backedge" sources अलग किए जाते हैं और मूल tag बरक़रार रखते हुए सबसे बाहरी END पर दोबारा जोड़े जाते हैं।

**extra_matcher**: हर बैकएंड अपने फ़ाइनल patterns जोड़ सकता है। इससे जेनेरिक पाइपलाइन बदले बिना हार्डवेयर-स्पेसिफ़िक ऑप्टिमाइज़ेशन मिलते हैं।

**Svod**: `optimizer/mod.rs`, `linearize/mod.rs`

---

## Stage 20: Add Control Flow

> **स्टेज एक नज़र में**
>
> **गोल**: कंट्रोल फ़्लो ग्राफ़ बनाएँ और range डिपेंडेंसी जोड़ें
> **मुख्य कॉन्सेप्ट**: तीन रिलेशनशिप टाइप (nested, dependent, independent)
> **प्रभाव**: सही इंस्ट्रक्शन ऑर्डरिंग

**यह क्या करता है**: कंट्रोल फ़्लो ग्राफ़ बनाता है और range डिपेंडेंसी जोड़ता है।

**यह क्यों ज़रूरी है**: ऑपरेशन सही ऑर्डर में एक्ज़ीक्यूट होने चाहिए। अगर load कोई RANGE की वैल्यू इस्तेमाल करता है, तो RANGE पहले आना चाहिए। यह स्टेज इन डिपेंडेंसीज़ को ट्रैक और एनफ़ोर्स करता है।

**Pattern**: `pm_add_control_flow` (bottom-up), जिससे पहले `pm_split_ends` दोबारा चलता है

```text
// Analyze which END operations depend on which
END(computation, [RANGE_A]) and END(other_computation, [RANGE_B]) are siblings
→ Creates edge: RANGE_B.src += END(computation)

// Add explicit dependency
RANGE_B waits for RANGE_A to complete
```

**तीन रिलेशनशिप टाइप**:

| रिलेशनशिप | शर्त | मतलब |
|------------|------|------|
| Nested | END_A, END_B की dep है **और** RANGE_B, END_A की dep है | A का लूप B के अंदर है, इसलिए A, B से पहले बंद होता है |
| Dependent | END_A, END_B की dep है पर वह nesting नहीं है | B का लूप A के बाद एमिट होना चाहिए |
| Independent | कोई END दूसरे पर निर्भर नहीं | ऑर्डर आज़ाद है; पैरेलल चल सकते हैं |

Bottom-up ट्रैवर्सल सुनिश्चित करता है कि डिपेंडेंसी leaves से roots तक सही बहे।

**Svod**: `schedule/src/linearize/mod.rs`, `schedule/src/linearize/cfg_context.rs`

---

## Stage 21: Linearize

> **स्टेज एक नज़र में**
>
> **गोल**: DAG को लीनियर इंस्ट्रक्शन सीक्वेंस में बदलें
> **मुख्य अल्गोरिदम**: Priority-aware topological sort
> **प्रभाव**: वैलिड एक्ज़ीक्यूशन ऑर्डर

**यह क्या करता है**: DAG को priority-aware topological sort से लीनियर इंस्ट्रक्शन सीक्वेंस में बदलता है।

**यह क्यों ज़रूरी है**: ग्राफ़ स्ट्रक्चर एक्ज़ीक्यूशन ऑर्डर specify नहीं करता। हमें डिपेंडेंसी respect करते हुए फ़्लैटन करना होगा। Priorities सेंसिबल ऑर्डरिंग सुनिश्चित करती हैं (definitions uses से पहले, loads कम्प्यूटेशन से पहले, stores बाद में)।

**फ़ंक्शन**: `linearize(sink)`

| ऑपरेशन | प्रायोरिटी | क्यों |
|---------|-----------|------|
| PARAM | -20 | Kernel आर्ग्युमेंट (और symbolic वेरिएबल) पहले डिफ़ाइन होने चाहिए; बराबरी पर parameter slot से तय |
| BUFFER | -18 | एलोकेशन पहले |
| BUFFER (`AddrSpace::Local`) | -17 | Global वालों के ठीक बाद लोकल एलोकेशन |
| END | -5 | Ranges बंद करता है |
| LOAD | -1 | इस्तेमाल से पहले Loads |
| बाकी सब (CONST, ALU, …) | 0 | अपने कंज़्यूमर के पास जाकर बैठता है |
| STORE | +1 | कम्प्यूटेशन के बाद Stores |
| RANGE | +5 | इस्तेमाल से पहले Ranges खुलें |

कम प्रायोरिटी = सीक्वेंस में पहले। इससे सुनिश्चित होता है:
- Definitions पहले आएँ
- Loads कम्प्यूटेशन से पहले हों
- Stores आखिर में हों
- Ranges अपने contents से पहले खुलें, बाद में बंद हों

**Run_count ऑर्डरिंग**: ऑपरेशन मुख्य रूप से एक्ज़ीक्यूशन फ़्रीक्वेंसी (run_count) से सॉर्ट होते हैं, फिर प्रायोरिटी से, फिर PARAM slot और tuplize rank से। कम एक्ज़ीक्यूशन फ़्रीक्वेंसी वाले ऑपरेशन (inner loops के बाहर) पहले शेड्यूल होते हैं, जबकि inner loops वाले (ज़्यादा run_count) बाद में। उदाहरण: 100 बार एक्ज़ीक्यूट होने वाला CONST, 1M बार वाले से पहले आता है।

**run_count कैलकुलेशन**:
```text
run_count = prod(int(r.vmax) + 1 for r in u.in_scope_ranges())
```
यह कैलकुलेट करता है कि enclosing in-scope ranges के आधार पर ऑपरेशन कितनी बार एक्ज़ीक्यूट होता है; जिस range का `vmax` कॉन्क्रीट integer न हो, वह 1 गिना जाता है।

**Svod**: `linearize()` in `schedule/src/linearize/linearize.rs`

---

## Stage 22: Cleanup IF/ENDIF

> **स्टेज एक नज़र में**
>
> **गोल**: लीनियर इंस्ट्रक्शन लिस्ट का फ़ाइनल क्लीनअप
> **मुख्य ट्रांसफ़ॉर्मेशन**: Gated STORE → IF/STORE/ENDIF
> **प्रभाव**: बिना predicated stores वाले हार्डवेयर को हैंडल करता है

**यह क्या करता है**: लीनियर इंस्ट्रक्शन लिस्ट का फ़ाइनल क्लीनअप।

**यह क्यों ज़रूरी है**: कुछ हार्डवेयर (मॉडर्न GPUs) "predicated stores" सपोर्ट करता है — मेमोरी में तभी लिखो जब condition true हो। पुराना हार्डवेयर नहीं करता। उनके लिए, store को IF स्टेटमेंट में रैप करना पड़ता है। यह स्टेज सिर्फ़ उन बैकएंड के लिए ज़रूरी है जिनमें predicated store सपोर्ट नहीं; LLVM, CUDA और Metal gate को नेटिवली हैंडल करते हैं, इसलिए `linearize_with_cfg()` इसे नहीं चलाता।

**Pattern**: `line_rewrite_cleanups` (`line_rewrite` से, `graph_rewrite` नहीं)

```text
// Gated STORE becomes a conditional store
STORE(INDEX(ptr, idx), value, gate=cond)
→ IF(cond) { STORE(INDEX(ptr, idx), value) } ENDIF
```

**नोट**: यह स्टेज `graph_rewrite` के बजाय `line_rewrite` इस्तेमाल करता है क्योंकि यह DAG के बजाय पहले से लीनियराइज़्ड इंस्ट्रक्शन लिस्ट पर ऑपरेट करता है।

इस पॉइंट पर, इंस्ट्रक्शन लिस्ट कोड जनरेशन के लिए तैयार है।

**Svod**: `line_rewrite_cleanups()` in `schedule/src/linearize/mod.rs`
