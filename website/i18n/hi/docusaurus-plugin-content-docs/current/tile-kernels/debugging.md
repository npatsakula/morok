---
sidebar_label: डीबगिंग
---

# कर्नेल को Debug और Verify करना

एक हाथ से लिखा कर्नेल उतना ही भरोसेमंद होता है जितनी उसे जाँच पाने की आपकी क़ाबिलियत।
[Flash Attention](./flash-attention) walkthrough ने दिखाया कि किस तरह का कर्नेल हाथ से लिखने लायक़ होता है;
यह chapter वह है जिससे आप उस पर भरोसा करना सीखते हैं। USE चेहरा आपको एक lazy `Tensor` देता है जो एक बड़े
graph में fuse हो जाता है — सुविधाजनक तो है, पर "क्या यह एक कर्नेल correct है, और यह कितना तेज़ है?" यह
पूछने के लिए बुरी जगह। `tk` का **DEBUG चेहरा** ठीक इसी के लिए है: एक अकेले कर्नेल को concrete buffers के
ख़िलाफ़ run करो, नतीजा वापस पढ़ो, उसे time करो, और साबित करो कि किसी refactor ने इसका behavior नहीं बदला।

---

## Direct dispatch: एक कर्नेल run करो, bytes देखो

direct-launch API (`tk/src/launch.rs`) tensor scheduler को पूरी तरह bypass कर देता है। आप इसे एक finished
`Kernel` और असली input buffers देते हैं; यह render, compile, और dispatch करता है, और नतीजा एक output buffer
में लिख देता है जिसे आप वापस पढ़ सकते हैं:

```rust
// The DEBUG face from tk/src/lib.rs. `outs` are written in place.
run_kernel("tile_add", [1, 1, 1], block, &mut [&mut out], &[&input_a, &input_b], build)?;
let values = out.as_vec::<f32>()?;   // read the GPU result straight back
assert_eq!(values, expected);
```

चूँकि यह scheduling, fusion, और dependency tracking को छोड़ देता है, इसलिए आप जो measure करते हैं वह *सिर्फ़
आपका कर्नेल* होता है — कोई ऐसा graph नहीं जिसमें यह बस शामिल हो। यही isolation असल मुद्दा है: जब कोई number
ग़लत निकले, तो आप जानना चाहते हैं कि वह *यहीं* ग़लत है, न कि किसी fused pipeline में कहीं और।

path पर एक छोटी-सी बात: *scheduler* को छोड़ देना *optimizer* को छोड़ देना नहीं है। `compile` आपके `SINK` पर
production वाला `optimize_kernel_with_config` अब भी चलाता है — जो हाथ से lower किए गए body पर शून्य schedule
opts apply करता है (यही `opts_to_apply: Some(vec![])` marker ख़रीदता है), पर render से पहले हर कर्नेल को
ज़रूरी वे साझा rewrites अब भी करता है, जिनमें index-dtype lowering भी है। scheduler के बिना भी आपको correct
code मिलता है।

---

## असली hardware पर timing

performance वाले काम के लिए, `CompiledLaunch` (`compile` / `compile_kernel` से) wall-clock अंदाज़ों के बजाय
hardware timestamps expose करता है:

```rust
// Render + compile once …
let launch = compile_kernel("matmul", grid, block, &mut [&mut c], &[&a, &b], build)?;
// … then dispatch in a loop, outside the timed region.
// SAFETY: the bound buffers stay allocated for `launch`'s lifetime.
unsafe { launch.dispatch(true) }?;
let ns = launch.dispatch_gpu_ns()?;   // Option<u64>: device-measured dispatch time
```

`dispatch_gpu_ns()` dispatch के इर्द-गिर्द GPU के अपने timestamp counters पढ़ता है, इसलिए आप device पर बीते
समय को measure कर रहे होते हैं, न कि इसे launch करने की round-trip latency को। criterion benches वही
device-time stamps एक layer ऊपर, `plan.profile` के ज़रिए पाते हैं, ताकि एक `tk` कर्नेल की तुलना graph-native
baseline से कर सकें। वही benches `cargo bench
--profile-time` के तहत इससे ज़्यादा करते हैं: हर benchmark किए गए plan को पूरे layered profiler से गुज़ारा
जाता है — device time, roofline, occupancy, और hardware counters — जिन्हें per-kernel minimum से accumulate
करके एक table में लिख दिया जाता है। tiers, env vars, और criterion wiring के लिए देखें
[Profiling और Benchmarking](./profiling)।

:::tip[GPU विशेषज्ञों के लिए]
`KernelFingerprint` `SINK` के UOp graph का एक *structural* hash है — यह shape (ops, dtypes, edges) को instance IDs से स्वतंत्र रूप से capture करता है, इसलिए यह runs और processes भर में stable रहता है। यही इसे एक golden-test key बनाता है: एक behavior-preserving refactor वही fingerprint दोबारा produce करता है, जबकि emitted IR में कोई भी बदलाव इसे हिला देता है। `dispatch_gpu_ns` dispatch के इर्द-गिर्द device के अपने timestamp counters पढ़ता है, इसलिए यह on-device समय measure करता है, launch latency नहीं।
:::

---

## Fingerprints: साबित करना कि एक refactor behavior-preserving है

हाथ से लिखे कर्नेल के साथ एक बारीक जोखिम रहता है: आप builder code "साफ़-सुथरा कर देते हैं", कर्नेल फिर भी
compile हो जाता है और वाजिब-से numbers भी देता है, पर *generated IR* किसी ऐसे तरीक़े से बदल जाता है जो बाद में
किसी ख़ास shape या किसी ख़ास architecture पर ही सामने आता है।

`KernelFingerprint` (`tk/src/fingerprint.rs`) इसी के ख़िलाफ़ guard करता है। यह एक कर्नेल के UOp graph का एक
deterministic, structural hash compute करता है — SINK का shape, न कि pointer identities। आप fingerprint को
एक golden value के रूप में snapshot कर लेते हैं, और जिस refactor का मक़सद बस cosmetic होना है, उसे यही
fingerprint दोबारा produce करना ही होगा:

```rust
let fp = kernel_fingerprint(&sink);
assert_eq!(fp.digest, GOLDEN_MATMUL_DIGEST);  // structure unchanged ⇒ behavior unchanged
```

अगर fingerprint हिल जाए, तो आपने emitted IR बदल दिया — चाहे जान-बूझकर या नहीं — और golden test आपको इसकी
ओर देखने पर मजबूर कर देता है। `tk/src/test/unit/golden.rs` के unit tests ठीक इसी का इस्तेमाल करके matmul और
Flash Attention graphs को lock करते हैं (digest *और* node count, दोनों)।

---

## किस सवाल के लिए कौन-सा tool

| आप क्या पूछ रहे हैं… | इस्तेमाल करें |
|----------------|-----|
| "क्या यह कर्नेल सही numbers देता है?" | `run_kernel` + `as_vec`, और एक reference से तुलना करें |
| "यह इस GPU पर कितना तेज़ है?" | `compile_kernel` + `dispatch_gpu_ns` |
| "क्या मेरे refactor ने emitted IR बदला?" | `KernelFingerprint` golden test |
| "कहीं *device/driver layer* ही तो गड़बड़ नहीं कर रहा?" | [AMD Backend → Debugging](../backends/amd/debugging), [CUDA Backend → Debugging](../backends/cuda/debugging) |

वह आख़िरी row मायने रखती है: यह chapter *कर्नेल* को debug करने के बारे में है — वह IR जो आपने author किया और
वे numbers जो यह देता है। जब समस्या उससे नीचे की हो — queue dispatch, memory faults, driver, PTX JIT — तो
per-backend chapters सही जगह हैं:
[AMD](../backends/amd/debugging) और [CUDA](../backends/cuda/debugging)।

---

## यह क्यों ज़रूरी है

हाथ से authoring में आप optimizer की safety net छोड़कर control हाथ में लेते हैं। DEBUG चेहरा वही है जिससे आप
यह सौदा safely करते हैं: correctness bugs को localize करने के लिए isolation, ऐसे performance claims करने के
लिए hardware timestamps जिनका आप बचाव कर सकें, और structural fingerprints — ताकि "मैंने तो बस code साफ़ किया
था" चुपचाप "मैंने कर्नेल बदल दिया" में न बदल जाए। इन तीनों के साथ, एक हाथ से लिखा कर्नेल एक autotuned कर्नेल
जितना ही verifiable है।
