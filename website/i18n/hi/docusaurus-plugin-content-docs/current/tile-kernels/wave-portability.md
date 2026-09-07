---
sidebar_label: Wave32 बनाम Wave64
---

# एक कर्नेल को दो architectures पर correct रखना

यह रहा एक ऐसा bug जो NVIDIA पर होता ही नहीं। आप एक tile कर्नेल लिखते हैं, इसे एक CDNA datacenter GPU पर
test करते हैं, और यह बिल्कुल सही चलता है। फिर *वही* कर्नेल आप एक RDNA laptop APU पर run करते हैं और
numbers बिल्कुल कचरा निकलते हैं — न कोई crash, न कोई error, बस ग़लत। code में देखने से कुछ अलग नहीं लगता।

[Tiling क्या है](./tiling) ने fragments और role-based selection से परिचय कराया था; यह chapter बताता है कि
उस indirection का होना आख़िर ज़रूरी क्यों है। असली दोषी है **wavefront size**, और इससे साफ़-सुथरे ढंग से
निपटना ही वह बात है जो एक chip पर चलने वाली tile library को एक सचमुच portable library से अलग करती है।

---

## 32-बनाम-64 का बँटवारा

एक wavefront (AMD का "warp") उन lanes का समूह है जो lockstep में execute होती हैं। AMD पर इसके दो sizes
हैं, और Svod दोनों को target करता है:

| Architecture | उदाहरण | Matrix op | Wavefront |
|--------------|---------|-----------|-----------|
| **CDNA** | gfx942 (datacenter) | MFMA | **wave64** — 64 lanes |
| **RDNA** | gfx1151 (RDNA3.5) | WMMA | **wave32** — 32 lanes |
| **CUDA** | sm_80+ (Ampere और उसके बाद) | `mma.sync` | **warp32** — 32 lanes |

बस यही एक number है जिसका असर हर चीज़ पर पड़ता है। एक `16×16` tile में 256 elements होते हैं। 64 lanes में
बाँटें तो प्रति lane 4 elements; 32 lanes में बाँटें तो 8। अलग-अलग lanes अलग-अलग elements की मालिक होती हैं।
तो:

- एक tile का **register layout** बदल जाता है,
- matrix instruction जिस **operand layout** की उम्मीद करता है, वह बदल जाता है (RDNA तो कुछ operands को
  lanes भर में *replicate* तक करता है),
- और कोई भी **cross-lane reduction** — softmax और layernorm की जान — के steps की संख्या अलग होती है और
  sibling pattern भी अलग।

जो कर्नेल यह hardcode कर देता है कि "64 lanes हैं, lanes 16, 32, 48 को xor करके reduce करो", वह एक
32-lane machine पर एक *अधूरी* reduction compute करता है और चुपचाप ग़लत values लौटा देता है।

---

## हल: shape नहीं, role माँगें

`tk` का जवाब है indirection की एक layer। कोई कर्नेल कभी "16×16, 4 elements per lane" जैसा कोई concrete
fragment shape नहीं लिखता। इसके बजाय यह एक **role** माँगता है, और architecture capabilities को उसे resolve
करने देता है:

```text
   kernel says:  "I need an accumulator fragment"   (FragRole::Accumulator)
                          │
                          ▼
   ArchCaps::frag(role)   ── on CDNA ──▶  the wave64 16×16 shape
                          └─ on RDNA ──▶  the wave32 16×16 shape (8 ept, replicated operands)
```

roles हैं `FragRole::{Accumulator, Operand, AccumulatorT}`, और resolver है `tk/src/arch.rs` में
`ArchCaps::frag(role)`। कर्नेल author बस "accumulator" और "operand" लिखता है; *physical* layout — प्रति
lane element count, interleave map, replication — target wave size के हिसाब से अपने-आप भर जाता है। एक बार
लिखो, दोनों पर चलाओ।

यही सबक़ HipKittens ने भी सीखा था (देखें [tk बनाम HipKittens बनाम CuTile](./comparison)): यह दो parallel
backends भेजता है, `cdna4` (wave64) और `udna1` (wave32), जो एक अकेले `WARP_THREADS` constant से keyed हैं,
ताकि tile types हर एक के लिए सही ढंग से recompile हों। `tk` इस सबको एक ही runtime-resolved `ArchCaps` में
समेट देता है।

---

## एक bug जो इसने सचमुच पकड़ा

यह indirection कोई किताबी बात नहीं है। एक शुरुआती `tk` cross-lane all-reduce — वही `shuffle_xor` primitive
जो एक value को पूरी wave भर में sum करने के काम आता है — एक hardcoded wave64 reduction tree के साथ लिखा गया
था। RDNA की 32-lane waves पर इसने उन lanes पर reduce कर दिया जो हिस्सा ही नहीं लेतीं, और ठीक उन्हीं
softmax-style reductions के लिए ग़लत sums निकाल दिए जिन पर attention टिका है। हल था — reduction को किसी
constant से नहीं, बल्कि `caps.wave_size` और role-resolved fragment से चलाना। अब `tk/src/group/shuffle.rs` में
shuffle primitives wave size पढ़ते हैं; पूरी bug class को design से ही बाहर कर दिया गया।

:::tip GPU विशेषज्ञों के लिए
`ArchCaps` (`tk/src/arch.rs`) पर दो capability methods अधिकांश wave-specific बोझ संभालती हैं:

- **fragment का `LaneMap`** fold को साथ लाता है। कोई reduction अपना tree resolve हुए fragment से पढ़ती
  है (`tk/src/group/reduce.rs` में `src.base.map.tree(...)`), किसी constant से नहीं: wave64 पर वह 4
  sub-fragments को fold करते तीन xor steps `[16, 32, 48]` हैं, RDNA के wave32 पर एक step `[16]`, और
  CUDA के `MmaSync` layout पर `[1, 2]` पर एक butterfly। `ArchCaps::reduce_tree()` अब भी मौजूद है, पर
  अब वह सिर्फ़ graph-shape tests वाला AMD रूप है।
- **`acc_reusable_as_input()`** यह जवाब देता है: "क्या एक matrix accumulator को सीधे अगले multiply के
  operand के रूप में वापस feed किया जा सकता है?" CDNA और CUDA पर यह `true` है — layouts मेल खाते हैं, इसलिए यह बिना
  किसी अलग लागत के एक register copy है। RDNA पर यह `false` है — accumulator और operand layouts अलग होते हैं,
  इसलिए value relayout के लिए LDS से होकर एक round-trip करती है। [Flash Attention](./flash-attention) इस
  बँटवारे को अपने दो matmuls के बीच handle करता है।

`BaseShape` पर `ept` field ([Tiling क्या है](./tiling) से) इसी वजह से मौजूद है: RDNA पर operands lanes भर
में replicate होते हैं, इसलिए elements-per-thread `element_count / wave_size` नहीं है और इसे explicitly
store करना ही पड़ता है।
:::

---

## यह क्यों ज़रूरी है

wave sizes भर में portability ही हाथ से लिखे कर्नेल पर AMD-specific tax है, और इसी वजह से किसी NVIDIA tile
library का naive port यूँ ही नहीं चल जाता। `tk` यह tax एक बार चुका देता है, `ArchCaps` abstraction में, ताकि
अलग-अलग कर्नेल पढ़ने लायक़ बने रहें: वे *roles* में बात करते हैं और lanes का हिसाब hardware table पर छोड़
देते हैं। [Flash Attention](./flash-attention) वह जगह है जहाँ आप इसे एक असली कर्नेल में रंग लाते देखते हैं।
