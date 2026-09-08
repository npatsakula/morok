---
sidebar_label: Flash Attention
---

# Worked Example: Flash Attention

Flash Attention वह कर्नेल है जो `tk` के होने को सही ठहराता है — वही जिसे [अवलोकन](./overview) ने एक
single schedulable reduction के रूप में *not* expressible बताया था, और जिसकी वजह से एक hand-authoring
surface का होना ज़रूरी हुआ। यह chapter इसी से होकर चलता है: इसे मुश्किल क्या बनाता है, tile abstractions
इसका जवाब कैसे देते हैं, और [Wave32 बनाम Wave64](./wave-portability) का बँटवारा असल production कर्नेल में कहाँ
सामने आता है।

हम `tk/src/kernels/fa.rs` के forward कर्नेल की बात कर रहे हैं, जहाँ तक USE-चेहरे के
`flash_attention(q, k, v)` से पहुँचा जाता है। यह gfx942 (CDNA3), gfx1151 (RDNA3.5) और CUDA `sm_80+` के लिए
बना है; per-warp `(q_blk, kv_blk)` tile हर device के लिए `FaPolicy` चुनता है — ऊँचा tile तभी, जब इसके
shared-memory buffers समा जाएँ और launch grid पहले से device की compute units को cover करता हो, वरना
baseline `{16, 32}`।

---

## attention को autotune क्यों नहीं किया जा सकता

Plain attention `softmax(QKᵀ) · V` है। naively लिखें तो इसका मतलब है: पूरी `N×N` score matrix बनाओ,
उसे softmax करो, फिर `V` से multiply करो। पर यह score matrix बहुत बड़ी होती है और इसे कभी एक साथ पूरी
मौजूद रहने की ज़रूरत भी नहीं — इसीलिए Flash Attention keys और values के blocks पर stream करता है और
softmax को *incrementally* बनाए रखता है।

यही शब्द — incrementally — असली पेच है। softmax normalization *सभी* keys पर के maximum और sum पर निर्भर
करता है, पर हमें एक बार में सिर्फ़ एक block दिखता है। इसलिए हम running statistics रखते हैं और चलते-चलते
नतीजे को ठीक करते जाते हैं। यही **online softmax** है, और यह एक recurrence है: हर KV block वही state
पढ़ता और अपडेट करता है जो पिछले block ने बनाया था।

optimizer का action space तो बस इतना है — "इस `REDUCE` को tile और unroll करो।" पर यहाँ tile करने लायक़
कोई `REDUCE` है ही नहीं — यहाँ तो एक loop है जिसकी body अपने ही पिछले iteration पर निर्भर करती है। search
इसे ढूँढ नहीं सकती। आपको इसे ख़ुद लिखना पड़ता है।

---

## algorithm, tiles में

कर्नेल हर wave को queries का एक block सौंपता है और keys/values को block-दर-block चलता है। हर KV block के
लिए यह loop body run करता है, पूरी की पूरी tiles में:

```text
for each block of K, V:                          ┌─ everything here is a tile op
    S   = Q · Kᵀ                                 │  (mma into a register accumulator)
    S   = mask(S)                                │  causal + key-padding masks
    m'  = max(m, rowmax(S))                      │  update running max  (cross-lane reduce)
    P   = exp2(S - m')                           │  rescale to the new max (base-2 exp)
    l   = l * exp2(m - m') + rowsum(P)           │  update running sum
    O   = O * exp2(m - m') + P · V               │  rescale accumulator, accumulate
    m   = m'                                     │
O = O / l                                        └─ final normalize
```

हर block पर दो matrix multiplies (`Q·Kᵀ` और `P·V`), दो cross-lane row reductions (max और sum), और जब भी
running max हिलता है, output accumulator का एक rescale। वह `exp2` — base-2 exponential — जान-बूझकर है:
temperature को पहले ही `Q` में fold कर दिया जाता है, ताकि hardware का तेज़ `exp2` unit सीधे काम आ सके।

इनमें से हर line tiles पर एक `Group` operation है: multiplies के लिए `mma`, row max/sum के लिए
एक `RV` (register-vector) reduction, और rescale के लिए एक elementwise `exp2`/`mul` map। कहीं कोई lane
arithmetic नज़र नहीं आती।

---

## Streaming: double-buffered KV

यह [FLOPS कहाँ छिपते हैं](./where-flops-hide) वाला gap 2 असल काम में है। जब तक matrix core मौजूदा KV block
पर काम करता है, तब तक अगला block पहले से shared memory की ओर रास्ते में होना चाहिए। कर्नेल **दो** LDS
buffers रखता है और उन्हें बारी-बारी इस्तेमाल करता है ("double-buffering" / software pipelining): buffer B
load होते समय buffer A पर compute, फिर swap।

```text
   load K/V block 0 --> LDS[A]
   ┌─────────────────────────────────────────────────┐
   │ compute on LDS[A]   ║   load block 1 --> LDS[B] │   <- overlap
   │ compute on LDS[B]   ║   load block 2 --> LDS[A] │
   │ ...                                             │
   └─────────────────────────────────────────────────┘
```

shared tiles अपना XOR swizzle (gap 3) साथ रखते हैं, इसलिए cooperative fill और per-lane read दोनों
bank-conflict-free रहते हैं।

---

## layout की बारीकी: दो matmuls के बीच relayout

यहीं [Wave32 बनाम Wave64](./wave-portability) theory से निकलकर असल में सामने आता है। कर्नेल दो
matrix multiplies करता है, और पहले का output (`S = Q·Kᵀ`, जो softmax के बाद `P` बन जाता है) दूसरे का
*input* है (`P·V`)। तो क्या score accumulator को सीधे एक operand की तरह वापस feed किया जा सकता है?

- **CDNA और CUDA पर** (`acc_reusable_as_input() == true`): हाँ। CDNA पर MFMA accumulator ख़ुद *ही* input
  fragment है, और two-half `mma.sync` f32 accumulator m16n8 C fragments को ठीक उसी A-operand register
  order में रखता है — इसलिए यह बस एक register copy है। सस्ता।
- **RDNA पर** (`acc_reusable_as_input() == false`): नहीं। even/odd accumulator और replicated operand अलग
  होते हैं, इसलिए दूसरे multiply से पहले `P` को relayout के लिए **LDS से होकर एक round-trip** करना पड़ता है
  (उसी per-warp softmax band से होकर जो policy का `att_band` allocate करता है)।

कर्नेल हर architecture पर सही काम करने के लिए `ArchCaps` पर branch करता है। algorithm वही, पर दो physical
realizations — ठीक वही portability tax जिसका ज़िक्र पिछले chapter में था, और वह भी सबसे अहम कर्नेल के
सबसे hot loop में।

---

## Masking

Causal masking (एक query किसी future key पर attend नहीं कर सकता) और key-padding masking (किसी batch की
padded positions को नज़रअंदाज़ करना) — दोनों softmax से पहले score tile `S` पर apply होते हैं। mask को
memory से load नहीं किया जाता, बल्कि tile के अपने lane/row coordinates से derive किया जाता है — हर score
element की position इसी से तय हो जाती है कि उसे कौन-सा fragment और lane रखता है, इसलिए mask compute होता है,
fetch नहीं।

:::tip[GPU विशेषज्ञों के लिए]
compute/memory overlap को `tk` में raw scheduling intrinsics के रूप में hand-emit नहीं किया जाता, जैसा
HipKittens के कर्नेल में होता है। इसके बजाय KV loop पर `sched::pipeline(SchedKind::Attention, …)`
(`tk/src/kernels/fa.rs`) का annotation लगा होता है — एक marker, जिसे codegen में एक post-linearization
scheduling pass उठाकर matrix, memory, और exponential instruction streams को interleave करता है। इससे
कर्नेल body पढ़ने लायक़ बनी रहती है — यह बस यह बताती है कि *क्या* overlap करना है, और concrete instruction
ordering का फ़ैसला एक बाद वाला pass करता है। बजाय इसके कि author को raw scheduling intrinsics ख़ुद algorithm
के बीच-बीच में हाथ से डालने पड़ें।
:::

---

## यह क्यों ज़रूरी है

Flash Attention पूरे section को एक ही file में समेट देता है:

- यह इसलिए है क्योंकि **online softmax एक recurrence है**, कोई tileable reduction नहीं
  ([अवलोकन](./overview));
- यह **streaming और overlap** के दम पर जीता या मरता है ([FLOPS कहाँ छिपते हैं](./where-flops-hide));
- यह पूरी तरह **tiles और roles** में व्यक्त होता है, कभी lane indices में नहीं ([Tiling क्या है](./tiling));
- यह बाक़ी सब चीज़ों जैसा **वही UOp IR** बनकर compile होता है और lazy graph में एक
  `Op::Call` के रूप में शामिल होता है ([IR में authoring](./lowering));
- और अपने hot loop में यह एक explicit **accumulator-reuse branch** साथ रखता है, हर fragment layout के लिए एक
  ([Wave32 बनाम Wave64](./wave-portability))।

इसीलिए यह हाथ से लिखा गया है, और इसीलिए इसे लिखने के लिए `tk` मौजूद है। इसे isolation में run करके इसके
numbers जाँचने के लिए, देखें [डीबगिंग](./debugging)।
