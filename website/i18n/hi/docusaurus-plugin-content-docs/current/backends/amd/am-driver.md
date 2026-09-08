---
sidebar_label: AM Driver
---

# AM Driver (Userspace)

**AM** driver एक दूसरा [`AmdIface`](./overview.md) बैकएंड है जो GPU के PCI BARs को सीधे drive
करता है, kernel `amdgpu`/KFD driver को पूरी तरह bypass करते हुए। यह tinygrad के userspace AM
driver का एक port है। प्रेरणा ठोस है: single-XCC gfx11+ parts पर lock-free
[multi-queue dispatch](./queues-and-dispatch.md) path CP micro-engines को ऐसे waits में
park कर सकता है जिन्हें kernel का MES firmware preempt नहीं कर पाता, और यह उसे एक
unrecoverable reset में wedge कर देता है। kernel scheduler ही साझा कमज़ोरी है — इसी श्रेणी की
failure वह वजह है जो lane pool को हर part पर conservative बनाए रखने पर मजबूर करती है। यदि हम
GPU के मालिक हैं — page tables, firmware, scheduling — तो kernel कभी dispatch path में नहीं
होता और wedge नहीं हो सकता।

:::caution[Work in progress — अभी selectable नहीं]
यह पेज आज जो मौजूद है और बाक़ी के लिए roadmap, दोनों का दस्तावेज़ीकरण करता है।
**`SVOD_AMD_BACKEND=am` फ़िलहाल एक error देता है** (`device.rs` केवल `kfd` स्वीकार करता है):
अभी तक कोई AM type [`AmdIface`](./overview.md) seam को implement नहीं करता, इसलिए आज पहुँच
योग्य bring-up केवल `am_*` examples के माध्यम से चलाया जाता है, `AmdDevice` के माध्यम से नहीं।
जो मौजूद है वह **engine hand-off तक** validated है; target पर अभी तक कोई GPU engine work
consume करने के लिए पुष्ट नहीं है (देखें [the VF boundary](#the-vf-boundary))। नीचे के
sections हर टुकड़े की status को explicit रूप से चिह्नित करते हैं।
:::

कोड `device/src/amd/am/` के अंतर्गत रहता है। यह **हर Unix host पर** compile होता है
(`cfg(unix)`, बाक़ी बैकएंड की तरह — देखें [runtime-detected provider model](./overview.md)),
इसलिए यह हमेशा type-checked, linted, और unit-tested होता है — बैकएंड को *runtime* पर चुना
जाता है, कभी किसी ऐसे cargo feature के पीछे नहीं जो rot हो सकता है।

---

## Target hardware: एक CDNA3 SR-IOV VF (gfx9.4.3)

driver का target है एक **CDNA3** GPU — **gfx9.4.3**, SPX mode में 8 XCCs — और
विशेष रूप से उसका **SR-IOV Virtual Function** रूप (GPU एक VF है जो एक KVM guest में pass किया
गया है)। `AmDev::open` बाक़ी हर चीज़ को सीधे अस्वीकार कर देता है: एक non-VF function, या एक
ऐसा GC version जिसका major.minor `(9, 4)` नहीं है, fast fail करता है
(`device/src/amd/am/dev.rs`)। gfx1151 (RDNA3.5) अब *target* नहीं है, पर gfx11 arch branch
implemented और unit-tested बनी रहती है — और उसकी page-table geometry तथा palloc-range
helpers ही वे चीज़ें हैं जिन्हें gfx9 path दोबारा उपयोग करता है।

> Bring-up hardware: एक AMD Instinct MI300X (gfx942 / GC 9.4.3) का SR-IOV VF।
> `AmDev::open` इसके अलावा कुछ भी स्वीकार नहीं करता।

एक **VF** होना (bare metal के बजाय) defining constraint है, और यह पूरे driver को आकार देता
है:

- **GC MMIO host-gated है।** किसी GC register का हर *direct* read `0xffffffff` लौटाता है।
  सभी GC / GCVM register access को **RLC के माध्यम से indirectly** जाना ही होता है (RLCG
  path) — value को RLC scratch में stage करें, `RLC_SPARE_INT` kick करें, completion के लिए
  poll करें।
- **VRAM/discovery grant तक gated है।** framebuffer (और इसलिए IP-discovery table) तब तक
  unreadable है जब तक host **GIM** (SR-IOV host driver) एक **mailbox handshake** के माध्यम
  से access न दे दे, जो इसलिए discovery से *पहले* चलता है।
- **host PF privileged subsystems का मालिक है:** PSP, SMU, clocks, firmware / world-switch,
  L2 cache config, system aperture, और — सबसे अहम — **doorbell aperture routing**। AM केवल
  उस per-VF state को program करता है जिसे guest को छूने की अनुमति है (page-table context0,
  per-engine invalidation ranges, TLB flushes, ring/queue MQDs), ठीक वैसे ही जैसे kernel का
  `*_v*` IP code `amdgpu_sriov_vf` के तहत इन blocks को skip कर देता है।

यह tinygrad के AM का उलट है, जो **केवल bare-metal** है (यह `amdgpu` को unbind करता है और
पूरे device का मालिक होता है)। VF रूप एक अलग driver है: mailbox + RLCG indirect register
access + per-VF-only hub programming।

---

## आज क्या मौजूद है

जहाँ यह pure logic है वहाँ सब कुछ **बिना GPU के compile और unit-tested** है;
hardware-facing टुकड़े अतिरिक्त रूप से live VF पर `device/examples/am_*.rs` programs के
माध्यम से validated हैं। page tables एक injectable `PhysMem` trait से back होते हैं (tests
में एक plain buffer, असली driver में BAR-mapped VRAM)।

| Group | Module(s) | यह क्या करता है | Status |
|---|---|---|---|
| **Discovery** | `pci.rs`, `discovery.rs` | sysfs BAR mmap (BAR0 VRAM / BAR2 doorbell / BAR5 MMIO), config-space r/w, bounds-checked IP-discovery parser (per-XCC segment bases, `gc_info` v1/v2) | **HW-validated**; discovery parser unit-tested है |
| **Register access** | `regaccess.rs`, `rlcg.rs`, `mailbox.rs`, `regs.rs`, `regs_gen.rs` | mxgpu VF↔GIM mailbox handshake, RLCG indirect GC/GCVM r/w (per-XCC), MMIO/RLCG router, vendored register tables | **HW-validated**; register-table select/encode logic unit-tested है |
| **Memory (GMMU)** | `mm/{tlsf,pagetable,manager,mod}.rs` | TLSF VA/PA/page-table allocators, 4-level/48-bit walk, gfx9 **और** gfx11 PTE/PDE encoding, huge-page selection, table reclaim, `valloc`/`vfree` | **Done** + tests (PTE write path HW-exercised) |
| **GMC bring-up** | `ip/gmc.rs` | दोनों hubs का context0 program करें (start/end/base + CNTL), MX_L1_TLB enable, per-engine invalidation ranges, ENG17 TLB flush, HDP flush, fault-status decode | **HW-validated** context-program level तक |
| **GFX bring-up** | `ip/gfx.rs` | MEC enable करें (icache invalidate, golden `GB_ADDR_CONFIG`, doorbell range, unhalt), एक v9 compute MQD बनाएँ, HQD activate करें (`CP_HQD_ACTIVE=1`), `WRITE_DATA` PM4 | **MEC HQD activate होता है**; queue अभी नहीं चलती |
| **SDMA bring-up** | `ip/sdma.rs` | F32 unhalt करें, RB base/rptr/wptr + doorbell program करें, submit + `wait_idle` | **ring programmed**; engine अभी consume नहीं करता |
| **Orchestrator** | `dev.rs` | `AmDev::open` = mailbox → discovery → GMMU → GMC context0 → flush; `valloc`, `vram_read/write`, `release` | **HW-validated** GMC तक |

### GMMU और gfx9

page-table geometry **4-level / 48-bit** है (`va_shifts = [12, 21, 30, 39]`), एक आकार जो
**gfx9/11/12 में साझा है** — इसलिए geometry ख़ुद arch पर branch नहीं करती। केवल leaf PTE
encoding (विशेष रूप से MTYPE memory-type field) arch-specific है, और **अब gfx9 (CDNA)
और gfx11 (RDNA3) दोनों implement और unit-tested हैं** — gfx9 MTYPE को bits 57–58 पर रखता
है, PDB1 table entries पर `bfs` और PDB0 table entries पर translate-further bit set करता है,
और PDB1/PDB2 leaves को `PDE_PTE` से चिह्नित करता है (एक 2 MiB PDB0 leaf का मतलब है
translate-further का *अभाव*)। **gfx12 ही एकमात्र शेष `unimplemented!` है** (constants captured हैं, अभी तक
hardware-validated नहीं; एक test assert करता है कि यह panic करता है)। `MemoryManager` तीन
TLSF sub-allocators (VA space, physical VRAM, page-table pool) चलाता है और table को `Inspect`
/ `Create` / `Free` modes में walk करता है, unmap पर empty tables को reclaim करते हुए।

### Register tables एक-बार generate होते हैं, फिर vendor किए जाते हैं

tinygrad एक कभी-कभी-अनुपस्थित submodule है, इसलिए build को कभी उस पर निर्भर नहीं होना चाहिए।
इसके बजाय `device/tools/gen_am_regs.py` को एक arch जोड़ते या update करते समय **manually**
चलाया जाता है: यह tinygrad के `autogen/am/regs.py` को parse करता है और committed
`am/regs_gen.rs` emit करता है। `regs.rs` बस उसे `include!` करता है। boot पर सही table को
discovered `ip_ver` से चुना जाता है (`select` वह सबसे बड़ा version `≤ ip_ver` चुनता है जो वही
major साझा करता है — tinygrad का `import_module` नियम)। committed tables अब gfx9.4.3/CDNA3
set (`gc 9.4.3`, `mmhub 1.8.0`, `osssys 4.4.2`, `sdma 4.4.2`, `nbio 7.9.0`, `hdp 4.4.2`,
`mp 11.0.0`/`13.0.0`) और gfx11.5.0 set दोनों को कवर करते हैं। एक arch जोड़ना generator की
module list को widen करना और उसे re-run करना है — कोई build या runtime logic change नहीं।

---

## The VF boundary

यह वह दीवार है जहाँ bring-up फ़िलहाल रुक जाता है। guest engines को **program** कर सकता है पर
उन्हें **drive** नहीं कर सकता, क्योंकि वह doorbell aperture जो किसी ring का write-pointer
command processor तक पहुँचाता है **PF-owned** है। VF से इसे enable करना (`_PF` BIF
doorbell-access registers लिखना) VF↔GIM mailbox को wedge कर देता है और एक full VM reboot
चाहता है — इसलिए `enable_doorbell_aperture` `ip/gfx.rs` में मौजूद है पर explicit रूप से **VF
पर do-not-call** चिह्नित है।

ठोस परिणाम, दोनों examples द्वारा reproduce किए गए:

- **MEC compute queue activate होती है पर execute नहीं करती** (`am_compute`): HQD
  `CP_HQD_ACTIVE = 1` report करता है, पर एक `WRITE_DATA` packet अपना sentinel VRAM में कभी
  land नहीं करता — CP कभी doorbell नहीं देखता।
- **SDMA ring programmed है पर consume नहीं करता** (`am_sdma`): read pointer अटका रहता है;
  MM-hub page-table walk faults अब भी gated हैं।

तो आज AM **engine hand-off तक HW-validated** है — discovery, ownership, GMMU, और GMC live VF
पर सिद्ध हैं — और KFD काम करता हुआ VF backend बना रहता है। इस boundary को पार करना ही बचे हुए
milestones का विषय है।

---

## आज hardware पर क्या चलता है

हर `am_*` example एक standalone bring-up oracle है, जो live VF पर चलाया गया है:

| Example | यह क्या सिद्ध करता है | Status |
|---|---|---|
| `am_discovery` | BAR map + IP discovery (8× GC 9.4.3, SDMA, AIDs), read-only — एक bound `amdgpu` के साथ coexist करता है | **works** |
| `am_own` | mailbox grant + RLCG scratch echo + सभी 8 XCC पर non-gated `GRBM_STATUS` | **works** |
| `am_gmc` | GC + MM context0 programmed; सभी 8 XCC पर ENG17 TLB-flush ACK; कोई protection fault latch नहीं हुआ | **works** |
| `am_sdma` | SDMA ring setup + submit | ring programmed, **engine consume नहीं करता** |
| `am_compute` | MEC enable + MQD activate + `WRITE_DATA` | **HQD activate होता है**, queue execute नहीं करती |

---

## अभी भी क्या स्थगित है

privileged, PF-owned subsystems **tree से अनुपस्थित हैं** — एक VF पर वे GIM के मालिकाने में
हैं और guest के लिए करने को कुछ नहीं है; bare metal पर वे आख़िरी, सबसे-अधिक-जोखिम वाला port
हैं:

- **PSP firmware load** — sOS bootloader handshake / TMR / per-IP firmware load। VF पर
  GIM-owned।
- **SMU / clocks** — power और clock management। VF पर GIM-owned।
- **interrupt handler (IH)** — कोई `ip/ih.rs` मौजूद नहीं; OSSSYS register table vendored है
  पर unused। bring-up interrupts लेने के बजाय poll करता है।
- **`AmIface` seam implementor** — अभी तक कोई AM type [`AmdIface`](./overview.md) implement
  नहीं करता, इसलिए AM को एक device backend के रूप में नहीं चुना जा सकता; `AmDev` केवल
  examples के माध्यम से पहुँच योग्य है।

---

## Roadmap

काम milestones के रूप में staged है, हर एक live VF पर स्वतंत्र रूप से testable (और, PF-owned
blocks के लिए, bare-metal tinygrad AM को oracle मानकर)। पहले वाले milestones implement हो चुके
हैं; पूरा AM end-to-end integration भविष्य का काम है।

एक बार जब कोई engine work consume कर ले और seam wire हो जाए, AM
`SVOD_AMD_BACKEND=am` के माध्यम से selectable बन जाता है और पूरे मौजूदा ऊपरी हिस्से को
unchanged चलाता है। तब वह crash-inducing concurrency जिसने AM को प्रेरित किया crash नहीं कर
सकती — kernel को bypass कर दिया गया है।

---

## यह क्यों ज़रूरी है

AM driver उस firmware-wedge समस्या का असली उत्तर है जिसे lane pool को clamp करना
([`SVOD_AMD_HW_QUEUES=1`](./queues-and-dispatch.md)) केवल sidestep करता है। महँगे, GPU-free हिस्से — GMMU, register tables, mailbox/RLCG
indirect-access machinery — live VF पर built और validated हैं, और page tables, GMC, और
ownership handshake सभी काम करते हैं। बचा हुआ gap एक hardware boundary है (PF-owned doorbell
aperture), design वाला नहीं। और चूँकि यह उसी [seam](./overview.md) के पीछे slot होता है —
पाँच required methods और तीन defaulted hooks — इसके उतरने पर dispatch, compile, या graph
machinery में से किसी को बदलना नहीं पड़ता।
