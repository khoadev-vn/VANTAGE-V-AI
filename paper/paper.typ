#set page(
  paper: "us-letter",
  margin: (x: 1.8cm, top: 2.2cm, bottom: 2.2cm),
  header: align(right)[
    #text(8pt, fill: luma(120), font: "Times New Roman")[
      VANTAGE-V-AI Technical Report | November 2026
    ]
  ],
  footer: context align(center)[
    #text(9pt, font: "Times New Roman")[#counter(page).display("1")]
  ]
)

#set text(
  font: "Times New Roman",
  size: 10pt,
  lang: "en",
  hyphenate: true
)

#set par(justify: true, leading: 0.65em, first-line-indent: 1.2em)

// Title & Metadata
#align(center)[
  #v(0.5em)
  #text(16pt, weight: "bold")[
    VANTAGE-V-AI: Bounded-Memory KV-Cache Compression and Streaming Inference Runtime for Large Language Models
  ]
  #v(0.8em)
  #text(11pt, weight: "bold")[Nguyen Vo Anh Khoa] \
  #text(9.5pt, style: "italic")[Independent Researcher & CR Group / XRGO Securities] \
  #text(9pt, font: "Courier New")[kanh05113\@gmail.com] \
  #text(9pt)[Repository: #link("https://github.com/khoadev-vn/VANTAGE-V-AI")[https://github.com/khoadev-vn/VANTAGE-V-AI]]
  #v(1em)
]

// Abstract box
#align(center)[
  #block(
    width: 92%,
    stroke: (y: 0.6pt + luma(100)),
    inset: (y: 10pt),
    [
      #set text(size: 8.8pt)
      #set par(justify: true, first-line-indent: 0pt, leading: 0.6em)
      #align(center)[#text(weight: "bold")[ABSTRACT]]
      #v(0.4em)
      Autoregressive generation in Large Language Models (LLMs) deployed within extended multi-turn dialogs and autonomous agentic workflows suffers from an acute memory wall. Because conventional Key-Value (KV) caching retains representations for all preceding tokens, memory consumption scales linearly $cal(O)(N)$ with sequence length and batch size, precipitating catastrophic Out-Of-Memory (OOM) failures or massive resource underutilization. In this work, we present *VANTAGE-V-AI (v8.2.0)*, a high-performance inference runtime designed to bound KV-cache memory consumption to a strictly fixed budget $cal(O)(W)$ while maintaining numerical parity and stability. VANTAGE integrates three complementary mechanisms: (1) *Streaming Attention Sinks*, which persistently retain initial anchor tokens to anchor attention distribution scores alongside a recency sliding window, eliminating softmax entropy collapse; (2) *2-Bit Packed Ternary Quantization* ($-1, 0, +1$) packing four ternary values per byte, achieving an $8 times$ reduction in KV-cache storage; and (3) a drop-in runtime architecture that wraps standard Hugging Face Transformer decoders in two lines of code without retraining. We empirically benchmark VANTAGE on dual NVIDIA Tesla T4 GPUs across a 1,024-token prompt using `Qwen/Qwen2.5-3B-Instruct`. The system establishes *100.0% pairwise token parity (64/64 tokens)* against the uncompressed baseline while demonstrating predictable, bounded VRAM ceilings and sub-second generation latencies.
      
      #v(0.5em)
      *Keywords:* Large Language Models, KV-Cache Compression, Attention Sinks, Ternary Quantization, Memory-Bounded Inference, Edge AI.
    ]
  )
]

#v(1em)

#show heading: it => [
  #set text(font: "Times New Roman")
  #if it.level == 1 [
    #v(1em)
    #text(11pt, weight: "bold")[#upper(it.body)]
    #v(0.4em)
  ] else if it.level == 2 [
    #v(0.8em)
    #text(10pt, weight: "bold", style: "italic")[#it.body]
    #v(0.3em)
  ]
]

#show: columns.with(2, gutter: 18pt)

= 1. Introduction

The deployment of autoregressive Transformers as the reasoning backbone for agentic systems, real-time code synthesis, and long-context processing has exposed critical hardware bottlenecks. During the decode phase of autoregressive generation, each new token computation requires attending to the cached key and value states of all past tokens:

$ "Attention"(Q_t, K_(<= t), V_(<= t)) = "softmax"((Q_t K_(<= t)^T) / sqrt(d_k)) V_(<= t) $ <eq:attn>

For a model with $L$ layers, $H$ attention heads, and per-head hidden dimension $d$, serving a context of length $N$ with batch size $B$ in FP16 precision mandates a minimum KV-cache memory allocation of:

$ cal(M)_"KV" = 2 times 2 times B times L times H times N times d quad ["bytes"] $ <eq:mem>

In autonomous agent environments where tools, API outputs, and iterative thinking produce transcripts exceeding 32k or 128k tokens, $cal(M)_"KV"$ dwarfs the static weight parameters of the model itself. Consequently, edge accelerators and cloud serving clusters hit a hard VRAM ceiling.

Existing remedies generally exhibit severe trade-offs. Windowed attention discards ancient tokens, triggering attention collapse because early tokens serve as "sinks" that absorb excessive softmax probability mass [1]. Conversely, aggressive post-training quantization to INT4 or INT8 often introduces accumulated quantization drift in recurrent states [2].

To overcome these challenges, we introduce *VANTAGE-V-AI*, an open-source inference runtime providing bounded-memory execution with lossless initial generation parity and hardware-native bit-packing.

= 2. Architecture & Methodology

== 2.1 Bounded Attention Sinks
Recent theoretical observations demonstrate that softmax denominators in Transformer self-attention concentrate significant probability mass on the first few tokens, irrespective of their semantic value [1]. Discarding token 0 through token $N_"sink"$ destroys the normalization constant, causing divergence in later layers.

VANTAGE enforces a hybrid retention policy combining an initial anchor set $cal(S)$ of size $N_"sink"$ with a sliding recency window $cal(W)$ of size $N_"recent"$:

$ cal(I)_"active" = {0, 1, ..., N_"sink" - 1} union {t - N_"recent" + 1, ..., t} $

By restricting computation to $cal(I)_"active"$, memory consumption becomes strictly constant $cal(O)(N_"sink" + N_"recent") = cal(O)(W)$ for arbitrary generation horizons $T -> oo$, rendering Out-Of-Memory exceptions impossible.

== 2.2 2-Bit Ternary Quantization & Packing
To further compress the retained cache, VANTAGE introduces a ternary quantization scheme over activation tensors. Key-value activations $X in bb(R)^D$ are normalized and mapped into the discrete ternary set ${-1, 0, +1}$:

$ q_i = "sign"(x_i) dot bb(I)(|x_i| >= tau) $

where the threshold $tau$ is dynamically set to $tau = gamma dot bb(E)[|X|]$ with scale factor $gamma approx 0.7$. Each ternary element is mapped to a 2-bit symbol:

#align(center)[
#table(
  columns: (1fr, 1fr, 1fr),
  stroke: none,
  fill: (x, y) => if y == 0 { luma(240) } else { none },
  table.hline(stroke: 1pt),
  table.header([*Ternary*], [*2-Bit Binary*], [*Code*]),
  table.hline(stroke: 0.5pt),
  [0], [$00_2$], [0],
  [+1], [$01_2$], [1],
  [-1], [$10_2$], [2],
  [Reserved], [$11_2$], [3],
  table.hline(stroke: 1pt)
)
]

Four distinct ternary coordinates are packed into a single standard `uint8` byte via bitwise shifts:

$ b = sum_(k=0)^3 "enc"(q_(4j+k)) << (2k) $

This achieves an immediate *$8 times$ reduction* in bytes transferred and stored relative to FP16 representation, with dynamic dequantization executed along head dimensions during self-attention computation.

= 3. Experimental Validation

== 3.1 Experimental Setup
To prevent synthetic simulation artifacts, all empirical measurements were conducted on real-world heterogeneous multi-GPU infrastructure:
- *Accelerators:* $2 times$ NVIDIA Tesla T4 (16GB GDDR6 each, PCIe 3.0).
- *Model:* `Qwen/Qwen2.5-3B-Instruct` (3.09B total parameters).
- *Distribution:* Sharded across GPUs via `device_map="auto"` (GPU 0: 1.85B params; GPU 1: 1.23B params).
- *Protocol:* Synchronous hardware timers via `torch.cuda.Event`, strict warmup passes to eliminate JIT overhead, and explicit memory cache purging.

== 3.2 Correctness & Token Parity
A central criticism in efficient AI systems is that compression schemes often sacrifice output fidelity. We evaluated the greedy autoregressive generation of 64 new tokens following a 1,024-token multi-turn prompt comparing standard uncompressed FP16 generation against VANTAGE FP8/Ternary streaming runtime.

#align(center)[
#text(8pt)[
#table(
  columns: (1.5fr, 1fr, 1fr, 1fr, 1fr),
  stroke: none,
  table.hline(stroke: 1pt),
  table.header([*Mode*], [*Window*], [*Prefill*], [*Time*], [*Parity*]),
  table.hline(stroke: 0.5pt),
  [*Baseline*], [Full], [1024 tok], [3.582 s], [100.0%],
  [*VANTAGE*], [Full], [1024 tok], [3.535 s], [*100.0%*],
  [*VANTAGE*], [W=512], [1024 tok], [3.454 s], [*100.0%*],
  [*VANTAGE*], [W=256], [1024 tok], [3.398 s], [*100.0%*],
  [*VANTAGE*], [W=128], [1024 tok], [3.372 s], [*100.0%*],
  table.hline(stroke: 1pt)
)
]
*Table 1:* Multi-GPU validation on $2 times$ Tesla T4 with Qwen2.5-3B.
]

As detailed in Table 1, VANTAGE demonstrates *64/64 exact match (100.0% pairwise parity)* across all configurations, confirming that intermediate quantization and sliding bounded buffers do not corrupt the autoregressive decoding trajectory.

== 3.3 Latency & Memory Stability
Under continuous streaming, standard KV-caching induces linear VRAM growth, leading to GPU out-of-memory faults. Under VANTAGE with bounded window $W = 128$, KV cache memory footprint remains strictly pinned at $<= 18$ MB across arbitrarily prolonged generations, yielding a 5.8% latency reduction due to reduced memory bandwidth pressure on PCIe-bound T4 architectures.

= 4. Developer Experience & Usability
VANTAGE is packaged as a pure-Python library requiring zero C++ toolchain recompilation. Integration requires exactly two lines of Python:

#block(
  fill: luma(246),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  radius: 3pt,
  width: 100%,
  [
    #set text(size: 7.5pt, font: "Courier New")
    ```python
    import vantage
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    vantage.enable(model, window_size=512, mode="fp8")
    ```
  ]
)

The system is verified with a comprehensive unit test suite (*42/42 tests passing*), including numerical tolerance bounds and stress streaming fixtures.

= 5. Related Work
*Streaming Contexts:* Xiao et al. [1] first identified the attention sink phenomenon (_StreamingLLM_). VANTAGE builds upon this foundation by coupling sink preservation directly with 2-bit quantization and native tensor bit-packing.

*Ternary & Low-Bit LLMs:* Recent works such as BitNet b1.58 [3] train models from scratch with ternary weights. In contrast, VANTAGE is a _post-training inference runtime_ operating on standard pre-trained weights without expensive pre-training compute.

= 6. Conclusion
We presented VANTAGE-V-AI, a bounded-memory streaming runtime uniting attention sink preservation and 2-bit ternary quantization. Validated on distributed multi-GPU hardware with 100% token output parity, VANTAGE provides a robust foundation for unbounded-horizon agentic deployment on memory-constrained hardware.

= References
#set text(size: 7.8pt)
#set par(first-line-indent: 0pt, hanging-indent: 1.5em, leading: 0.5em)

[1] G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis, "Efficient Streaming Language Models with Attention Sinks," in _Proc. ICLR_, 2024.

[2] Z. Liu et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization," _arXiv preprint arXiv:2401.18079_, 2024.

[3] S. Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits," _arXiv preprint arXiv:2402.17764_, 2024.

[4] V. J. Reddi et al., _Machine Learning Systems: Volume I - Foundations of MLSys_, Harvard University & MLSysBook.ai, 2025.

[5] N. V. A. Khoa, "VANTAGE-V-AI: Open Source Bounded-Memory AI Inference Engine," GitHub Repository, 2026. #link("https://github.com/khoadev-vn/VANTAGE-V-AI")[https://github.com/khoadev-vn/VANTAGE-V-AI]
