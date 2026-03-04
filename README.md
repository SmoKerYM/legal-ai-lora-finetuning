# Legal AI LoRA Fine-Tuning: Form vs Facts in Legal Language Models

> **Why does legal AI need both RAG and fine-tuning?** This notebook answers that question empirically by fine-tuning a small LLM with LoRA and observing what it learns — and what it can't.

## TL;DR

Fine-tuning teaches a model **how** to respond (empathy, structure, actionable guidance). It does **not** teach the model legal **facts** (statutes, case law, phone numbers). For legal AI, you need:

- **Fine-tuning** → response style, tone, domain reasoning patterns
- **RAG** → grounded, citable, up-to-date legal knowledge
- **Human-in-the-loop** → lawyer review before any output reaches a real user

This notebook demonstrates all three points through a hands-on experiment.

## Motivation

This project was built while preparing for a role involving language models for the legal space.  
The central question: if ~90% of people globally cannot access a lawyer when they need one, can AI help close this justice gap — and what's the right architecture to do it safely?

## What This Notebook Does

| Step | What Happens |
|------|-------------|
| 1. Load base model | Qwen2.5-1.5B-Instruct, 4-bit quantised via BitsAndBytes |
| 2. Test before fine-tuning | Ask legal aid questions → observe generic/hallucinated responses |
| 3. Prepare training data | 6 synthetic examples teaching empathetic, structured legal aid style |
| 4. Apply LoRA | Freeze base weights, add trainable adapters (~1.2% of parameters) |
| 5. Train | ~60 steps, loss drops from 2.160 → 0.009 |
| 6. Test after fine-tuning | Same questions → structured, empathetic, actionable responses |
| 7. Analyse | Compare outputs to understand what changed and what didn't |

## Key Findings

### What Fine-Tuning Fixed
- **Tone**: Base model said *"talk to a grown-up"* for a domestic violence question; fine-tuned model responds with appropriate empathy and respect
- **Structure**: Responses gained clear headings, immediate steps, legal options, and calls to action
- **Safety**: Base model hallucinated a fake crisis hotline number (987); fine-tuned model learned to use safer generic phrasing

### What Fine-Tuning Couldn't Fix
- **Factual grounding**: The fine-tuned model dropped a correct Article 19 ICCPR citation that the base model had, replacing it with vaguer language
- **Domain terminology**: Replace "protection order" incorrectly with "injunction" and "restraining".
- **Language stability**: On an unseen topic (refugee rights), the model code-switched to Chinese mid-sentence — a sign of instability when the adapter is pushed beyond its training distribution

### The Conclusion

| Aspect | Base Model | Fine-Tuned | What's Needed |
|--------|-----------|------------|---------------|
| Tone & empathy | ❌ Generic | ✅ Appropriate | Fine-tuning |
| Response structure | ❌ Unstructured | ✅ Clear & actionable | Fine-tuning |
| Factual accuracy | ❌ Hallucinated | ❌ Still unreliable | **RAG** |
| Legal citations | ⚠️ Sometimes correct | ❌ Less specific | **RAG** |
| Generalisation | ✅ Stable | ⚠️ Breaks on new topics | More training data |

**Fine-tuning and RAG solve orthogonal problems.** Neither alone is sufficient for legal AI.

## Technical Details

### Stack
- **Model**: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **Fine-tuning**: LoRA via [PEFT](https://github.com/huggingface/peft) (r=16, alpha=32, targeting all attention + MLP projections)
- **Quantisation**: 4-bit NF4 via [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- **Trainer**: [TRL SFTTrainer](https://github.com/huggingface/trl)
- **Hardware**: Google Colab A100 GPU

### LoRA Configuration
```python
LoraConfig(
    r=16,              # Rank — balances expressiveness vs efficiency
    lora_alpha=32,     # Scaling factor (2× rank)
    target_modules=[   # All attention + MLP projections
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

### Training
- **Dataset**: 6 synthetic legal aid Q&A pairs (style-focused, not fact-focused)
- **Epochs**: 20 (intentional overfitting for style transfer on tiny dataset)
- **Batch size**: 2
- **Learning rate**: 2e-4
- **Loss curve**: 2.160 → 0.009 over ~60 steps

### Debugging Notes
Two issues encountered and resolved during development:

1. **dtype mismatch on T4 GPU**: Layer norm and embedding layers converted tensors to bfloat16, which T4 doesn't support. Fix: switched to A100 and set `torch_dtype=torch.bfloat16` consistently across model loading and training config.

2. **Nonsense output at inference**: Model produced repetitive tokens ("systemsystemsystem...") after training. Root cause: `model.eval()` was not called before inference, so LoRA dropout (5%) was still active, destabilising autoregressive generation. Fix: call `model.eval()` and add `repetition_penalty=1.2` as a safety net.


## Running This Notebook

### Prerequisites
- Google Colab account (A100 GPU recommended; available via Colab Pro)
- No API keys required — uses open-source model from Hugging Face

### Quick Start
1. Open `sft_lora_A100.ipynb` in Google Colab
2. Set runtime to **A100 GPU** (Runtime → Change runtime type)
3. Run all cells sequentially

## References & Further Reading

- [Why Aren't We Using AI to Advance Justice?](https://time.com/collections/davos-2026/7339221/ai-justice-gap-womens-rights-legal/) — TIME, Jan 2026
- [Oxford Institute of Technology and Justice](https://www.techandjustice.bsg.ox.ac.uk/)
- [RAFT: Adapting Language Model to Domain Specific RAG](https://arxiv.org/abs/2403.10131) — UC Berkeley, 2024
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [Building AI Applications with Microsoft Semantic Kernel](https://www.amazon.com/Building-Applications-Microsoft-Semantic-Kernel/dp/1835463703) — Lucas A. Meyer

## Author

**Mingwei Yan** — MSc Applied Computational Science & Engineering, Imperial College London

- [LinkedIn](https://www.linkedin.com/in/mingwei-yan-my324)
- [AI Experience Matcher (RAG Project)](https://github.com/SmoKerYM/AI_Powered_Experience_Matcher)

---

*AI should assist — not replace — qualified legal professionals.*