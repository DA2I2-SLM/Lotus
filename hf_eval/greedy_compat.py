"""greedy_compat.py — force deterministic greedy decoding for SP model evaluation.

ROOT CAUSE ADDRESSED:
  google/gemma-3-270m ships with generation_config.json:
    {"do_sample": true, "top_k": 64, "top_p": 0.95, "cache_implementation": "hybrid"}

  OpenTSLMSP.generate() calls self.llm.generate() WITHOUT passing do_sample, so
  these stochastic defaults are inherited.  Result: the same input produces a
  DIFFERENT output on every call, and coherence is ~12–20% (stochastic luck).

  Forcing do_sample=False (greedy) improves measured coherence from 1/8 to 3/8
  on Sleep test samples (encoder_compat also applied).

USAGE (in run_eval.py or any eval script, after model construction):

    import greedy_compat
    greedy_compat.apply(model)

  "model" is an OpenTSLM (or OpenTSLMSP) instance whose .llm is the
  PeftModelForCausalLM wrapping Gemma3ForCausalLM.

WHAT IT DOES:
  Mutates model.llm.generation_config in-place:
    do_sample   → False    (greedy argmax, not stochastic top-k/p)
    top_k       → None     (disabled; ignored with do_sample=False anyway)
    top_p       → None     (disabled)
    temperature → None     (unset; irrelevant for greedy, avoids scale confusion)
  No repo source files are modified.
"""


def apply(model) -> None:
    """Patch model.llm.generation_config to use greedy (do_sample=False).

    Call this once after OpenTSLM.load_pretrained() and before any model.generate() calls.

    Args:
        model: OpenTSLM (or OpenTSLMSP) instance.
    """
    gc = model.llm.generation_config
    gc.do_sample = False
    gc.top_k = None
    gc.top_p = None
    gc.temperature = None
    print(
        "[greedy_compat] Forced greedy decoding: do_sample=False, top_k=None, "
        "top_p=None, temperature=None"
    )
