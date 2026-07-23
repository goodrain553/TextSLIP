"""Optional downstream evaluation hook.

The standalone TextSLIP training repository intentionally excludes the
downstream evaluation suites from the source UniMed-CLIP workspace. This
stub keeps the training loop import-compatible while making evaluation a
no-op unless users add their own evaluator.
"""


def slip_evaluate(args, model, val_transform, tokenizer, epoch=0):
    return {}
