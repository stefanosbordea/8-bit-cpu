"""Quantization-aware training of the integer model (SPEC Parts B, C, E).

    from lsllm.qat.calibrate import calibrate      float model -> QatModel
    from lsllm.qat.model import QatModel           the integer graph as a torch module
    from lsllm.qat.export import export            QatModel -> model.json and model.npz
    python -m lsllm.qat.train                      calibrate, fine-tune, export
"""
