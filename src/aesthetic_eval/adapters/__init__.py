def build_adapter(base_cfg: dict, model_cfg: dict):
    adapter_name = str(model_cfg.get("adapter", "")).lower().strip()
    if adapter_name == "internvl":
        from aesthetic_eval.adapters.internvl import InternVLAdapter

        return InternVLAdapter(base_cfg, model_cfg)
    if adapter_name == "qwen":
        from aesthetic_eval.adapters.qwen import QwenAdapter

        return QwenAdapter(base_cfg, model_cfg)
    if adapter_name == "llava":
        from aesthetic_eval.adapters.llava import LlavaAdapter

        return LlavaAdapter(base_cfg, model_cfg)
    if adapter_name == "aesexpert":
        from aesthetic_eval.adapters.aesexpert import AesExpertAdapter

        return AesExpertAdapter(base_cfg, model_cfg)
    if adapter_name == "artquant":
        from aesthetic_eval.adapters.artquant import ArtQuantAdapter

        return ArtQuantAdapter(base_cfg, model_cfg)
    if adapter_name == "onealign":
        from aesthetic_eval.adapters.onealign import OneAlignAdapter

        return OneAlignAdapter(base_cfg, model_cfg)
    if adapter_name == "qsit":
        from aesthetic_eval.adapters.qsit import QSITAdapter

        return QSITAdapter(base_cfg, model_cfg)
    if adapter_name == "unipercept":
        from aesthetic_eval.adapters.unipercept import UniPerceptAdapter

        return UniPerceptAdapter(base_cfg, model_cfg)
    raise ValueError(f"Unknown adapter: {adapter_name}")
