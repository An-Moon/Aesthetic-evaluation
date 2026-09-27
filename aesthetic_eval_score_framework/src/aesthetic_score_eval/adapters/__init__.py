def build_adapter(base_cfg: dict, model_cfg: dict):
    adapter_name = str(model_cfg.get("adapter", "")).lower().strip()
    if adapter_name == "artimuse":
        from aesthetic_score_eval.adapters.artimuse import ArtiMuseScoreAdapter

        return ArtiMuseScoreAdapter(base_cfg, model_cfg)
    if adapter_name == "prompt_numeric":
        from aesthetic_score_eval.adapters.prompt_numeric import PromptNumericScoreAdapter

        return PromptNumericScoreAdapter(base_cfg, model_cfg)
    if adapter_name == "artquant":
        from aesthetic_score_eval.adapters.artquant import ArtQuantAdapter

        return ArtQuantAdapter(base_cfg, model_cfg)
    if adapter_name == "aesexpert":
        from aesthetic_score_eval.adapters.aesexpert import AesExpertScoreAdapter

        return AesExpertScoreAdapter(base_cfg, model_cfg)
    if adapter_name in {"qalign", "onealign"}:
        from aesthetic_score_eval.adapters.qalign import QAlignScoreAdapter

        return QAlignScoreAdapter(base_cfg, model_cfg)
    if adapter_name == "unipercept":
        from aesthetic_score_eval.adapters.unipercept import UniPerceptScoreAdapter

        return UniPerceptScoreAdapter(base_cfg, model_cfg)
    if adapter_name == "qsit":
        from aesthetic_score_eval.adapters.qsit import QSITScoreAdapter

        return QSITScoreAdapter(base_cfg, model_cfg)
    if adapter_name == "qwen3vl_prompt":
        from aesthetic_score_eval.adapters.qwen3vl_prompt import Qwen3VLPromptScoreAdapter

        return Qwen3VLPromptScoreAdapter(base_cfg, model_cfg)
    if adapter_name == "internvl_prompt":
        from aesthetic_score_eval.adapters.internvl_prompt import InternVLPromptScoreAdapter

        return InternVLPromptScoreAdapter(base_cfg, model_cfg)
    if adapter_name == "llava_onevision_prompt":
        from aesthetic_score_eval.adapters.llava_onevision_prompt import LLaVAOneVisionPromptScoreAdapter

        return LLaVAOneVisionPromptScoreAdapter(base_cfg, model_cfg)
    raise ValueError(f"Unknown adapter: {adapter_name}")
