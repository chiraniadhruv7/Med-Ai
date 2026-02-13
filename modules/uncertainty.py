import math
from typing import Optional


class UncertaintyEstimator:

    def estimate(
        self,
        cnn_predictions: dict,
        llm_token_confidence: Optional[float] = None,
        agreement_ratio: float = 1.0,
    ) -> dict:
        cnn_confidence = self._cnn_confidence(cnn_predictions)
        llm_confidence = self._llm_confidence(llm_token_confidence)
        fused_confidence = self._fused_confidence(cnn_confidence, llm_confidence, agreement_ratio)

        return {
            "cnn": round(cnn_confidence, 4),
            "llm": round(llm_confidence, 4),
            "fused": round(fused_confidence, 4),
            "interpretation": self._interpret(fused_confidence),
        }

    def _cnn_confidence(self, predictions: dict) -> float:
        if not predictions:
            return 0.0

        probs = list(predictions.values())
        entropy = 0.0
        for p in probs:
            p_clamped = max(min(p, 0.9999), 0.0001)
            entropy -= p_clamped * math.log2(p_clamped) + (1 - p_clamped) * math.log2(1 - p_clamped)

        max_entropy = len(probs) * 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        confidence = 1.0 - normalized_entropy

        return max(0.0, min(1.0, confidence))

    def _llm_confidence(self, token_confidence: Optional[float]) -> float:
        if token_confidence is not None:
            return max(0.0, min(1.0, token_confidence))
        return 0.7

    def _fused_confidence(
        self,
        cnn_conf: float,
        llm_conf: float,
        agreement_ratio: float,
    ) -> float:
        CNN_WEIGHT = 0.45
        LLM_WEIGHT = 0.35
        AGREEMENT_WEIGHT = 0.20

        fused = (
            CNN_WEIGHT * cnn_conf
            + LLM_WEIGHT * llm_conf
            + AGREEMENT_WEIGHT * agreement_ratio
        )
        return max(0.0, min(1.0, fused))

    def _interpret(self, confidence: float) -> str:
        if confidence >= 0.8:
            return "High confidence — findings are consistent across models"
        elif confidence >= 0.5:
            return "Moderate confidence — some uncertainty in findings"
        else:
            return "Low confidence — significant uncertainty, interpret with caution"
