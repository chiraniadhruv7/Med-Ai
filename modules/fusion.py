from typing import Optional


class FusionModule:

    AGREEMENT_BOOST = 0.10
    DISAGREEMENT_PENALTY = 0.15
    LLM_MENTION_THRESHOLD = 0.4

    def fuse(
        self,
        cnn_predictions: dict,
        llm_output: dict,
    ) -> dict:
        visual_findings = {}
        contradictions = []
        agreement_count = 0
        total_significant = 0

        llm_text = (
            llm_output.get("clinical_reasoning", "")
            + " "
            + " ".join(llm_output.get("key_observations", []))
            + " "
            + " ".join(llm_output.get("differential_diagnosis", []))
        ).lower()

        for label, cnn_prob in cnn_predictions.items():
            label_lower = label.lower().replace("_", " ")
            llm_mentions = label_lower in llm_text

            adjusted_prob = cnn_prob

            if cnn_prob > self.LLM_MENTION_THRESHOLD:
                total_significant += 1

                if llm_mentions:
                    adjusted_prob = min(1.0, cnn_prob + self.AGREEMENT_BOOST)
                    agreement_count += 1
                else:
                    adjusted_prob = max(0.0, cnn_prob - self.DISAGREEMENT_PENALTY)
                    contradictions.append({
                        "pathology": label,
                        "cnn_probability": round(cnn_prob, 4),
                        "llm_mentions": False,
                        "adjustment": "reduced",
                    })

            elif cnn_prob <= self.LLM_MENTION_THRESHOLD and llm_mentions:
                adjusted_prob = min(1.0, cnn_prob + self.AGREEMENT_BOOST * 0.5)

            visual_findings[label] = round(adjusted_prob, 4)

        agreement_ratio = (
            agreement_count / total_significant if total_significant > 0 else 1.0
        )

        return {
            "visual_findings": visual_findings,
            "contradictions": contradictions,
            "agreement_ratio": round(agreement_ratio, 4),
            "total_significant_findings": total_significant,
        }
