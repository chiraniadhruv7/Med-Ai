
HIGH_RISK_PATHOLOGIES = {"Pneumothorax", "Mass", "Nodule"}
HIGH_RISK_THRESHOLD = 0.5
MODERATE_RISK_THRESHOLD = 0.3
ANY_HIGH_THRESHOLD = 0.7


class RiskStratifier:

    def assess(self, visual_findings: dict, confidence_scores: dict = None) -> dict:
        risk_level = "Low"
        risk_factors = []
        max_prob = 0.0
        max_label = ""

        for label, prob in visual_findings.items():
            if prob > max_prob:
                max_prob = prob
                max_label = label

            if label in HIGH_RISK_PATHOLOGIES and prob > HIGH_RISK_THRESHOLD:
                risk_level = "High"
                risk_factors.append(f"{label} detected with probability {prob:.2f}")

            elif prob > ANY_HIGH_THRESHOLD:
                risk_level = "High"
                risk_factors.append(f"{label} detected with high probability {prob:.2f}")

            elif prob > MODERATE_RISK_THRESHOLD and risk_level != "High":
                risk_level = "Moderate"
                risk_factors.append(f"{label} detected with moderate probability {prob:.2f}")

        if confidence_scores:
            fused_conf = confidence_scores.get("fused", 1.0)
            if fused_conf < 0.4 and risk_level == "High":
                risk_level = "Moderate"
                risk_factors.append("Downgraded due to low model confidence")

        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "dominant_finding": max_label,
            "dominant_probability": round(max_prob, 4),
        }
