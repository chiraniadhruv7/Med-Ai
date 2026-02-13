import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from modules.cnn_model import CXRClassifier, PATHOLOGY_LABELS
from modules.multimodal_reasoning import MedGemmaReasoner
from modules.fusion import FusionModule
from modules.risk_module import RiskStratifier
from modules.uncertainty import UncertaintyEstimator
from modules.explainability import GradCAMExplainer


def test_cnn_prediction():
    print("[TEST] CNN prediction...")
    classifier = CXRClassifier(use_pretrained=True)
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    predictions = classifier.predict(dummy_image)
    assert isinstance(predictions, dict)
    assert len(predictions) == len(PATHOLOGY_LABELS)
    for label, prob in predictions.items():
        assert 0.0 <= prob <= 1.0, f"{label}: {prob} out of range"
    print(f"  ✓ {len(predictions)} pathologies predicted")
    return predictions


def test_medgemma_dummy():
    print("[TEST] MedGemma dummy mode...")
    reasoner = MedGemmaReasoner(mode="dummy")
    result = reasoner.reason(image_path=None, symptoms="cough and fever")
    assert "clinical_reasoning" in result
    assert "differential_diagnosis" in result
    assert len(result["differential_diagnosis"]) > 0
    print(f"  ✓ Reasoning: {result['clinical_reasoning'][:60]}...")
    return result


def test_fusion():
    print("[TEST] Fusion module...")
    fusion = FusionModule()
    cnn_preds = {"Pneumonia": 0.85, "Effusion": 0.12, "Cardiomegaly": 0.05}
    llm_output = {
        "clinical_reasoning": "findings suggest possible pneumonia",
        "key_observations": ["opacity in right lower lobe"],
        "differential_diagnosis": ["pneumonia"],
    }
    result = fusion.fuse(cnn_preds, llm_output)
    assert "visual_findings" in result
    assert result["visual_findings"]["Pneumonia"] >= 0.85
    print(f"  ✓ Fused findings: {result['visual_findings']}")
    return result


def test_uncertainty():
    print("[TEST] Uncertainty estimation...")
    estimator = UncertaintyEstimator()
    predictions = {"Pneumonia": 0.85, "Effusion": 0.12, "Cardiomegaly": 0.05}
    scores = estimator.estimate(predictions, llm_token_confidence=0.8, agreement_ratio=0.9)
    assert "cnn" in scores
    assert "llm" in scores
    assert "fused" in scores
    assert 0.0 <= scores["fused"] <= 1.0
    print(f"  ✓ Confidence: CNN={scores['cnn']}, LLM={scores['llm']}, Fused={scores['fused']}")
    return scores


def test_risk():
    print("[TEST] Risk stratification...")
    stratifier = RiskStratifier()
    findings_high = {"Pneumothorax": 0.75, "Pneumonia": 0.3}
    result = stratifier.assess(findings_high)
    assert result["risk_level"] == "High"
    findings_low = {"Pneumonia": 0.1, "Effusion": 0.05}
    result_low = stratifier.assess(findings_low)
    assert result_low["risk_level"] == "Low"
    print(f"  ✓ High risk test: {result['risk_level']}, Low risk test: {result_low['risk_level']}")
    return result


def test_gradcam():
    print("[TEST] Grad-CAM explainability...")
    classifier = CXRClassifier(use_pretrained=True)
    explainer = GradCAMExplainer()
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    input_tensor = classifier.preprocess(dummy_image)
    input_tensor.requires_grad_(True)
    predictions = {"Pneumonia": 0.85, "Effusion": 0.12}
    heatmap_b64 = explainer.generate_for_top_pathology(
        model=classifier.get_model(),
        image=dummy_image,
        input_tensor=input_tensor,
        predictions=predictions,
    )
    assert isinstance(heatmap_b64, str)
    assert len(heatmap_b64) > 100
    print(f"  ✓ Heatmap base64 length: {len(heatmap_b64)}")
    return heatmap_b64


def run_all_tests():
    print("=" * 60)
    print("  Explainable Clinical AI — Pipeline Tests")
    print("=" * 60)
    preds = test_cnn_prediction()
    llm = test_medgemma_dummy()
    fusion_result = test_fusion()
    scores = test_uncertainty()
    risk_result = test_risk()
    heatmap = test_gradcam()
    print("=" * 60)
    print("  ✅ ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
