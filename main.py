import io
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from modules.cnn_model import CXRClassifier
from modules.multimodal_reasoning import MedGemmaReasoner
from modules.fusion import FusionModule
from modules.risk_module import RiskStratifier
from modules.uncertainty import UncertaintyEstimator
from modules.explainability import GradCAMExplainer

cnn: CXRClassifier = None
reasoner: MedGemmaReasoner = None
fusion: FusionModule = None
risk: RiskStratifier = None
uncertainty: UncertaintyEstimator = None
explainer: GradCAMExplainer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cnn, reasoner, fusion, risk, uncertainty, explainer
    print("[Startup] Loading modules...")
    cnn = CXRClassifier(use_pretrained=True)
    reasoner = MedGemmaReasoner()
    fusion = FusionModule()
    risk = RiskStratifier()
    uncertainty = UncertaintyEstimator()
    explainer = GradCAMExplainer()
    print("[Startup] All modules loaded.")
    yield
    print("[Shutdown] Cleaning up...")


app = FastAPI(
    title="Explainable Multimodal Clinical AI Assistant",
    description="Chest X-ray analysis with CNN + MedGemma fusion. For research use only.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "cnn_ready": cnn.ready if cnn else False,
        "medgemma_status": reasoner.get_status() if reasoner else {},
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    symptoms: str = Form(""),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    cnn_predictions = cnn.predict(pil_image)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_image.save(tmp, format="PNG")
            tmp_path = tmp.name

        llm_output = reasoner.reason(image_path=tmp_path, symptoms=symptoms)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    fusion_result = fusion.fuse(cnn_predictions, llm_output)

    confidence_scores = uncertainty.estimate(
        cnn_predictions=cnn_predictions,
        llm_token_confidence=llm_output.get("token_confidence"),
        agreement_ratio=fusion_result.get("agreement_ratio", 1.0),
    )

    risk_result = risk.assess(
        visual_findings=fusion_result["visual_findings"],
        confidence_scores=confidence_scores,
    )

    heatmap_b64 = ""
    try:
        input_tensor = cnn.preprocess(pil_image)
        input_tensor.requires_grad_(True)
        heatmap_b64 = explainer.generate_for_top_pathology(
            model=cnn.get_model(),
            image=pil_image,
            input_tensor=input_tensor,
            predictions=cnn_predictions,
        )
    except Exception as e:
        print(f"[Explainability] Grad-CAM failed: {e}")

    response = {
        "visual_findings": fusion_result["visual_findings"],
        "clinical_reasoning": llm_output.get("clinical_reasoning", ""),
        "differential_diagnosis": llm_output.get("differential_diagnosis", []),
        "confidence_scores": {
            "cnn": confidence_scores["cnn"],
            "llm": confidence_scores["llm"],
            "fused": confidence_scores["fused"],
        },
        "risk_level": risk_result["risk_level"],
        "risk_factors": risk_result["risk_factors"],
        "contradictions": fusion_result.get("contradictions", []),
        "heatmap_available": bool(heatmap_b64),
        "heatmap_base64": heatmap_b64,
        "disclaimer": "For research and educational use only. Not a clinical diagnostic tool.",
    }

    return JSONResponse(content=response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
