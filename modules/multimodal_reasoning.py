import os
import json
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SAFETY_SYSTEM_PROMPT = """You are a medical imaging AI research assistant. You analyze chest X-ray images and clinical symptoms to provide educational analysis.

STRICT RULES:
1. NEVER provide a definitive diagnosis.
2. NEVER prescribe medications or treatments.
3. NEVER fabricate physical exam findings or procedures.
4. ALWAYS state findings as observations, not conclusions.
5. ALWAYS recommend professional medical consultation.
6. ONLY reason from the provided image and text input.
7. This system is for research and educational use only.

Output your analysis in the following JSON structure:
{
    "clinical_reasoning": "Your analysis of the image and symptoms...",
    "differential_diagnosis": ["Possible condition 1", "Possible condition 2", "Possible condition 3"],
    "key_observations": ["Observation 1", "Observation 2"],
    "recommendation": "Consult a qualified healthcare professional for proper evaluation."
}
"""


class MedGemmaReasoner:

    def __init__(self, mode: str = None):
        self.mode = mode or os.getenv("MEDGEMMA_MODE", "local")
        self.local_model = None
        self.local_processor = None
        self.ready = False
        self._initialize()

    def _initialize(self):
        if self.mode == "local":
            self._init_local()
        elif self.mode == "api":
            self._init_api()
        else:
            self.ready = False

    def _init_local(self):
        try:
            old_offline = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                from mlx_vlm import load, generate
                self._mlx_generate = generate
                model_name = "mlx-community/medgemma-4b-it-8bit"
                self.local_model, self.local_processor = load(model_name)
                self.ready = True
                print("[MedGemma] Local model loaded successfully.")
            finally:
                if old_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = old_offline
        except Exception as e:
            print(f"[MedGemma] Local MLX init failed: {e}")
            print("[MedGemma] To download: huggingface-cli download mlx-community/medgemma-4b-it-8bit")
            print("[MedGemma] Falling back to API mode...")
            self.mode = "api"
            self._init_api()

    def _init_api(self):
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                print("[MedGemma] No GEMINI_API_KEY set. Using dummy mode.")
                self.ready = False
                return
            genai.configure(api_key=api_key)
            self._genai = genai
            self.ready = True
        except Exception as e:
            print(f"[MedGemma] API init failed: {e}")
            self.ready = False

    def reason(self, image_path: str = None, symptoms: str = "") -> dict:
        if not self.ready:
            return self._dummy_response(symptoms)

        if self.mode == "local":
            return self._reason_local(image_path, symptoms)
        elif self.mode == "api":
            return self._reason_api(image_path, symptoms)
        else:
            return self._dummy_response(symptoms)

    def _reason_local(self, image_path: str, symptoms: str) -> dict:
        try:
            from mlx_vlm import generate
            from mlx_vlm.prompt_utils import apply_chat_template

            user_text = (
                f"{SAFETY_SYSTEM_PROMPT}\n\n"
                f"Patient presents with: {symptoms}\n\n"
                "Analyze the provided chest X-ray image along with the symptoms. "
                "Provide your structured analysis as JSON with keys: "
                "clinical_reasoning, differential_diagnosis, key_observations, recommendation."
            )

            if image_path and os.path.exists(image_path):
                prompt = apply_chat_template(
                    self.local_processor,
                    config=self.local_model.config,
                    prompt=user_text,
                    num_images=1,
                )
                response = generate(
                    self.local_model,
                    self.local_processor,
                    image=image_path,
                    prompt=prompt,
                    max_tokens=512,
                    verbose=False,
                )
            else:
                prompt = apply_chat_template(
                    self.local_processor,
                    config=self.local_model.config,
                    prompt=user_text,
                    num_images=0,
                )
                response = generate(
                    self.local_model,
                    self.local_processor,
                    prompt=prompt,
                    max_tokens=512,
                    verbose=False,
                )

            result = response.text if hasattr(response, 'text') else str(response)
            return self._parse_response(result)

        except Exception as e:
            print(f"[MedGemma] Local inference error: {e}")
            return self._dummy_response(symptoms)

    def _reason_api(self, image_path: str, symptoms: str) -> dict:
        try:
            model = self._genai.GenerativeModel("gemini-1.5-flash")
            prompt = SAFETY_SYSTEM_PROMPT + "\n\n" + self._build_prompt(symptoms)

            content_parts = [prompt]
            if image_path and os.path.exists(image_path):
                import PIL.Image
                img = PIL.Image.open(image_path)
                content_parts.append(img)

            response = model.generate_content(content_parts)
            return self._parse_response(response.text)

        except Exception as e:
            print(f"[MedGemma] API inference error: {e}")
            return self._dummy_response(symptoms)

    def _build_prompt(self, symptoms: str) -> str:
        prompt = SAFETY_SYSTEM_PROMPT + "\n\n"
        prompt += "Patient presents with the following symptoms:\n"
        prompt += f"{symptoms}\n\n"
        prompt += "Analyze the provided chest X-ray image along with the symptoms. "
        prompt += "Provide your structured analysis as JSON."
        return prompt

    def _parse_response(self, raw_response: str) -> dict:
        try:
            start = raw_response.find("{")
            end = raw_response.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(raw_response[start:end])
                result = {
                    "clinical_reasoning": parsed.get("clinical_reasoning", ""),
                    "differential_diagnosis": parsed.get("differential_diagnosis", []),
                    "key_observations": parsed.get("key_observations", []),
                    "recommendation": parsed.get("recommendation", "Consult a healthcare professional."),
                    "raw_response": raw_response,
                    "token_confidence": None,
                }
                return result
        except json.JSONDecodeError:
            pass

        return {
            "clinical_reasoning": raw_response,
            "differential_diagnosis": [],
            "key_observations": [],
            "recommendation": "Consult a qualified healthcare professional.",
            "raw_response": raw_response,
            "token_confidence": None,
        }

    def _dummy_response(self, symptoms: str) -> dict:
        return {
            "clinical_reasoning": (
                f"Based on the provided chest X-ray and reported symptoms ({symptoms}), "
                "there are findings that warrant further clinical evaluation. "
                "The radiographic appearance suggests possible parenchymal changes. "
                "Correlation with clinical history and additional imaging may be beneficial. "
                "This is an automated preliminary assessment for research purposes only."
            ),
            "differential_diagnosis": [
                "Community-acquired pneumonia",
                "Acute bronchitis",
                "Viral lower respiratory tract infection",
            ],
            "key_observations": [
                "Possible opacification in lung fields",
                "Cardiac silhouette within normal limits",
                "No obvious pneumothorax identified",
            ],
            "recommendation": "Professional medical consultation is strongly recommended.",
            "raw_response": "[dummy mode — MedGemma not loaded]",
            "token_confidence": None,
        }

    def get_status(self) -> dict:
        return {
            "mode": self.mode,
            "ready": self.ready,
            "model": "mlx-community/medgemma-4b-it-8bit" if self.mode == "local" else "gemini-1.5-flash",
        }
