from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import base64
import re
from io import BytesIO


class ModelService:
    def __init__(self, model_path: Path, supports_ocr: bool = False):
        self.model_path = model_path
        self.supports_ocr = supports_ocr
        self.bundle = None
        self._reader = None

    def get_ocr_reader(self):
        if not self.supports_ocr:
            raise RuntimeError("OCR is only available for the anemia model.")
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError(
                    "OCR dependencies are not installed. Install requirements-ocr.txt to enable image extraction."
                ) from exc
            self._reader = easyocr.Reader(['en'])
        return self._reader

    def extract_from_image(self, b64: str):
        try:
            from PIL import Image
            
            img_data = base64.b64decode(b64)
            img = Image.open(BytesIO(img_data)).convert('RGB')
            img_np = np.array(img)
            
            reader = self.get_ocr_reader()
            results = reader.readtext(img_np)
            
            text_lines = [r[1] for r in results]
            joined_text = " ".join(text_lines).upper()
            
            extracted = {}
            if self.bundle is None:
                self.load()
                
            features = self.bundle["features"]
            aliases = {
                "WBC": ["WBC", "LEUCOCYTES", "WHITE BLOOD CELLS"],
                "RBC": ["RBC", "ERYTHROCYTES", "RED BLOOD CELLS"],
                "HGB": ["HGB", "HB", "HEMOGLOBIN", "HAEMOGLOBIN"],
                "HCT": ["HCT", "HEMATOCRIT"],
                "MCV": ["MCV"],
                "MCH": ["MCH"],
                "MCHC": ["MCHC"],
                "RDW": ["RDW"],
                "PLT": ["PLT", "PLATELETS", "THROMBOCYTES"],
                "MPV": ["MPV"],
                "PCT": ["PCT"],
                "PDW": ["PDW"],
                "NE#": ["NE#", "NEUTROPHILS", "NEU", "NEUT#"],
                "LY#": ["LY#", "LYMPHOCYTES", "LYM", "LYMPH#"],
                "MO#": ["MO#", "MONOCYTES", "MONO", "MONO#"],
                "EO#": ["EO#", "EOSINOPHILS", "EOS", "EOS#"],
                "BA#": ["BA#", "BASOPHILS", "BASO", "BASO#"],
                "FERRITTE": ["FERRITIN", "FERR", "FERRITTE"],
                "FOLATE": ["FOLATE", "ACIDE FOLIQUE"],
                "B12": ["B12", "VITAMIN B12", "VIT B12"]
            }
            
            for f in features:
                if f in aliases:
                    for alias in aliases[f]:
                        pattern = r'\b' + re.escape(alias) + r'\b[^\d]{0,15}?(\d+[\.,]\d+|\d+)'
                        match = re.search(pattern, joined_text)
                        if match:
                            val_str = match.group(1).replace(',', '.')
                            try:
                                extracted[f] = float(val_str)
                                break
                            except ValueError:
                                pass
                                
            return extracted
        except Exception as e:
            print("OCR Error:", e)
            return {}

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. Train first: python train_model.py"
            )
        self.bundle = joblib.load(self.model_path)

    def get_info(self):
        if self.bundle is None:
            self.load()
        return {
            "features": self.bundle["features"],
            "classes": self.bundle["classes"],
            "title": self.bundle.get("title", self.model_path.stem),
            "description": self.bundle.get("description", ""),
            "supportsImageOcr": self.supports_ocr,
            "negativeClassValue": int(self.bundle.get("negativeClassValue", 0)),
        }

    def _df_from_samples(self, samples):
        if self.bundle is None:
            self.load()
        features = self.bundle["features"]
        rows = []
        for s in samples:
            row = {f: np.nan for f in features}
            for k, v in (s or {}).items():
                if k in row:
                    try:
                        row[k] = float(v)
                    except Exception:
                        row[k] = np.nan
            rows.append(row)
        return pd.DataFrame(rows, columns=features)

    def predict_batch(self, samples):
        if self.bundle is None:
            self.load()

        pipeline = self.bundle["pipeline"]
        df = self._df_from_samples(samples)

        proba = pipeline.predict_proba(df)
        class_values = [int(c) for c in pipeline.classes_]

        # Prefer labels saved during training
        saved = {int(c["value"]): str(c["name"]) for c in self.bundle.get("classes", [])}
        value_to_name = {v: saved.get(v, f"Class_{v}") for v in class_values}

        ANEMIA_THRESHOLD = 0.12  # if any anemia class reaches this prob, prefer it over No_Anemia

        results = []
        for i in range(proba.shape[0]):
            p = proba[i]

            # Index of No_Anemia class (value == 0)
            no_anemia_idx = next((j for j, v in enumerate(class_values) if v == 0), None)

            # Check if any anemia class has probability >= threshold
            anemia_candidates = [
                (j, p[j]) for j, v in enumerate(class_values)
                if v != 0 and p[j] >= ANEMIA_THRESHOLD
            ]

            if anemia_candidates and no_anemia_idx is not None:
                # Pick the anemia class with the highest probability
                best_i = max(anemia_candidates, key=lambda x: x[1])[0]
            else:
                best_i = int(np.argmax(p))

            value = class_values[best_i]
            predicted_class = {"value": value, "name": value_to_name[value]}
            negative_value = int(self.bundle.get("negativeClassValue", 0))
            results.append(
                {
                    "anemia": value != 0,
                    "hasCondition": value != negative_value,
                    "predictedClass": predicted_class,
                    "probability": float(p[best_i]),
                    "probabilities": [
                        {
                            "value": class_values[j],
                            "name": value_to_name[class_values[j]],
                            "p": float(p[j]),
                        }
                        for j in range(len(class_values))
                    ],
                }
            )
        return results

    def predict_one(self, sample):
        return self.predict_batch([sample])[0]


def create_api(model_path: Path):
    service = ModelService(model_path=model_path)

    class Api:
        def get_model_info(self):
            return service.get_info()

        def predict(self, sample):
            return service.predict_one(sample)

        def predict_batch(self, samples):
            return service.predict_batch(samples)

        def extract_from_image(self, b64):
            return service.extract_from_image(b64)

    return Api()


def create_multi_model_api(model_paths: dict[str, dict]):
    services = {
        key: ModelService(
            model_path=cfg["path"],
            supports_ocr=cfg.get("supports_ocr", False),
        )
        for key, cfg in model_paths.items()
    }

    class Api:
        def _service(self, model_key: str):
            if model_key not in services:
                raise KeyError(f"Unknown model: {model_key}")
            return services[model_key]

        def list_models(self):
            models = []
            for key, cfg in model_paths.items():
                service = self._service(key)
                info = service.get_info()
                models.append(
                    {
                        "key": key,
                        "title": cfg.get("title", info.get("title", key)),
                        "description": cfg.get("description", info.get("description", "")),
                        "supportsImageOcr": cfg.get("supports_ocr", False),
                    }
                )
            return models

        def get_model_info(self, model_key: str):
            return self._service(model_key).get_info()

        def predict(self, model_key: str, sample):
            return self._service(model_key).predict_one(sample)

        def predict_batch(self, model_key: str, samples):
            return self._service(model_key).predict_batch(samples)

        def extract_from_image(self, model_key: str, b64):
            return self._service(model_key).extract_from_image(b64)

    return Api()
