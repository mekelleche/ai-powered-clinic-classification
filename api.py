from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import base64
import os
import re
from io import BytesIO


class ModelService:
    def __init__(self, model_path: Path, supports_ocr: bool = False):
        self.model_path = model_path
        self.supports_ocr = supports_ocr
        self.bundle = None
        self._easyocr_reader = None
        self._ocr_backend = os.getenv("OCR_BACKEND", "auto").strip().lower()

    def get_ocr_reader(self):
        if not self.supports_ocr:
            raise RuntimeError("OCR is only available for the anemia model.")
        if self._easyocr_reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError(
                    "OCR dependencies are not installed. Install requirements-ocr.txt to enable image extraction."
                ) from exc
            self._easyocr_reader = easyocr.Reader(["en"])
        return self._easyocr_reader

    def _ocr_text_easyocr(self, img_np):
        reader = self.get_ocr_reader()
        results = reader.readtext(img_np)
        return " ".join(r[1] for r in results).upper()

    def _ocr_text_tesseract(self, pil_img):
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError(
                "pytesseract is not installed. Install requirements-ocr.txt to use tesseract OCR."
            ) from exc
        text = pytesseract.image_to_string(pil_img, lang="eng")
        return text.upper()

    def _extract_text(self, pil_img, img_np):
        backend = self._ocr_backend

        if backend == "easyocr":
            return self._ocr_text_easyocr(img_np)
        if backend in {"tesseract", "pytesseract"}:
            return self._ocr_text_tesseract(pil_img)
        if backend != "auto":
            raise RuntimeError(
                f"Unknown OCR_BACKEND='{self._ocr_backend}'. Use easyocr, tesseract, or auto."
            )

        try:
            return self._ocr_text_tesseract(pil_img)
        except Exception:
            return self._ocr_text_easyocr(img_np)

    def extract_from_image(self, b64: str):
        try:
            from PIL import Image
            
            img_data = base64.b64decode(b64)
            img = Image.open(BytesIO(img_data)).convert('RGB')
            img_np = np.array(img)
            
            joined_text = self._extract_text(img, img_np)
            
            extracted = {}
            if self.bundle is None:
                self.load()
                
            features = self.bundle["features"]
            aliases = {
                # Anemia / CBC aliases
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
                "B12": ["B12", "VITAMIN B12", "VIT B12"],
                # Diabetes feature aliases
                "HighBP": ["HIGHBP", "HIGH BP", "HIGH BLOOD PRESSURE", "HYPERTENSION", "BP"],
                "HighChol": ["HIGHCHOL", "HIGH CHOL", "HIGH CHOLESTEROL", "HYPERCHOLESTEROLEMIA", "CHOLESTEROL"],
                "CholCheck": ["CHOLCHECK", "CHOL CHECK", "CHOLESTEROL CHECK", "LIPID CHECK"],
                "BMI": ["BMI", "BODY MASS INDEX"],
                "Smoker": ["SMOKER", "SMOKING", "TOBACCO", "CIGARETTE"],
                "Stroke": ["STROKE", "CVA"],
                "HeartDiseaseorAttack": ["HEARTDISEASEORATTACK", "HEART DISEASE OR ATTACK", "HEART DISEASE", "HEART ATTACK", "MI", "CAD"],
                "PhysActivity": ["PHYSACTIVITY", "PHYSICAL ACTIVITY", "EXERCISE", "ACTIVE"],
                "Fruits": ["FRUITS", "FRUIT INTAKE"],
                "Veggies": ["VEGGIES", "VEGETABLES", "VEGETABLE INTAKE"],
                "HvyAlcoholConsump": ["HVYALCOHOLCONSUMP", "HEAVY ALCOHOL", "ALCOHOL CONSUMPTION", "ALCOHOL"],
                "AnyHealthcare": ["ANYHEALTHCARE", "ANY HEALTHCARE", "HEALTH COVERAGE", "HEALTH INSURANCE"],
                "NoDocbcCost": ["NODOCBCCOST", "NO DOC BC COST", "COULD NOT SEE DOCTOR", "NO DOCTOR COST"],
                "GenHlth": ["GENHLTH", "GENERAL HEALTH", "OVERALL HEALTH"],
                "MentHlth": ["MENTHLTH", "MENTAL HEALTH", "MENTAL UNHEALTHY DAYS"],
                "PhysHlth": ["PHYSHLTH", "PHYSICAL HEALTH", "PHYSICAL UNHEALTHY DAYS"],
                "DiffWalk": ["DIFFWALK", "DIFFICULTY WALKING", "WALKING DIFFICULTY", "MOBILITY LIMITATION"],
                "Sex": ["SEX", "GENDER"],
                "Age": ["AGE"],
                "Education": ["EDUCATION", "EDUC LEVEL", "SCHOOLING"],
                "Income": ["INCOME", "HOUSEHOLD INCOME"],
                # Diabetes-v3 (xgboost) feature aliases
                "gender": ["GENDER", "SEX"],
                "age": ["AGE"],
                "hypertension": ["HYPERTENSION", "HIGH BP", "HIGH BLOOD PRESSURE", "BP"],
                "heart_disease": ["HEART DISEASE", "HEART_DISEASE", "HEART ATTACK", "CAD", "MI"],
                "smoking_history": ["SMOKING HISTORY", "SMOKING", "SMOKER"],
                "bmi": ["BMI", "BODY MASS INDEX"],
                "HbA1c_level": ["HBA1C", "HBA1C LEVEL", "A1C", "HBA1C_LEVEL"],
                "blood_glucose_level": ["BLOOD GLUCOSE", "GLUCOSE", "GLUCOSE LEVEL", "BLOOD SUGAR"],
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
                    row[k] = self._coerce_feature_value(k, v)
            rows.append(row)
        return pd.DataFrame(rows, columns=features)

    @staticmethod
    def _coerce_feature_value(feature, value):
        if value is None:
            return np.nan

        text = str(value).strip()
        if text == "":
            return np.nan

        if feature == "smoking_history":
            smoking_map = {
                "no info": 0,
                "current": 1,
                "ever": 2,
                "former": 3,
                "never": 4,
                "not current": 5,
            }
            key = text.lower()
            if key in smoking_map:
                return float(smoking_map[key])

        if feature == "gender":
            gender_map = {
                "female": 0,
                "male": 1,
            }
            key = text.lower()
            if key in gender_map:
                return float(gender_map[key])

        try:
            return float(text.replace(",", "."))
        except Exception:
            return np.nan

    def predict_batch(self, samples):
        if self.bundle is None:
            self.load()

        df = self._df_from_samples(samples)
        pipeline = self.bundle.get("pipeline")

        if pipeline is not None:
            proba = pipeline.predict_proba(df)
            class_values = [int(c) for c in pipeline.classes_]
        else:
            model = self.bundle.get("model")
            imputer = self.bundle.get("imputer")
            scaler = self.bundle.get("scaler")
            if model is None or imputer is None or scaler is None:
                raise RuntimeError(
                    "Model bundle is missing 'pipeline' or ('model', 'imputer', 'scaler')."
                )
            try:
                x_imp = imputer.transform(df)
            except Exception:
                # Fallback for cross-version sklearn pickle incompatibilities.
                x_imp = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                if np.isnan(x_imp).any():
                    col_medians = np.nanmedian(x_imp, axis=0)
                    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
                    nan_mask = np.isnan(x_imp)
                    x_imp[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

            try:
                x_scaled = scaler.transform(x_imp)
            except Exception:
                x_scaled = x_imp

            proba = model.predict_proba(x_scaled)
            class_values = [int(c) for c in model.classes_]

        # Prefer labels saved during training
        saved = {int(c["value"]): str(c["name"]) for c in self.bundle.get("classes", [])}
        value_to_name = {v: saved.get(v, f"Class_{v}") for v in class_values}

        ANEMIA_THRESHOLD = 0.12  # used only for anemia-style multiclass models
        use_anemia_bias = len(class_values) > 2 and any("ANEMIA" in n.upper() for n in value_to_name.values())

        results = []
        for i in range(proba.shape[0]):
            p = proba[i]

            if use_anemia_bias:
                no_anemia_idx = next((j for j, v in enumerate(class_values) if v == 0), None)
                anemia_candidates = [
                    (j, p[j]) for j, v in enumerate(class_values)
                    if v != 0 and p[j] >= ANEMIA_THRESHOLD
                ]
                if anemia_candidates and no_anemia_idx is not None:
                    best_i = max(anemia_candidates, key=lambda x: x[1])[0]
                else:
                    best_i = int(np.argmax(p))
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


class LeukemiaImageService:
    IMG_SIZE = 224
    VAL_SIZE = 96
    VALID_THR = 0.55
    UNC_LOW = 0.30
    UNC_HIGH = 0.60
    ENSEMBLE_THR = 0.40

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.bundle = None

    def get_info(self):
        return {
            "features": [],
            "classes": [
                {"value": 0, "name": "Normal"},
                {"value": 1, "name": "Leukemia"},
            ],
            "title": "Leukemia Image Classifier",
            "description": "Detects suspected leukemia from microscopic blood smear images.",
            "supportsImageOcr": False,
            "supportsImagePrediction": True,
            "negativeClassValue": 0,
        }

    def load(self):
        if self.bundle is not None:
            return

        required = ["best_validator.pth", "best_b5.pth", "best_b4.pth"]
        missing = [name for name in required if not (self.model_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Leukemia model files missing in {self.model_dir}: {', '.join(missing)}"
            )

        try:
            import torch
            import torch.nn as nn
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "Leukemia image prediction needs torch and timm. Install requirements.txt first."
            ) from exc

        class ImageValidator(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = timm.create_model(
                    "mobilenetv3_small_100",
                    pretrained=False,
                    num_classes=0,
                    global_pool="avg",
                )
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, LeukemiaImageService.VAL_SIZE, LeukemiaImageService.VAL_SIZE)
                    feat_dim = self.backbone(dummy).shape[1]
                self.head = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(feat_dim, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(128, 1),
                )

            def forward(self, x):
                return self.head(self.backbone(x))

        class LeukemiaClassifier(nn.Module):
            def __init__(self, backbone_name: str = "efficientnet_b5"):
                super().__init__()
                self.backbone = timm.create_model(
                    backbone_name,
                    pretrained=False,
                    num_classes=0,
                    global_pool="avg",
                    drop_rate=0.4,
                )
                f = self.backbone.num_features
                self.attention = nn.Sequential(
                    nn.Linear(f, f // 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(f // 4, f),
                    nn.Sigmoid(),
                )
                self.head = nn.Sequential(
                    nn.Linear(f, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.45),
                    nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.35),
                    nn.Linear(256, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(0.25),
                    nn.Linear(64, 1),
                )

            def forward(self, x):
                feat = self.backbone(x)
                feat = feat * self.attention(feat)
                return self.head(feat)

        class LeukemiaClassifierB4(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = timm.create_model(
                    "efficientnet_b4",
                    pretrained=False,
                    num_classes=0,
                    global_pool="avg",
                    drop_rate=0.35,
                )
                f = self.backbone.num_features
                self.head = nn.Sequential(
                    nn.Linear(f, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.4),
                    nn.Linear(512, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.3),
                    nn.Linear(128, 1),
                )

            def forward(self, x):
                return self.head(self.backbone(x))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        validator = ImageValidator().to(device)
        model_b5 = LeukemiaClassifier().to(device)
        model_b4 = LeukemiaClassifierB4().to(device)

        validator.load_state_dict(self._torch_load(torch, self.model_dir / "best_validator.pth", device))
        model_b5.load_state_dict(self._torch_load(torch, self.model_dir / "best_b5.pth", device))
        model_b4.load_state_dict(self._torch_load(torch, self.model_dir / "best_b4.pth", device))
        validator.eval()
        model_b5.eval()
        model_b4.eval()

        self.bundle = {
            "torch": torch,
            "device": device,
            "validator": validator,
            "model_b5": model_b5,
            "model_b4": model_b4,
        }

    @staticmethod
    def _torch_load(torch, path: Path, device):
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=device)

    def _image_from_b64(self, b64: str):
        from PIL import Image

        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_data = base64.b64decode(b64)
        return Image.open(BytesIO(img_data)).convert("RGB")

    def _quality(self, img_np: np.ndarray):
        gray = np.dot(img_np[..., :3], [0.299, 0.587, 0.114])
        brightness = float(gray.mean())
        gy, gx = np.gradient(gray.astype(np.float32))
        sharpness = float((gx * gx + gy * gy).var())
        issues = []
        if sharpness < 80.0:
            issues.append(f"Blurry (sharp={sharpness:.0f})")
        if brightness < 20:
            issues.append(f"Dark (brt={brightness:.0f})")
        if brightness > 240:
            issues.append(f"Overexposed (brt={brightness:.0f})")
        return {
            "sharpness": sharpness,
            "brightness": brightness,
            "is_good": not issues,
            "issues": issues,
        }

    def _preprocess_blood(self, img_np: np.ndarray):
        try:
            import cv2

            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
            img_np = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
            return cv2.GaussianBlur(img_np, (3, 3), 0)
        except Exception:
            from PIL import Image, ImageFilter, ImageOps

            img = Image.fromarray(img_np)
            img = ImageOps.autocontrast(img).filter(ImageFilter.GaussianBlur(radius=0.6))
            return np.asarray(img)

    def _tensor_from_image(self, image, size: int):
        torch = self.bundle["torch"]
        image = image.resize((size, size))
        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        return torch.from_numpy(arr.transpose(2, 0, 1)).float()

    def _tta_tensors(self, img_np: np.ndarray):
        from PIL import Image, ImageOps

        img = Image.fromarray(img_np)
        bigger = img.resize((int(self.IMG_SIZE * 1.1), int(self.IMG_SIZE * 1.1)))
        left = (bigger.width - self.IMG_SIZE) // 2
        top = (bigger.height - self.IMG_SIZE) // 2
        center_crop = bigger.crop((left, top, left + self.IMG_SIZE, top + self.IMG_SIZE))
        variants = [
            img,
            ImageOps.mirror(img),
            ImageOps.flip(img),
            img.rotate(90, expand=True),
            center_crop,
        ]
        return [self._tensor_from_image(v, self.IMG_SIZE) for v in variants]

    def _predict_tta(self, model, img_np: np.ndarray):
        torch = self.bundle["torch"]
        device = self.bundle["device"]
        probs = []
        with torch.no_grad():
            for tensor in self._tta_tensors(img_np):
                logits = model(tensor.unsqueeze(0).to(device))
                probs.append(torch.sigmoid(logits).item())
        return float(np.mean(probs))

    def _last_conv_layer(self, model):
        import torch.nn as nn

        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Conv2d):
                return module
        raise RuntimeError("No convolutional layer found for Grad-CAM.")

    def _gradcam_overlay(self, model, image, size: int, title: str, note: str):
        torch = self.bundle["torch"]
        device = self.bundle["device"]
        layer = self._last_conv_layer(model)
        activations = {}
        gradients = {}

        def save_activation(_module, _inputs, output):
            activations["value"] = output
            output.register_hook(lambda grad: gradients.__setitem__("value", grad))

        forward_handle = layer.register_forward_hook(save_activation)
        try:
            tensor = self._tensor_from_image(image, size).unsqueeze(0).to(device)
            tensor.requires_grad_(True)
            model.zero_grad(set_to_none=True)
            logits = model(tensor)
            score = logits.reshape(-1)[0]
            prob = float(torch.sigmoid(score.detach()).item())
            score.backward()

            acts = activations["value"].detach()
            grads = gradients["value"].detach()
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)
            cam = torch.nn.functional.interpolate(
                cam,
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            )
            cam_np = cam.squeeze().cpu().numpy()
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
            return {
                "model": title,
                "note": note,
                "probability": prob,
                "image": self._overlay_b64(image, cam_np, size),
            }
        finally:
            forward_handle.remove()
            model.zero_grad(set_to_none=True)

    def _overlay_b64(self, image, cam: np.ndarray, size: int):
        from PIL import Image

        base = np.asarray(image.resize((size, size))).astype(np.float32)
        heat = np.zeros_like(base)
        heat[..., 0] = 255.0 * cam
        heat[..., 1] = 190.0 * np.sqrt(cam)
        heat[..., 2] = 45.0 * (1.0 - cam)
        overlay = np.clip(0.58 * base + 0.42 * heat, 0, 255).astype(np.uint8)
        out = Image.fromarray(overlay)
        buf = BytesIO()
        out.save(buf, format="JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _build_explanations(self, image, preprocessed, validator, model_b5, model_b4, include_classifiers=True):
        from PIL import Image

        explanations = []
        try:
            explanations.append(
                self._gradcam_overlay(
                    validator,
                    image,
                    self.VAL_SIZE,
                    "Validator",
                    "Blood smear validity focus",
                )
            )
        except Exception as exc:
            explanations.append({"model": "Validator", "error": str(exc)})

        if include_classifiers:
            preprocessed_image = Image.fromarray(preprocessed)
            for model, title in [(model_b5, "EfficientNet-B5"), (model_b4, "EfficientNet-B4")]:
                try:
                    explanations.append(
                        self._gradcam_overlay(
                            model,
                            preprocessed_image,
                            self.IMG_SIZE,
                            title,
                            "Leukemia classification focus",
                        )
                    )
                except Exception as exc:
                    explanations.append({"model": title, "error": str(exc)})
        return explanations

    def predict_image(self, b64: str):
        self.load()
        torch = self.bundle["torch"]
        device = self.bundle["device"]
        validator = self.bundle["validator"]
        model_b5 = self.bundle["model_b5"]
        model_b4 = self.bundle["model_b4"]

        image = self._image_from_b64(b64)
        img_np = np.asarray(image)

        val_tensor = self._tensor_from_image(image, self.VAL_SIZE).unsqueeze(0).to(device)
        with torch.no_grad():
            valid_prob = float(torch.sigmoid(validator(val_tensor)).item())

        if valid_prob < self.VALID_THR:
            explanations = self._build_explanations(
                image=image,
                preprocessed=img_np,
                validator=validator,
                model_b5=model_b5,
                model_b4=model_b4,
                include_classifiers=False,
            )
            return self._format_result(
                value=-1,
                name="Invalid blood smear image",
                probability=1.0 - valid_prob,
                has_condition=True,
                probabilities=[],
                message="The image does not look like a valid microscopic blood smear.",
                details={"bloodSmearProbability": valid_prob},
                explanations=explanations,
            )

        quality = self._quality(img_np)
        preprocessed = self._preprocess_blood(img_np)
        prob_b5 = self._predict_tta(model_b5, preprocessed)
        prob_b4 = self._predict_tta(model_b4, preprocessed)
        probability = 0.60 * prob_b5 + 0.40 * prob_b4
        explanations = self._build_explanations(image, preprocessed, validator, model_b5, model_b4)

        if self.UNC_LOW < probability < self.UNC_HIGH:
            value, name, has_condition = 2, "Uncertain", True
        elif probability >= self.ENSEMBLE_THR:
            value, name, has_condition = 1, "Leukemia suspected", True
        else:
            value, name, has_condition = 0, "Normal", False

        return self._format_result(
            value=value,
            name=name,
            probability=probability,
            has_condition=has_condition,
            probabilities=[
                {"value": 0, "name": "Normal", "p": float(1.0 - probability)},
                {"value": 1, "name": "Leukemia", "p": float(probability)},
            ],
            message=(
                "Specialist review is recommended."
                if has_condition
                else "No signs of leukemia detected by the model."
            ),
            details={
                "bloodSmearProbability": valid_prob,
                "probB5": prob_b5,
                "probB4": prob_b4,
                "quality": quality,
            },
            explanations=explanations,
        )

    @staticmethod
    def _format_result(value, name, probability, has_condition, probabilities, message, details, explanations=None):
        return {
            "anemia": False,
            "hasCondition": has_condition,
            "predictedClass": {"value": value, "name": name},
            "probability": float(probability),
            "probabilities": probabilities,
            "message": message,
            "details": details,
            "explanations": explanations or [],
        }


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
    services = {}
    for key, cfg in model_paths.items():
        if cfg.get("type") == "combined":
            continue
        if cfg.get("type") == "leukemia_image":
            services[key] = LeukemiaImageService(model_dir=cfg["path"])
        else:
            services[key] = ModelService(
                model_path=cfg["path"],
                supports_ocr=cfg.get("supports_ocr", False),
            )

    class Api:
        def _combined_cfg(self, model_key: str):
            cfg = model_paths.get(model_key)
            if cfg and cfg.get("type") == "combined":
                return cfg
            return None

        @staticmethod
        def _has_value(v):
            if v is None:
                return False
            if isinstance(v, str):
                return v.strip() != ""
            return True

        def _combined_parts(self, model_key: str):
            cfg = self._combined_cfg(model_key)
            if not cfg:
                raise KeyError(f"Unknown combined model: {model_key}")
            keys = [k for k in cfg.get("models", []) if k in services]
            if not keys:
                raise RuntimeError(f"Combined model '{model_key}' has no valid sub-models.")
            return keys

        def _combined_features(self, model_keys):
            infos = {k: services[k].get_info() for k in model_keys}
            feature_sets = {k: list(infos[k].get("features", [])) for k in model_keys}
            shared_aliases = {
                "age": ["age", "Age"],
                "gender": ["gender", "GENDER", "Sex"],
            }
            shared = set()
            for canonical, aliases in shared_aliases.items():
                if any(any(a in feature_sets[k] for a in aliases) for k in model_keys):
                    shared.add(canonical)

            model_shared_features = {}
            for mk in model_keys:
                model_shared_features[mk] = set()
                for canonical in shared:
                    for alias in shared_aliases[canonical]:
                        if alias in feature_sets[mk]:
                            model_shared_features[mk].add(alias)

            feature_to_shared = {}
            for mk in model_keys:
                feature_to_shared[mk] = {}
                for f in model_shared_features[mk]:
                    for canonical in shared:
                        if f in shared_aliases[canonical]:
                            feature_to_shared[mk][f] = canonical
                            break
            ordered = []
            seen = set()
            for k in model_keys:
                for f in feature_sets[k]:
                    if f not in seen:
                        seen.add(f)
                        ordered.append(f)
            return infos, feature_sets, shared, ordered, model_shared_features, feature_to_shared, shared_aliases

        def _predict_combined_one(self, model_key: str, sample):
            model_keys = self._combined_parts(model_key)
            infos, feature_sets, shared, _, model_shared_features, feature_to_shared, shared_aliases = self._combined_features(model_keys)
            sample = sample or {}
            combined = {
                "combined": True,
                "sharedFeatures": sorted(shared),
                "results": {},
            }

            for mk in model_keys:
                features = feature_sets[mk]
                specific = [f for f in features if f not in model_shared_features[mk]]
                provided_features = [f for f in features if self._has_value(sample.get(f))]
                specific_present = [f for f in provided_features if f in specific]

                if len(specific_present) == 0:
                    combined["results"][mk] = {
                        "status": "skipped",
                        "reason": "no_model_specific_features_provided",
                        "missingFeatures": [],
                        "result": None,
                        "title": infos[mk].get("title", mk),
                    }
                    continue

                missing = []
                optional = set(model_paths.get(mk, {}).get("optional_features", []))
                for f in features:
                    if f in optional:
                        continue
                    if self._has_value(sample.get(f)):
                        continue
                    canonical = feature_to_shared[mk].get(f)
                    if canonical:
                        if any(self._has_value(sample.get(alias)) for alias in shared_aliases[canonical]):
                            continue
                    missing.append(f)
                if missing:
                    combined["results"][mk] = {
                        "status": "incomplete",
                        "reason": "missing_features",
                        "missingFeatures": missing,
                        "result": None,
                        "title": infos[mk].get("title", mk),
                    }
                    continue

                combined["results"][mk] = {
                    "status": "ok",
                    "reason": "",
                    "missingFeatures": [],
                    "result": services[mk].predict_one(sample),
                    "title": infos[mk].get("title", mk),
                }

            return combined

        def _service(self, model_key: str):
            if model_key not in services:
                raise KeyError(f"Unknown model: {model_key}")
            return services[model_key]

        def list_models(self):
            models = []
            for key, cfg in model_paths.items():
                if cfg.get("internal"):
                    continue
                if cfg.get("type") == "combined":
                    model_keys = self._combined_parts(key)
                    _infos, _feature_sets, _shared, ordered, _a, _b, _c = self._combined_features(model_keys)
                    info = {
                        "title": cfg.get("title", key),
                        "description": cfg.get("description", ""),
                        "supportsImagePrediction": False,
                    }
                    features = ordered
                else:
                    service = self._service(key)
                    info = service.get_info()
                    features = info.get("features", [])
                models.append(
                    {
                        "key": key,
                        "title": cfg.get("title", info.get("title", key)),
                        "description": cfg.get("description", info.get("description", "")),
                        "supportsImageOcr": cfg.get("supports_ocr", False),
                        "supportsImagePrediction": info.get("supportsImagePrediction", False),
                        "features": features,
                    }
                )
            return models

        def get_model_info(self, model_key: str):
            combined_cfg = self._combined_cfg(model_key)
            if combined_cfg:
                model_keys = self._combined_parts(model_key)
                infos, _feature_sets, shared, ordered, _a, _b, _c = self._combined_features(model_keys)
                classes = []
                for mk in model_keys:
                    classes.extend(infos[mk].get("classes", []))
                return {
                    "features": ordered,
                    "classes": classes,
                    "title": combined_cfg.get("title", model_key),
                    "description": combined_cfg.get("description", ""),
                    "supportsImageOcr": combined_cfg.get("supports_ocr", False),
                    "supportsImagePrediction": False,
                    "combined": True,
                    "subModels": model_keys,
                    "sharedFeatures": sorted(shared),
                }
            return self._service(model_key).get_info()

        def predict(self, model_key: str, sample):
            if self._combined_cfg(model_key):
                return self._predict_combined_one(model_key, sample)
            return self._service(model_key).predict_one(sample)

        def predict_batch(self, model_key: str, samples):
            if self._combined_cfg(model_key):
                return [self._predict_combined_one(model_key, s) for s in (samples or [])]
            return self._service(model_key).predict_batch(samples)

        def extract_from_image(self, model_key: str, b64):
            combined_cfg = self._combined_cfg(model_key)
            if combined_cfg:
                merged = {}
                for mk in self._combined_parts(model_key):
                    try:
                        data = services[mk].extract_from_image(b64)
                        if isinstance(data, dict):
                            merged.update(data)
                    except Exception:
                        continue
                return merged
            return self._service(model_key).extract_from_image(b64)

        def predict_image(self, model_key: str, b64):
            service = self._service(model_key)
            if not hasattr(service, "predict_image"):
                raise RuntimeError("Image prediction is only available for image-based models.")
            return service.predict_image(b64)

    return Api()
