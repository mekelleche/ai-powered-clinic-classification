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

    def _build_explanations(self, image, preprocessed, validator, model_b5, model_b4):
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
        if cfg.get("type") == "leukemia_image":
            services[key] = LeukemiaImageService(model_dir=cfg["path"])
        else:
            services[key] = ModelService(
                model_path=cfg["path"],
                supports_ocr=cfg.get("supports_ocr", False),
            )

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
                        "supportsImagePrediction": info.get("supportsImagePrediction", False),
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

        def predict_image(self, model_key: str, b64):
            service = self._service(model_key)
            if not hasattr(service, "predict_image"):
                raise RuntimeError("Image prediction is only available for image-based models.")
            return service.predict_image(b64)

    return Api()
