import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.cm as cm
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ---------------- CONFIG ----------------

MODEL_PATH = r"C:\Users\aksha\OneDrive\Desktop\Age and Gender Detection 2\Data\processed_advanced\best_model_advanced.pth"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- MODEL DEFINITIONS ----------------

class ChannelAttention(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c, c // r, 1),
            nn.ReLU(),
            nn.Conv2d(c // r, c, 1),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return x * self.sig(self.mlp(self.avg(x)) + self.mlp(self.max(x)))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([avg, maxv], 1)))

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

class AgeGenderNet(nn.Module):
    def __init__(self):
        super().__init__()
        w = EfficientNet_B0_Weights.IMAGENET1K_V1
        net = efficientnet_b0(weights=w)
        self.features = net.features
        self.cbam = CBAM(1280)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.4)

        self.gender_head = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        self.age_head = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        feat = self.cbam(x)
        pooled = self.pool(feat)
        flat = torch.flatten(pooled, 1)
        flat = self.drop(flat)

        gender = self.gender_head(flat).squeeze(1)
        age = self.age_head(flat).squeeze(1)
        return gender, age, feat

def load_model():
    model = AgeGenderNet().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

# ---------------- TRANSFORMS ----------------

def get_transform():
    w = EfficientNet_B0_Weights.IMAGENET1K_V1
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=w.transforms().mean, std=w.transforms().std),
    ])

# ---------------- FACE DETECTION + CLAHE (same as training style) ----------------

haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_largest_face(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    # largest face
    areas = [(w*h, (x, y, w, h)) for (x, y, w, h) in faces]
    _, (x, y, w, h) = max(areas, key=lambda t: t[0])
    return x, y, w, h

def crop_face_with_clahe(frame_bgr, box):
    x, y, w, h = box
    h_img, w_img = frame_bgr.shape[:2]

    # expand box by 25%
    expand = 0.25
    cx = x + w / 2
    cy = y + h / 2
    size = max(w, h) * (1.0 + expand)

    x1 = int(cx - size / 2)
    y1 = int(cy - size / 2)
    x2 = int(cx + size / 2)
    y2 = int(cy + size / 2)

    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w_img, x2); y2 = min(h_img, y2)

    face = frame_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None, None

    # CLAHE on Y channel
    ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
    y_chan, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y_chan)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    face_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    face_resized = cv2.resize(face_eq, (IMG_SIZE, IMG_SIZE))
    return face_resized, (x1, y1, x2, y2)

# ---------------- GRAD-CAM++ ----------------

last_grad = None
last_feat = None

def save_gradients_hook(grad):
    global last_grad
    last_grad = grad

def save_features_hook(module, inp, out):
    global last_feat
    last_feat = out

def gradcam_pp(features, gradients):
    gradients = gradients.cpu().numpy()
    features = features.cpu().numpy()

    grads2 = gradients ** 2
    grads3 = gradients ** 3
    eps = 1e-8
    denom = 2 * grads2 + (features * grads3).sum(axis=(1, 2), keepdims=True)
    denom = np.where(denom != 0, denom, eps)
    alpha = grads2 / denom
    weights = (alpha * np.maximum(gradients, 0)).sum(axis=(1, 2))

    cam = (weights.reshape(-1, 1, 1) * features).sum(axis=0)
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    return cam  # (H,W) normalized

def overlay_cam_on_face(face_bgr, cam):
    h, w = face_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = (cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(face_bgr, 0.55, heatmap_bgr, 0.45, 0)
    return blended

# ---------------- IMAGE MODE ----------------

def run_on_image(model, transform, image_path: str):
    global last_grad, last_feat

    if not os.path.exists(image_path):
        print("❌ Image path does not exist:", image_path)
        return

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("❌ Could not read image:", image_path)
        return

    # 🔥 Resize like webcam
    h, w = img_bgr.shape[:2]
    scale_factor = 640 / w if w > 640 else 1.0
    if scale_factor != 1.0:
        img_bgr = cv2.resize(img_bgr, (640, int(h * scale_factor)))

    # 🔥 SAME processing as webcam
    box = detect_largest_face(img_bgr)
    if box is None:
        print("❌ No face detected in image.")
        return

    face_bgr, (x1, y1, x2, y2) = crop_face_with_clahe(img_bgr, box)
    if face_bgr is None:
        print("❌ Face crop failed.")
        return

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)

    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)

    model.zero_grad()
    last_grad = None
    last_feat = None

    gender_logit, age_pred, _ = model(tensor)
    target = age_pred.mean()
    target.backward()

    if last_feat is None or last_grad is None:
        print("❌ GradCAM failed.")
        return

    feat = last_feat.squeeze(0).detach()
    grad = last_grad.squeeze(0).detach()
    cam = gradcam_pp(feat, grad)

    # overlay
    face_cam = overlay_cam_on_face(face_bgr, cam)

    prob = torch.sigmoid(gender_logit).item()
    gender_str = "Female" if prob > 0.5 else "Male"
    age_int = int(age_pred.item())

    label = f"{gender_str}, Age {age_int}"

    disp = face_cam.copy()
    cv2.putText(disp, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    print("\n============== RESULT ==============")
    print("Gender:", gender_str)
    print("Age:", age_int)
    print("====================================\n")

    cv2.imshow("Custom Image Prediction", disp)

    base, ext = os.path.splitext(image_path)
    out_path = base + "_gradcam" + ext
    cv2.imwrite(out_path, disp)
    print("Saved GradCAM++ as:", out_path)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------- WEBCAM MODE ----------------

def webcam_mode(model, transform):
    global last_grad, last_feat

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("🎥 Webcam running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_disp = frame.copy()
        box = detect_largest_face(frame)

        if box is not None:
            face_bgr, (x1, y1, x2, y2) = crop_face_with_clahe(frame, box)
            if face_bgr is not None:
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(face_rgb)
                tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
                tensor.requires_grad_(True)

                model.zero_grad()
                last_grad = None
                last_feat = None

                gender_logit, age_pred, _ = model(tensor)
                target = age_pred.mean()
                target.backward()

                if last_feat is not None and last_grad is not None:
                    feat = last_feat.squeeze(0).detach()
                    grad = last_grad.squeeze(0).detach()
                    cam = gradcam_pp(feat, grad)
                    face_cam = overlay_cam_on_face(face_bgr, cam)
                else:
                    face_cam = face_bgr

                face_resized = cv2.resize(face_cam, (x2 - x1, y2 - y1))
                frame_disp[y1:y2, x1:x2] = face_resized

                prob_female = torch.sigmoid(gender_logit).item()
                gender_str = "Female" if prob_female > 0.5 else "Male"
                age_int = int(age_pred.item())
                label = f"{gender_str}, Age {age_int}"

                cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame_disp, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame_disp, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Live Age & Gender + GradCAM++", frame_disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- MAIN ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to image for prediction")
    parser.add_argument("--webcam", action="store_true", help="Run live webcam demo")
    args = parser.parse_args()

    model = load_model()
    transform = get_transform()

    # hook last conv layer once
    target_layer = model.features[-1][0]
    target_layer.register_forward_hook(save_features_hook)
    target_layer.register_full_backward_hook(
        lambda m, gin, gout: save_gradients_hook(gout[0])
    )

    if args.webcam:
        webcam_mode(model, transform)
    elif args.image:
        run_on_image(model, transform, args.image)
    else:
        print("❌ Please use either --image <path> or --webcam")

if __name__ == "__main__":
    main()
