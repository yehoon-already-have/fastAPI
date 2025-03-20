from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from model.network import DrawAingCNN  # 모델 클래스 임포트. 필요하다니 불러옴...
from fastapi.middleware.cors import CORSMiddleware
import base64  # 추가

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (필요에 따라 수정 가능)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GPU가 있다면 GPU 사용, 없다면 CPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드 (사전에 학습된 CNN 모델)
model = DrawAingCNN(num_classes=30)  # 클래스 수를 맞춰서 초기화
model.load_state_dict(torch.load("draw_classify_model1.pth", map_location=device))
model.to(device)  # 모델을 GPU 또는 CPU로 이동
model.eval()

# 클래스 라벨 리스트 (실제 학습 데이터에 맞게 수정)
class_labels = ['ant', 'apple', 'axe', 'backpack', 'banana', 
                'barn', 'basket', 'bear', 'bed', 'bee', 
                'bench', 'bread', 'bridge', 'broccoli', 'broom', 
                'bucket', 'bush', 'butterfly', 'carrot', 'cat', 
                'chair', 'cloud', 'cow', 'cup', 'dog', 
                'donut', 'door', 'duck', 'feather', 'fence']

# 이미지 변환 함수
def transform_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert('RGB')  # RGBA나 RGB 이미지를 먼저 불러온 후
    print(f"Original Image Size: {image.size}")  # 이미지 크기 확인

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),  # 흑백 이미지로 변환
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 학습 시 사용된 정규화 값 맞추기
    ])

    image_tensor = transform(image).unsqueeze(0)
    print(f"Transformed Image Tensor Shape: {image_tensor.shape}")  # 변환된 텐서 크기 확인

    return image_tensor

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_tensor = transform_image(image_bytes).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # 확률 변환
        top_probs, top_indices = torch.topk(probabilities, 10, dim=1)  # 상위 10개 예측

    top_predictions = [
        {"class": class_labels[idx.item()], "probability": round(prob.item(), 4)}  # 확률값을 포함
        for idx, prob in zip(top_indices[0], top_probs[0])
    ]

    # 이미지를 Base64로 변환해 클라이언트에서 표시할 수 있도록 함
    image_base64 = base64.b64encode(image_bytes).decode()

    return {
        "predictions": top_predictions,
        "image": f"data:image/png;base64,{image_base64}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)