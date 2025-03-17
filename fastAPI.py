from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO

app = FastAPI()

# 모델 로드 (사전에 학습된 CNN 모델)
model = torch.load("draw_classify_model.pth", map_location=torch.device('cpu'))
model.eval()

# 이미지 변환 함수
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_tensor = transform_image(image_bytes)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return {"prediction": predicted.item()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)