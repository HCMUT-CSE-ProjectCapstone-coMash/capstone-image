from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter
from pydantic import BaseModel
import clip
import torch
from PIL import Image
import requests
from io import BytesIO

app = FastAPI()
router = APIRouter(prefix="/api")

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

class ImageRequest(BaseModel):
    url: str

@router.get("/")
def read_root():
    return {"message": "Hello, World!"}

@router.post("/vectorize")
def vectorize_image(body: ImageRequest):
    try:
        response = requests.get(body.url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")

        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            vector = model.encode_image(image_tensor)
            vector = vector / vector.norm(dim=-1, keepdim=True)

        return {
            "vector": vector.cpu().numpy().tolist()[0],
            "dimensions": vector.shape[-1]
        }

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)