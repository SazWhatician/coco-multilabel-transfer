# app.py
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from agent_graph import agent

app = FastAPI(
    title       = "Enhanced Vision API",
    description = "VOC Multi-Label CNN + LangGraph + Groq Multimodal",
    version     = "1.0.0",
)


@app.get("/")
def root():
    return {"status": "running", "endpoint": "POST /enhanced-vision"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/enhanced-vision")
async def enhanced_vision(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    try:
        result = agent.invoke({
            "image"                  : img,
            "cnn_predictions"        : {},
            "multimodal_llm_response": "",
            "final_description"      : "",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    return JSONResponse(content={
        "cnn_predictions"        : result["cnn_predictions"],
        "multimodal_enhancement" : result["multimodal_llm_response"],
        "final_enhanced_response": result["final_description"],
    })