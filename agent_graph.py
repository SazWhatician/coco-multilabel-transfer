# agent_graph.py
import os
import io
import base64
import torch
import numpy as np
from PIL import Image
from typing import TypedDict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from groq import Groq
from langgraph.graph import StateGraph, END

from model import MultiLabelVOCNet, VOC_CLASSES, NUM_CLASSES, IMG_SIZE, load_model


WEIGHTS_PATH = "voc_resnet50_best.pth"
THRESHOLD    = 0.25
TOP_K        = 5
DEVICE       = "cpu"


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client  = Groq(api_key=GROQ_API_KEY)



class AgentState(TypedDict):
    image                   : Image.Image
    cnn_predictions         : dict[str, float]
    multimodal_llm_response : str
    final_description       : str



val_tfm = A.Compose([
    A.Resize(360, 360),
    A.CenterCrop(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])



_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model(WEIGHTS_PATH, device=DEVICE)
        print("CNN model loaded")
    return _model



def cnn_node(state: AgentState) -> AgentState:
    print("Node 1: CNN inference")
    model  = get_model()
    img    = state["image"]

    img_np = np.array(img.convert("RGB"))
    tensor = val_tfm(image=img_np)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits)[0].cpu().numpy()

    indices     = np.argsort(probs)[::-1]
    predictions = {}
    for i in indices:
        if probs[i] >= THRESHOLD and len(predictions) < TOP_K:
            predictions[VOC_CLASSES[i]] = round(float(probs[i]), 4)

    
    if not predictions:
        top_i = int(np.argmax(probs))
        predictions[VOC_CLASSES[top_i]] = round(float(probs[top_i]), 4)

    print(f"   CNN predictions: {predictions}")
    return {**state, "cnn_predictions": predictions}



def multimodal_llm_node(state: AgentState) -> AgentState:
    print("Node 2: Multimodal LLM (Groq)")

    img       = state["image"]
    cnn_preds = state["cnn_predictions"]

    pred_text = ", ".join([f"{k}({v*100:.0f}%)" for k, v in cnn_preds.items()])

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_b64 = base64.standard_b64encode(buffered.getvalue()).decode("utf-8")

    prompt = (
        f"A CNN model analyzed this image and detected: {pred_text}.\n\n"
        f"Please:\n"
        f"1. Describe what you actually see in the image in 2-3 sentences.\n"
        f"2. Validate whether the CNN predictions are accurate or not.\n"
        f"3. Mention any important objects the CNN may have missed.\n"
        f"Keep your response concise and factual."
    )

    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        max_tokens=512
    )

    llm_response = response.choices[0].message.content
    print(f"   LLM response: {llm_response[:120]}...")
    return {**state, "multimodal_llm_response": llm_response}



def description_node(state: AgentState) -> AgentState:
    print("Node 3: Building final description")

    cnn_preds    = state["cnn_predictions"]
    llm_response = state["multimodal_llm_response"]

    cnn_summary = ", ".join([
        f"{cls}({score*100:.0f}%)"
        for cls, score in cnn_preds.items()
    ])

    final = (
        f"CNN detected: {cnn_summary}. "
        f"Multimodal LLM analysis: {llm_response.strip()}"
    )

    print(f"   Final: {final[:120]}...")
    return {**state, "final_description": final}



def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("cnn_node",            cnn_node)
    graph.add_node("multimodal_llm_node", multimodal_llm_node)
    graph.add_node("description_node",    description_node)

    graph.set_entry_point("cnn_node")
    graph.add_edge("cnn_node",            "multimodal_llm_node")
    graph.add_edge("multimodal_llm_node", "description_node")
    graph.add_edge("description_node",    END)

    return graph.compile()



agent = build_graph()
print("LangGraph agent compiled — 3 nodes ready")



if __name__ == "__main__":
    import sys

    if not GROQ_API_KEY:
        print("❌ Set your key first:")
        print("   $env:GROQ_API_KEY=your_key_here")
        sys.exit(1)

    img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    print(f"\nTesting with: {img_path}\n")

    img    = Image.open(img_path).convert("RGB")
    result = agent.invoke({
        "image"                  : img,
        "cnn_predictions"        : {},
        "multimodal_llm_response": "",
        "final_description"      : "",
    })

  
    print("CNN Predictions     :", result["cnn_predictions"])
    print("LLM Response        :", result["multimodal_llm_response"])
    print("Final Description   :", result["final_description"])