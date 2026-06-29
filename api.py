# ─── SILHOUETTE API ───────────────────────────────────────────────────────────

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import asyncio
from avatar_engine  import AvatarPipeline
from garment_engine import GarmentPipeline

app = FastAPI(title="Silhouette API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

avatar_pipeline  = AvatarPipeline()
garment_pipeline = GarmentPipeline()

# Connected Room sessions
room_sessions: dict = {}


@app.post("/avatar/generate")
async def generate_avatar(file: UploadFile = File(...)):
    """
    POST a photo → returns 3D mesh JSON for Three.js.
    """
    if file.content_type not in ["image/jpeg","image/png","image/webp"]:
        raise HTTPException(400, "Image files only.")
    try:
        image_bytes = await file.read()
        result      = avatar_pipeline.run(image_bytes)
        return JSONResponse(content={"status": "ok", "avatar": result})
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Avatar generation failed: {str(e)}")


@app.post("/garment/process")
async def process_garment(
    file:     UploadFile = File(...),
    category: str        = None
):
    """
    POST a clothing photo → returns processed garment asset.
    """
    try:
        image_bytes = await file.read()
        result      = garment_pipeline.process(image_bytes, category)
        return JSONResponse(content={
            "status":          "ok",
            "category":        result.category.value,
            "dominant_colour": list(result.dominant_colour),
            "texture":         result.texture_b64,
            "thumbnail":       result.thumbnail_b64,
            "mask":            result.mask_b64,
            "uv_region":       result.uv_region,
            "metadata":        result.metadata,
        })
    except Exception as e:
        raise HTTPException(500, f"Garment processing failed: {str(e)}")


@app.websocket("/room/live/{session_id}")
async def room_websocket(ws: WebSocket, session_id: str):
    """
    Real-time wardrobe state sync for the Room screen.
    Client sends: { action: "wear"|"remove", garment_id: str }
    Server broadcasts: { worn_items: [...], model_state: {...} }
    """
    await ws.accept()
    room_sessions[session_id] = {"worn_items": {}, "ws": ws}

    try:
        while True:
            data    = await ws.receive_text()
            payload = json.loads(data)
            action  = payload.get("action")
            session = room_sessions[session_id]

            if action == "wear":
                category = payload["category"]
                garment  = payload["garment"]
                # One garment per category
                session["worn_items"][category] = garment
            elif action == "remove":
                category = payload.get("category")
                session["worn_items"].pop(category, None)
            elif action == "clear":
                session["worn_items"] = {}

            # Broadcast updated state
            await ws.send_json({
                "worn_items":  session["worn_items"],
                "piece_count": len(session["worn_items"])
            })
    except Exception:
        del room_sessions[session_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
@app.get("/health")
def health():
    return {"status": "ok"}
