from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from api.websocket_handler import stream_live_data

app = FastAPI(title="Holter Monitor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router)


@app.get("/")
async def root():
    return FileResponse("static/dashboard.html")


@app.websocket("/ws/live/{mode_name}")
async def websocket_live(websocket: WebSocket, mode_name: str):
    await websocket.accept()
    try:
        await stream_live_data(websocket, mode_name)
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
