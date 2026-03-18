import asyncio

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.routes import router
from api.simulator_state import simulator_state

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


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(simulator_state.run_forever())


@app.get("/")
async def root():
    return FileResponse("static/dashboard.html")


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    simulator_state.clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        simulator_state.clients.discard(websocket)
        try:
            await websocket.close()
        except Exception:
            pass


class ModeRequest(BaseModel):
    mode: str


@app.post("/control/mode")
async def set_mode(body: ModeRequest):
    simulator_state.set_mode(body.mode)
    return {"status": "ok", "mode": body.mode}


@app.get("/control/status")
async def get_status():
    return {
        "current_mode": simulator_state.current_mode,
        "latest": simulator_state.latest_reading,
    }
