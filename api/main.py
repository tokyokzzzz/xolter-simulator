import asyncio

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.routes import router
from api.simulator_state import simulator_state
from api.auth import verify_supervisor_token
from api.firebase_notifier import initialize_firebase

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
    initialize_firebase()
    asyncio.create_task(simulator_state.run_forever())


@app.get("/")
async def root():
    return FileResponse("static/dashboard.html")


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    token = websocket.query_params.get("token")

    if not token:
        await websocket.close(code=4001, reason="No token provided")
        return

    supervisor = await verify_supervisor_token(token)

    if not supervisor:
        await websocket.close(code=4003, reason="Invalid or expired token")
        return

    await websocket.accept()
    simulator_state.clients.add(websocket)
    print(f"Supervisor connected: {supervisor.get('username')}")

    try:
        while True:
            try:
                await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                continue
    except Exception:
        pass
    finally:
        simulator_state.clients.discard(websocket)
        print(f"Supervisor disconnected: {supervisor.get('username')}")


@app.websocket("/ws/admin")
async def websocket_admin(websocket: WebSocket):
    admin_key = websocket.query_params.get("key")

    if admin_key != "holter-admin-2026":
        await websocket.close(code=4003, reason="Invalid admin key")
        return

    await websocket.accept()
    simulator_state.clients.add(websocket)
    print("Admin dashboard connected")

    try:
        while True:
            try:
                await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                continue
    except Exception:
        pass
    finally:
        simulator_state.clients.discard(websocket)
        print("Admin dashboard disconnected")


class ModeRequest(BaseModel):
    mode: str


class FCMTokenRequest(BaseModel):
    token: str
    fcm_token: str


@app.post("/device/register-fcm")
async def register_fcm_token(request: FCMTokenRequest):
    supervisor = await verify_supervisor_token(request.token)
    if not supervisor:
        raise HTTPException(status_code=403, detail="Invalid token")
    simulator_state.fcm_tokens.add(request.fcm_token)
    print(f"FCM token registered for: {supervisor.get('username')}")
    return {"status": "registered"}


@app.delete("/device/unregister-fcm")
async def unregister_fcm_token(fcm_token: str):
    simulator_state.fcm_tokens.discard(fcm_token)
    return {"status": "unregistered"}


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
