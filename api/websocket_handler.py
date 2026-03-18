import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from simulator.signal_generator import SignalGenerator
from ai.analyzer import HolterAnalyzer

logger = logging.getLogger(__name__)

# Loaded once when the module is imported — shared across all connections
analyzer = HolterAnalyzer()


async def stream_live_data(websocket: WebSocket, mode_name: str):
    generator = SignalGenerator(mode_name=mode_name)
    try:
        while True:
            reading = generator.get_live_reading()

            if analyzer.ready:
                analysis = analyzer.analyze_live_reading(reading)
            else:
                analysis = {
                    "diagnosis": "UNKNOWN",
                    "confidence": 0.0,
                    "is_alert": False,
                    "alert_message": "Model not loaded",
                }

            message = {**reading, **analysis}
            await websocket.send_text(json.dumps(message))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except ConnectionClosedOK:
        pass
    except ConnectionClosedError:
        pass
    except Exception as e:
        logger.debug("WebSocket stream ended: %s", e)
    finally:
        print("Client disconnected, stream stopped", flush=True)
