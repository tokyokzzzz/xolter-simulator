import asyncio
from simulator.signal_generator import SignalGenerator
from ai.analyzer import HolterAnalyzer


class SimulatorState:
    def __init__(self):
        self.current_mode = "NORMAL"
        self.generator = SignalGenerator("NORMAL")
        self.analyzer = HolterAnalyzer()
        self.latest_reading = None
        self.clients = set()
        self.fcm_tokens = set()
        self.running = False

    def set_mode(self, mode_name: str):
        self.current_mode = mode_name
        self.generator = SignalGenerator(mode_name)

    async def run_forever(self):
        self.running = True
        while self.running:
            try:
                reading = self.generator.get_live_reading()
                analysis = self.analyzer.analyze_live_reading(reading)
                self.latest_reading = {**reading, **analysis}

                if self.latest_reading.get("is_alert") and self.fcm_tokens:
                    from api.firebase_notifier import send_alert_notification
                    for token in list(self.fcm_tokens):
                        send_alert_notification(
                            token,
                            "🚨 МАҢЫЗДЫ ЕСКЕРТУ",
                            self.latest_reading.get("alert_message", "Критикалық жағдай"),
                            self.latest_reading.get("bpm", 0),
                        )

                if self.clients:
                    dead_clients = set()
                    for client in list(self.clients):
                        try:
                            await client.send_json(self.latest_reading)
                        except Exception:
                            dead_clients.add(client)
                    self.clients -= dead_clients

            except Exception as e:
                print(f"Simulator loop error: {e}")

            await asyncio.sleep(1)


simulator_state = SimulatorState()
