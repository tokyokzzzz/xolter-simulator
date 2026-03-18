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
        self.running = False

    def set_mode(self, mode_name: str):
        self.current_mode = mode_name
        self.generator = SignalGenerator(mode_name)

    async def run_forever(self):
        self.running = True
        while self.running:
            reading = self.generator.get_live_reading()
            analysis = self.analyzer.analyze_live_reading(reading)
            self.latest_reading = {**reading, **analysis}

            dead_clients = set()
            for client in self.clients:
                try:
                    await client.send_json(self.latest_reading)
                except Exception:
                    dead_clients.add(client)

            self.clients -= dead_clients
            await asyncio.sleep(1)


simulator_state = SimulatorState()
