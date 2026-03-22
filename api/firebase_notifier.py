import os

import firebase_admin
from firebase_admin import credentials, messaging


def initialize_firebase():
    import os
    cred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "firebase-admin.json")
    if os.path.exists(cred_path) and not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        print("Firebase initialized successfully")
    else:
        print("Firebase credentials not found - notifications disabled")


def send_alert_notification(fcm_token: str, title: str, body: str, bpm: float):
    if not firebase_admin._apps:
        return False
    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            android=messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(
                    sound="default",
                    priority="max",
                    channel_id="holter_alerts",
                    color="#FF0000",
                    vibrate_timings_millis=[0, 800, 200, 800, 200, 800],
                ),
            ),
            data={
                "bpm": str(bpm),
                "alert": "true",
                "screen": "monitoring",
            },
            token=fcm_token,
        )
        response = messaging.send(message)
        print(f"Alert sent to device: {response}")
        return True
    except Exception as e:
        print(f"Failed to send notification: {e}")
        return False


def send_status_notification(fcm_token: str, diagnosis: str, bpm: float):
    if not firebase_admin._apps:
        return False
    try:
        message = messaging.Message(
            data={
                "bpm": str(bpm),
                "diagnosis": diagnosis,
                "alert": "false",
            },
            android=messaging.AndroidConfig(priority="normal"),
            token=fcm_token,
        )
        messaging.send(message)
        return True
    except Exception as e:
        print(f"Status update failed: {e}")
        return False
