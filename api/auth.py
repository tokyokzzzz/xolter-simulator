import httpx

DJANGO_VERIFY_URL = "http://16.171.171.100:8001/api/verify-token/"


async def verify_supervisor_token(token: str) -> dict:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                DJANGO_VERIFY_URL,
                json={"token": token},
                timeout=5.0
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("is_valid"):
                    return data
    except Exception as e:
        print(f"Token verification error: {e}")
    return None
