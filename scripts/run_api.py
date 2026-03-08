"""Run MANTRA FastAPI service in production mode."""

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "mantra.interfaces.api.api_server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
    )
