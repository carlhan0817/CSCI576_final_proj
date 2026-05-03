"""FastAPI application factory for the player backend."""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.server.config import ServerConfig
from backend.server import routes


def create_app(config: ServerConfig | None = None) -> FastAPI:
    config = config or ServerConfig.from_env()
    app = FastAPI(title="Multimedia Segmentation Player")
    app.state.config = config
    app.include_router(routes.router)

    # Static MP4 files (browser uses HTTP Range to seek)
    config.videos_root.mkdir(parents=True, exist_ok=True)
    app.mount("/videos", StaticFiles(directory=config.videos_root), name="videos")

    # Static frontend assets + index.html
    static_dir = config.frontend_root / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    index_path = config.frontend_root / "index.html"

    @app.get("/")
    def index():
        if index_path.exists():
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="frontend index.html missing")

    return app


# Module-level app for `uvicorn backend.server.app:app`
app = create_app()
