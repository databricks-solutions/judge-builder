"""FastAPI application for Judge Builder."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server.routers import router


def load_env_file(filepath: str) -> None:
    """Load environment variables from a file."""
    if Path(filepath).exists():
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, _, value = line.partition('=')
                    if key and value:
                        os.environ[key] = value


# Load .env files
load_env_file('.env')
load_env_file('.env.local')


def setup_logging() -> None:
    """Simple logging configuration."""
    debug_evaluation = os.getenv('DEBUG_EVALUATION', 'false').lower() == 'true'
    log_level = logging.DEBUG if debug_evaluation else logging.INFO

    logging.basicConfig(
        level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True
    )

    # Silence urllib3 connection pool warnings
    logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)

    print('Judge Builder - Server starting up')
    logging.info('Judge Builder - Logging initialized')


# Setup logging
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    from server.services.judge_service import judge_service
    await judge_service.load_all_judges_on_startup()
    
    yield
    
    # Shutdown (if needed in the future)
    pass


app = FastAPI(
    title='Judge Builder API',
    description='API for LLM Judge Builder with MLflow integration',
    version='0.1.0',
    lifespan=lifespan,
    servers=[{'url': 'http://localhost:8001', 'description': 'Development server'}],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000', 'http://127.0.0.1:3000'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(router, prefix='/api', tags=['api'])


@app.get('/health')
async def health():
    """Health check endpoint."""
    return {'status': 'healthy'}


# Serve static files from client build directory (MUST be LAST!)
if os.path.exists('client/build'):
    app.mount('/', StaticFiles(directory='client/build', html=True), name='static')
