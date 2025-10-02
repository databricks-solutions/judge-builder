"""Judges API router."""

import logging
import traceback
from typing import List

from fastapi import APIRouter, HTTPException

from server.models import (
    JudgeCreateRequest,
    JudgeResponse,
)
from server.services.judge_service import judge_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post('/', response_model=JudgeResponse)
async def create_judge(request: JudgeCreateRequest):
    """Create a new judge (direct judge creation, not full orchestration)."""
    try:
        logger.info(f'Creating judge: {request.name}')
        return judge_service.create_judge(request)
    except Exception as e:
        logger.error(f'Failed to create judge: {e}\n{traceback.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/', response_model=List[JudgeResponse])
async def list_judges():
    """List all judges."""
    try:
        return judge_service.list_judges()
    except Exception as e:
        logger.error(f'Failed to list judges: {e}\n{traceback.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/{judge_id}', response_model=JudgeResponse)
async def get_judge(judge_id: str):
    """Get a judge by ID."""
    try:
        judge = judge_service.get_judge(judge_id)
        if not judge:
            raise HTTPException(status_code=404, detail='Judge not found')
        return judge
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f'Failed to get judge {judge_id}: {e}\n{traceback.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))


@router.delete('/{judge_id}')
async def delete_judge(judge_id: str):
    """Delete a judge."""
    try:
        success = judge_service.delete_judge(judge_id)
        if not success:
            raise HTTPException(status_code=404, detail='Judge not found')
        return {'message': 'Judge deleted successfully'}
    except Exception as e:
        logger.error(f'Failed to delete judge {judge_id}: {e}\n{traceback.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))
