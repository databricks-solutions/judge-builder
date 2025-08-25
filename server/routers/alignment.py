"""Alignment API router."""

import logging
import traceback

from fastapi import APIRouter, HTTPException

from server.models import (
    AlignmentResponse,
    EvaluationResult,
    TestJudgeRequest,
    TestJudgeResponse,
    TraceRequest,
)
from server.services.alignment_service import alignment_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post('/{judge_id}/align', response_model=AlignmentResponse)
async def run_alignment(judge_id: str):
    """Run alignment for a judge."""
    try:
        return alignment_service.run_alignment(judge_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f'Request failed: {e}\n{traceback.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/{judge_id}/evaluate', response_model=EvaluationResult)
async def evaluate_judge(judge_id: str, request: TraceRequest):
    """Run judge evaluation on traces and log to MLflow."""
    try:
        return alignment_service.evaluate_judge(judge_id, request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f'Request failed: {e}\n{traceback.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/{judge_id}/test', response_model=TestJudgeResponse)
async def test_judge(judge_id: str, request: TestJudgeRequest):
    """Test judge on a single trace (for play buttons)."""
    try:
        return alignment_service.test_judge(judge_id, request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f'Request failed: {e}\n{traceback.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/{judge_id}/alignment-comparison')
async def get_alignment_comparison(judge_id: str):
    """Get alignment comparison data including metrics and confusion matrix."""
    try:
        return alignment_service.get_alignment_comparison(judge_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f'Request failed: {e}\n{traceback.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))
