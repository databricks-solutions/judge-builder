# Alignment Model Separation Plan

## Overview
Enable DSPy alignment to use a different model than the judge evaluation, with user-configurable serving endpoints.

## Current Architecture Analysis

### Judge Model Usage
- Judges use `InstructionJudge` which leverages MLflow's `make_judge` API
- Judge evaluation uses chat completions endpoint via `managed_rag_client.get_chat_completions_result()`
- **Current DSPy LM (`AgentEvalLM`):**
  - Extends `dspy.BaseLM` with dummy model name `databricks:/databricks-llama-4-maverick`
  - The model name is not used - actual LLM calls happen via `managed_rag_client.get_chat_completions_result()`
  - The `managed_rag_client` gets its endpoint from the eval context
  - Judge evaluation continues to use this path (requirement #1 ✓)

### DSPy Alignment
- DSPy is configured globally in `AlignmentService.__init__()` with `AgentEvalLM`
- Alignment uses `scorer_func.align(traces=traces)` from MLflow's native capability (server/judges/instruction_judge.py:103)
- **The alignment process internally uses DSPy which calls the configured LM**
- Currently shares the same model as judge evaluation because both use `AgentEvalLM`

## Key Insight
The alignment model separation needs to happen at the `managed_rag_client` level, not the DSPy LM level. We need to:
1. Allow users to select a serving endpoint for alignment
2. Configure the eval context to use that endpoint during alignment operations
3. Keep judge evaluation using the default eval context endpoint

## Requirements
1. ✅ Judge model continues to use the chat completions endpoint (no change needed)
2. 🎯 DSPy alignment now uses a different model than the judge evaluation
3. 🎯 Alignment model should be configurable based on any endpoint in the user's workspace

## Key Architecture Decisions

1. **Custom Alignment Optimizer:** Extend MLflow's `SIMBAAlignmentOptimizer` to accept custom model parameter. Pass to `judge.align(optimizer=custom_optimizer)`.

2. **Model Format:** `databricks:/endpoint-name`

3. **Endpoint Caching:** 5-minute TTL cache with manual entry fallback

## Implementation Plan

### Phase 1: Model Configuration

**Add to server/models.py:**
```python
class ServingEndpointConfig(BaseModel):
    endpoint_name: str

class AlignmentModelConfig(BaseModel):
    model_type: str = "default"  # or "serving_endpoint"
    serving_endpoint: Optional[ServingEndpointConfig] = None

# Extend JudgeCreateRequest and JudgeResponse with:
alignment_model_config: Optional[AlignmentModelConfig] = None
```

**Add to requirements.txt:**
```
databricks-dspy>=0.1.0
```

### Phase 2: Core Implementation

**2.1 Custom SIMBA Optimizer (server/judges/custom_simba_optimizer.py):**
```python
class CustomSIMBAAlignmentOptimizer(SIMBAAlignmentOptimizer):
    def __init__(self, model: Optional[str] = None, **kwargs):
        # Model format: databricks:/endpoint-name
        # The parent class stores it in self._model and DSPy uses it for optimization
        super().__init__(model=model, **kwargs)
```

**2.2 Update InstructionJudge.optimize():**
```python
def optimize(self, traces, alignment_model: Optional[str] = None):
    if alignment_model:
        optimizer = CustomSIMBAAlignmentOptimizer(model=alignment_model)
        self.scorer_func = self.scorer_func.align(traces=traces, optimizer=optimizer)
    else:
        self.scorer_func = self.scorer_func.align(traces=traces)  # default
```

**2.3 Update AlignmentService.run_alignment():**
```python
def run_alignment(self, judge_id: str):
    # ... get judge, traces, run evaluation ...

    # Get alignment model if configured
    alignment_model = None
    if judge.alignment_model_config and judge.alignment_model_config.model_type == "serving_endpoint":
        endpoint_name = judge.alignment_model_config.serving_endpoint.endpoint_name
        # Format: databricks:/endpoint-name
        alignment_model = f"databricks:/{endpoint_name}"

    # Run alignment with optional custom model
    judge_instance.optimize(fresh_traces, alignment_model=alignment_model)

    # ... create new version, evaluate, tag ...
```

**2.4 Update JudgeService:**
- Store `alignment_model_config` in judge instance
- Include in `_judge_to_response()`
- Restore from metadata in `_get_or_recreate_judge()`

**2.5 ServingEndpointService (server/services/serving_endpoint_service.py):**
```python
class ServingEndpointService(BaseService):
    def __init__(self):
        self._endpoints_cache = TTLCache(maxsize=1, ttl=300)  # 5 min cache

    def list_serving_endpoints(self, force_refresh=False):
        # Return cached or fetch from workspace_client.serving_endpoints.list()

    def validate_endpoint_name(self, name: str) -> bool:
        # Check if endpoint exists
```

### Phase 3: API Endpoints

**server/routers/serving_endpoints.py:**
```python
@router.get("/")
async def list_serving_endpoints():
    # Return list of endpoints from serving_endpoint_service

@router.get("/{endpoint_name}")
async def get_serving_endpoint(endpoint_name: str):
    # Return specific endpoint details

@router.post("/{endpoint_name}/validate")
async def validate_endpoint(endpoint_name: str):
    # Validate endpoint exists
```

**Register in server/app.py:**
```python
app.include_router(serving_endpoints.router, prefix="/api/serving-endpoints")
```

### Phase 4: Metadata Storage

- Add `alignment_model_config` parameter to `InstructionJudge.__init__()`
- Store in experiment metadata via `_update_judge_metadata()`
- Restore from metadata in `_get_or_recreate_judge()`

### Phase 5: Testing

**Unit Tests:**
- Custom optimizer initialization with model formats
- Endpoint service caching and validation
- Metadata storage/retrieval

**Integration Tests:**
- Full alignment flow with custom endpoint
- Judge evaluation unaffected by alignment model config
- Backward compatibility (default config)
- Manual endpoint name entry

**E2E Tests:**
- Create judge → run alignment with custom model → verify new version

## Backward Compatibility

- Existing judges: `alignment_model_config = None` → uses default alignment model
- No changes required to existing workflows

## Implementation Notes

1. **Model Format:** `databricks:/endpoint-name` (passed to optimizer which stores in `self._model`)
2. **Auth:** Handled by environment variables
3. **Caching:** 5-minute TTL for endpoint lists
4. **Frontend:** Endpoint selector on judge creation + alignment page (dropdown + manual entry)

## Success Criteria

- Judge evaluation uses chat completions endpoint (unchanged)
- Alignment uses configurable serving endpoint
- Backward compatible with existing judges
- All tests pass

