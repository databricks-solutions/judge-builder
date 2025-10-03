/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { SchemaInfo } from './SchemaInfo';
/**
 * Response model for judge information.
 */
export type JudgeResponse = {
    /**
     * Unique judge identifier
     */
    id: string;
    /**
     * Human-readable name for the judge
     */
    name: string;
    /**
     * User-provided evaluation criteria (always shown to user)
     */
    instruction: string;
    /**
     * MLflow experiment ID
     */
    experiment_id: string;
    /**
     * Judge version number
     */
    version?: number;
    /**
     * MLflow run ID for labeling session
     */
    labeling_run_id?: (string | null);
    /**
     * Cached schema analysis for consistent use
     */
    schema_info?: (SchemaInfo | null);
};

