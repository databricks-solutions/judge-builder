/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Response model for alignment results.
 */
export type AlignmentResponse = {
    /**
     * Judge identifier
     */
    judge_id: string;
    /**
     * Whether alignment succeeded
     */
    success: boolean;
    /**
     * Result message
     */
    message: string;
    /**
     * New judge version number
     */
    new_version: number;
    /**
     * Performance improvement metrics
     */
    improvement_metrics?: (Record<string, any> | null);
};

