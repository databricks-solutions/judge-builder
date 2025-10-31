/**
 * Utility functions for validating MLflow template variables in judge instructions
 */

// Supported MLflow template variables
export const TEMPLATE_VARIABLES = {
  INPUTS: '{{ inputs }}',
  OUTPUTS: '{{ outputs }}',
  EXPECTATIONS: '{{ expectations }}',
  TRACE: '{{ trace }}'
} as const

export const TEMPLATE_VARIABLE_LIST = Object.values(TEMPLATE_VARIABLES)

// Regular expression to match template variables
const TEMPLATE_VARIABLE_REGEX = /\{\{\s*(\w+)\s*\}\}/g

/**
 * Extract all template variables from text
 */
export function extractTemplateVariables(text: string): string[] {
  const matches = text.match(TEMPLATE_VARIABLE_REGEX)
  return matches || []
}

/**
 * Check if text contains at least one supported template variable
 */
export function hasTemplateVariables(text: string): boolean {
  const variables = extractTemplateVariables(text)
  return variables.some(variable => 
    (TEMPLATE_VARIABLE_LIST as readonly string[]).includes(variable)
  )
}

/**
 * Validate judge instructions for template variables
 */
export function validateTemplateVariables(instructions: string): {
  isValid: boolean
  error?: string
  suggestions?: string[]
} {
  const trimmed = instructions.trim()
  
  if (!trimmed) {
    return {
      isValid: false,
      error: 'Instructions are required'
    }
  }

  if (!hasTemplateVariables(trimmed)) {
    return {
      isValid: false,
      error: 'Instructions must contain at least one template variable',
      suggestions: [
        'Use {{ inputs }} to reference the input/request being judged',
        'Use {{ outputs }} to reference the output/response being judged',
        'Use {{ expectations }} to reference expected behavior or criteria',
        'Use {{ trace }} to reference the full trace context'
      ]
    }
  }

  // Check for invalid template variable syntax
  const foundVariables = extractTemplateVariables(trimmed)
  const invalidVariables = foundVariables.filter(variable => 
    !(TEMPLATE_VARIABLE_LIST as readonly string[]).includes(variable)
  )

  if (invalidVariables.length > 0) {
    return {
      isValid: false,
      error: `Invalid template variable(s): ${invalidVariables.join(', ')}`,
      suggestions: [
        `Supported variables are: ${TEMPLATE_VARIABLE_LIST.join(', ')}`
      ]
    }
  }

  return { isValid: true }
}

/**
 * Get description for a template variable
 */
export function getTemplateVariableDescription(variable: string): string {
  switch (variable) {
    case TEMPLATE_VARIABLES.INPUTS:
      return 'The input/request to be judged'
    case TEMPLATE_VARIABLES.OUTPUTS:
      return 'The output/response to be judged'
    case TEMPLATE_VARIABLES.EXPECTATIONS:
      return 'Expected behavior or criteria'
    case TEMPLATE_VARIABLES.TRACE:
      return 'Full trace context including metadata'
    default:
      return 'Unknown template variable'
  }
}