# MLflow Template Variable Support Implementation Plan

## Overview
Add UI support for MLflow's make_judge template variables (`{{ inputs }}`, `{{ outputs }}`, `{{ expectations }}`, `{{ trace }}`) with validation, insertion buttons, and user guidance.

## Current State Analysis
- Judge creation happens in `client/src/pages/WelcomePage.tsx` (lines 355-367)
- Current instruction textarea has no validation or template variable support
- Backend already supports template variables via MLflow's make_judge API
- Tests show required template variables: `{{ inputs }}`, `{{ outputs }}`, `{{ expectations }}`, `{{ trace }}`

## Implementation Plan

### 1. Create Template Variable Validation Utility
**File:** `client/src/lib/templateValidation.ts`
- Function to validate that at least one template variable is present
- Function to extract template variables from text
- Constants for supported template variables
- Validation error messages

### 2. Create Template Variable Insertion Component
**File:** `client/src/components/TemplateVariableButtons.tsx`
- Buttons for each template variable
- Click handler to insert at cursor position
- Ref forwarding to parent textarea
- Styled button group with proper spacing

### 3. Update Judge Instruction Input Component
**File:** `client/src/components/JudgeInstructionInput.tsx`
- Extract current textarea logic from WelcomePage
- Add template variable buttons above textarea
- Add validation state and error messages
- Add instructional text blurbs
- Handle cursor position tracking for insertion

### 4. Update WelcomePage
**File:** `client/src/pages/WelcomePage.tsx`
- Replace existing textarea with new JudgeInstructionInput component
- Add validation to judge creation logic
- Update form submission to check template variables
- Add proper error handling for validation failures

### 5. Add Validation Messages and Help Text
- Add instructional text about using template variables
- Add guidance about output types (pass/fail or 1-5 likert scale)
- Add real-time validation feedback
- Add proper error states and styling

### 6. Backend Validation Enhancement (Optional)
**File:** `server/models.py` or validation layer
- Add server-side validation for template variables
- Return helpful error messages if validation fails
- Ensure consistency between frontend and backend validation

### 7. Testing
**Files:** 
- `client/src/lib/__tests__/templateValidation.test.ts`
- `client/src/components/__tests__/TemplateVariableButtons.test.tsx`
- `client/src/components/__tests__/JudgeInstructionInput.test.tsx`
- Integration tests for judge creation workflow

## Key Features

### Template Variable Support
- `{{ inputs }}` - The input/request to be judged
- `{{ outputs }}` - The output/response to be judged  
- `{{ expectations }}` - Expected behavior/criteria
- `{{ trace }}` - Full trace context

### Validation Rules
- At least one template variable must be present
- Template variables must use correct syntax `{{ variable }}`
- Clear error messages for missing or invalid variables

### User Experience
- One-click insertion of template variables at cursor position
- Real-time validation feedback
- Clear instructional text
- Helpful error messages
- Visual distinction of template variables in text

### UI Components
- Styled button group for template variables
- Validation state indicators
- Help text and examples
- Responsive design for mobile

## File Structure
```
client/src/
├── lib/
│   ├── templateValidation.ts (NEW)
│   └── __tests__/
│       └── templateValidation.test.ts (NEW)
├── components/
│   ├── TemplateVariableButtons.tsx (NEW)
│   ├── JudgeInstructionInput.tsx (NEW)
│   └── __tests__/
│       ├── TemplateVariableButtons.test.tsx (NEW)
│       └── JudgeInstructionInput.test.tsx (NEW)
└── pages/
    └── WelcomePage.tsx (MODIFIED)
```

## Implementation Phases

### Phase 1: Core Components
1. Create template validation utility
2. Create template variable buttons component
3. Create judge instruction input component

### Phase 2: Integration
1. Update WelcomePage to use new components
2. Add validation to form submission
3. Style and polish UI/UX

### Phase 3: Testing & Polish
1. Write comprehensive tests
2. Add error handling and edge cases
3. Code review and mentor approval

## Expected Lines of Code
- Template validation: ~50 lines
- Template buttons component: ~80 lines  
- Judge instruction input: ~120 lines
- WelcomePage updates: ~50 lines
- Tests: ~200 lines
- **Total: ~500 lines across multiple files**

## Considerations
- Maintain backward compatibility
- Ensure accessibility (keyboard navigation, screen readers)
- Mobile-responsive design
- Performance (avoid unnecessary re-renders)
- Consistent with existing UI patterns and styling