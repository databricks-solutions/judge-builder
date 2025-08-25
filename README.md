# Judge Builder

A comprehensive platform for building and optimizing LLM judges.

Judge Builder enables you to align custom LLM judges or the built-in Databricks LLM judges with human preferences using state-of-the-art optimization techniques. The platform provides seamless integration with MLflow experiments and Databricks workspaces, supporting the full lifecycle from judge creation to production evaluation workflows.

## Video Overview

Include a GIF overview of what your project does. Use a service like Quicktime, Zoom or Loom to create the video, then convert to a GIF.

## Installation

### Prerequisites

- Python 3.9+
- Node.js 18+
- uv (Python package manager)

### Setup

```bash
git clone <repository-url>
cd judge-builder-v1
./setup.sh
```

This will
   - Install Python dependencies using uv
   - Install Node.js dependencies
   - Set up environment configuration

#### Deploy

To deploy the application to Databricks Apps:

```bash
./deploy.sh
```

This will:
- Build the frontend
- Sync code to Databricks workspace
- Create and deploy the Databricks App

#### Develop
```bash
./dev/watch.sh
```
This runs both the FastAPI backend (port 8001) and React frontend (port 3000) in development mode. The API documentation can be found at: http://localhost:8001/docs

## Usage

1. **Create a Judge**: Start by creating a new judge with a name, instruction, and experiment ID
2. **Add Examples**: Import traces from MLflow experiments to use as evaluation examples
3. **Labeling**: Invite SMEs to provide human feedback
4. **Align Judge**: Run alignment to optimize the judge based on human feedback
5. **Evaluate**: Test the aligned judge on new examples and review performance

## How to get help

For questions or bugs, please contact agents-outreach@databricks.com and the team will reach out shortly.

## License

&copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| React                                  | Frontend framework      | MIT        | https://github.com/facebook/react                  |
| FastAPI                                | Backend web framework   | MIT        | https://github.com/tiangolo/fastapi               |
| Tailwind CSS                           | Utility-first CSS       | MIT        | https://github.com/tailwindlabs/tailwindcss       |
| shadcn/ui                              | UI component library    | MIT        | https://github.com/shadcn-ui/ui                   |
| MLflow                                 | ML lifecycle management | Apache 2.0 | https://github.com/mlflow/mlflow                  |
| DSPy                                   | LM programming framework| MIT        | https://github.com/stanfordnlp/dspy               |
| Lucide React                           | Icon library            | ISC        | https://github.com/lucide-icons/lucide            |