# Judge Builder

A comprehensive platform for building and optimizing LLM judges.

Judge Builder enables you to align custom LLM judges or the built-in Databricks LLM judges with human preferences using state-of-the-art optimization techniques. The platform provides seamless integration with MLflow experiments and Databricks workspaces, supporting the full lifecycle from judge creation to production evaluation workflows.

![judge-builder-v0 3 3](https://github.com/user-attachments/assets/9a4c0443-5368-4838-846a-1ea934d4edc3)


## Installation

### Prerequisites

- Python 3.9+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/) v0.230.0+

### Setup

```bash
git clone https://github.com/databricks-solutions/judge-builder.git
cd judge-builder
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

1. **Create a Judge**: Start by creating a new judge with a name, instruction, and experiment ID. You will also invite SMEs to provide human feedback.
2. **Add Examples**: Add traces from the attached experiment to use as examples for your judge to learn from.
3. **Labeling**: The SME will provide human feedback over the added examples.
4. **Align Judge**: Run alignment to optimize the judge based on human feedback. You can review the performance of the judge before and after alignment.

**Note:** In the existing MVP, we only support binary outcome judges (pass/fail) over the request and response. We will introduce support for additional fields soon.

To retrieve the judge to use in online/offline evaluations, use the `list_scorers()` API:

```python
# Run `pip install -U "mlflow[databricks]>=3.2.0"` to get the `list_scorers` API

from mlflow.genai.scorers import list_scorers

mlflow.set_experiment(experiment_id="<YOUR_EXPERIMENT_ID>")
list_scorers()
```

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
