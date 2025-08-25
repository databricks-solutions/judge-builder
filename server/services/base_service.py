"""Base service class with shared authentication and MLflow setup."""

import os

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient


class BaseService:
    """Base service class with authentication and MLflow client setup."""

    def __init__(self):
        load_dotenv('.env.local')
        self._validate_auth()
        self._setup_mlflow()

    def _validate_auth(self):
        """Validate Databricks authentication credentials."""
        databricks_host = os.getenv('DATABRICKS_HOST')
        databricks_token = os.getenv('DATABRICKS_TOKEN')
        databricks_client_id = os.getenv('DATABRICKS_CLIENT_ID')
        databricks_client_secret = os.getenv('DATABRICKS_CLIENT_SECRET')

        has_token_auth = databricks_host and databricks_token
        has_oauth_auth = databricks_host and databricks_client_id and databricks_client_secret

        if not (has_token_auth or has_oauth_auth):
            raise ValueError(
                'Databricks authentication required: Set DATABRICKS_HOST and '
                '(DATABRICKS_TOKEN or DATABRICKS_CLIENT_ID+DATABRICKS_CLIENT_SECRET)'
            )

    def _setup_mlflow(self):
        """Setup MLflow tracking URI and client."""
        mlflow.set_tracking_uri('databricks')
        self.client = MlflowClient()
