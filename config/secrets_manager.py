import os

from dotenv import load_dotenv

AVAILABLE_CLOUD_SECRET_MANAGER = []
load_dotenv()

try:
    import boto3
    AVAILABLE_CLOUD_SECRET_MANAGER.append("aws")
except ModuleNotFoundError as e:
    print("No module found for AWS secret store")

try:
    import secretmanager
    AVAILABLE_CLOUD_SECRET_MANAGER.append("gcp")
except ModuleNotFoundError as e:
    print("No module found for GCP secret store")

try:
    import SecretClient
    import DefaultAzureCredential
    AVAILABLE_CLOUD_SECRET_MANAGER.append("azure")
except ModuleNotFoundError as e:
    print("No module found for Azure secret store")

def get_secret(name: str, default: str | None = None) -> str | None:
    backend = os.getenv("SECRET_BACKEND", "env").lower()

    # 1. Docker/K8s file secrets
    secret_file = f"/run/secrets/{name}"
    if os.path.isfile(secret_file):
        with open(secret_file) as f:
            return f.read().strip()

    # 2. Environment variable
    if backend == "env":
        return os.getenv(name, default)

    # 3. Cloud backends
    if backend == "aws" and "aws" in AVAILABLE_CLOUD_SECRET_MANAGER:
        client = boto3.client("secretsmanager")
        resp = client.get_secret_value(SecretId=name)
        return resp.get("SecretString")

    if backend == "gcp" and "gcp" in AVAILABLE_CLOUD_SECRET_MANAGER:
        client = secretmanager.SecretManagerServiceClient()
        project_id = os.getenv("GCP_PROJECT_ID")
        secret_path = f"projects/{project_id}/secrets/{name}/versions/latest"
        resp = client.access_secret_version(name=secret_path)
        return resp.payload.data.decode("UTF-8")

    if backend == "azure" and "azure" in AVAILABLE_CLOUD_SECRET_MANAGER:
        keyvault_url = os.getenv("AZURE_KEYVAULT_URL")
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=keyvault_url, credential=credential)
        secret = client.get_secret(name)
        return secret.value

    return default