import os, yaml, re
from config.settings import PATH_CONFIG_RAG

def load_rag_config():
    """
    Load the RAG configuration file and validate that all required attributes
    are present for each source type (Confluence, Notion, etc.).
    Environment variables in the form ${VAR} are automatically expanded.
    """

    config_path = os.getenv("RAG_CONFIG_PATH", "/app/config/rag_config.yml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ RAG config not found at {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if not raw or "sources" not in raw:
        raise ValueError("❌ Invalid config: missing top-level 'sources' key")

    sources = raw["sources"]

    if not isinstance(sources, list) or not sources:
        raise ValueError("❌ Invalid config: 'sources' must be a non-empty list")

    # 🔒 Required keys per source type
    required_fields = {
        "confluence": ["name", "type", "base_url", "root_ids", "api_token"],
        "notion": ["name", "type", "integration_token", "database_ids"],
    }

    for idx, src in enumerate(sources, start=1):
        if "type" not in src:
            raise ValueError(f"❌ Source #{idx} missing 'type' field")

        src_type = src["type"].lower()

        if src_type not in required_fields:
            raise ValueError(f"❌ Unsupported source type '{src_type}' in source #{idx}")

        # 🔍 Expand environment variables like ${VAR}
        for key, val in src.items():
            if isinstance(val, str):
                match = re.match(r"\$\{(\w+)\}", val)
                if match:
                    env_var = match.group(1)
                    env_value = os.getenv(env_var)
                    if env_value is None:
                        raise ValueError(f"❌ Missing environment variable: {env_var} (used in {key})")
                    src[key] = env_value

        # 🔍 Check for missing mandatory attributes
        missing = [field for field in required_fields[src_type] if field not in src or src[field] in (None, "")]
        if missing:
            raise ValueError(f"❌ Missing required fields for {src['name']} ({src_type}): {', '.join(missing)}")

    print(f"✅ Loaded {len(sources)} sources from config:")
    for src in sources:
        print(f"   • {src['name']} ({src['type']})")

    return sources