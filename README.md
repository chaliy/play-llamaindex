## How to run

- Python 3.11+ installed
- poetry installed

```bash
poetry install
cp sample.env .env
poetry run python api_query.py
PRESET=bedrock poetry run python api_query.py
```