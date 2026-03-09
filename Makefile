.PHONY: dev run ingest verify qdrant

# Start Qdrant
qdrant:
	docker compose up -d

# Run API in dev mode — watch only src/ and ui/ to avoid HuggingFace cache reload loops
dev:
	uvicorn phflow.api.main:app --reload --reload-dir src --reload-dir ui --port 8000

# Run API in production mode (no reload)
run:
	uvicorn phflow.api.main:app --port 8000

# Ingest data
ingest:
	python scripts/ingest.py --reset

# Verify Qdrant collection
verify:
	python scripts/verify_collection.py
