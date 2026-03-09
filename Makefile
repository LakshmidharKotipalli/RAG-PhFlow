.PHONY: dev run ingest verify qdrant

# Absolute path to the project root — works regardless of where make is invoked from
ROOT := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Start Qdrant
qdrant:
	docker compose -f $(ROOT)/docker-compose.yml up -d

# Run API in dev mode — absolute --reload-dir paths so watchfiles never
# watches the HuggingFace cache or other unrelated directories
dev:
	cd $(ROOT) && uvicorn phflow.api.main:app \
		--reload \
		--reload-dir $(ROOT)/src \
		--reload-dir $(ROOT)/ui \
		--port 8000

# Run API in production mode (no reload)
run:
	cd $(ROOT) && uvicorn phflow.api.main:app --port 8000

# Ingest data
ingest:
	cd $(ROOT) && python scripts/ingest.py --reset

# Verify Qdrant collection
verify:
	cd $(ROOT) && python scripts/verify_collection.py
