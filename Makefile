.PHONY: dev-up dev-down dev-logs install build test lint typecheck clean

# Development environment
dev-up:
	docker-compose -f docker-compose.dev.yml up -d
	@echo "🚀 Development environment is starting..."
	@echo "📊 Services:"
	@echo "  - Postgres: localhost:5432"
	@echo "  - Redis: localhost:6379"
	@echo "  - NATS: localhost:4222 (monitoring: localhost:8222)"
	@echo "  - MinIO: localhost:9000 (console: localhost:9001)"

dev-down:
	docker-compose -f docker-compose.dev.yml down

dev-logs:
	docker-compose -f docker-compose.dev.yml logs -f

dev-clean:
	docker-compose -f docker-compose.dev.yml down -v
	docker system prune -f

# Project management
install:
	npm install
	cd apps/frontend && npm install
	cd apps/backend && npm install

build:
	npm run build

dev:
	npm run dev

test:
	npm run test

lint:
	npm run lint

typecheck:
	npm run typecheck

clean:
	npm run clean
	rm -rf node_modules apps/*/node_modules packages/*/node_modules
	rm -rf apps/*/.next apps/*/dist packages/*/dist
