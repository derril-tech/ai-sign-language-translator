-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create database if it doesn't exist
SELECT 'CREATE DATABASE sign_translator'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'sign_translator')\gexec

-- Connect to the database
\c sign_translator;

-- Enable pgvector extension in the target database
CREATE EXTENSION IF NOT EXISTS vector;

-- Create initial schema (will be managed by TypeORM migrations in production)
-- This is just for development setup
