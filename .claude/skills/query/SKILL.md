---
name: query
description: Answer questions about the Velox codebase or pull requests. Use when asked a question via "/query" or when the user wants to understand code, architecture, or implementation details.
---

# Velox Query Skill

Answer questions about the Velox project codebase or specific pull requests.

## Key Context

- Velox is a C++ execution engine library for analytical data processing
- Uses C++20 standard with heavy use of templates and SFINAE
- Custom memory management with MemoryPool
- Vectorized execution with custom Vector types
- Follows Google C++ style with some modifications

## Guidelines

- Read CLAUDE.md and CODING_STYLE.md for project-specific conventions
- Answer thoroughly and accurately
- If the question is about PR changes, analyze the diff carefully
- If it's about the codebase, explore relevant files to provide a complete answer
- Be specific and reference exact file paths and line numbers when relevant
