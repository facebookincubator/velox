These PEM files are deterministic, loopback-only fixtures for the isolated
ABFS TLS transport tests. They have no external use or credentials.

The certificates and keys use P-256 with SHA-256 signatures.

The root CA is included in test-ca-bundle.pem. The server fixtures are signed
by that root and are valid from 2026-07-20 through 2036-07-17. The normal
server has SAN DNS:localhost. The negative server has SAN DNS:wrong-host.test.