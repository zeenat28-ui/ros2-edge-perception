"""
Authentication and Role-Based Access Control (RBAC) Security Manager.

Implements zero-trust token authentication and access control (ISO 21434, UNECE R155):
- HMAC-SHA256 JWT generation with constant-time signature verification
- Role-Based Access Control (RBAC) permission resolution
- Token-bucket rate limiting for endpoint protection against denial-of-service
- Real-time token revocation registry
"""

import os
import time
import hmac
import hashlib
import base64
import json
import secrets
from typing import Dict, List, Optional, Tuple, Set


class SecurityRole:
    SUPER_ADMIN = "SUPER_ADMIN"
    FLEET_DISPATCHER = "FLEET_DISPATCHER"
    SAFETY_OFFICER = "SAFETY_OFFICER"
    MAINTENANCE_TECH = "MAINTENANCE_TECH"
    MONITORING_CLIENT = "MONITORING_CLIENT"


class Permission:
    DISPATCH_ORDER = "DISPATCH_ORDER"
    TRIGGER_ESTOP = "TRIGGER_ESTOP"
    RESET_INTERLOCK = "RESET_INTERLOCK"
    READ_TELEMETRY = "READ_TELEMETRY"
    UPDATE_CONFIG = "UPDATE_CONFIG"
    EXECUTE_DIAGNOSTICS = "EXECUTE_DIAGNOSTICS"


ROLE_PERMISSIONS: Dict[str, Set[str]] = {
    SecurityRole.SUPER_ADMIN: {
        Permission.DISPATCH_ORDER,
        Permission.TRIGGER_ESTOP,
        Permission.RESET_INTERLOCK,
        Permission.READ_TELEMETRY,
        Permission.UPDATE_CONFIG,
        Permission.EXECUTE_DIAGNOSTICS,
    },
    SecurityRole.SAFETY_OFFICER: {
        Permission.TRIGGER_ESTOP,
        Permission.RESET_INTERLOCK,
        Permission.READ_TELEMETRY,
        Permission.EXECUTE_DIAGNOSTICS,
    },
    SecurityRole.FLEET_DISPATCHER: {
        Permission.DISPATCH_ORDER,
        Permission.TRIGGER_ESTOP,
        Permission.READ_TELEMETRY,
    },
    SecurityRole.MAINTENANCE_TECH: {
        Permission.READ_TELEMETRY,
        Permission.EXECUTE_DIAGNOSTICS,
        Permission.TRIGGER_ESTOP,
    },
    SecurityRole.MONITORING_CLIENT: {
        Permission.READ_TELEMETRY,
    },
}


class TokenBucketRateLimiter:
    """Token-Bucket Rate Limiter preventing DoS and brute-force attacks on AMR endpoints."""

    def __init__(self, rate_per_sec: float = 50.0, capacity: float = 100.0):
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()

    def allow_request(self, cost: float = 1.0) -> bool:
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now

        # Replenish tokens
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


class SecurityAuthManager:
    """
    HMAC-SHA256 JWT Authentication & Granular RBAC Engine.
    Ensures zero plaintext credentials and prevents timing attacks.
    """

    def __init__(self, secret_key: Optional[str] = None, token_validity_sec: int = 3600):
        # Retrieve secret from environment variable or generate secure 256-bit ephemeral key
        env_secret = os.getenv("AURA_AUTH_SECRET_KEY")
        if secret_key:
            self._secret_key = secret_key.encode("utf-8")
        elif env_secret:
            self._secret_key = env_secret.encode("utf-8")
        else:
            # Cryptographically secure random key for local ephemeral run
            self._secret_key = secrets.token_bytes(32)

        self.token_validity_sec = token_validity_sec
        self.rate_limiter = TokenBucketRateLimiter(rate_per_sec=50.0, capacity=100.0)
        self.revoked_token_hashes: Set[str] = set()

    def _b64_encode(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

    def _b64_decode(self, data: str) -> bytes:
        padding = 4 - (len(data) % 4)
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data.encode("utf-8"))

    def create_token(self, subject: str, role: str, custom_claims: Optional[dict] = None) -> str:
        """
        Generate an HMAC-SHA256 signed JSON Web Token (JWT).
        """
        if role not in ROLE_PERMISSIONS:
            raise ValueError(f"Invalid security role: {role}")

        header = {"alg": "HS256", "typ": "JWT"}
        now = int(time.time())
        payload = {
            "sub": subject,
            "role": role,
            "iat": now,
            "exp": now + self.token_validity_sec,
            "jti": secrets.token_hex(8),
        }
        if custom_claims:
            payload.update(custom_claims)

        hdr_b64 = self._b64_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        pay_b64 = self._b64_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        signing_input = f"{hdr_b64}.{pay_b64}".encode("utf-8")

        signature = hmac.new(self._secret_key, signing_input, hashlib.sha256).digest()
        sig_b64 = self._b64_encode(signature)

        return f"{hdr_b64}.{pay_b64}.{sig_b64}"

    def verify_token(self, token: str) -> Tuple[bool, Optional[dict], str]:
        """
        Verify JWT signature using constant-time digest comparison.
        Returns: (is_valid, payload_dict, error_or_success_message)
        """
        # Enforce rate-limiting
        if not self.rate_limiter.allow_request():
            return False, None, "RATE_LIMIT_EXCEEDED: Too many authentication attempts."

        parts = token.split(".")
        if len(parts) != 3:
            return False, None, "MALFORMED_TOKEN: Token must have 3 sections."

        hdr_b64, pay_b64, sig_b64 = parts

        # Verify signature in constant time
        signing_input = f"{hdr_b64}.{pay_b64}".encode("utf-8")
        expected_sig = hmac.new(self._secret_key, signing_input, hashlib.sha256).digest()
        expected_sig_b64 = self._b64_encode(expected_sig)

        if not hmac.compare_digest(sig_b64, expected_sig_b64):
            return False, None, "INVALID_SIGNATURE: Cryptographic verification failed."

        try:
            payload = json.loads(self._b64_decode(pay_b64).decode("utf-8"))
        except Exception as e:
            return False, None, f"PAYLOAD_DECODE_ERROR: {str(e)}"

        # Verify expiration
        now = int(time.time())
        if payload.get("exp", 0) < now:
            return False, None, "TOKEN_EXPIRED: Token lifetime exceeded."

        # Verify token revocation
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        if token_hash in self.revoked_token_hashes:
            return False, None, "TOKEN_REVOKED: This token was explicitly revoked."

        return True, payload, "TOKEN_VERIFIED"

    def authorize(self, token: str, required_permission: str) -> Tuple[bool, str]:
        """
        Verify token and check if the associated role possesses the required permission.
        """
        is_valid, payload, msg = self.verify_token(token)
        if not is_valid or not payload:
            return False, f"AUTH_FAILED: {msg}"

        role = payload.get("role")
        permissions = ROLE_PERMISSIONS.get(role, set())

        if required_permission not in permissions:
            return False, f"PERMISSION_DENIED: Role '{role}' lacks permission '{required_permission}'."

        return True, f"AUTHORIZED: {payload.get('sub')} granted '{required_permission}'."

    def revoke_token(self, token: str):
        """Revoke a token immediately."""
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        self.revoked_token_hashes.add(token_hash)


if __name__ == "__main__":
    auth = SecurityAuthManager()
    token = auth.create_token(subject="user_dispatcher_01", role=SecurityRole.FLEET_DISPATCHER)
    print("Created JWT:", token)

    # Test authorization
    allowed, reason = auth.authorize(token, Permission.DISPATCH_ORDER)
    print("Dispatch Order:", allowed, reason)
    assert allowed

    # Test unauthorized action
    denied, reason = auth.authorize(token, Permission.RESET_INTERLOCK)
    print("Reset Interlock (should fail):", denied, reason)
    assert not denied
    print("Security Auth Manager self-test passed.")
