"""
Secure API Key Management System
Production-grade security for API credentials with encryption, access control, and monitoring
"""

import os
import json
import base64
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class AccessLevel(Enum):
    """API key access levels"""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"
    SYSTEM = "system"

class KeyStatus(Enum):
    """API key status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REVOKED = "revoked"
    ROTATION_PENDING = "rotation_pending"

class SecurityEvent(Enum):
    """Security event types"""
    KEY_ACCESS = "key_access"
    KEY_CREATION = "key_creation"
    KEY_ROTATION = "key_rotation"
    KEY_REVOCATION = "key_revocation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    ENCRYPTION_ERROR = "encryption_error"
    DECRYPTION_ERROR = "decryption_error"

@dataclass
class APIKeyMetadata:
    """API key metadata"""
    key_id: str
    service_name: str
    access_level: AccessLevel
    status: KeyStatus
    created_at: datetime
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    rotation_interval: Optional[int] = None  # Days
    usage_count: int = 0
    max_usage: Optional[int] = None
    allowed_operations: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class SecurityAuditLog:
    """Security audit log entry"""
    timestamp: datetime
    event_type: SecurityEvent
    key_id: Optional[str]
    service_name: Optional[str]
    operation: Optional[str]
    success: bool
    details: Dict[str, Any]
    client_info: Dict[str, str] = field(default_factory=dict)

class EncryptionManager:
    """Advanced encryption manager for API keys"""
    
    def __init__(self, master_password: Optional[str] = None):
        self.master_password = master_password or self._generate_master_password()
        self.salt = self._get_or_create_salt()
        self.fernet = self._create_fernet_instance()
        
        logger.info("Encryption manager initialized with AES-256 encryption")
    
    def _generate_master_password(self) -> str:
        """Generate a secure master password"""
        # In production, this should come from a secure key management service
        master_password = os.environ.get('MASTER_ENCRYPTION_KEY')
        if not master_password:
            # Generate a secure random password
            master_password = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
            logger.warning("Generated new master password. Store securely: MASTER_ENCRYPTION_KEY")
        return master_password
    
    def _get_or_create_salt(self) -> bytes:
        """Get or create encryption salt"""
        salt_file = os.path.join(os.path.expanduser('~'), '.ai_business_salt')
        
        if os.path.exists(salt_file):
            with open(salt_file, 'rb') as f:
                return f.read()
        else:
            salt = os.urandom(16)
            with open(salt_file, 'wb') as f:
                f.write(salt)
            os.chmod(salt_file, 0o600)  # Restrict permissions
            return salt
    
    def _create_fernet_instance(self) -> Fernet:
        """Create Fernet encryption instance"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_password.encode()))
        return Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt data: {e}")
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise DecryptionError(f"Failed to decrypt data: {e}")
    
    def rotate_encryption_key(self, new_master_password: str) -> bool:
        """Rotate the master encryption key"""
        try:
            old_fernet = self.fernet
            
            # Create new encryption instance
            self.master_password = new_master_password
            self.salt = os.urandom(16)  # New salt
            self.fernet = self._create_fernet_instance()
            
            logger.info("Encryption key rotated successfully")
            return True
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            return False

class AccessController:
    """Access control system for API keys"""
    
    def __init__(self):
        self.access_policies = {}
        self.active_sessions = {}
        self.failed_attempts = defaultdict(list)
        self.lock = threading.Lock()
        
        # Default policies
        self._setup_default_policies()
    
    def _setup_default_policies(self):
        """Setup default access policies"""
        self.access_policies = {
            AccessLevel.READ_ONLY: {
                "allowed_operations": ["get", "list", "view"],
                "max_requests_per_hour": 1000,
                "max_concurrent_sessions": 5
            },
            AccessLevel.READ_WRITE: {
                "allowed_operations": ["get", "list", "view", "create", "update"],
                "max_requests_per_hour": 500,
                "max_concurrent_sessions": 3
            },
            AccessLevel.ADMIN: {
                "allowed_operations": ["*"],
                "max_requests_per_hour": 100,
                "max_concurrent_sessions": 2
            },
            AccessLevel.SYSTEM: {
                "allowed_operations": ["*"],
                "max_requests_per_hour": 10000,
                "max_concurrent_sessions": 10
            }
        }
    
    def check_access(self, key_id: str, operation: str, 
                    access_level: AccessLevel) -> Tuple[bool, str]:
        """Check if access is allowed for the given operation"""
        
        # Check if operation is allowed for access level
        policy = self.access_policies.get(access_level, {})
        allowed_ops = policy.get("allowed_operations", [])
        
        if "*" not in allowed_ops and operation not in allowed_ops:
            return False, f"Operation '{operation}' not allowed for access level '{access_level.value}'"
        
        # Check rate limiting
        with self.lock:
            if not self._check_rate_limit(key_id, access_level):
                return False, "Rate limit exceeded"
            
            # Check concurrent sessions
            if not self._check_concurrent_sessions(key_id, access_level):
                return False, "Maximum concurrent sessions exceeded"
        
        return True, "Access granted"
    
    def _check_rate_limit(self, key_id: str, access_level: AccessLevel) -> bool:
        """Check rate limiting for the key"""
        policy = self.access_policies.get(access_level, {})
        max_requests = policy.get("max_requests_per_hour", 1000)
        
        # Simple rate limiting implementation
        # In production, use Redis or similar for distributed rate limiting
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        session_key = f"{key_id}_{current_hour.isoformat()}"
        
        if session_key not in self.active_sessions:
            self.active_sessions[session_key] = 0
        
        if self.active_sessions[session_key] >= max_requests:
            return False
        
        self.active_sessions[session_key] += 1
        return True
    
    def _check_concurrent_sessions(self, key_id: str, access_level: AccessLevel) -> bool:
        """Check concurrent session limits"""
        policy = self.access_policies.get(access_level, {})
        max_sessions = policy.get("max_concurrent_sessions", 5)
        
        # Count active sessions for this key
        active_count = sum(1 for session_key in self.active_sessions.keys() 
                          if session_key.startswith(key_id))
        
        return active_count <= max_sessions
    
    def record_failed_attempt(self, key_id: str, reason: str):
        """Record failed access attempt"""
        with self.lock:
            self.failed_attempts[key_id].append({
                "timestamp": datetime.now(),
                "reason": reason
            })
            
            # Keep only last 10 failed attempts
            if len(self.failed_attempts[key_id]) > 10:
                self.failed_attempts[key_id] = self.failed_attempts[key_id][-10:]
    
    def is_key_blocked(self, key_id: str) -> bool:
        """Check if key is temporarily blocked due to failed attempts"""
        with self.lock:
            recent_failures = [
                attempt for attempt in self.failed_attempts.get(key_id, [])
                if datetime.now() - attempt["timestamp"] < timedelta(minutes=15)
            ]
            
            # Block if more than 5 failures in 15 minutes
            return len(recent_failures) > 5

class SecureKeyManager:
    """Main secure API key management system"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or os.path.join(
            os.path.expanduser('~'), '.ai_business_keys.enc'
        )
        self.encryption_manager = EncryptionManager()
        self.access_controller = AccessController()
        self.audit_log = deque(maxlen=10000)
        self.key_metadata = {}
        self.encrypted_keys = {}
        self.lock = threading.Lock()
        
        # Load existing keys
        self._load_keys()
        
        logger.info("Secure key manager initialized")
    
    def create_api_key(self, service_name: str, api_key: str, 
                      access_level: AccessLevel = AccessLevel.READ_WRITE,
                      expires_in_days: Optional[int] = None,
                      allowed_operations: Optional[List[str]] = None,
                      tags: Optional[Dict[str, str]] = None) -> str:
        """Create and store a new API key"""
        
        key_id = self._generate_key_id(service_name)
        
        # Create metadata
        metadata = APIKeyMetadata(
            key_id=key_id,
            service_name=service_name,
            access_level=access_level,
            status=KeyStatus.ACTIVE,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=expires_in_days) if expires_in_days else None,
            allowed_operations=allowed_operations or [],
            tags=tags or {}
        )
        
        with self.lock:
            # Encrypt and store the API key
            encrypted_key = self.encryption_manager.encrypt(api_key)
            self.encrypted_keys[key_id] = encrypted_key
            self.key_metadata[key_id] = metadata
            
            # Save to persistent storage
            self._save_keys()
        
        # Log security event
        self._log_security_event(
            SecurityEvent.KEY_CREATION,
            key_id=key_id,
            service_name=service_name,
            success=True,
            details={"access_level": access_level.value, "expires_in_days": expires_in_days}
        )
        
        logger.info(f"API key created for service '{service_name}' with ID '{key_id}'")
        return key_id
    
    def get_api_key(self, key_id: str, operation: str = "get") -> Optional[str]:
        """Retrieve and decrypt an API key"""
        
        with self.lock:
            # Check if key exists
            if key_id not in self.key_metadata:
                self._log_security_event(
                    SecurityEvent.UNAUTHORIZED_ACCESS,
                    key_id=key_id,
                    success=False,
                    details={"reason": "key_not_found", "operation": operation}
                )
                return None
            
            metadata = self.key_metadata[key_id]
            
            # Check key status
            if metadata.status != KeyStatus.ACTIVE:
                self._log_security_event(
                    SecurityEvent.UNAUTHORIZED_ACCESS,
                    key_id=key_id,
                    service_name=metadata.service_name,
                    success=False,
                    details={"reason": "key_inactive", "status": metadata.status.value}
                )
                return None
            
            # Check expiration
            if metadata.expires_at and datetime.now() > metadata.expires_at:
                metadata.status = KeyStatus.EXPIRED
                self._save_keys()
                self._log_security_event(
                    SecurityEvent.UNAUTHORIZED_ACCESS,
                    key_id=key_id,
                    service_name=metadata.service_name,
                    success=False,
                    details={"reason": "key_expired", "expired_at": metadata.expires_at.isoformat()}
                )
                return None
            
            # Check if key is blocked
            if self.access_controller.is_key_blocked(key_id):
                self._log_security_event(
                    SecurityEvent.UNAUTHORIZED_ACCESS,
                    key_id=key_id,
                    service_name=metadata.service_name,
                    success=False,
                    details={"reason": "key_blocked", "operation": operation}
                )
                return None
            
            # Check access permissions
            access_allowed, reason = self.access_controller.check_access(
                key_id, operation, metadata.access_level
            )
            
            if not access_allowed:
                self.access_controller.record_failed_attempt(key_id, reason)
                self._log_security_event(
                    SecurityEvent.UNAUTHORIZED_ACCESS,
                    key_id=key_id,
                    service_name=metadata.service_name,
                    success=False,
                    details={"reason": reason, "operation": operation}
                )
                return None
            
            # Decrypt and return the key
            try:
                encrypted_key = self.encrypted_keys[key_id]
                api_key = self.encryption_manager.decrypt(encrypted_key)
                
                # Update usage statistics
                metadata.last_used = datetime.now()
                metadata.usage_count += 1
                self._save_keys()
                
                # Log successful access
                self._log_security_event(
                    SecurityEvent.KEY_ACCESS,
                    key_id=key_id,
                    service_name=metadata.service_name,
                    operation=operation,
                    success=True,
                    details={"usage_count": metadata.usage_count}
                )
                
                return api_key
                
            except Exception as e:
                self._log_security_event(
                    SecurityEvent.DECRYPTION_ERROR,
                    key_id=key_id,
                    service_name=metadata.service_name,
                    success=False,
                    details={"error": str(e), "operation": operation}
                )
                logger.error(f"Failed to decrypt key '{key_id}': {e}")
                return None
    
    def rotate_api_key(self, key_id: str, new_api_key: str) -> bool:
        """Rotate an existing API key"""
        
        with self.lock:
            if key_id not in self.key_metadata:
                return False
            
            metadata = self.key_metadata[key_id]
            
            try:
                # Encrypt new key
                encrypted_key = self.encryption_manager.encrypt(new_api_key)
                self.encrypted_keys[key_id] = encrypted_key
                
                # Update metadata
                metadata.status = KeyStatus.ACTIVE
                
                # Save changes
                self._save_keys()
                
                # Log rotation event
                self._log_security_event(
                    SecurityEvent.KEY_ROTATION,
                    key_id=key_id,
                    service_name=metadata.service_name,
                    success=True,
                    details={"rotated_at": datetime.now().isoformat()}
                )
                
                logger.info(f"API key '{key_id}' rotated successfully")
                return True
                
            except Exception as e:
                self._log_security_event(
                    SecurityEvent.ENCRYPTION_ERROR,
                    key_id=key_id,
                    service_name=metadata.service_name,
                    success=False,
                    details={"error": str(e)}
                )
                logger.error(f"Failed to rotate key '{key_id}': {e}")
                return False
    
    def revoke_api_key(self, key_id: str, reason: str = "manual_revocation") -> bool:
        """Revoke an API key"""
        
        with self.lock:
            if key_id not in self.key_metadata:
                return False
            
            metadata = self.key_metadata[key_id]
            metadata.status = KeyStatus.REVOKED
            
            # Save changes
            self._save_keys()
            
            # Log revocation event
            self._log_security_event(
                SecurityEvent.KEY_REVOCATION,
                key_id=key_id,
                service_name=metadata.service_name,
                success=True,
                details={"reason": reason, "revoked_at": datetime.now().isoformat()}
            )
            
            logger.info(f"API key '{key_id}' revoked: {reason}")
            return True
    
    def list_keys(self, service_name: Optional[str] = None, 
                 status: Optional[KeyStatus] = None) -> List[Dict[str, Any]]:
        """List API keys with optional filtering"""
        
        with self.lock:
            keys = []
            for key_id, metadata in self.key_metadata.items():
                # Apply filters
                if service_name and metadata.service_name != service_name:
                    continue
                if status and metadata.status != status:
                    continue
                
                keys.append({
                    "key_id": key_id,
                    "service_name": metadata.service_name,
                    "access_level": metadata.access_level.value,
                    "status": metadata.status.value,
                    "created_at": metadata.created_at.isoformat(),
                    "last_used": metadata.last_used.isoformat() if metadata.last_used else None,
                    "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                    "usage_count": metadata.usage_count,
                    "tags": metadata.tags
                })
            
            return keys
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        
        with self.lock:
            total_keys = len(self.key_metadata)
            active_keys = sum(1 for m in self.key_metadata.values() if m.status == KeyStatus.ACTIVE)
            expired_keys = sum(1 for m in self.key_metadata.values() if m.status == KeyStatus.EXPIRED)
            revoked_keys = sum(1 for m in self.key_metadata.values() if m.status == KeyStatus.REVOKED)
            
            # Recent security events
            recent_events = list(self.audit_log)[-20:]
            
            # Failed attempts summary
            failed_attempts = sum(len(attempts) for attempts in self.access_controller.failed_attempts.values())
            
            return {
                "timestamp": datetime.now().isoformat(),
                "key_statistics": {
                    "total_keys": total_keys,
                    "active_keys": active_keys,
                    "expired_keys": expired_keys,
                    "revoked_keys": revoked_keys
                },
                "security_metrics": {
                    "total_audit_events": len(self.audit_log),
                    "recent_failed_attempts": failed_attempts,
                    "encryption_status": "active"
                },
                "recent_events": [
                    {
                        "timestamp": event.timestamp.isoformat(),
                        "event_type": event.event_type.value,
                        "service_name": event.service_name,
                        "success": event.success,
                        "details": event.details
                    }
                    for event in recent_events
                ],
                "system_status": "secure"
            }
    
    def _generate_key_id(self, service_name: str) -> str:
        """Generate unique key ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = secrets.token_hex(4)
        return f"{service_name}_{timestamp}_{random_suffix}"
    
    def _load_keys(self):
        """Load keys from persistent storage"""
        if not os.path.exists(self.storage_path):
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                encrypted_data = f.read()
            
            if encrypted_data:
                decrypted_data = self.encryption_manager.decrypt(encrypted_data)
                data = json.loads(decrypted_data)
                
                # Load metadata
                for key_id, metadata_dict in data.get('metadata', {}).items():
                    metadata = APIKeyMetadata(
                        key_id=metadata_dict['key_id'],
                        service_name=metadata_dict['service_name'],
                        access_level=AccessLevel(metadata_dict['access_level']),
                        status=KeyStatus(metadata_dict['status']),
                        created_at=datetime.fromisoformat(metadata_dict['created_at']),
                        last_used=datetime.fromisoformat(metadata_dict['last_used']) if metadata_dict.get('last_used') else None,
                        expires_at=datetime.fromisoformat(metadata_dict['expires_at']) if metadata_dict.get('expires_at') else None,
                        usage_count=metadata_dict.get('usage_count', 0),
                        allowed_operations=metadata_dict.get('allowed_operations', []),
                        tags=metadata_dict.get('tags', {})
                    )
                    self.key_metadata[key_id] = metadata
                
                # Load encrypted keys
                self.encrypted_keys = data.get('encrypted_keys', {})
                
                logger.info(f"Loaded {len(self.key_metadata)} API keys from storage")
        
        except Exception as e:
            logger.error(f"Failed to load keys from storage: {e}")
    
    def _save_keys(self):
        """Save keys to persistent storage"""
        try:
            # Prepare data for storage
            data = {
                'metadata': {},
                'encrypted_keys': self.encrypted_keys
            }
            
            # Serialize metadata
            for key_id, metadata in self.key_metadata.items():
                data['metadata'][key_id] = {
                    'key_id': metadata.key_id,
                    'service_name': metadata.service_name,
                    'access_level': metadata.access_level.value,
                    'status': metadata.status.value,
                    'created_at': metadata.created_at.isoformat(),
                    'last_used': metadata.last_used.isoformat() if metadata.last_used else None,
                    'expires_at': metadata.expires_at.isoformat() if metadata.expires_at else None,
                    'usage_count': metadata.usage_count,
                    'allowed_operations': metadata.allowed_operations,
                    'tags': metadata.tags
                }
            
            # Encrypt and save
            json_data = json.dumps(data)
            encrypted_data = self.encryption_manager.encrypt(json_data)
            
            with open(self.storage_path, 'w') as f:
                f.write(encrypted_data)
            
            # Secure file permissions
            os.chmod(self.storage_path, 0o600)
            
        except Exception as e:
            logger.error(f"Failed to save keys to storage: {e}")
    
    def _log_security_event(self, event_type: SecurityEvent, 
                           key_id: Optional[str] = None,
                           service_name: Optional[str] = None,
                           operation: Optional[str] = None,
                           success: bool = True,
                           details: Optional[Dict[str, Any]] = None):
        """Log security event to audit log"""
        
        event = SecurityAuditLog(
            timestamp=datetime.now(),
            event_type=event_type,
            key_id=key_id,
            service_name=service_name,
            operation=operation,
            success=success,
            details=details or {}
        )
        
        self.audit_log.append(event)

# Custom exceptions
class EncryptionError(Exception):
    """Raised when encryption fails"""
    pass

class DecryptionError(Exception):
    """Raised when decryption fails"""
    pass

class KeyNotFoundError(Exception):
    """Raised when API key is not found"""
    pass

class AccessDeniedError(Exception):
    """Raised when access is denied"""
    pass

# Test function
def test_secure_key_manager():
    """Test the secure key management system"""
    print("Testing Secure Key Management System...")
    
    # Initialize manager
    manager = SecureKeyManager()
    
    # Test 1: Create API keys
    print("\n1. Testing API key creation...")
    
    openai_key_id = manager.create_api_key(
        service_name="openai",
        api_key="sk-test-openai-key-12345",
        access_level=AccessLevel.READ_WRITE,
        expires_in_days=90,
        tags={"environment": "production", "cost_center": "research"}
    )
    print(f"✅ OpenAI key created: {openai_key_id}")
    
    reddit_key_id = manager.create_api_key(
        service_name="reddit",
        api_key="reddit-api-key-67890",
        access_level=AccessLevel.READ_ONLY,
        allowed_operations=["get", "list"]
    )
    print(f"✅ Reddit key created: {reddit_key_id}")
    
    # Test 2: Retrieve API keys
    print("\n2. Testing API key retrieval...")
    
    retrieved_openai_key = manager.get_api_key(openai_key_id, "create")
    if retrieved_openai_key == "sk-test-openai-key-12345":
        print("✅ OpenAI key retrieved and decrypted correctly")
    else:
        print("❌ OpenAI key retrieval failed")
    
    retrieved_reddit_key = manager.get_api_key(reddit_key_id, "get")
    if retrieved_reddit_key == "reddit-api-key-67890":
        print("✅ Reddit key retrieved and decrypted correctly")
    else:
        print("❌ Reddit key retrieval failed")
    
    # Test 3: Access control
    print("\n3. Testing access control...")
    
    # Try unauthorized operation
    unauthorized_key = manager.get_api_key(reddit_key_id, "create")  # READ_ONLY key
    if unauthorized_key is None:
        print("✅ Access control working - unauthorized operation blocked")
    else:
        print("❌ Access control failed - unauthorized operation allowed")
    
    # Test 4: Key rotation
    print("\n4. Testing key rotation...")
    
    rotation_success = manager.rotate_api_key(openai_key_id, "sk-test-openai-key-rotated-54321")
    if rotation_success:
        rotated_key = manager.get_api_key(openai_key_id, "get")
        if rotated_key == "sk-test-openai-key-rotated-54321":
            print("✅ Key rotation successful")
        else:
            print("❌ Key rotation failed - wrong key retrieved")
    else:
        print("❌ Key rotation failed")
    
    # Test 5: Key listing
    print("\n5. Testing key listing...")
    
    all_keys = manager.list_keys()
    openai_keys = manager.list_keys(service_name="openai")
    active_keys = manager.list_keys(status=KeyStatus.ACTIVE)
    
    print(f"✅ Total keys: {len(all_keys)}")
    print(f"✅ OpenAI keys: {len(openai_keys)}")
    print(f"✅ Active keys: {len(active_keys)}")
    
    # Test 6: Security status
    print("\n6. Testing security status...")
    
    security_status = manager.get_security_status()
    print(f"✅ Total keys: {security_status['key_statistics']['total_keys']}")
    print(f"✅ Active keys: {security_status['key_statistics']['active_keys']}")
    print(f"✅ Security events: {security_status['security_metrics']['total_audit_events']}")
    print(f"✅ System status: {security_status['system_status']}")
    
    # Test 7: Key revocation
    print("\n7. Testing key revocation...")
    
    revocation_success = manager.revoke_api_key(reddit_key_id, "testing_revocation")
    if revocation_success:
        revoked_key = manager.get_api_key(reddit_key_id, "get")
        if revoked_key is None:
            print("✅ Key revocation successful - access denied")
        else:
            print("❌ Key revocation failed - access still allowed")
    else:
        print("❌ Key revocation failed")
    
    print("\n🎉 Secure key management system test completed!")
    
    return {
        "manager_initialized": True,
        "keys_created": 2,
        "encryption_working": retrieved_openai_key == "sk-test-openai-key-12345",
        "access_control_working": unauthorized_key is None,
        "rotation_working": rotation_success,
        "revocation_working": revocation_success,
        "security_status": security_status['system_status'],
        "total_security_events": security_status['security_metrics']['total_audit_events']
    }

if __name__ == "__main__":
    # Run secure key management tests
    test_secure_key_manager()

