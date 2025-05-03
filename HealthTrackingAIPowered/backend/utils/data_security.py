"""
Data security utilities
- Encryption/decryption
- Key management
- Secure storage
"""

from cryptography.fernet import Fernet
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

KEY_FILE = "data/secret.key"

def generate_key():
    """Generate encryption key if not exists"""
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        Path(KEY_FILE).write_bytes(key)
        logger.info("Generated new encryption key")

def get_cipher():
    """Get Fernet cipher instance"""
    generate_key()
    return Fernet(Path(KEY_FILE).read_bytes())

def encrypt_data(data: dict) -> bytes:
    """Encrypt dictionary data"""
    try:
        cipher = get_cipher()
        return cipher.encrypt(json.dumps(data).encode())
    except Exception as e:
        logger.error(f"Encryption failed: {str(e)}")
        raise

def decrypt_data(encrypted_data: bytes) -> dict:
    """Decrypt data to dictionary"""
    try:
        cipher = get_cipher()
        return json.loads(cipher.decrypt(encrypted_data).decode())
    except Exception as e:
        logger.error(f"Decryption failed: {str(e)}")
        raise