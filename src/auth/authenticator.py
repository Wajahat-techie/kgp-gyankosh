"""
Authentication & Access Control Layer for KGP Gyankosh
Enforces role-based internal access control for college administration staff
using streamlit-authenticator with secure bcrypt-hashed passwords.
Credentials are kept strictly in local gitignored configuration.
"""

import os
import yaml
import bcrypt
import logging
from typing import Tuple, Dict, Any, Optional
import streamlit as st
import streamlit_authenticator as stauth

logger = logging.getLogger("kgp_gyankosh.auth")


def hash_password(password: str) -> str:
    """Utility function to hash raw passwords with bcrypt for addition to auth_config.yaml."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


@st.cache_data(show_spinner=False)
def load_auth_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads authentication YAML configuration.
    If the real config does not exist, copies from auth_config.yaml.example.
    """
    path = config_path or os.getenv("AUTH_CONFIG_PATH", "config/auth_config.yaml")
    
    # Resolve relative to project root
    if not os.path.isabs(path):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        path = os.path.join(base_dir, path)

    if not os.path.exists(path):
        example_path = path + ".example"
        if os.path.exists(example_path):
            import shutil
            shutil.copy(example_path, path)
            logger.info(f"Initialized auth_config.yaml from template {example_path}")
        else:
            raise FileNotFoundError(f"Authentication config file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config


def get_authenticator(config: Optional[Dict[str, Any]] = None) -> stauth.Authenticate:
    """
    Initializes and returns a Streamlit Authenticate instance.
    """
    if config is None:
        config = load_auth_config()

    authenticator = stauth.Authenticate(
        credentials=config["credentials"],
        cookie_name=config["cookie"]["name"],
        key=config["cookie"]["key"],
        cookie_expiry_days=config["cookie"]["expiry_days"]
    )
    return authenticator
