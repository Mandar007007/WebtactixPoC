"""Centralized configuration loader for enterprise app configs.

Loads JSON config files (e.g. veeva_config.json, jira_config.json)
and provides typed access to users, app URLs, and other settings.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UserCredentials:
    """Login credentials for a single user role.

    Attributes:
        role: The role key (e.g. "Owner", "Editor1").
        username: Login username / email.
        password: Account password.
    """

    role: str
    username: str
    password: str


class ConfigManager:
    """Loads and provides typed access to an app config JSON.

    Typical usage::

        cfg = ConfigManager.from_file("veeva_config.json")
        creds = cfg.get_user("Owner")
        print(creds.username, creds.password)

    Attributes:
        raw: The full parsed JSON dict for direct access.
    """

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw

    @classmethod
    def from_file(
        cls,
        path: str | Path = "veeva_config.json",
    ) -> ConfigManager:
        """Load configuration from a JSON file.

        Args:
            path: Path to the JSON config file.

        Returns:
            A ConfigManager instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        config_path = Path(path)
        with open(config_path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    @property
    def app_url(self) -> str:
        """Base application URL."""
        return str(self.raw.get("app_url", ""))

    @property
    def app_name(self) -> str:
        """Application display name."""
        return str(self.raw.get("app_name", ""))

    @property
    def available_roles(self) -> list[str]:
        """List of all user role keys in the config."""
        users = self.raw.get("users", {})
        return list(users.keys())

    def get_user(self, role: str) -> UserCredentials:
        """Look up credentials for a user role.

        Args:
            role: The role key (e.g. "Owner", "Editor1",
                "DocControl"). Case-sensitive.

        Returns:
            UserCredentials with username and password.

        Raises:
            KeyError: If the role is not found in the config.
        """
        users = self.raw.get("users", {})
        if role not in users:
            raise KeyError(
                f"User role '{role}' not found in config. "
                f"Available roles: {list(users.keys())}"
            )
        entry = users[role]
        return UserCredentials(
            role=role,
            username=entry["username"],
            password=entry["password"],
        )

    def extract_login_role(
        self,
        goal: str,
    ) -> tuple[str | None, str]:
        """Parse a login role from goal text and return clean goal.

        Scans for patterns like "login with Owner",
        "login as Editor1", "login using DocControl",
        "sign in as QA", or quoted variants like
        'login with "Owner"'. Only matches role names
        that exist in this config.

        Args:
            goal: The raw goal / intent text.

        Returns:
            A tuple of (matched_role, cleaned_goal).
            matched_role is None if no login instruction
            was found. cleaned_goal has the login phrase
            stripped so the planner can focus on the real
            task.
        """
        roles = self.available_roles
        if not roles:
            return None, goal

        escaped = [re.escape(r) for r in roles]
        role_alt = "|".join(escaped)

        pattern = re.compile(
            r'(?:log\s*in|login|sign\s*in)\s+'
            r'(?:with|as|using)\s+'
            rf'["\']?({role_alt})["\']?',
            re.IGNORECASE,
        )
        match = pattern.search(goal)
        if match is None:
            return None, goal

        matched_text = match.group(0)
        raw_role = match.group(1)

        canonical = next(
            (r for r in roles if r.lower() == raw_role.lower()),
            None,
        )
        if canonical is None:
            return None, goal

        cleaned = goal.replace(matched_text, "").strip()
        cleaned = re.sub(
            r'^[\s,.\-;:and]+|[\s,.\-;:and]+$',
            '',
            cleaned,
        )

        return canonical, cleaned

    def get(self, key: str, default: Any = None) -> Any:
        """Get any top-level config value by key.

        Args:
            key: The config key to look up.
            default: Value to return if the key is missing.

        Returns:
            The config value, or default if not found.
        """
        return self.raw.get(key, default)
