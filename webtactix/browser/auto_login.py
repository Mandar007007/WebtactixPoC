"""In-session Veeva Vault login using credentials from config.

Performs the full login flow (username -> continue -> password ->
submit) on an already-open Playwright page, so the agent can
handle "login with Owner" style goals without a separate
pre-login step.
"""
from __future__ import annotations

from pathlib import Path

from playwright.async_api import Page

from webtactix.core.config_manager import (
    ConfigManager,
    UserCredentials,
)


async def perform_veeva_login(
    page: Page,
    *,
    app_url: str,
    creds: UserCredentials,
) -> Page:
    """Drive the Veeva Vault login flow on an existing page.

    Args:
        page: An open Playwright page.
        app_url: The Veeva login URL (e.g.
            ``https://login.veevavault.com/``).
        creds: Username and password to use.

    Returns:
        The same page, now authenticated and on the vault
        dashboard.
    """
    login_url = app_url.rstrip("/")
    print(
        f"[AUTO-LOGIN] Logging in as '{creds.role}' "
        f"({creds.username})"
    )
    await page.goto(
        login_url, wait_until="domcontentloaded",
    )
    await page.wait_for_timeout(5000)

    user_input = page.locator(
        "input[name='username'], "
        "input[id='username'], "
        "input[type='text']"
    ).first
    await user_input.wait_for(
        state="visible", timeout=30000,
    )
    await user_input.fill(creds.username)

    print("[AUTO-LOGIN] Clicking Continue...")
    continue_btn = page.locator(
        "button[name='continue']",
    )
    await continue_btn.click()
    await page.wait_for_timeout(5000)

    pwd_input = page.locator(
        "input[name='password'], "
        "input[id='password'], "
        "input[type='password']"
    ).first
    await pwd_input.wait_for(
        state="visible", timeout=15000,
    )
    await pwd_input.fill(creds.password)

    print("[AUTO-LOGIN] Submitting login...")
    login_btn = page.locator(
        "button[type='submit'], "
        "input[type='submit'], "
        "button:has-text('Log In'), "
        "button:has-text('Login'), "
        "button:has-text('Sign In'), "
        "#login-btn"
    ).first
    await login_btn.click()

    await page.wait_for_timeout(10000)

    target_host = login_url.split("//")[-1]
    if target_host not in page.url:
        print(
            "[AUTO-LOGIN] Waiting for vault redirect..."
        )
        try:
            await page.wait_for_url(
                f"**{target_host}**", timeout=30000,
            )
        except Exception:
            await page.wait_for_timeout(10000)

    await page.wait_for_timeout(3000)
    print(f"[AUTO-LOGIN] Logged in at {page.url}")
    return page


async def auto_login_if_needed(
    page: Page,
    *,
    goal: str,
    config_path: str | Path = "veeva_config.json",
) -> tuple[Page, str]:
    """Check the goal for a login instruction and handle it.

    If the goal contains a phrase like "login with Owner",
    this function looks up the credentials from the config
    and performs the full login. The login phrase is stripped
    from the returned goal so the planner focuses on the
    real task.

    Args:
        page: An open Playwright page.
        goal: The raw intent / goal text.
        config_path: Path to the app config JSON.

    Returns:
        A tuple of (page, cleaned_goal). If no login was
        needed, the goal is returned unchanged.
    """
    try:
        cfg = ConfigManager.from_file(config_path)
    except (FileNotFoundError, OSError):
        return page, goal

    role, cleaned_goal = cfg.extract_login_role(goal)

    if role is None:
        return page, goal

    creds = cfg.get_user(role)
    page = await perform_veeva_login(
        page,
        app_url=cfg.app_url,
        creds=creds,
    )
    return page, cleaned_goal
