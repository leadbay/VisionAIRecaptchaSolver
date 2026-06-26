"""Asynchronous RecaptchaSolver implementation."""

from __future__ import annotations

import asyncio
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from vision_ai_recaptcha_solver.browser.navigation import (
    click_checkbox,
    click_reload_button,
    click_verify_button,
    get_challenge_title,
    get_target_keyword,
    is_solved,
    wait_for_challenge_tiles,
    wait_for_verify_result,
)
from vision_ai_recaptcha_solver.captcha.dynamic_handler import DynamicCaptchaHandler
from vision_ai_recaptcha_solver.captcha.selection_handler import SelectionCaptchaHandler
from vision_ai_recaptcha_solver.captcha.square_handler import SquareCaptchaHandler
from vision_ai_recaptcha_solver.config import SolverConfig
from vision_ai_recaptcha_solver.detector.yolo_detector import YOLODetector
from vision_ai_recaptcha_solver.exceptions import (
    CaptchaNotFoundError,
    ElementNotFoundError,
    LowConfidenceError,
    RecaptchaSolverError,
    TokenExtractionError,
    UnsupportedCaptchaError,
)
from vision_ai_recaptcha_solver.logging_config import setup_logging
from vision_ai_recaptcha_solver.resource_allocation import (
    release_solver_resources,
    reserve_solver_resources,
)
from vision_ai_recaptcha_solver.types import CaptchaType, SolveResult
from vision_ai_recaptcha_solver.utils import human_delay

if TYPE_CHECKING:
    from vision_ai_recaptcha_solver.captcha.base_handler import BaseCaptchaHandler

_DOWNLOAD_DIR_MARKER = ".vision_ai_recaptcha_solver_owned"


class AsyncRecaptchaSolver:
    """Asynchronous reCAPTCHA solver using YOLO object detection.

    This class provides an async interface for solving reCAPTCHA challenges.
    Browser operations run in a thread pool to avoid blocking the event loop.

    Example:
        ```python
        import asyncio
        from vision_ai_recaptcha_solver import AsyncRecaptchaSolver, SolverConfig

        async def main():
            config = SolverConfig(timeout=120)

            async with AsyncRecaptchaSolver(config) as solver:
                result = await solver.solve(
                    website_key="6Le-wvkSAAAAAPBMRTvw0Q4Muexq9bi0DJwx_mJ-",
                    website_url="https://www.google.com/recaptcha/api2/demo"
                )
                print(f"Token: {result.token}")

        asyncio.run(main())
        ```
    """

    def __init__(self, config: SolverConfig | None = None) -> None:
        """Initialize the AsyncRecaptchaSolver.

        Args:
            config: Solver configuration. If None, uses default configuration.
        """
        self.config = config or SolverConfig()
        self.logger = setup_logging(self.config.log_level, "vision_ai_recaptcha_solver")
        reserve_solver_resources(self, self.config, self.logger)
        self._closed: bool = False
        self._executor: ThreadPoolExecutor | None = None
        self._detector: YOLODetector | None = None
        self._handlers: dict[CaptchaType, BaseCaptchaHandler] | None = None
        self._replicator: Any = None
        self._owns_download_dir: bool = False
        self.last_attempt_trace: list[dict[str, object]] = []
        self.debug_artifacts: list[str] = []
        self._init_download_dir()

    def _init_download_dir(self) -> None:
        """Ensure download_dir exists and mark ownership if created by solver."""
        download_dir = self.config.download_dir
        marker_path = download_dir / _DOWNLOAD_DIR_MARKER

        if download_dir.exists():
            if self.config._download_dir_explicit:
                self._owns_download_dir = False
            else:
                self._owns_download_dir = marker_path.exists()
            return

        try:
            download_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.logger.debug("Failed to create download directory '%s': %s", download_dir, e)
            return

        if not self.config._download_dir_explicit and self.config.cleanup_tmp_on_close:
            try:
                marker_path.write_text("owned\n", encoding="utf-8")
                self._owns_download_dir = True
            except OSError as e:
                self.logger.debug("Failed to create download dir marker '%s': %s", marker_path, e)

    async def __aenter__(self) -> AsyncRecaptchaSolver:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager and cleanup resources."""
        await self.close()

    async def close(self) -> None:
        """Close the async solver and cleanup resources."""
        if self._closed:
            return
        self._closed = True

        # Cleanup replicator in thread pool
        if self._replicator:
            await self._run_in_executor(self._cleanup_replicator_sync)

        # Cleanup temporary directory
        if self.config.cleanup_tmp_on_close:
            await self._run_in_executor(self._cleanup_tmp_directory_sync)

        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

        release_solver_resources(self)

    def _cleanup_replicator_sync(self) -> None:
        """Synchronously cleanup the replicator."""
        if self._replicator:
            try:
                self._replicator.close_browser()
            except Exception as e:
                self.logger.debug(f"Error closing browser: {e}")
            try:
                self._replicator.stop_http_server()
            except Exception as e:
                self.logger.debug(f"Error stopping server: {e}")
            self._replicator = None

    def _cleanup_tmp_directory_sync(self) -> None:
        """Cleanup the temporary download directory."""
        if not self._owns_download_dir:
            self.logger.debug(
                "Skipping download directory cleanup: %s",
                self.config.download_dir,
            )
            return
        try:
            if self.config.download_dir.exists():
                shutil.rmtree(self.config.download_dir)
                self.logger.debug(f"Cleaned up temporary directory: {self.config.download_dir}")
        except Exception as e:
            self.logger.debug(f"Error cleaning up temporary directory: {e}")

    def _get_executor(self) -> ThreadPoolExecutor:
        """Get or create the thread pool executor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="async_solver")
        return self._executor

    async def _run_in_executor(self, func: Any, *args: Any) -> Any:
        """Run a synchronous function in the thread pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._get_executor(), partial(func, *args))

    def _init_detector_and_handlers(self) -> None:
        """Initialize detector and handlers (sync, runs in thread pool)."""
        self._detector = YOLODetector(
            model_path=self.config.model_path,
            detection_model_path=self.config.detection_model_path,
            verbose=self.config.verbose,
            logger=self.logger,
            conf_threshold=self.config.conf_threshold,
            fourth_cell_threshold=self.config.fourth_cell_threshold,
            detection_conf_threshold=self.config.detection_conf_threshold,
        )

        # Initialize handlers
        self._handlers = {
            CaptchaType.DYNAMIC_3X3: DynamicCaptchaHandler(
                self._detector, self.config, self.logger
            ),
            CaptchaType.SELECTION_3X3: SelectionCaptchaHandler(
                self._detector, self.config, self.logger
            ),
            CaptchaType.SQUARE_4X4: SquareCaptchaHandler(self._detector, self.config, self.logger),
        }

    async def solve(
        self,
        website_key: str,
        website_url: str,
        *,
        is_invisible: bool = False,
        action: str | None = None,
        is_enterprise: bool = False,
        api_domain: str = "google.com",
        bypass_domain_check: bool = True,
        use_ssl: bool = True,
        cookies: list[dict[str, Any]] | None = None,
        user_agent: str | None = None,
    ) -> SolveResult:
        """Asynchronously solve a reCAPTCHA challenge.

        Args:
            website_key: The reCAPTCHA site key.
            website_url: The URL of the page containing the captcha.
            is_invisible: Whether the captcha is invisible reCAPTCHA.
            action: Action name for invisible reCAPTCHA v3.
            is_enterprise: Whether this is an enterprise reCAPTCHA.
            api_domain: API domain (google.com or recaptcha.net).
            bypass_domain_check: Whether to bypass domain verification.
            use_ssl: Whether to use HTTPS for the local server.
            cookies: Optional cookies to set in the browser.
            user_agent: Optional custom user agent string.

        Returns:
            SolveResult with token, cookies, and timing information.

        Raises:
            RecaptchaSolverError: If solving fails or solver is closed.
            TokenExtractionError: If token cannot be extracted.
        """
        self.last_attempt_trace = []
        self.debug_artifacts = []

        # Validate inputs
        if not website_key or not website_key.strip():
            raise ValueError("website_key cannot be empty")
        if not website_url or not website_url.strip():
            raise ValueError("website_url cannot be empty")

        parsed = urlparse(website_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL format: {website_url}")

        if self._closed:
            raise RecaptchaSolverError("Solver has been closed")

        start_time = time.time()
        attempts = 0
        last_captcha_type = CaptchaType.UNKNOWN

        try:
            if self._detector is None:
                await self._run_in_executor(self._init_detector_and_handlers)

            # Cleanup previous replicator
            if self._replicator:
                await self._run_in_executor(self._cleanup_replicator_sync)

            # Initialize browser
            browser, token_handle = await self._run_in_executor(
                self._init_browser,
                website_key,
                website_url,
                is_invisible,
                action,
                is_enterprise,
                api_domain,
                bypass_domain_check,
                use_ssl,
                cookies,
                user_agent,
            )

            if not browser:
                raise CaptchaNotFoundError("Failed to initialize browser session")

            # reCAPTCHA v3 (invisible), no challenge to solve
            if is_invisible:
                self.logger.info("Invisible reCAPTCHA (v3) detected - waiting for token...")
                last_captcha_type = CaptchaType.INVISIBLE

                token = await self._run_in_executor(
                    lambda: token_handle.wait(timeout=self.config.timeout) if token_handle else None
                )

                if not token:
                    raise TokenExtractionError("Failed to extract reCAPTCHA v3 token")

                result_cookies = await self._run_in_executor(self._get_cookies, browser)
                time_taken = round(time.time() - start_time, 2)

                self.logger.info("reCAPTCHA token obtained successfully!")
                return SolveResult(
                    token=token,
                    cookies=result_cookies,
                    time_taken=time_taken,
                    captcha_type=CaptchaType.INVISIBLE,
                    attempts=0,
                )

            await self._run_in_executor(human_delay, 0.8, 0.2)

            # Click checkbox to trigger the challenge
            try:
                await self._run_in_executor(click_checkbox, browser)
            except ElementNotFoundError as e:
                raise CaptchaNotFoundError(f"Could not find captcha checkbox: {e}") from e

            await self._run_in_executor(human_delay, 0.5, 0.3)

            # Check if captcha was solved immediately
            if await self._run_in_executor(is_solved, browser, 2):
                self.logger.info("Captcha solved immediately on checkbox click (no challenge)")

                token = await self._run_in_executor(
                    lambda: token_handle.wait(timeout=self.config.timeout) if token_handle else None
                )

                if not token:
                    raise TokenExtractionError(
                        "Failed to extract reCAPTCHA token after immediate solve"
                    )

                result_cookies = await self._run_in_executor(self._get_cookies, browser)
                time_taken = round(time.time() - start_time, 2)

                return SolveResult(
                    token=token,
                    cookies=result_cookies,
                    time_taken=time_taken,
                    captcha_type=CaptchaType.NO_CHALLENGE,
                    attempts=0,
                )

            # Ensure model warmup is complete before detection
            assert self._detector is not None
            await self._run_in_executor(self._detector.ensure_warmup_complete)

            # Solve loop
            solved_successfully = False
            attempt_trace: list[dict[str, object]] = []
            self.last_attempt_trace = attempt_trace
            while attempts < self.config.max_attempts:
                attempts += 1
                self.logger.debug(f"Solve attempt {attempts}/{self.config.max_attempts}")

                handler: BaseCaptchaHandler | None = None
                try:
                    # Determine captcha type and get target
                    captcha_type = await self._run_in_executor(
                        self._determine_captcha_type, browser
                    )
                    last_captcha_type = captcha_type
                    keyword, target_class, coco_target_class = await self._run_in_executor(
                        self._get_target_info, browser, captcha_type
                    )
                    attempt_info: dict[str, object] = {
                        "attempt": attempts,
                        "captcha_type": captcha_type.value,
                        "keyword": keyword,
                        "target_class": target_class,
                        "coco_target_class": coco_target_class,
                    }
                    attempt_trace.append(attempt_info)
                    self.logger.info("Attempt %s target info: %s", attempts, attempt_info)
                    before_screenshot = await self._run_in_executor(
                        self._capture_browser_screenshot_sync,
                        browser,
                        attempts,
                        "before-click",
                    )
                    if before_screenshot:
                        attempt_info["before_screenshot"] = before_screenshot

                    if target_class is None:
                        attempt_info["action"] = "reload_unknown_target"
                        self.logger.info("Unknown target, reloading captcha")
                        await self._run_in_executor(click_reload_button, browser)
                        await self._run_in_executor(
                            human_delay,
                            self.config.human_delay_mean,
                            self.config.human_delay_sigma,
                        )
                        # Get new challenge
                        await self._run_in_executor(
                            wait_for_challenge_tiles,
                            browser,
                            self.config.default_timeout,
                        )
                        continue

                    if captcha_type == CaptchaType.SQUARE_4X4 and coco_target_class is None:
                        attempt_info["action"] = "reload_unsupported_4x4_target"
                        self.logger.info(
                            "Unsupported 4x4 target '%s' for COCO detection, reloading captcha",
                            keyword,
                        )
                        await self._run_in_executor(click_reload_button, browser)
                        await self._run_in_executor(
                            human_delay,
                            self.config.human_delay_mean,
                            self.config.human_delay_sigma,
                        )
                        await self._run_in_executor(
                            wait_for_challenge_tiles,
                            browser,
                            self.config.default_timeout,
                        )
                        continue

                    # Get handler and solve
                    handler = self._get_handler(captcha_type)
                    clicked_cells = await self._run_in_executor(
                        handler.solve, browser, target_class
                    )
                    attempt_info["clicked_cells"] = clicked_cells
                    if getattr(handler, "last_debug", None):
                        attempt_info["detection_debug"] = handler.last_debug
                        self._refresh_debug_artifacts()

                    if not clicked_cells:
                        attempt_info["action"] = "reload_no_cells_clicked"
                        self.logger.info("No cells clicked, reloading")
                        await self._run_in_executor(click_reload_button, browser)
                        await self._run_in_executor(
                            human_delay,
                            self.config.human_delay_mean,
                            self.config.human_delay_sigma,
                        )
                        await self._run_in_executor(
                            wait_for_challenge_tiles,
                            browser,
                            self.config.default_timeout,
                        )
                        continue

                    after_screenshot = await self._run_in_executor(
                        self._capture_browser_screenshot_sync,
                        browser,
                        attempts,
                        "after-click",
                    )
                    if after_screenshot:
                        attempt_info["after_screenshot"] = after_screenshot

                    # Click verify
                    await self._run_in_executor(human_delay, 0.3, 0.2)
                    await self._run_in_executor(click_verify_button, browser)

                    # Wait for verify result
                    if await self._run_in_executor(
                        wait_for_verify_result, browser, self.config.default_timeout
                    ):
                        attempt_info["action"] = "verify_solved"
                        solved_successfully = True
                        self.logger.info("Captcha solved successfully!")
                        break

                    attempt_info["action"] = "verify_not_solved"
                    self.logger.info("Attempt %s verify did not solve captcha", attempts)

                    # Not solved, continue to next attempt
                    await self._run_in_executor(human_delay, 0.2, 0.1)

                except LowConfidenceError as e:
                    if attempt_trace:
                        attempt_trace[-1]["action"] = "reload_low_confidence"
                        attempt_trace[-1]["error"] = str(e)
                        if handler is not None and getattr(handler, "last_debug", None):
                            attempt_trace[-1]["detection_debug"] = handler.last_debug
                            self._refresh_debug_artifacts()
                    self.logger.info(f"Low confidence detection, reloading: {e}")
                    await self._run_in_executor(click_reload_button, browser)
                    await self._run_in_executor(
                        human_delay,
                        self.config.human_delay_mean,
                        self.config.human_delay_sigma,
                    )
                    await self._run_in_executor(
                        wait_for_challenge_tiles,
                        browser,
                        self.config.default_timeout,
                    )

                except (ElementNotFoundError, UnsupportedCaptchaError) as e:
                    if attempt_trace:
                        attempt_trace[-1]["action"] = "attempt_exception"
                        attempt_trace[-1]["error"] = str(e)
                        if handler is not None and getattr(handler, "last_debug", None):
                            attempt_trace[-1]["detection_debug"] = handler.last_debug
                            self._refresh_debug_artifacts()
                    self.logger.warning(f"Attempt {attempts} failed: {e}")
                    await self._run_in_executor(human_delay, 0.5, 0.1)

            if not solved_successfully and not await self._run_in_executor(is_solved, browser, 1):
                captcha_type_value = last_captcha_type.value if last_captcha_type else "unknown"
                raise TokenExtractionError(
                    "Captcha not solved after "
                    f"{attempts}/{self.config.max_attempts} attempts "
                    f"(last_captcha_type={captcha_type_value}, trace={attempt_trace})"
                )

            # Extract token. Once the visible challenge is solved, token propagation
            # should be quick; do not burn the full solver timeout here because that
            # turns unsupported/failed challenges into Cloud Run 504s.
            token_timeout = min(self.config.default_timeout, self.config.timeout)
            token = await self._run_in_executor(
                lambda: token_handle.wait(timeout=token_timeout) if token_handle else None
            )

            if not token:
                raise TokenExtractionError(
                    f"Failed to extract reCAPTCHA token within {token_timeout}s after solve"
                )

            result_cookies = await self._run_in_executor(self._get_cookies, browser)
            time_taken = round(time.time() - start_time, 2)

            return SolveResult(
                token=token,
                cookies=result_cookies,
                time_taken=time_taken,
                captcha_type=last_captcha_type,
                attempts=attempts,
            )

        except RecaptchaSolverError:
            raise
        except (RuntimeError, OSError, ValueError) as e:
            raise RecaptchaSolverError(f"Solve failed: {e}") from e

    def _init_browser(
        self,
        website_key: str,
        website_url: str,
        is_invisible: bool,
        action: str | None,
        is_enterprise: bool,
        api_domain: str,
        bypass_domain_check: bool,
        use_ssl: bool,
        cookies: list[dict[str, Any]] | None,
        user_agent: str | None,
    ) -> tuple[Any, Any]:
        """Initialize browser and replicate captcha."""
        from recaptcha_domain_replicator import RecaptchaDomainReplicator

        self._replicator = RecaptchaDomainReplicator(
            download_dir=str(self.config.download_dir),
            server_port=self.config.server_port,
            persist_html=self.config.persist_html,
            proxy=self.config.proxy,
            browser_path=self.config.browser_path,
        )

        browser, token_handle = self._replicator.replicate_captcha(
            website_key=website_key,
            website_url=website_url,
            is_invisible=is_invisible,
            action=action,
            is_enterprise=is_enterprise,
            api_domain=api_domain,
            bypass_domain_check=bypass_domain_check,
            use_ssl=use_ssl,
            cookies=cookies,
            user_agent=user_agent,
            headless=self.config.headless,
        )

        return browser, token_handle

    def _determine_captcha_type(self, browser: Any) -> CaptchaType:
        """Determine the type of captcha challenge."""
        title = get_challenge_title(browser)
        title_lower = title.lower()

        if "squares" in title_lower:
            return CaptchaType.SQUARE_4X4
        elif "none" in title_lower:
            return CaptchaType.DYNAMIC_3X3
        else:
            return CaptchaType.SELECTION_3X3

    def _get_target_info(
        self, browser: Any, captcha_type: CaptchaType
    ) -> tuple[str | None, int | None, int | None]:
        """Get target keyword plus classifier/detector class mappings."""
        keyword = get_target_keyword(browser)
        if not keyword:
            self.logger.info("No target keyword found in challenge")
            return None, None, None
        if self._detector is None:
            self.logger.info("No detector available for target keyword '%s'", keyword)
            return keyword, None, None

        target_class = self._detector.get_target_class(keyword)
        coco_target_class = None
        if captcha_type == CaptchaType.SQUARE_4X4:
            coco_target_class = self._detector.get_coco_target_class(keyword)
        self.logger.info(
            "Target keyword '%s' mapped to class=%s coco_class=%s captcha_type=%s",
            keyword,
            target_class,
            coco_target_class,
            captcha_type.value,
        )
        return keyword, target_class, coco_target_class

    def _get_handler(self, captcha_type: CaptchaType) -> BaseCaptchaHandler:
        """Get the appropriate handler for a captcha type."""
        if self._handlers is None:
            raise RecaptchaSolverError("Handlers not initialized")

        handler = self._handlers.get(captcha_type)
        if not handler:
            raise UnsupportedCaptchaError(f"Unsupported captcha type: {captcha_type}")
        return handler

    def _debug_dir(self) -> Path | None:
        if not self.config.debug_artifacts_enabled or self.config.debug_artifacts_dir is None:
            return None
        debug_dir = Path(self.config.debug_artifacts_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir

    def _refresh_debug_artifacts(self) -> None:
        debug_dir = self._debug_dir()
        if debug_dir is None:
            return
        self.debug_artifacts = sorted(
            path.name for path in debug_dir.iterdir() if path.is_file()
        )

    def _capture_browser_screenshot_sync(
        self,
        browser: Any,
        attempt: int,
        label: str,
    ) -> str | None:
        debug_dir = self._debug_dir()
        if debug_dir is None:
            return None

        filename = f"{int(time.time() * 1000)}-attempt-{attempt}-{label}-page.png"
        path = debug_dir / filename

        candidates = []
        if hasattr(browser, "latest_tab"):
            candidates.append(browser.latest_tab)
        candidates.append(browser)

        for target in candidates:
            calls = []
            if hasattr(target, "get_screenshot"):
                calls.extend(
                    (
                        ("get_screenshot", (), {"path": str(path), "full_page": True}),
                        ("get_screenshot", (), {"path": str(path)}),
                        ("get_screenshot", (str(path),), {}),
                    )
                )
            if hasattr(target, "screenshot"):
                calls.extend(
                    (
                        ("screenshot", (), {"path": str(path)}),
                        ("screenshot", (str(path),), {}),
                    )
                )

            for method_name, args, kwargs in calls:
                try:
                    getattr(target, method_name)(*args, **kwargs)
                    if path.exists() and path.stat().st_size > 0:
                        self._refresh_debug_artifacts()
                        return filename
                except Exception:
                    continue

        self.logger.debug("Could not capture browser screenshot for attempt %s label %s", attempt, label)
        return None

    def _get_cookies(self, browser: Any) -> list[dict[str, Any]]:
        """Get cookies from the browser."""
        try:
            if hasattr(browser, "cookies"):
                cookies = browser.cookies(all_info=True)
                return list(cookies) if cookies else []

            # Try latest_tab
            if hasattr(browser, "latest_tab"):
                tab = browser.latest_tab
                if hasattr(tab, "cookies"):
                    cookies = tab.cookies(all_info=True)
                    return list(cookies) if cookies else []

            return []
        except (AttributeError, RuntimeError) as e:
            self.logger.debug(f"Error getting cookies: {e}")
            return []
