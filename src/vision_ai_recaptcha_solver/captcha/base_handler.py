"""Abstract base class for captcha handlers."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from vision_ai_recaptcha_solver.browser.navigation import (
    click_tile,
    get_captcha_image_urls,
)
from vision_ai_recaptcha_solver.captcha.image_utils import download_image, load_image_as_array
from vision_ai_recaptcha_solver.config import SolverConfig
from vision_ai_recaptcha_solver.detector.yolo_detector import YOLODetector
from vision_ai_recaptcha_solver.utils import human_delay as shared_human_delay

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BaseCaptchaHandler(ABC):
    """Abstract base class for handling different captcha types.

    Subclasses implement the specific solving logic for each captcha type
    (dynamic 3x3, selection 3x3, square 4x4).
    """

    def __init__(
        self,
        detector: YOLODetector,
        config: SolverConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the captcha handler.

        Args:
            detector: YOLO detector instance for image analysis.
            config: Solver configuration.
            logger: Logger instance. If None, creates a new one.
        """
        self.detector = detector
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._work_dir = config.download_dir
        self.last_debug: dict[str, Any] = {}

    @abstractmethod
    def solve(self, browser: Any, target_class: int) -> list[int]:
        """Solve the captcha and return the cells that were clicked.

        Args:
            browser: Browser instance from recaptcha_domain_replicator.
            target_class: YOLO class of the target object.

        Returns:
            List of cell indices that were clicked.
        """
        pass

    def reset_debug(self) -> None:
        self.last_debug = {}

    def _debug_dir(self) -> Path | None:
        if not self.config.debug_artifacts_enabled or self.config.debug_artifacts_dir is None:
            return None
        debug_dir = Path(self.config.debug_artifacts_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir

    def save_debug_image(self, image: NDArray[np.uint8], label: str) -> str | None:
        debug_dir = self._debug_dir()
        if debug_dir is None:
            return None
        safe_label = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)[:80]
        filename = f"{int(time.time() * 1000)}-{safe_label}.png"
        path = debug_dir / filename
        try:
            cv2.imwrite(str(path), image)
            return filename
        except Exception as e:
            self.logger.debug("Failed to save debug image %s: %s", path, e)
            return None

    def save_grid_confidence_image(
        self,
        image: NDArray[np.uint8],
        grid_size: int,
        confidences: list[tuple[int, float]],
        selected_cells: list[int],
        label: str,
    ) -> str | None:
        annotated = image.copy()
        img_height, img_width = annotated.shape[:2]
        tile_h = img_height // grid_size
        tile_w = img_width // grid_size
        conf_by_cell = dict(confidences)
        selected = set(selected_cells)

        for row in range(grid_size):
            for col in range(grid_size):
                cell = row * grid_size + col + 1
                x1, y1 = col * tile_w, row * tile_h
                x2, y2 = x1 + tile_w, y1 + tile_h
                color = (0, 200, 0) if cell in selected else (80, 80, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                text = f"{cell}:{conf_by_cell.get(cell, 0.0):.2f}"
                cv2.putText(
                    annotated,
                    text,
                    (x1 + 5, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        return self.save_debug_image(annotated, label)

    def click_cells(self, browser: Any, cells: list[int]) -> None:
        """Click on the specified grid cells.

        Args:
            browser: Browser instance.
            cells: List of 1-indexed cell numbers to click.
        """
        for cell in cells:
            if click_tile(browser, cell, timeout=self.config.default_timeout):
                self.human_delay(0.3, 0.2)
            else:
                self.logger.warning(f"Failed to click cell {cell}")

    def get_image_urls(self, browser: Any) -> list[str]:
        """Get all captcha image URLs from the browser.

        Args:
            browser: Browser instance.

        Returns:
            List of image URLs.
        """
        return get_captcha_image_urls(browser, timeout=self.config.default_timeout)

    def download_main_image(self, url: str) -> tuple[Path, NDArray[np.uint8]]:
        """Download the captcha image and return the path and image as an numpy array.

        Args:
            url: URL of the image.

        Returns:
            Tuple of (path, image_array).
        """
        self._work_dir.mkdir(parents=True, exist_ok=True)
        path = self._work_dir / "main.png"
        download_image(
            url,
            path,
            retries=self.config.image_download_retries,
            retry_delay=self.config.image_download_retry_delay,
        )
        image = load_image_as_array(path)
        return path, image

    def human_delay(self, mean: float | None = None, sigma: float | None = None) -> None:
        """Add a random human like delay.

        Args:
            mean: Mean delay in seconds. Uses config default if None.
            sigma: Standard deviation. Uses config default if None.
        """
        mu = self.config.human_delay_mean if mean is None else mean
        sig = self.config.human_delay_sigma if sigma is None else sigma
        shared_human_delay(mean=mu, sigma=sig)
