from unittest.mock import patch
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

class TestKedroRun:
    @patch("src.kedro_road_sign.pipelines.ocr_pipeline.nodes.pytesseract.get_tesseract_version")
    def test_kedro_run(self, mock_get_version):
        # Simuler que Tesseract est bien installé en retournant une version fictive
        mock_get_version.return_value = "5.3.0"

        bootstrap_project(Path.cwd())

        with KedroSession.create(project_path=Path.cwd()) as session:
            assert session.run() is not None
