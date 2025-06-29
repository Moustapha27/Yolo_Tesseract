from unittest.mock import patch
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


class TestKedroRun:
    @patch("src.kedro_road_sign.pipelines.ocr_pipeline.nodes.pytesseract.get_tesseract_version")
    @patch("torch.load")
    def test_kedro_run(self, mock_torch_load, mock_get_version):
        mock_get_version.return_value = "5.3.0"
        mock_torch_load.return_value = "mocked_model_object"  # ou un objet factice
        
        bootstrap_project(Path.cwd())
        with KedroSession.create(project_path=Path.cwd()) as session:
            assert session.run() is not None
