from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from unittest.mock import patch

class TestKedroRun:
    @patch("kedro.framework.session.KedroSession.run", return_value=None)
    def test_kedro_run(self, mock_run):
        # Initialise le projet Kedro sans lancer de pipelines
        bootstrap_project(Path.cwd())

        with KedroSession.create(project_path=Path.cwd()) as session:
            result = session.run()
            assert result is None
