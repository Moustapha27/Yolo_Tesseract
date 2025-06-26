import unittest
from unittest.mock import patch, MagicMock
import os

# On importe la fonction à tester
from  src.kedro_road_sign.pipelines.ocr_pipeline.nodes import run_ocr_on_detection

class TestRunOcrOnDetection(unittest.TestCase):

    @patch("os.listdir")
    @patch("cv2.imread")
    @patch("pytesseract.image_to_string")
    def test_run_ocr_on_detection(self, mock_ocr, mock_imread, mock_listdir):
        # Données de test
        mock_listdir.return_value = ["image1.jpg", "image2.png", "not_image.txt"]
        mock_imread.return_value = MagicMock()  # un faux objet image
        mock_ocr.side_effect = ["Text from image1", "Text from image2"]

        folder_path = "/fake/path"
        expected_result = {
            "image1.jpg": "Text from image1",
            "image2.png": "Text from image2"
        }

        result = run_ocr_on_detection(folder_path)
        self.assertEqual(result, expected_result)

        mock_listdir.assert_called_once_with(folder_path)
        self.assertEqual(mock_ocr.call_count, 2)
        self.assertEqual(mock_imread.call_count, 2)

if __name__ == "__main__":
    unittest.main()
