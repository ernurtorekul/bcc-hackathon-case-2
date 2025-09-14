import cv2
import numpy as np
from paddleocr import PaddleOCR
import Levenshtein
import fitz  # PyMuPDF
from typing import List, Dict, Union
from .base_module import BaseModule
from .llm_enhancement import LLMEnhancementModule
import os

class OCRModule(BaseModule):
    def __init__(self, lang='ru'):
        super().__init__("OCR Check")
        self.lang = lang
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        self.llm_enhancement = LLMEnhancementModule()
        print(f"OCR initialized for language: {lang}")

    def process(self, file_path: str) -> str:
        if file_path.lower().endswith('.pdf'):
            return self._process_pdf(file_path)
        else:
            return self._process_image(file_path)

    def _process_image(self, image_path: str) -> str:
        result = self.ocr.ocr(image_path, cls=True)

        if not result or not result[0]:
            return ""

        text_lines = []
        for line in result[0]:
            if len(line) >= 2:
                text_lines.append(line[1][0])

        raw_text = ' '.join(text_lines)

        # Enhance OCR text with LLM if available
        if self.llm_enhancement.enabled and raw_text.strip():
            try:
                enhanced_text = self.llm_enhancement.enhance_ocr_text(raw_text)
                return enhanced_text
            except Exception as e:
                print(f"LLM OCR enhancement failed: {e}")
                return raw_text

        return raw_text

    def _process_pdf(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            all_text = []

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)

                # Convert PDF page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better quality
                img_data = pix.tobytes("png")

                # Save temporarily for OCR processing
                temp_img_path = f"/tmp/pdf_page_{page_num}.png"
                with open(temp_img_path, "wb") as f:
                    f.write(img_data)

                # Process with OCR
                page_text = self._process_image(temp_img_path)
                if page_text.strip():
                    all_text.append(f"[Page {page_num + 1}] {page_text}")

                # Clean up temporary file
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)

            doc.close()
            return '\n\n'.join(all_text)

        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    def process_with_noise_handling(self, file_path: str, noise_reduction: bool = True) -> str:
        if file_path.lower().endswith('.pdf'):
            return self._process_pdf(file_path)

        if noise_reduction:
            image = cv2.imread(file_path)
            image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            temp_path = "/tmp/denoised_image.jpg"
            cv2.imwrite(temp_path, image)
            return self._process_image(temp_path)
        else:
            return self._process_image(file_path)

    def calculate_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")

        cer_scores = []
        wer_scores = []
        nld_scores = []

        for pred, gt in zip(predictions, ground_truth):
            pred_chars = list(pred.lower())
            gt_chars = list(gt.lower())

            char_distance = Levenshtein.distance(pred_chars, gt_chars)
            cer = char_distance / max(len(gt_chars), 1)
            cer_scores.append(cer)

            pred_words = pred.lower().split()
            gt_words = gt.lower().split()
            word_distance = Levenshtein.distance(pred_words, gt_words)
            wer = word_distance / max(len(gt_words), 1)
            wer_scores.append(wer)

            max_len = max(len(pred), len(gt))
            nld = Levenshtein.distance(pred.lower(), gt.lower()) / max(max_len, 1)
            nld_scores.append(nld)

        return {
            'CER': np.mean(cer_scores),
            'WER': np.mean(wer_scores),
            'NLD': np.mean(nld_scores),
            'CER_std': np.std(cer_scores),
            'WER_std': np.std(wer_scores),
            'NLD_std': np.std(nld_scores)
        }