from modules.ocr_module import OCRModule
from modules.extraction_module import DataExtractionModule
from analytics import AnalyticsLayer
import time
from typing import Dict, Any, List

class OCRPipeline:
    def __init__(self):
        self.ocr_ru = OCRModule(lang='ru')
        self.ocr_en = OCRModule(lang='en')
        self.extraction_module = DataExtractionModule()
        self.analytics = AnalyticsLayer()

    def run_ocr(self, file_path: str, lang: str = 'ru', with_noise_reduction: bool = False) -> Dict[str, Any]:
        start_time = time.time()

        try:
            ocr_module = self.ocr_ru if lang == 'ru' else self.ocr_en

            if with_noise_reduction:
                result = ocr_module.process_with_noise_handling(file_path, True)
            else:
                result = ocr_module.process(file_path)

            execution_time = time.time() - start_time

            self.analytics.log_result(
                module_name=f"OCR Check ({lang.upper()})",
                input_path=file_path,
                output=result,
                execution_time=execution_time
            )

            return {
                'text': result,
                'execution_time': execution_time,
                'success': True
            }

        except Exception as e:
            execution_time = time.time() - start_time
            error_result = {'error': str(e)}

            self.analytics.log_result(
                module_name=f"OCR Check ({lang.upper()})",
                input_path=file_path,
                output=error_result,
                execution_time=execution_time
            )

            return {
                'error': str(e),
                'execution_time': execution_time,
                'success': False
            }

    def run_extraction(self, image_path: str, task_prompt: str = "<s_cord-v2>") -> Dict[str, Any]:
        start_time = time.time()

        try:
            result = self.extraction_module.process(image_path, task_prompt)
            execution_time = time.time() - start_time

            self.analytics.log_result(
                module_name="Data Extraction",
                input_path=image_path,
                output=result,
                execution_time=execution_time
            )

            return {
                'data': result,
                'execution_time': execution_time,
                'success': 'error' not in result
            }

        except Exception as e:
            execution_time = time.time() - start_time
            error_result = {'error': str(e)}

            self.analytics.log_result(
                module_name="Data Extraction",
                input_path=image_path,
                output=error_result,
                execution_time=execution_time
            )

            return {
                'error': str(e),
                'execution_time': execution_time,
                'success': False
            }

    def evaluate_ocr_batch(self, file_paths: List[str], ground_truth: List[str], lang: str = 'ru') -> Dict[str, Any]:
        predictions = []
        execution_times = []

        for file_path in file_paths:
            result = self.run_ocr(file_path, lang)
            predictions.append(result.get('text', ''))
            execution_times.append(result.get('execution_time', 0))

        ocr_module = self.ocr_ru if lang == 'ru' else self.ocr_en
        metrics = ocr_module.calculate_metrics(predictions, ground_truth)

        return {
            'metrics': metrics,
            'predictions': predictions,
            'average_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
            'total_processing_time': sum(execution_times)
        }

    def evaluate_extraction_batch(self, image_paths: List[str], ground_truth: List[Dict]) -> Dict[str, Any]:
        predictions = []
        execution_times = []

        for image_path in image_paths:
            result = self.run_extraction(image_path)
            predictions.append(result.get('data', {}))
            execution_times.append(result.get('execution_time', 0))

        metrics = self.extraction_module.calculate_metrics(predictions, ground_truth)

        return {
            'metrics': metrics,
            'predictions': predictions,
            'average_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
            'total_processing_time': sum(execution_times)
        }

    def get_analytics_summary(self) -> Dict[str, Any]:
        return {
            'ocr_ru_performance': self.analytics.get_module_performance_summary("OCR Check (RU)"),
            'ocr_en_performance': self.analytics.get_module_performance_summary("OCR Check (EN)"),
            'extraction_performance': self.analytics.get_module_performance_summary("Data Extraction"),
            'recent_results': self.analytics.get_recent_results(5)
        }