import openai
import json
import os
from typing import Dict, Any
from .base_module import BaseModule

class LLMEnhancementModule(BaseModule):
    def __init__(self, api_key: str = None):
        super().__init__("LLM Enhancement")

        # Set OpenAI API key
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")

        if not openai.api_key:
            print("Warning: OpenAI API key not found. LLM enhancement will be disabled.")
            self.enabled = False
        else:
            self.enabled = True

    def enhance_ocr_text(self, raw_ocr_text: str) -> str:
        """Enhance OCR text using ChatGPT to fix errors and improve readability"""
        if not self.enabled:
            return raw_ocr_text

        try:
            prompt = f"""
Ты специалист по обработке банковских документов. Исправь ошибки OCR в тексте и сделай его более читаемым.

Правила:
1. Исправь очевидные ошибки распознавания (замена символов, неправильные цифры)
2. Сохрани все важные банковские термины и числа
3. Структурируй текст, если это возможно
4. НЕ добавляй информацию, которой нет в оригинале
5. Сохрани русский язык

Исходный OCR текст:
{raw_ocr_text}

Исправленный текст:"""

            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1
            )

            enhanced_text = response.choices[0].message.content.strip()
            return enhanced_text

        except Exception as e:
            print(f"LLM OCR enhancement failed: {e}")
            return raw_ocr_text

    def enhance_json_extraction(self, raw_json: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """Enhance extracted JSON using ChatGPT to improve accuracy and completeness"""
        if not self.enabled:
            return raw_json

        try:
            prompt = f"""
Ты эксперт по извлечению данных из банковских документов. Улучши извлеченные данные.

ОБЯЗАТЕЛЬНАЯ СХЕМА JSON:
{{
  "transaction_id": "строка",
  "transaction_date": "YYYY-MM-DD или YYYY-MM-DD HH:MM:SS",
  "amount": "строка (только числа)",
  "currency": "₸",
  "parties": {{
    "sender_name": "строка",
    "sender_account": "строка",
    "receiver_name": "строка",
    "receiver_account": "строка"
  }},
  "bank": "строка",
  "purpose": "строка"
}}

Правила:
1. Если поле пустое, оставь пустую строку ""
2. НЕ добавляй поля, которых нет в схеме
3. Исправь очевидные ошибки в данных
4. Улучши неполные данные, используя контекст из оригинального текста

Исходные данные:
{json.dumps(raw_json, ensure_ascii=False, indent=2)}

Оригинальный текст документа:
{original_text[:500]}...

Улучшенный JSON:"""

            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.1
            )

            enhanced_json_str = response.choices[0].message.content.strip()

            # Extract JSON from response
            if enhanced_json_str.startswith('```json'):
                enhanced_json_str = enhanced_json_str[7:-3]
            elif enhanced_json_str.startswith('```'):
                enhanced_json_str = enhanced_json_str[3:-3]

            try:
                enhanced_json = json.loads(enhanced_json_str)

                # Ensure currency is always tenge
                enhanced_json["currency"] = "₸"

                # Add metadata about enhancement
                if "_metadata" not in enhanced_json:
                    enhanced_json["_metadata"] = {}

                enhanced_json["_metadata"].update({
                    "llm_enhanced": True,
                    "enhancement_method": "ChatGPT-4",
                    "original_confidence": raw_json.get("_metadata", {}).get("confidence", "unknown")
                })

                return enhanced_json

            except json.JSONDecodeError:
                print("LLM returned invalid JSON, using original")
                return raw_json

        except Exception as e:
            print(f"LLM JSON enhancement failed: {e}")
            return raw_json

    def process(self, data: Any, enhancement_type: str = "json", original_text: str = "") -> Any:
        """Main processing method for LLM enhancement"""
        if enhancement_type == "ocr":
            return self.enhance_ocr_text(data)
        elif enhancement_type == "json":
            return self.enhance_json_extraction(data, original_text)
        else:
            return data

    def calculate_metrics(self, predictions, ground_truth):
        """Placeholder method to satisfy abstract base class"""
        return {"enhancement_success_rate": 1.0}