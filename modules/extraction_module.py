import json
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import fitz  # PyMuPDF
from typing import List, Dict, Any
import numpy as np
import os
import re
from difflib import SequenceMatcher
from .base_module import BaseModule
from .ocr_module import OCRModule
from .llm_enhancement import LLMEnhancementModule

class DataExtractionModule(BaseModule):
    def __init__(self):
        super().__init__("Data Extraction")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Russian OCR for text extraction
        self.ocr_module = OCRModule(lang='ru')

        # Initialize LLM enhancement module
        self.llm_enhancement = LLMEnhancementModule()

        # Load banking/financial terms dictionary
        self.dictionary = self._load_dictionary()

        # Bank-specific extraction templates
        self.bank_templates = self._load_bank_templates()

        try:
            self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
            self.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
            self.model.to(self.device)
        except Exception as e:
            print(f"Warning: Could not load Donut model: {e}")
            self.processor = None
            self.model = None

    def _load_dictionary(self) -> Dict[str, List[str]]:
        """Load banking/financial terms dictionary"""
        try:
            dict_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'termin.json')
            with open(dict_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load dictionary: {e}")
            return {"banks": [], "common_fields": [], "currencies": [], "actions": [], "document_types": []}

    def _load_bank_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load bank-specific extraction templates"""
        return {
            "Kaspi": {
                "amount": [r'Сумма\s+([\d.,]+)', r'К оплате\s+([\d.,]+)', r'Итого\s+([\d.,]+)'],
                "sender": [r'ИП\s+([А-Яа-яA-Za-z\s]+)', r'Отправитель:?\s*([А-Яа-я\s\.]+)'],
                "transaction_id": [r'(?:идентификатор|Номер операции)\s+(\d+)', r'ID операции:\s*([A-Za-z0-9_\-]+)'],
                "receiver": [r'Получатель:?\s*([А-Яа-я\s\.,ООО АО ТОО ИП]+)', r'В пользу\s+([А-Яа-я\s\.]+)'],
                "purpose": [r'Назначение:?\s*([^\n]{5,80})', r'За что:?\s*([^\n]{5,80})']
            },
            "Freedom": {
                "amount": [r'Сумма операции\s+([\d.,]+)', r'К списанию\s+([\d.,]+)'],
                "sender": [r'Клиент:?\s*([А-Яа-я\s\.]+)', r'ФИО:?\s*([А-Яа-я\s\.]+)'],
                "transaction_id": [r'Номер операции:?\s*(\d+)', r'Операция\s+№\s*([A-Za-z0-9]+)'],
                "receiver": [r'Организация:?\s*([А-Яа-я\s\.,ООО АО ТОО]+)', r'Получатель:?\s*([^\n]+)']
            },
            "BCC": {
                "amount": [r'Сумма\s+([\d.,]+)', r'Итого к оплате\s+([\d.,]+)'],
                "sender": [r'Плательщик:?\s*([А-Яа-я\s\.]+)', r'От:?\s*([А-Яа-я\s\.]+)'],
                "transaction_id": [r'BCC[_\-]?(\w+)', r'Документ\s+№\s*([A-Za-z0-9_\-]+)'],
                "receiver": [r'Получатель:?\s*([А-Яа-я\s\.,ООО АО ТОО]+)', r'Куда:?\s*([^\n]+)']
            },
            "Halyk": {
                "amount": [r'Сумма платежа\s+([\d.,]+)', r'К доплате\s+([\d.,]+)'],
                "sender": [r'Плательщик:?\s*([А-Яа-я\s\.]+)', r'Клиент банка:?\s*([^\n]+)'],
                "transaction_id": [r'Номер документа:?\s*([A-Za-z0-9_\-]+)', r'Ref\s+([A-Za-z0-9]+)']
            }
        }

    def _ocr_post_correction(self, text: str) -> str:
        """Minimal OCR post-correction - only fix obvious OCR errors"""
        corrections = {
            # Only fix clear OCR character misreads
            'оо': '00', 'ОО': '00', 'o0': '00', '0o': '00',  # Number corrections
            'Kaspl': 'Kaspi', 'Каsрi': 'Kaspi',  # Bank name corrections
        }

        corrected_text = text
        for wrong, correct in corrections.items():
            corrected_text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, corrected_text, flags=re.IGNORECASE)

        return corrected_text

    def _apply_bank_templates(self, text: str, bank_name: str) -> Dict[str, str]:
        """Apply bank-specific extraction templates"""
        results = {}

        if bank_name in self.bank_templates:
            templates = self.bank_templates[bank_name]

            for field, patterns in templates.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if matches and field not in results:
                        results[field] = matches[0].strip() if isinstance(matches[0], str) else matches[0]
                        break

        return results

    def _fallback_extraction(self, text: str, field_name: str) -> str:
        """Apply fallback rules when primary extraction fails"""
        fallback_patterns = {
            "amount": [
                r'(\d{2,8}[.,]\d{1,2})\s*(?:тенге|₸)',  # Looser amount pattern
                r'(\d{3,8})\s*(?:тенге|₸)',  # Amount without decimals
                r'(\d+[.,]\d+)'  # Any decimal number
            ],
            "sender_name": [
                r'ИП\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*)',  # ИП pattern
                r'([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.(?:[А-ЯЁ]\.)?)',  # Name with initials
                r'Плательщик[:\s]*([А-Яа-я\s\.]+)'  # Generic payer pattern
            ],
            "transaction_id": [
                r'(\d{8,20})',  # Long numeric ID
                r'([A-Z]{2,5}\d{5,15})',  # Bank prefix + numbers
                r'([A-Za-z0-9_\-]{8,25})'  # Mixed alphanumeric
            ],
            "receiver_name": [
                r'((?:ООО|АО|ТОО|ИП)\s+[А-Яа-я\s\.]+)',  # Legal entity pattern
                r'Получатель[:\s]*([А-Яа-я\s\.]+)'  # Generic receiver
            ],
            "currency": [
                r'(₸|тенге)',  # Prefer tenge
                r'(KZT)',  # Kazakhstani tenge code
                r'(USD|EUR|RUB)',  # Other currency codes
                r'([₸$€₽])'  # Currency symbols
            ]
        }

        patterns = fallback_patterns.get(field_name, [])
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0].strip() if isinstance(matches[0], str) else matches[0]

        # Default currency to tenge if nothing found
        if field_name == "currency":
            return "₸"

        return ""

    def _confidence_based_recheck(self, file_path: str, low_confidence_result: Dict) -> Dict[str, Any]:
        """Re-run extraction with different settings if confidence is low"""
        try:
            # Try with English OCR as backup
            ocr_en = OCRModule(lang='en')
            en_text = ocr_en.process(file_path)

            if en_text.strip():
                # Apply same processing pipeline to English OCR result
                cleaned_en = self._clean_text(en_text)
                corrected_en = self._ocr_post_correction(cleaned_en)

                # Extract again with mixed text
                mixed_text = f"{low_confidence_result['_metadata'].get('original_text', '')} {corrected_en}"
                new_result = self._extract_structured_data_from_text(mixed_text)

                if new_result and float(new_result['_metadata']['confidence']) > float(low_confidence_result['_metadata']['confidence']):
                    new_result['_metadata']['fallback_used'] = 'English OCR'
                    return new_result

        except Exception as e:
            print(f"Fallback OCR failed: {e}")

        return low_confidence_result

    def _detect_bank(self, text: str) -> str:
        """Detect which bank from the text"""
        text_lower = text.lower()

        # Check for bank names in order of priority
        bank_indicators = {
            "Kaspi": ["kaspi", "каспи", "kaspi bank"],
            "BCC": ["bcc", "банк центр кредит", "bank centercredit"],
            "Freedom": ["freedom", "фридом", "freedom finance"],
            "Halyk": ["halyk", "халык", "народный банк"]
        }

        for bank, indicators in bank_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    return bank

        return ""

    def _get_similarity_score(self, token: str, dictionary_word: str) -> float:
        """Calculate similarity score between two strings"""
        return SequenceMatcher(None, token.lower(), dictionary_word.lower()).ratio()

    def _correct_token(self, token: str, min_similarity: float = 0.7) -> str:
        """Find the most similar dictionary word for a given token - only use termin.json"""
        if not token or len(token) < 3:  # Skip very short tokens
            return token

        best_match = token
        best_score = 0.0

        # Only use termin.json dictionary terms for correction
        all_terms = []
        for category in self.dictionary.values():
            if isinstance(category, list):
                all_terms.extend(category)

        # Find best matching term only if similarity is high enough
        for term in all_terms:
            score = self._get_similarity_score(token, term)
            if score > best_score and score >= min_similarity:
                best_score = score
                best_match = term

        # Only return corrected word if it's significantly similar (70%+)
        return best_match

    def _correct_text(self, text: str) -> str:
        """Apply token correction to the entire text"""
        # Split text into tokens (words)
        tokens = re.findall(r'\b\w+\b', text)
        corrected_tokens = []

        for token in tokens:
            corrected_token = self._correct_token(token)
            corrected_tokens.append(corrected_token)

        # Reconstruct text with corrections
        corrected_text = text
        for original, corrected in zip(tokens, corrected_tokens):
            if original != corrected:
                corrected_text = re.sub(r'\b' + re.escape(original) + r'\b', corrected, corrected_text)

        return corrected_text

    def _is_garbage_word(self, word: str) -> bool:
        """Check if a word is garbage (not related to banking)"""
        word_lower = word.lower()

        # Skip very short words
        if len(word) < 2:
            return True

        # Check if word is in our banking dictionary
        for category_terms in self.dictionary.values():
            for term in category_terms:
                if word_lower in term.lower() or term.lower() in word_lower:
                    return False

        # Common meaningful words that should be kept
        meaningful_words = [
            'иванов', 'петров', 'сидоров', 'мария', 'анна', 'елена',
            'ооо', 'ао', 'тоо', 'ип', 'зао', 'пао',
            'москва', 'алматы', 'астана', 'нур-султан',
            'улица', 'дом', 'кв', 'офис',
            'коммунальные', 'интернет', 'телефон', 'электричество',
            'зарплата', 'аренда', 'штраф', 'налог'
        ]

        for meaningful in meaningful_words:
            if meaningful in word_lower:
                return False

        # If word contains mostly numbers, it might be important
        if re.search(r'\d{3,}', word):
            return False

        return True

    def _clean_text(self, text: str) -> str:
        """Remove garbage words and clean the text"""
        words = text.split()
        cleaned_words = []

        for word in words:
            # Remove punctuation for garbage check but keep original for output
            clean_word = re.sub(r'[^\w\d]', '', word)
            if not self._is_garbage_word(clean_word):
                cleaned_words.append(word)

        return ' '.join(cleaned_words)

    def _calculate_field_confidence(self, field_name: str, field_value: str) -> float:
        """Calculate confidence score using specific banking rules"""
        if not field_value:
            return 0.0

        field_value = field_value.strip()

        # Specific validation patterns for banking documents
        validation_patterns = {
            "transaction_id": r'^[A-Za-z0-9_\-]{5,30}$',
            "transaction_date": r'^\d{4}-\d{2}-\d{2}',
            "amount": r'^\d+([.,]\d{1,2})?$',
            "currency": r'^[₸$€₽]$|^(KZT|USD|EUR|RUB)$',
            "sender_name": r'^[А-Яа-яA-Za-z\s\.]{3,50}$',
            "sender_account": r'^\d{12}$|^[A-Za-z]{2}\d{2}[A-Za-z0-9]{13,29}$',  # IIN/BIN or IBAN
            "receiver_name": r'^[А-Яа-яA-Za-z\s\.ООО АО ТОО ИП]{3,80}$',
            "receiver_account": r'^\d{12}$|^[A-Za-z]{2}\d{2}[A-Za-z0-9]{13,29}$',
            "bank": r'^[А-Яа-яA-Za-z\s]{3,30}$',
            "purpose": r'^.{5,100}$'
        }

        pattern = validation_patterns.get(field_name)
        if not pattern:
            return 0.5

        # Base confidence from pattern matching
        if re.match(pattern, field_value):
            base_confidence = 0.9
        else:
            base_confidence = 0.3

        # Special checks for specific fields
        if field_name in ['sender_account', 'receiver_account']:
            # Check for IIN/BIN (12 digits) or IBAN format
            if re.match(r'^\d{12}$', field_value):
                base_confidence = 0.95  # High confidence for 12-digit IIN/BIN
            elif re.match(r'^KZ\d{2}[A-Za-z0-9]{13,}$', field_value):
                base_confidence = 0.95  # High confidence for Kazakhstani IBAN

        elif field_name == 'amount':
            # Amount should be reasonable (not too many digits)
            if re.match(r'^\d{1,8}([.,]\d{1,2})?$', field_value):
                base_confidence = 0.95

        elif field_name == 'bank':
            # Check if it's a known bank from dictionary
            for bank in self.dictionary.get('banks', []):
                if bank.lower() in field_value.lower():
                    base_confidence = 0.95
                    break

        return round(base_confidence, 2)

    def process(self, file_path: str, task_prompt: str = "<s_cord-v2>") -> Dict[str, Any]:
        # First try OCR-based extraction for Russian documents
        try:
            ocr_text = self.ocr_module.process(file_path)
            if ocr_text.strip():
                # Step 1: Remove garbage words not related to banking
                cleaned_text = self._clean_text(ocr_text)

                # Step 2: Apply enhanced OCR post-correction
                post_corrected_text = self._ocr_post_correction(cleaned_text)

                # Step 3: Apply token correction to improve OCR accuracy
                corrected_text = self._correct_text(post_corrected_text)

                # Step 4: Detect bank from text to apply bank-specific templates
                detected_bank = self._detect_bank(corrected_text)

                # Try to extract structured data from processed Russian OCR text
                structured_data = self._extract_structured_data_from_text(corrected_text, detected_bank)

                if structured_data:
                    # Check confidence and apply fallback if needed
                    confidence = float(structured_data["_metadata"]["confidence"])

                    if confidence < 0.5:
                        # Try confidence-based re-check
                        structured_data = self._confidence_based_recheck(file_path, structured_data)
                        structured_data["_metadata"]["fallback_attempted"] = True

                    # Step 5: Enhance with LLM if available
                    if self.llm_enhancement.enabled:
                        try:
                            enhanced_data = self.llm_enhancement.enhance_json_extraction(
                                structured_data,
                                corrected_text
                            )
                            structured_data = enhanced_data
                        except Exception as e:
                            print(f"LLM enhancement failed: {e}")
                            structured_data["_metadata"]["llm_enhancement_failed"] = True

                    # Add processing info to metadata
                    structured_data["_metadata"]["text_cleaned"] = (cleaned_text != ocr_text)
                    structured_data["_metadata"]["text_corrected"] = (corrected_text != post_corrected_text)
                    structured_data["_metadata"]["bank_detected"] = detected_bank
                    if ocr_text != corrected_text:
                        structured_data["_metadata"]["original_text"] = ocr_text[:150] + "..." if len(ocr_text) > 150 else ocr_text

                    return structured_data

        except Exception as e:
            print(f"OCR extraction failed: {e}")

        # Fallback to Donut model if available
        if not self.model or not self.processor:
            return {
                "transaction_id": "",
                "transaction_date": "",
                "amount": "",
                "currency": "",
                "parties": {
                    "sender_name": "",
                    "sender_account": "",
                    "receiver_name": "",
                    "receiver_account": ""
                },
                "bank": "",
                "purpose": "",
                "_metadata": {
                    "extraction_method": "None",
                    "language": "Unknown",
                    "confidence": "0"
                }
            }

        try:
            # Handle PDF files by converting to image first
            if file_path.lower().endswith('.pdf'):
                image_path = self._convert_pdf_to_image(file_path)
                if not image_path:
                    return {"error": "Failed to convert PDF to image"}
            else:
                image_path = file_path

            image = Image.open(image_path).convert("RGB")

            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)

            decoder_input_ids = self.processor.tokenizer(
                task_prompt,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids.to(self.device)

            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model.decoder.config.max_position_embeddings,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")

            sequence = sequence.split(task_prompt)[-1].strip()

            try:
                extracted_data = json.loads(sequence)
                result = self._convert_to_standard_format(extracted_data)

                # Add PDF processing info if it was a PDF
                if file_path.lower().endswith('.pdf'):
                    result["_metadata"]["source_type"] = "PDF"
                    # Clean up temporary image file
                    if image_path != file_path and os.path.exists(image_path):
                        os.remove(image_path)

                return result
            except json.JSONDecodeError:
                print(f"Donut model raw output: {repr(sequence)}")

                # Return in standard format even for failures
                result = {
                    "transaction_id": "",
                    "transaction_date": "",
                    "amount": "",
                    "currency": "",
                    "parties": {
                        "sender_name": "",
                        "sender_account": "",
                        "receiver_name": "",
                        "receiver_account": ""
                    },
                    "bank": "",
                    "purpose": "",
                    "_metadata": {
                        "extraction_method": "Donut Model",
                        "language": "Mixed",
                        "confidence": "0"
                    }
                }

                # Clean up temporary image file if PDF processing failed
                if file_path.lower().endswith('.pdf') and image_path != file_path and os.path.exists(image_path):
                    os.remove(image_path)

                return result

        except Exception as e:
            return {
                "transaction_id": "",
                "transaction_date": "",
                "amount": "",
                "currency": "",
                "parties": {
                    "sender_name": "",
                    "sender_account": "",
                    "receiver_name": "",
                    "receiver_account": ""
                },
                "bank": "",
                "purpose": "",
                "_metadata": {
                    "extraction_method": "None",
                    "language": "Unknown",
                    "confidence": "0"
                }
            }

    def _convert_pdf_to_image(self, pdf_path: str) -> str:
        """Convert first page of PDF to image for processing"""
        try:
            doc = fitz.open(pdf_path)
            if doc.page_count == 0:
                doc.close()
                return None

            # Process first page only for structured data extraction
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # High resolution for better extraction

            # Save as temporary image
            temp_img_path = f"/tmp/extraction_temp_{os.path.basename(pdf_path)}.png"
            pix.save(temp_img_path)

            doc.close()
            return temp_img_path

        except Exception as e:
            print(f"Error converting PDF to image: {e}")
            return None

    def _validate_and_clean_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cleaned_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                cleaned_data[key] = value.strip()
            elif isinstance(value, (int, float)):
                cleaned_data[key] = value
            elif isinstance(value, dict):
                cleaned_data[key] = self._validate_and_clean_json(value)
            else:
                cleaned_data[key] = str(value)

        cleaned_data["_metadata"] = {
            "field_count": len(cleaned_data) - 1,
            "extraction_success": True
        }

        return cleaned_data

    def _convert_to_standard_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Donut model output to standard transaction format"""
        result = {
            "transaction_id": "",
            "transaction_date": "",
            "amount": "",
            "currency": "",
            "parties": {
                "sender_name": "",
                "sender_account": "",
                "receiver_name": "",
                "receiver_account": ""
            },
            "bank": "",
            "purpose": ""
        }

        # Try to map common fields from Donut output
        field_mappings = {
            'transaction_id': ['number', 'receipt_number', 'id', 'transaction_id'],
            'transaction_date': ['date', 'transaction_date', 'receipt_date'],
            'amount': ['total', 'amount', 'sum', 'price'],
            'currency': ['currency', 'curr'],
            'sender_name': ['company', 'store_name', 'sender', 'merchant'],
            'receiver_name': ['receiver', 'customer'],
            'bank': ['bank'],
            'purpose': ['purpose', 'description', 'note']
        }

        for standard_field, possible_keys in field_mappings.items():
            for key in possible_keys:
                if key in data and data[key]:
                    if standard_field in ['sender_name', 'sender_account', 'receiver_name', 'receiver_account']:
                        result['parties'][standard_field] = str(data[key]).strip()
                    else:
                        result[standard_field] = str(data[key]).strip()
                    break

        # Calculate confidence
        total_fields = 10
        filled_fields = sum(1 for v in [result["transaction_id"], result["transaction_date"], result["amount"],
                                      result["currency"], result["bank"], result["purpose"]] if v) + \
                       sum(1 for v in result["parties"].values() if v)
        confidence = round(filled_fields / total_fields, 2) if total_fields > 0 else 0

        result["_metadata"] = {
            "extraction_method": "Donut Model",
            "language": "Mixed",
            "confidence": str(confidence)
        }

        return result

    def _extract_structured_data_from_text(self, text: str, detected_bank: str = "") -> Dict[str, Any]:
        """Extract structured data from Russian OCR text using pattern matching"""
        try:
            # Initialize the new detailed JSON structure
            extracted_data = {
                "transaction_id": "",
                "transaction_date": "",
                "amount": "",
                "currency": "",
                "parties": {
                    "sender_name": "",
                    "sender_account": "",
                    "receiver_name": "",
                    "receiver_account": ""
                },
                "bank": "",
                "purpose": ""
            }

            # Banking-specific extraction patterns with improved accuracy
            patterns = {
                # Transaction ID patterns
                'transaction_id': [
                    r'(?:№\s*квитанции|номер\s*квитанции)[\s:]*([A-Za-z0-9_\-]{5,30})',
                    r'(?:номер\s*операции|операция\s*№)[\s:]*([A-Za-z0-9_\-]{5,30})',
                    r'(?:чек\s*№|№\s*чек)[\s:]*([A-Za-z0-9_\-]{5,30})',
                    r'(?:id|номер)[\s:]*([A-Za-z0-9_\-]{5,30})'
                ],

                # Date patterns - DD.MM.YYYY or "дд месяц гггг, hh:mm"
                'transaction_date': [
                    r'(?:дата\s*и\s*время|дата\s*время)[\s:]*(\d{1,2}\.\d{1,2}\.\d{4}(?:\s+\d{1,2}:\d{2})?)',
                    r'(?:дата|date)[\s:]*(\d{1,2}\.\d{1,2}\.\d{4}(?:\s+\d{1,2}:\d{2})?)',
                    r'(\d{1,2}\.\d{1,2}\.\d{4}(?:\s+\d{1,2}:\d{2})?)',
                    r'(\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4}(?:,\s*\d{1,2}:\d{2})?)'
                ],

                # Sender name patterns
                'sender_name': [
                    r'(?:отправитель|фио\s*плательщика|плательщик)[\s:]*([А-Яа-яA-Za-z\s\.]{3,50})',
                    r'(?:от|from|sender)[\s:]*([А-Яа-яA-Za-z\s\.]{3,50})'
                ],

                # Sender account patterns - IIN/BIN (12 digits) or IBAN
                'sender_account': [
                    r'(?:счет\s*отправителя|счет\s*плательщика|иин|бин)[\s:]*(\d{12})',
                    r'(?:с\s*карты|карта\s*отправителя)[\s:]*([A-Za-z]{2}\d{2}[A-Za-z0-9]{13,29})',
                    r'(?:account|acc)[\s:]*([A-Za-z0-9]{10,30})'
                ],

                # Receiver name patterns
                'receiver_name': [
                    r'(?:куда|наименование\s*получателя|получатель)[\s:]*([А-Яа-яA-Za-z\s\.,ООО АО ТОО ИП]{3,80})',
                    r'(?:организация|company|получить|to|receiver)[\s:]*([А-Яа-яA-Za-z\s\.,ООО АО ТОО ИП]{3,80})'
                ],

                # Receiver account patterns - IIN/BIN (12 digits) or IBAN
                'receiver_account': [
                    r'(?:счет\s*получателя|иин\s*получателя|бин\s*получателя)[\s:]*(\d{12})',
                    r'(?:на\s*карту|карта\s*получателя)[\s:]*([A-Za-z]{2}\d{2}[A-Za-z0-9]{13,29})',
                    r'(?:to\s*account|to\s*card)[\s:]*([A-Za-z0-9]{10,30})'
                ],

                # Bank patterns - match against known banks
                'bank': [
                    r'(?:банк|bank)[\s:]*([А-Яа-яA-Za-z\s]{3,30})',
                    r'(?:через|via)[\s:]*([А-Яа-яA-Za-z\s]*(?:банк|bank)[А-Яа-яA-Za-z\s]*)'
                ],

                # Purpose patterns - limit length
                'purpose': [
                    r'(?:назначение|purpose|цель|за\s*что|для)[\s:]*([^\n]{5,100})',
                    r'(?:комментарий|comment|note|описание)[\s:]*([^\n]{5,100})'
                ]
            }

            # Step 1: Try bank-specific templates first if bank is detected
            bank_extracted = {}
            if detected_bank:
                bank_extracted = self._apply_bank_templates(text, detected_bank)

            # Step 2: Extract data using standard patterns
            for field, pattern_list in patterns.items():
                # Skip if already extracted by bank template
                if field in bank_extracted:
                    if field in ['sender_name', 'sender_account', 'receiver_name', 'receiver_account']:
                        extracted_data['parties'][field] = bank_extracted[field]
                    else:
                        extracted_data[field] = bank_extracted[field]
                    continue

                if field in ['sender_name', 'sender_account', 'receiver_name', 'receiver_account']:
                    # Handle nested parties structure
                    for pattern in pattern_list:
                        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                        if matches and not extracted_data['parties'][field]:
                            extracted_data['parties'][field] = matches[0].strip()
                            break
                else:
                    # Handle top-level fields
                    for pattern in pattern_list:
                        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                        if matches and not extracted_data[field]:
                            extracted_data[field] = matches[0].strip()
                            break

            # Amount patterns - numbers followed by currency symbol
            amount_patterns = [
                r'(?:сумма|итого|total|amount)[\s:]*(\d+(?:[.,]\d{1,2})?)[\s]*([₸$€₽]|KZT|USD|EUR|RUB)',
                r'(\d+(?:[.,]\d{1,2})?)[\s]*([₸$€₽]|KZT|USD|EUR|RUB|тенге|руб)',
                r'(?:сумма|итого|total|amount)[\s:]*(\d+(?:[.,]\d{1,2})?)'
            ]

            # Currency patterns - specific currency symbols and codes
            currency_patterns = [
                r'(?:валюта|currency)[\s:]*([₸$€₽]|KZT|USD|EUR|RUB|тенге|руб)',
                r'([₸$€₽])(?:\s|$|[А-Яа-я])',
                r'\b(KZT|USD|EUR|RUB|тенге|руб)\b'
            ]

            for pattern in amount_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches and not extracted_data["amount"]:
                    if len(matches[0]) == 2:  # Pattern with currency
                        amount_value, currency = matches[0]
                        extracted_data["amount"] = amount_value.strip()
                        # Don't set currency here - will be set to tenge below
                    else:  # Pattern without currency
                        extracted_data["amount"] = matches[0].strip()
                    break

            # Always set currency to tenge (₸) regardless of what was detected
            extracted_data["currency"] = "₸"

            # Convert date to YYYY-MM-DD format if possible
            if extracted_data["transaction_date"]:
                date_str = extracted_data["transaction_date"]
                # Try to convert DD.MM.YYYY or DD/MM/YYYY to YYYY-MM-DD
                date_match = re.match(r'(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})', date_str)
                if date_match:
                    day, month, year = date_match.groups()
                    if len(year) == 2:
                        year = '20' + year if int(year) < 50 else '19' + year
                    try:
                        formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        # Preserve time if present
                        time_match = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?)', date_str)
                        if time_match:
                            formatted_date += f" {time_match.group(1)}"
                        extracted_data["transaction_date"] = formatted_date
                    except:
                        pass  # Keep original if conversion fails

            # Step 3: Apply fallback extraction for missing critical fields
            critical_fields = ["amount", "currency", "transaction_id", "sender_name", "receiver_name"]
            for field in critical_fields:
                if field in ["sender_name", "receiver_name"]:
                    if not extracted_data["parties"][field]:
                        fallback_value = self._fallback_extraction(text, field)
                        if fallback_value:
                            extracted_data["parties"][field] = fallback_value
                elif not extracted_data[field]:
                    fallback_value = self._fallback_extraction(text, field)
                    if fallback_value:
                        extracted_data[field] = fallback_value

            # Apply confidence filtering - leave fields empty if confidence is too low
            min_confidence_threshold = 0.3  # Minimum confidence to include field
            field_confidences = {}

            # Calculate individual field confidence based on pattern matching quality
            for field_name, field_value in [
                ("transaction_id", extracted_data["transaction_id"]),
                ("transaction_date", extracted_data["transaction_date"]),
                ("amount", extracted_data["amount"]),
                ("currency", extracted_data["currency"]),
                ("bank", extracted_data["bank"]),
                ("purpose", extracted_data["purpose"])
            ]:
                if field_value:
                    # Simple confidence based on whether value looks reasonable
                    field_conf = self._calculate_field_confidence(field_name, field_value)
                    field_confidences[field_name] = field_conf

                    # Clear field if confidence too low
                    if field_conf < min_confidence_threshold:
                        extracted_data[field_name] = ""

            # Check parties fields
            for party_field, party_value in extracted_data["parties"].items():
                if party_value:
                    field_conf = self._calculate_field_confidence(party_field, party_value)
                    field_confidences[party_field] = field_conf

                    if field_conf < min_confidence_threshold:
                        extracted_data["parties"][party_field] = ""

            # Calculate overall confidence based on filled fields after filtering
            total_fields = 10
            filled_fields = sum(1 for v in [extracted_data["transaction_id"], extracted_data["transaction_date"],
                                          extracted_data["amount"], extracted_data["currency"], extracted_data["bank"],
                                          extracted_data["purpose"]] if v) + \
                           sum(1 for v in extracted_data["parties"].values() if v)

            confidence = round(filled_fields / total_fields, 2) if total_fields > 0 else 0

            extracted_data["_metadata"] = {
                "extraction_method": "OCR + Pattern Matching",
                "language": "Russian",
                "confidence": str(confidence)
            }

            return extracted_data

        except Exception as e:
            print(f"Error in text extraction: {e}")
            return {
                "transaction_id": "",
                "transaction_date": "",
                "amount": "",
                "currency": "",
                "parties": {
                    "sender_name": "",
                    "sender_account": "",
                    "receiver_name": "",
                    "receiver_account": ""
                },
                "bank": "",
                "purpose": "",
                "_metadata": {
                    "extraction_method": "OCR + Pattern Matching",
                    "language": "Russian",
                    "confidence": "0"
                }
            }

    def _extract_items_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract item details from receipt-like text"""
        items = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for lines with price patterns
            price_match = re.search(r'([0-9,.\s]+)[\s]*(?:руб|₽|rub)', line, re.IGNORECASE)
            if price_match:
                price = price_match.group(1).strip()
                # Extract item name (everything before the price)
                item_name = re.sub(r'[0-9,.\s]*(?:руб|₽|rub).*', '', line, flags=re.IGNORECASE).strip()

                if item_name and len(item_name) > 2:  # Avoid single characters
                    items.append({
                        'название': item_name,
                        'цена': price
                    })

        return items[:10]  # Limit to first 10 items to avoid noise

    def calculate_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")

        field_accuracies = {}
        exact_matches = 0
        total_f1_scores = []

        for pred, gt in zip(predictions, ground_truth):
            if "error" in pred:
                continue

            pred_clean = {k: v for k, v in pred.items() if k != "_metadata"}
            gt_clean = {k: v for k, v in gt.items() if k != "_metadata"}

            exact_match = pred_clean == gt_clean
            if exact_match:
                exact_matches += 1

            all_fields = set(pred_clean.keys()) | set(gt_clean.keys())

            correct_fields = 0
            tp = fp = fn = 0

            for field in all_fields:
                pred_val = str(pred_clean.get(field, "")).strip().lower()
                gt_val = str(gt_clean.get(field, "")).strip().lower()

                is_correct = pred_val == gt_val
                if is_correct:
                    correct_fields += 1

                if field not in field_accuracies:
                    field_accuracies[field] = []
                field_accuracies[field].append(1.0 if is_correct else 0.0)

                if pred_val and gt_val:
                    if pred_val == gt_val:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                elif pred_val:
                    fp += 1
                elif gt_val:
                    fn += 1

            if tp + fp + fn > 0:
                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / (tp + fn) if tp + fn > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                total_f1_scores.append(f1)

        metrics = {
            'exact_match_ratio': exact_matches / len(predictions) if predictions else 0,
            'average_f1': np.mean(total_f1_scores) if total_f1_scores else 0,
            'f1_std': np.std(total_f1_scores) if total_f1_scores else 0
        }

        for field, accuracies in field_accuracies.items():
            metrics[f'field_accuracy_{field}'] = np.mean(accuracies)

        return metrics