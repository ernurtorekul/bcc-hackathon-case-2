import streamlit as st
import os
import time
import json
from PIL import Image
from modules.ocr_module import OCRModule
from modules.extraction_module import DataExtractionModule
from modules.llm_enhancement import LLMEnhancementModule
from analytics import AnalyticsLayer

st.set_page_config(page_title="Обработка банковских документов", layout="wide")

@st.cache_resource
def load_modules():
    return {
        'ocr_ru': OCRModule(lang='ru'),
        'ocr_en': OCRModule(lang='en'),
        'extraction': DataExtractionModule()
    }

@st.cache_resource
def load_analytics():
    return AnalyticsLayer()

def main():
    st.title("🏦 Система обработки банковских документов")

    # Sidebar for LLM configuration
    with st.sidebar:
        st.header("⚡ LLM Enhancement")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Введите ключ OpenAI API для улучшения результатов с помощью ChatGPT"
        )

        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            st.success("✅ LLM Enhancement активен")
        else:
            st.info("💡 Добавьте API ключ для улучшения результатов")

    modules = load_modules()
    analytics = load_analytics()

    tab1, tab2 = st.tabs(["📄 Обработка документов", "📊 Аналитика"])

    with tab1:
        st.header("Обработка документов")

        col1_ui, col2_ui = st.columns(2)

        with col1_ui:
            module_choice = st.selectbox(
                "Выберите модуль",
                ["Распознавание текста", "Извлечение данных"]
            )

        with col2_ui:
            language = st.selectbox(
                "Язык",
                ["Русский", "Английский"]
            ) if module_choice == "Распознавание текста" else "Русский"

        uploaded_file = st.file_uploader(
            "Загрузите документ",
            type=['png', 'jpg', 'jpeg', 'pdf']
        )

        if uploaded_file is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Входной документ")

                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                if uploaded_file.name.lower().endswith('.pdf'):
                    st.info(f"📄 PDF загружен: {uploaded_file.name}")
                    st.write(f"**Размер:** {uploaded_file.size:,} байт")

                    try:
                        import fitz
                        doc = fitz.open(temp_path)
                        if doc.page_count > 0:
                            page = doc.load_page(0)
                            pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
                            img_data = pix.tobytes("png")
                            st.image(img_data, caption=f"Страница 1 из {doc.page_count}", width=400)
                        doc.close()
                    except:
                        st.warning("Не удалось создать превью PDF")
                else:
                    image = Image.open(uploaded_file)
                    st.image(image, width=400)

            with col2:
                st.subheader("Результаты обработки")

                if st.button("Обработать документ", type="primary"):
                    start_time = time.time()

                    try:
                        if module_choice == "Распознавание текста":
                            ocr_module = modules['ocr_ru'] if language == "Русский" else modules['ocr_en']

                            with st.spinner(f"Обработка OCR ({language})..."):
                                if st.checkbox("Применить шумоподавление"):
                                    result = ocr_module.process_with_noise_handling(temp_path, True)
                                else:
                                    result = ocr_module.process(temp_path)

                                st.text_area("Извлеченный текст:", result, height=300)

                        else:
                            with st.spinner("Извлечение структурированных данных..."):
                                result = modules['extraction'].process(temp_path)

                                if 'error' in result:
                                    st.error(f"Ошибка: {result['error']}")
                                else:
                                    st.json(result)

                        execution_time = time.time() - start_time

                        analytics.log_result(
                            module_name=f"{module_choice} ({language})",
                            input_path=uploaded_file.name,
                            output=result,
                            execution_time=execution_time
                        )

                        st.success(f"Обработка завершена за {execution_time:.2f} секунд")

                    except Exception as e:
                        st.error(f"Ошибка обработки: {str(e)}")

                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

    with tab2:
        st.header("Панель аналитики")

        if analytics.results_history:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("OCR модуль")
                ocr_summary = analytics.get_module_performance_summary("Распознавание текста")
                if 'error' not in ocr_summary:
                    st.metric("Успешность", f"{ocr_summary['success_rate']:.2%}")
                    st.metric("Всего запусков", ocr_summary['total_runs'])
                    if ocr_summary['average_execution_time']:
                        st.metric("Среднее время", f"{ocr_summary['average_execution_time']:.2f}с")

            with col2:
                st.subheader("Модуль извлечения")
                ext_summary = analytics.get_module_performance_summary("Извлечение данных")
                if 'error' not in ext_summary:
                    st.metric("Успешность", f"{ext_summary['success_rate']:.2%}")
                    st.metric("Всего запусков", ext_summary['total_runs'])
                    if ext_summary['average_execution_time']:
                        st.metric("Среднее время", f"{ext_summary['average_execution_time']:.2f}с")

            st.subheader("История обработки")
            recent_results = analytics.get_recent_results(10)
            for result in reversed(recent_results):
                with st.expander(f"{result['module']} - {result['timestamp'][:19]}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Файл:** {result['input_path']}")
                    with col2:
                        st.write(f"**Статус:** {'✅ Успех' if result['success'] else '❌ Ошибка'}")
                    with col3:
                        if result['execution_time']:
                            st.write(f"**Время:** {result['execution_time']:.2f}с")

            if st.button("Экспорт аналитики"):
                export_data = analytics.export_results('json')
                st.download_button(
                    label="Скачать отчет JSON",
                    data=export_data,
                    file_name=f"analytics_report_{int(time.time())}.json",
                    mime="application/json"
                )
        else:
            st.info("История обработки пуста. Обработайте документы для просмотра аналитики.")

if __name__ == "__main__":
    main()