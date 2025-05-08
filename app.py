from telegram import Update, ForceReply
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from rag_processor import RAGProcessor
from config import MAX_CONTEXT_LENGTH, DEEPSEEK_API_URL, DEEPSEEK_MODEL
load_dotenv()


class AIChatBot:
    def __init__(self):
        self.rag = RAGProcessor()
        self.rag.load_and_process_documents()
        self.user_contexts = {}
        self.log_file = "logs.txt"  # Перенесено из _init_logging
        self._init_logging()  # Явный вызов метода инициализации

    def _init_logging(self):
        """Инициализация системы логирования"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding='utf-8') as f:
                f.write("Timestamp|User ID|Duration|Status|Request|Response\n")

    async def _log_request(self, user_id: int, start_time: float, status: str, request: dict, response: str = ""):
        """Логирование запроса"""
        log_entry = (
            f"{datetime.now().isoformat()}|"
            f"{user_id}|"
            f"{time.time() - start_time:.2f}|"
            f"{status}|"
            f"{str(request)[:500].replace('\n', ' ')}|"
            f"{str(response)[:500].replace('\n', ' ')}\n"
        )
        with open(self.log_file, "a", encoding='utf-8') as f:
            f.write(log_entry)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Отправляет сообщение при получении команды /start"""
        user = update.effective_user
        await update.message.reply_html(
            f"""Здравствуйте, {user.mention_html()}!\nЯ ваш юридический помощник.\nЯ могу проконсультировать Вас по следующим законам:\nГражданский кодекс\nКодекс РФ об административных нарушениях\nСемейный кодекс\nТрудовой кодекс\nУголовный кодекс.\n\nЧем могу помочь?""",
            reply_markup=ForceReply(selective=True),
        )

    async def reset_context(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Сбрасывает контекст диалога"""
        user_id = update.message.from_user.id
        self.user_contexts[user_id] = []
        await update.message.reply_text("Контекст диалога сброшен. Начинаем новый диалог.")

    def _get_user_context(self, user_id: int) -> list:
        """Возвращает контекст пользователя или создает новый"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = []
        return self.user_contexts[user_id]

    def _update_user_context(self, user_id: int, role: str, message: str) -> None:
        """Обновляет контекст пользователя"""
        context = self._get_user_context(user_id)
        context.append({"role": role, "text": message})

        # Ограничиваем длину контекста
        if len(context) > MAX_CONTEXT_LENGTH * 2:
            context = context[-MAX_CONTEXT_LENGTH * 2:]
            self.user_contexts[user_id] = context

    async def _call_deepseek(self, user_id: int, message: str) -> str:
        """Вызывает Deepseek API с ретраями и логированием"""
        relevant_docs = self.rag.search_relevant_documents(message)
        docs_context = "\n\n".join(relevant_docs)

        context = self._get_user_context(user_id)
        messages = [{"role": msg["role"], "content": msg["text"]}
                    for msg in context]
        messages.append({"role": "user", "content": message})

        system_prompt = """ 
        Вы высококвалифицированный юридический помощник Yur-bot. Правила:
            1. Отвечайте строго по российскому законодательству
            2. Указывайте конкретные статьи законов
            3. Пишите профессиональным, но понятным языком
            4. Добавляйте в конце: "Мои ответы не заменяют необходимость консультироваться с профессиональным юристом!"

            Контекст: {context}""".format(context=docs_context if docs_context else "нет дополнительного контекста")  # Ваш промпт
        messages.insert(0, {"role": "system", "content": system_prompt})

        headers = {
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 3000
        }

        max_retries = 3
        retry_delay = 5
        last_error = "Неизвестная ошибка"

        for attempt in range(max_retries):
            start_time = time.time()
            try:
                response = requests.post(
                    DEEPSEEK_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response_time = time.time() - start_time

                if response.status_code == 200:
                    result = response.json().get("choices", [{}])[0].get(
                        "message", {}).get("content", "")
                    await self._log_request(
                        user_id,
                        start_time,
                        "SUCCESS",
                        # Логируем только часть данных
                        {"messages": messages[:3], "model": DEEPSEEK_MODEL},
                        result  # Логируем только начало ответа
                    )
                    return result

                last_error = f"HTTP {response.status_code}"
                await self._log_request(
                    user_id,
                    start_time,
                    f"ERROR_{response.status_code}",
                    {"messages": messages[:3], "model": DEEPSEEK_MODEL},
                    response.text
                )

            except requests.exceptions.Timeout:
                last_error = "Timeout"
                await self._log_request(
                    user_id,
                    start_time,
                    "TIMEOUT",
                    {"messages": messages[:3], "model": DEEPSEEK_MODEL}
                )
            except Exception as e:
                last_error = str(e)
                await self._log_request(
                    user_id,
                    start_time,
                    "CONNECTION_ERROR",
                    {"messages": messages[:3], "model": DEEPSEEK_MODEL},
                    str(e)
                )

            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))

        return f"Ошибка сервиса: {last_error}. Попробуйте позже."

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обрабатывает текстовые сообщения"""
        user_id = update.message.from_user.id
        user_message = update.message.text

        self._update_user_context(user_id, "user", user_message)

        try:
            bot_response = await self._call_deepseek(user_id, user_message)
            self._update_user_context(user_id, "assistant", bot_response)
            await update.message.reply_text(bot_response)
        except Exception as e:
            print(f"Ошибка: {e}")
            await update.message.reply_text("Произошла ошибка. Попробуйте позже.")


def main() -> None:
    """Запускает бота"""
    bot = AIChatBot()
    application = Application.builder().token(os.getenv('TELEGRAM_TOKEN')).build()
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("reset", bot.reset_context))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, bot.handle_message))

    print("Бот запущен...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()


# from telegram import Update, ForceReply
# from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
# import requests
# import json
# import os
# from dotenv import load_dotenv
# from rag_processor import RAGProcessor
# from config import MAX_CONTEXT_LENGTH, YANDEX_GPT_API_URL, YANDEX_GPT_MODEL

# load_dotenv()

# class AIChatBot:
#     def __init__(self):
#         self.rag = RAGProcessor()
#         self.rag.load_and_process_documents()
#         self.user_contexts = {}  # Хранит контексты диалогов по user_id

#     async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#         """Отправляет сообщение при получении команды /start"""
#         user = update.effective_user
#         await update.message.reply_html(
#             f"Здравствуйте, {user.mention_html()}! Я ваш юридический помощник. Я могу помочь вам с ответом по следующим законам: Гражданский кодекс, Кодекс РФ об административных нарушениях, Семейный кодекс, Трудовой кодекс, Уголовный кодекс. Чем могу помочь?",
#             reply_markup=ForceReply(selective=True),
#         )

#     async def reset_context(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#         """Сбрасывает контекст диалога"""
#         user_id = update.message.from_user.id
#         self.user_contexts[user_id] = []
#         await update.message.reply_text("Контекст диалога сброшен. Начинаем новый диалог.")

#     def _get_user_context(self, user_id: int) -> list:
#         """Возвращает контекст пользователя или создает новый"""
#         if user_id not in self.user_contexts:
#             self.user_contexts[user_id] = []
#         return self.user_contexts[user_id]

#     def _update_user_context(self, user_id: int, role: str, message: str) -> None:
#         """Обновляет контекст пользователя"""
#         context = self._get_user_context(user_id)
#         context.append({"role": role, "text": message})

#         # Ограничиваем длину контекста
#         if len(context) > MAX_CONTEXT_LENGTH * 2:  # Умножаем на 2, так как храним и user и assistant сообщения
#             context = context[-MAX_CONTEXT_LENGTH * 2:]
#             self.user_contexts[user_id] = context

#     async def _call_yandex_gpt(self, user_id: int, message: str) -> str:
#         """Вызывает YandexGPT API с улучшенным промптом для юридических вопросов"""
#         # Получаем релевантные документы
#         relevant_docs = self.rag.search_relevant_documents(message)
#         docs_context = "\n\n".join(relevant_docs)

#         # Формируем контекст диалога
#         context = self._get_user_context(user_id)
#         messages = [{"role": msg["role"], "text": msg["text"]} for msg in context]
#         messages.append({"role": "user", "text": message})

#         # Создаем системный промпт для юридического помощника
#         system_prompt = """
#         ###ВАША РОЛЬ###
#             Вы высококвалифицированный юридический помощник. Вас зовут "Yur-bot”.
#             Вы разговариваете с людьми 14-70 лет и помогаете им с юридическими вопросами.

#         ###ПРАВИЛА ВЫВОДА###
#             1. Пишите только на русском языке, пишите формально, как юрист, но помните, что вы -
#             профессионал, объясняющий пользователю сложную тему простым языком;
#             2. Ответ должен быть развернутым и давать полную картину пользователю по заданному вопросу.
#             3. Ответ должен содержать ссылки на конкретные статьи и законы, на основе которых он сформулирован
#             нормативные акты, на которых основан ответ.
#             4. Ответ обязательно должен содержать дисклеймер: В конце каждого ответа добавляйте фразу:
#             "Данные ответы не заменяют необходимость консультироваться с профессиональными юристами"

#             ###ЗАПРЕЩЕНО###
#             1. Использовать "**" и "##" для форматирования вашего ответа;
#             2. Ссылаться на любые сайты в интернете;

#             ###КОНТЕКСТ ОТВЕТА###
#             {context}""".format(context=docs_context if docs_context else "нет дополнительного контекста")

#         messages.insert(0, {"role": "system", "text": system_prompt})

#         payload = {
#             "modelUri": f"gpt://{os.getenv('YANDEX_FOLDER_ID')}/yandexgpt-lite",
#             "completionOptions": {
#                 "stream": False,
#                 "temperature": 0.6,  # Чуть меньше креативности для юридических ответов
#                 "maxTokens": 3000
#             },
#             "messages": messages
#         }

#         headers = {
#             "Authorization": f"Api-Key {os.getenv('YANDEX_API_KEY')}",
#             "Content-Type": "application/json"
#         }

#         try:
#             response = requests.post(
#                 "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
#                 headers=headers,
#                 json=payload
#             )
#             response.raise_for_status()
#             return response.json()["result"]["alternatives"][0]["message"]["text"]
#         except Exception as e:
#             print(f"Error calling YandexGPT: {str(e)}")
#             return "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."

#     async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#         """Обрабатывает текстовые сообщения"""
#         user_id = update.message.from_user.id
#         user_message = update.message.text

#         # Обновляем контекст
#         self._update_user_context(user_id, "user", user_message)

#         try:
#             # Получаем ответ от YandexGPT
#             bot_response = await self._call_yandex_gpt(user_id, user_message)

#             # Обновляем контекст ответом
#             self._update_user_context(user_id, "assistant", bot_response)

#             await update.message.reply_text(bot_response)
#         except Exception as e:
#             print(f"Ошибка: {e}")
#             await update.message.reply_text("Произошла ошибка при обработке запроса. Попробуйте позже.")

# def main() -> None:
#     """Запускает бота"""
#     bot = AIChatBot()

#     # Создаем Application и передаем токен бота
#     application = Application.builder().token(os.getenv('TELEGRAM_TOKEN')).build()

#     # Регистрируем обработчики команд
#     application.add_handler(CommandHandler("start", bot.start))
#     application.add_handler(CommandHandler("reset", bot.reset_context))

#     # Регистрируем обработчик текстовых сообщений
#     application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

#     # Запускаем бота
#     print("Бот запущен...")
#     application.run_polling(allowed_updates=Update.ALL_TYPES)

# if __name__ == "__main__":
#     main()
