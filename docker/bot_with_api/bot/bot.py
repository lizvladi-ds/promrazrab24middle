import logging
import os
import pandas as pd
import requests

from telegram import ForceReply, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    ConversationHandler,
    CallbackQueryHandler,
)
from telegram import (
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

_APP_ADDRESS = "http://app:5001/predict"
CHOICE, FILE, PREDICT, LOOP = range(4)

keyboard = [
    [InlineKeyboardButton("Да", callback_data="Да")],
    [InlineKeyboardButton("Нет", callback_data="Нет")],
]
_REPLY_MARKUP = InlineKeyboardMarkup(keyboard)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Вывод подсказок.
    """
    await update.message.reply_text(
        "Чтобы загрузить файлы для предсказания введи /predict"
    )


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Прекращение диалога.
    """
    user = update.message.from_user
    logger.info("Пользователь %s остановил диалог.", user.first_name)

    await update.message.reply_text(
        "Будем на связи.", reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Начало диалога.
    """
    user = update.effective_user
    logger.info("Пользователь %s начал диалог.", user.first_name)

    await update.message.reply_text(
        f"Привет, {user.first_name}. "
        "В любой момент диалог можно прекратить с помощью команды /stop.\n\n"
        "Для получения подсказок введи /help.\n"
        "Хочешь получить предсказание?.\n\n",
        reply_markup=_REPLY_MARKUP,
    )
    return CHOICE


async def choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Развилка да/нет.
    """
    query = update.callback_query
    await query.answer()
    answer = query.data
    logger.info("Пользователь ответил %s." % answer)

    if answer.lower() == "нет":
        logger.info("Пользователь ответил %s. Окончание диалога." % answer)
        await update.callback_query.message.reply_text(
            "Будем на связи.", reply_markup=ReplyKeyboardRemove()
        )
        return ConversationHandler.END
    else:
        await update.callback_query.message.reply_text("Загрузи файл для предсказания")
        return PREDICT


async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Принятие пользовательского датасета.
    """
    logger.info(str(update.message.document))
    logger.info(update.message.document.file_name)
    file_name = update.message.document.file_name
    new_file = await update.message.effective_attachment.get_file(
        read_timeout=60, write_timeout=60, connect_timeout=60, pool_timeout=60
    )
    await new_file.download_to_drive(file_name)
    context.user_data["file_name"] = file_name

    logger.info("Файл сохранен в память.")
    await update.message.reply_text(
        "Файл получен, готовим предсказание. Введи любое сообщение."
    )

    return LOOP


async def loop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Подготовка и отправка предсказания.
    """
    user = update.message.from_user
    file_name = context.user_data["file_name"]

    files = {"file": open(file_name, "rb")}
    r = requests.post(_APP_ADDRESS, files=files).json()
    logger.info("Предсказание получено.")

    result_name = f"{user.first_name}_result.csv"
    pd.DataFrame(r["data"], columns=r["columns"], index=r["index"]).to_csv(
        result_name, index=False
    )
    logger.info("Результат сохранен локально")

    chat_id = update.effective_chat.id
    await context.bot.send_document(chat_id=chat_id, document=result_name)
    logger.info("Результат возвращен пользователю")

    await update.message.reply_text(
        "Сделать еще одно предсказание?", reply_markup=_REPLY_MARKUP
    )

    return CHOICE


def main(TOKEN:str) -> None:
    application = Application.builder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        allow_reentry=True,
        entry_points=[CommandHandler("start", start)],
        states={
            CHOICE: [CallbackQueryHandler(choice)],
            PREDICT: [MessageHandler(filters.ATTACHMENT, predict)],
            LOOP: [MessageHandler(filters.TEXT, loop)],
        },
        fallbacks=[CommandHandler("stop", stop_command)],
    )

    application.add_handler(conv_handler)

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    TOKEN = os.getenv("BOT_TOKEN", "")
    logger.info(TOKEN)
    main(TOKEN)
