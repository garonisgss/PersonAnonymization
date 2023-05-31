import logging
import io
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from src.anonymization.anonymizer import anonymize, get_mask
from src.models.model import get_detectron_model

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


model = get_detectron_model("cpu", 0.25)


# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        "Send me a picture! I will blur all people captured in this picture."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Send me a picture! I will blur all people captured in this picture.")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    if len(update.message.photo) == 0:
        await update.message.reply_text("No pictures in message.")
    else:
        image_file = await update.message.photo[-1].get_file()
        in_memory_file = io.BytesIO(await image_file.download_as_bytearray())
        img = np.asarray(Image.open(in_memory_file))

        blur_kernel_size = int(max(img.shape[0], img.shape[1]) / 35)
        blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        human_mask = get_mask(img, model)
        img_anon = anonymize(img, human_mask, "median", blur_kernel_size, True)

        in_memory_file_out = io.BytesIO()
        img_out = Image.fromarray(np.uint8(img_anon))
        img_out.save(in_memory_file_out, "JPEG")
        in_memory_file_out.seek(0)
        await update.message.reply_photo(in_memory_file_out)


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token("6289760635:AAEM4HzYVplQl654TGhBiXutBS6nS9hGkbs").build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler((filters.PHOTO | filters.TEXT) & ~filters.COMMAND, echo))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()
