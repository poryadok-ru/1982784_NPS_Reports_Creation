import uuid
from datetime import datetime
from bitrix24_sdk import BitrixClient
from config import BITRIX_TOKEN, BITRIX_USER_ID, BITRIX_NPS_REPORTS_FOLDER_ID


def upload_to_bitrix(pdf_path: str, store_name: str, period_str: str, logger) -> str:
    """Загрузить PDF отчет в Bitrix24 и вернуть публичную ссылку."""
    try:
        client = BitrixClient(token=BITRIX_TOKEN, user_id=BITRIX_USER_ID)
        folder_name = f"ПРК_{store_name}"
        target_folder_id = None

        nps_children = client.disk.get_children(id=BITRIX_NPS_REPORTS_FOLDER_ID)
        for item in nps_children.result:
            if item.name == folder_name and item.type == "folder":
                target_folder_id = item.id
                break
        if target_folder_id is None:
            new_folder = client.disk.add_subfolder(BITRIX_NPS_REPORTS_FOLDER_ID, {"NAME": folder_name})
            target_folder_id = new_folder.result.id

        with open(pdf_path, "rb") as f:
            file_content = f.read()

        # Генерируем уникальное имя файла с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        file_name = f"NPS_отчет_{store_name}_{period_str}_{timestamp}_{unique_id}.pdf"

        upload_result = client.disk.upload_file_complete(
            folder_id=target_folder_id,
            file_content=file_content,
            file_name=file_name
        )
        download_url = getattr(
            upload_result.result,
            'download_url',
            getattr(
                upload_result.result, 'url',
                f"https://bitrix24.com/disk/downloadFile/{upload_result.result.id}/"
            )
        )
        return download_url
    except Exception as e:
        logger.error(f"Ошибка при загрузке в Bitrix24 для {store_name}: {e}")
        raise