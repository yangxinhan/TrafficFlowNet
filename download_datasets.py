import os
import requests
import zipfile
from tqdm import tqdm
import gdown
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file_with_progress(url, filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 檢查是否成功
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            desc=filename
        ) as pbar:
            for data in response.iter_content(chunk_size=8192):
                size = f.write(data)
                pbar.update(size)
    except Exception as e:
        logger.error(f"下載 {filename} 時發生錯誤: {str(e)}")
        return False
    return True

def download_ua_detrac():
    data_dir = 'data/UA-DETRAC'
    os.makedirs(data_dir, exist_ok=True)
    
    # UA-DETRAC 官方下載連結
    files = {
        'DETRAC-Train-Data.zip': 'https://detrac-db.rit.albany.edu/Data/DETRAC-train-data.zip',
        'DETRAC-Train-Annotations-XML.zip': 'https://detrac-db.rit.albany.edu/Data/DETRAC-Train-Annotations-XML.zip'
    }
    
    for filename, url in files.items():
        output_path = os.path.join(data_dir, filename)
        if not os.path.exists(output_path):
            logger.info(f"下載 {filename}...")
            try:
                if download_file_with_progress(url, output_path):
                    logger.info(f"解壓 {filename}...")
                    try:
                        with zipfile.ZipFile(output_path, 'r') as zip_ref:
                            zip_ref.extractall(data_dir)
                        logger.info(f"{filename} 解壓完成")
                    except zipfile.BadZipFile:
                        logger.error(f"{filename} 解壓失敗，檔案可能已損壞")
                        if os.path.exists(output_path):
                            os.remove(output_path)
            except Exception as e:
                logger.error(f"下載 {filename} 失敗: {str(e)}")
                logger.info("""
                請手動下載 UA-DETRAC 數據集：
                1. 訪問 https://detrac-db.rit.albany.edu/
                2. 註冊並登入
                3. 下載 DETRAC-train-data.zip 和 DETRAC-Train-Annotations-XML.zip
                4. 將檔案放在 data/UA-DETRAC/ 目錄下
                """)
                return
        else:
            logger.info(f"{filename} 已存在")

def download_bdd():
    data_dir = 'data/BDD'
    os.makedirs(data_dir, exist_ok=True)
    
    # BDD100K 直接下載連結
    files = {
        'bdd100k_images.zip': 'https://bdd-data-storage.s3.amazonaws.com/bdd100k_images.zip',
        'bdd100k_labels.zip': 'https://bdd-data-storage.s3.amazonaws.com/bdd100k_labels.zip'
    }
    
    for filename, url in files.items():
        output_path = os.path.join(data_dir, filename)
        if not os.path.exists(output_path):
            logger.info(f"下載 {filename}...")
            if download_file_with_progress(url, output_path):
                logger.info(f"解壓 {filename}...")
                try:
                    with zipfile.ZipFile(output_path, 'r') as zip_ref:
                        zip_ref.extractall(data_dir)
                    logger.info(f"{filename} 解壓完成")
                except zipfile.BadZipFile:
                    logger.error(f"{filename} 解壓失敗")
        else:
            logger.info(f"{filename} 已存在")

def main():
    try:
        # 檢查並創建數據目錄
        os.makedirs('data', exist_ok=True)
        
        logger.info("開始下載 UA-DETRAC 數據集...")
        download_ua_detrac()
        
        logger.info("開始下載 Berkeley DeepDrive 數據集...")
        download_bdd()
        
        logger.info("所有數據集下載完成！")
        
    except Exception as e:
        logger.error(f"發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main()
