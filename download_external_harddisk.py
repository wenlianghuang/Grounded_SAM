import os
import requests

# 配置
API_KEY = '53877501-4a0695420783524274c4069f7'
QUERY = 'external hard drive'
DOWNLOAD_DIR = '/Volumes/T7_SSD/Object_Image/external_harddisk'
LIMIT = 100

def download_images():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    # Pixabay 每頁最多 200 張
    url = f"https://pixabay.com/api/?key={API_KEY}&q={QUERY}&image_type=photo&per_page={LIMIT}"
    
    response = requests.get(url)
    
    # 檢查 HTTP 狀態碼
    if response.status_code != 200:
        print(f"API 請求失敗，狀態碼: {response.status_code}")
        print(f"響應內容: {response.text[:500]}")
        return
    
    # 檢查響應是否為有效的 JSON
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError as e:
        print(f"無法解析 JSON 響應: {e}")
        print(f"響應內容前 500 字符: {response.text[:500]}")
        return

    if 'hits' not in data:
        print("未能獲取數據，請檢查 API Key")
        if 'error' in data:
            print(f"API 錯誤訊息: {data.get('error', '未知錯誤')}")
        return

    for i, hit in enumerate(data['hits']):
        img_url = hit['largeImageURL']
        try:
            img_data = requests.get(img_url).content
            file_name = f"{DOWNLOAD_DIR}/hard_drive_{i+1}.jpg"
            with open(file_name, 'wb') as f:
                f.write(img_data)
            print(f"已下載: {file_name}")
        except Exception as e:
            print(f"下載失敗 {img_url}: {e}")

if __name__ == "__main__":
    download_images()