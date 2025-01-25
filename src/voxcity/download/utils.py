import requests
import gdown

def download_file(url, filename):
    """Download a file from a URL and save it locally.
    
    Args:
        url (str): URL of the file to download
        filename (str): Local path where the downloaded file will be saved
        
    Returns:
        None
        
    Prints:
        Success or failure message with status code
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved as {filename}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def download_file_google_drive(file_id, output_path):
    """Download a file from Google Drive using its file ID.
    
    Args:
        file_id (str): Google Drive file ID
        output_path (str): Local path where the downloaded file will be saved
        
    Returns:
        bool: True if download successful, False otherwise
        
    Prints:
        Error message if download fails
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        print(f"Error downloading file {file_id}: {str(e)}")
        return False