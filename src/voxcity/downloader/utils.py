# Utility functions for downloading files from various sources
import requests
import gdown

def download_file(url, filename):
    """Download a file from a URL and save it locally.
    
    This function uses the requests library to download a file from any publicly 
    accessible URL and save it to the local filesystem. It handles the download 
    process and provides feedback on the operation's success or failure.
    
    Args:
        url (str): URL of the file to download. Must be a valid, accessible URL.
        filename (str): Local path where the downloaded file will be saved.
                       Include the full path and filename with extension.
        
    Returns:
        None
        
    Prints:
        - Success message with filename if download is successful (status code 200)
        - Error message with status code if download fails
        
    Example:
        >>> download_file('https://example.com/file.pdf', 'local_file.pdf')
        File downloaded successfully and saved as local_file.pdf
    """
    # Attempt to download the file from the provided URL
    response = requests.get(url)
    
    # Check if the download was successful (HTTP status code 200)
    if response.status_code == 200:
        # Open the local file in binary write mode and save the content
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved as {filename}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def download_file_google_drive(file_id, output_path):
    """Download a file from Google Drive using its file ID.
    
    This function specifically handles downloads from Google Drive using the gdown
    library, which is designed to bypass Google Drive's download restrictions.
    It's useful for downloading large files or files that require authentication.
    
    Args:
        file_id (str): Google Drive file ID. This is the unique identifier in the 
                       sharing URL after '/d/' or 'id='.
        output_path (str): Local path where the downloaded file will be saved.
                          Include the full path and filename with extension.
        
    Returns:
        bool: True if download was successful, False if any error occurred
        
    Prints:
        Error message with exception details if download fails
        
    Example:
        >>> success = download_file_google_drive('1234abcd...', 'downloaded_file.zip')
        >>> if success:
        >>>     print("Download completed successfully")
    """
    # Construct the direct download URL using the file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        # Use gdown to handle the Google Drive download
        # quiet=False enables download progress display
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        print(f"Error downloading file {file_id}: {str(e)}")
        return False