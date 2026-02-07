# Utility functions for downloading files from various sources
import requests


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