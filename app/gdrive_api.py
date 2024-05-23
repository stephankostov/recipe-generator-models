from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.http import MediaFileUpload
import io
from googleapiclient.errors import HttpError
from pathlib import Path
import json
import os
import streamlit as st

scope = ['https://www.googleapis.com/auth/drive']

with open('./env/gdrive-creds.json', 'r') as f: 
    service_account_info = json.load(f)

service_account_info = {
    "client_email": st.secrets['GDRIVE_CLIENT_EMAIL'],
    "client_id": st.secrets['GDRIVE_CLIENT_ID'],
    "private_key_id": st.secrets['GDRIVE_API_KEY_ID'],
    "private_key": st.secrets['GDRIVE_API_KEY'],
    **service_account_info,
}

credentials = service_account.Credentials.from_service_account_info(
                              service_account_info, 
                              scopes=scope)
service = build('drive', 'v3', credentials=credentials)

def download_gdrive_folder(folder_name, download_dir):

    results = service.files().list(pageSize=1000, fields="nextPageToken, files(id, name, mimeType)", q=f'name contains "{folder_name}" and mimeType = "application/vnd.google-apps.folder"').execute()
    folder_id = results.get('files', [])[0]['id']

    results = service.files().list(pageSize=1000, fields="nextPageToken, files(id, name, mimeType)", q=f"'{folder_id}' in parents").execute()
    files = results.get('files', [])

    for file in files:
        try:
            request_file = service.files().get_media(fileId=file['id'])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request_file)
            done = False
            while done == False:
                status, done = downloader.next_chunk()
                print(F'Download {int(status.progress() * 100)}.')
            with io.open(download_dir/file['name'],'wb') as f:
                fh.seek(0)
                f.write(fh.read())
        except HttpError as error:
            print(f"An error occurred: {error}")
            file = None