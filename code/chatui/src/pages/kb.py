"""This module contains the frontend gui for chat."""
from pathlib import Path
from typing import List

import os
import gradio as gr

from src import assets, chat_client

PATH = "/kb"
TITLE = "Knowledge Base Management"
STATE_FILE = '/tmp/uploaded_files.txt'


def build_page(client: chat_client.ChatClient) -> gr.Blocks:
    """Buiild the gradio page to be mounted in the frame."""
    kui_theme, kui_styles = assets.load_theme("kaizen")

    with gr.Blocks(title=TITLE, theme=kui_theme, css=kui_styles) as page:
        # create the page header
        gr.Markdown(f"# {TITLE}")

        with gr.Row():
            upload_button = gr.UploadButton(
                "Add File", file_types=["pdf"], file_count="multiple"
            )
        with gr.Row():
            file_output = gr.File()

        with gr.Row():
            gr.Dataframe(
                headers=["File Uploaded"],
                datatype=["str"],
                col_count=(1, "fixed"),
                value=get_uploaded_files,
                every=1,
            )

        # form actions
        upload_button.upload(
            lambda files: upload_file(files, client), upload_button, file_output
        )

    page.queue()
    return page


def upload_file(files: List[Path], client: chat_client.ChatClient) -> List[str]:
    """Use the client to upload a file to the knowledge base."""
    try:
        file_paths = [file.name for file in files]
        client.upload_documents(file_paths = file_paths)

        # Save the uploaded file names to the state file
        with open(STATE_FILE, 'a') as file:
            for file_path in file_paths:
                file_path = os.path.basename(file_path)
                file.write(file_path + '\n')

        return file_paths
    except Exception as e:
        raise gr.Error(f"{e}")

def get_uploaded_files():
    """Load previously uploaded files if the file exists"""
    uploaded_files = [["No Files uploaded"]]
    if os.path.exists(STATE_FILE):
        uploaded_files = []
        with open(STATE_FILE, 'r') as file:
            for line in file.read().splitlines():
                uploaded_files.append([line])
    return uploaded_files