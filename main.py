import os
import flet as ft
from mobile_app import main  # imports the main UI function from your mobile_app.py

if __name__ == "__main__":
    # Render assigns the port dynamically using the PORT environment variable
    port = int(os.environ.get("PORT", 8550))
    ft.app(target=main, view=ft.AppView.WEB_BROWSER, host="0.0.0.0", port=port)