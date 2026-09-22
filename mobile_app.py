import flet as ft
import requests

# Put your computer's local Wi-Fi IP address here (e.g., from `ipconfig`)
SERVER_URL = "http://192.168.1.10:8000/scan"

def main(page: ft.Page):
    page.title = "VisionPass Mobile"
    page.theme_mode = ft.ThemeMode.DARK
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    status_text = ft.Text("Ready to Scan", size=16, color=ft.Colors.BLUE_400)
    log_column = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)

    def handle_upload(e: ft.FilePickerResultEvent, action: str):
        if not e.files:
            return

        status_text.value = f"Processing {action}..."
        status_text.color = ft.Colors.AMBER_400
        page.update()

        file_path = e.files[0].path
        try:
            with open(file_path, "rb") as img:
                res = requests.post(SERVER_URL, data={"action": action}, files={"file": img})
                data = res.json()

                if data.get("status") == "success":
                    status_text.value = f"Verified: {data['user']} ({action})"
                    status_text.color = ft.Colors.GREEN_400
                    log_column.controls.insert(
                        0,
                        ft.Text(f"[{data['timestamp']}] {data['user']} - {action}", color=ft.Colors.WHITE70)
                    )
                else:
                    status_text.value = f"Error: {data.get('message')}"
                    status_text.color = ft.Colors.RED_400
        except Exception as err:
            status_text.value = f"Connection Failed: {err}"
            status_text.color = ft.Colors.RED_400

        page.update()

    picker_in = ft.FilePicker(on_result=lambda e: handle_upload(e, "ENTRY"))
    picker_out = ft.FilePicker(on_result=lambda e: handle_upload(e, "EXIT"))
    page.overlay.extend([picker_in, picker_out])

    page.add(
        ft.Container(
            content=ft.Column(
                [
                    ft.Text("VisionPass Mobile", size=24, weight=ft.FontWeight.BOLD),
                    ft.Text("Biometric Attendance Scanner", size=13, color=ft.Colors.GREY_400),
                    ft.Divider(height=20),
                    status_text,
                    ft.ElevatedButton(
                        "📷 Scan Entry (Take Photo)",
                        bgcolor=ft.Colors.GREEN_700,
                        color=ft.Colors.WHITE,
                        width=300,
                        height=50,
                        on_click=lambda _: picker_in.pick_files(allow_multiple=False, file_type=ft.FilePickerFileType.IMAGE)
                    ),
                    ft.ElevatedButton(
                        "📷 Scan Exit (Take Photo)",
                        bgcolor=ft.Colors.RED_700,
                        color=ft.Colors.WHITE,
                        width=300,
                        height=50,
                        on_click=lambda _: picker_out.pick_files(allow_multiple=False, file_type=ft.FilePickerFileType.IMAGE)
                    ),
                    ft.Divider(height=20),
                    ft.Text("Recent Mobile Logs", size=16, weight=ft.FontWeight.W_600),
                    ft.Container(
                        content=log_column,
                        bgcolor=ft.Colors.SURFACE_VARIANT,
                        border_radius=10,
                        padding=10,
                        height=200,
                        width=320
                    )
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=20
        )
    )

if __name__ == "__main__":
    ft.app(target=main)