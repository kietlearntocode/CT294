import os
import numpy as np
import pandas as pd
import tkinter as tk
from joblib import load
from tkinter import messagebox
from PIL import Image, ImageTk

# Tải mô hình đã huấn luyện
model = load(r"rforest_model.pkl")


# Hàm dự đoán trạng thái bệnh Parkinson
def predict_status(values):
    y_pred = model.predict([values])
    if y_pred == 1:
        return "Người này có dấu hiệu của bệnh parkinson"
    else:
        return "Người này không có dấu hiệu của bệnh parkinson"


# Hàm thêm placeholder vào entry
def add_placeholder(entry, placeholder_text):
    entry.insert(0, placeholder_text)
    entry.bind("<FocusIn>", lambda event: clear_placeholder(entry, placeholder_text))
    entry.bind("<FocusOut>", lambda event: restore_placeholder(entry, placeholder_text))


# Hàm xóa placeholder khi có sự kiện focus in
def clear_placeholder(entry, placeholder_text):
    if entry.get() == placeholder_text:
        entry.delete(0, tk.END)
        entry.config(bg="white",
                     fg="black",
                     highlightcolor="gray",
                     highlightbackground="gray")


# Hàm khôi phục placeholder nếu ô trống
def restore_placeholder(entry, placeholder_text):
    if not entry.get():
        entry.insert(0, placeholder_text)
        entry.config(bg="white",
                     fg="gray",
                     highlightcolor="gray",
                     highlightbackground="gray")


# Hàm tải dữ liệu từ file CSV
def load_csv_row(index):
    df_path = os.path.abspath(os.path.join(os.getcwd(), "..", "data", "processed", "X_test.csv"))
    df = pd.read_csv(df_path)
    return df.values[index]


# Hàm validate sau khi nhập
def validate_after(event):
    widget = event.widget
    value = widget.get()
    if value == "":
        widget.config(bg="white",
                      fg="gray",
                      highlightcolor="gray",
                      highlightbackground="gray")
        return
    try:
        float(value)
        widget.config(
            fg="black",
            bg="#69f18a",
            highlightcolor="#69f18a",
            highlightbackground="#69f18a"
        )
    except:
        if value.strip() == "":
            widget.config(bg="white")
        else:
            widget.config(
                bg="#ff8787",
                highlightcolor="#ff8787",
                highlightbackground="#ff8787"
            )


# Hàm tạo nhóm entry
def create_entry_group(parent, title, rows, cols, names):
    label = tk.Label(parent, text=title, font=("Helvetica", 14, "bold"), bg="#f0f2f5")
    label.pack(anchor="w", pady=(20, 5), padx=30)

    container = tk.Frame(parent, bg="#f0f2f5")
    container.pack()

    for i in range(rows):
        for j in range(cols):
            frame = tk.Frame(container, bg="#f0f2f5")
            frame.grid(row=i, column=j, padx=8, pady=6)

            entry = tk.Entry(
                frame,
                font=("Helvetica", 11),
                fg="gray",
                width=30,
                relief="solid",
                bd=1,
                highlightthickness=1,
                bg="white",
                justify="center",
                highlightcolor="gray",
                highlightbackground="gray"
            )
            entry.pack(padx=1, pady=1, ipady=6)
            entry.bind("<KeyRelease>", validate_after)
            add_placeholder(entry, f"{names[i][j]}")

            all_entries.append(entry)


# Hàm submit dữ liệu
def submit_data():
    values = []
    errors = []

    for idx, entry in enumerate(all_entries):
        val = entry.get().strip()

        if val.lower() == f"{namesProperties[idx]}" or val == "":
            errors.append(entry)
            continue

        try:
            num = float(val)
            values.append(num)
        except ValueError:
            errors.append(entry)

    if errors:
        messagebox.showerror("Lỗi nhập liệu",
                             "Vui lòng kiểm tra các ô bị tô đỏ:\n- Không được để trống\n- Chỉ được nhập số")
        for entry in errors:
            entry.config(bg="#ff8787",
                         highlightcolor="#ff8787",
                         highlightbackground="#ff8787")
        return
    predict_status(values)
    messagebox.showinfo("Kết quả", predict_status(values))


# Hover cho button
def on_enter(event, color):
    event.widget["foreground"] = "white"
    event.widget["activebackground"] = color


def on_leave(event, color, fg="black"):
    event.widget["background"] = color
    event.widget["foreground"] = fg


# Hàm xóa dữ liệu nhập
def clear_input():
    for idx, entry in enumerate(all_entries):
        entry.delete(0, tk.END)
        entry.config(bg="white",
                     fg="gray",
                     highlightcolor="gray",
                     highlightbackground="gray")
        add_placeholder(entry, f"{namesProperties[idx]}")


# Hàm chuyển đổi giữa các chức năng
def swap_option():
    global option
    option = (option + 1) % 3
    option_btn.config(text=option_text[option])
    for entry in all_entries:
        entry.config(state="normal")
    if option == 0:
        clear_input()
    else:
        data = load_csv_row(option - 1)
        for entry in all_entries:
            entry.delete(0, tk.END)
            entry.insert(0, str(data[all_entries.index(entry)]))
            entry.config(
                fg="black",
                bg="#69f18a",
                highlightcolor="#69f18a",
                highlightbackground="#69f18a",
                state="disabled"
            )


# Tạo cửa sổ chính
root = tk.Tk()
root.title("Chuẩn đoán Parkinson")
root.geometry("1280x800")
root.configure(bg="#f0f2f5")
root.resizable(False, False)
# root.iconbitmap("icon.ico")

main_frame = tk.Frame(root, bg="#f0f2f5")
main_frame.pack(padx=40, pady=20, fill="both", expand=True)

bg_image = Image.open(r"../imgs/background.png")
bg_image = bg_image.resize((1280, 800))
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(main_frame, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Danh sách chứa tất cả entry và tên thuộc tính
all_entries = []
namesProperties = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
                   "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
                   "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
                   "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
                   "NHR", "HNR", "RPDE", "DFA", "spread1",
                   "spread2", "D2", "PPE"]
# Tạo các nhóm
create_entry_group(main_frame, "Nhóm tần số cơ bản và dao động tần số", 2, 4,
                   [["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)"],
                    ["MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP"]])
create_entry_group(main_frame, "Nhóm biên độ và dao động biên độ", 2, 3,
                   [["MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3"], ["Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA"]])
create_entry_group(main_frame, "Nhóm tỷ lệ tiếng ồn – âm điều hòa", 1, 2, [["NHR", "HNR"]])
create_entry_group(main_frame, "Nhóm đặc trưng phi tuyến và hỗn loạn", 2, 3,
                   [["RPDE", "DFA", "spread1"], ["spread2", "D2", "PPE"]])

# Tạo nút và khung chứa nút
button_frame = tk.Frame(main_frame, bg="#f0f2f5")
button_frame.pack(pady=20, fill="x", expand=True)

btn_style = {
    "font": ("Helvetica", 14, "bold"),
    "width": 30,
    "height": 3,
    "relief": "groove",
    "bd": 2,
    "cursor": "hand2",
}

clear_btn = tk.Button(
    button_frame,
    text="Xóa dữ liệu",
    command=clear_input,
    bg="#ff8787",
    fg="black",
    **btn_style,
)
clear_btn.pack(side="left", padx=20, pady=10, expand=True)

submit_btn = tk.Button(
    button_frame,
    text="Chẩn đoán Parkinson",
    command=submit_data,
    bg="#69f18a",
    fg="black",
    **btn_style,
)
submit_btn.pack(side="right", padx=20, pady=10, expand=True)

option_text = ["Nhập dữ liệu thủ công", "Dữ liệu người không bệnh", "Dữ liệu người bị bệnh"]
option = 0
option_btn = tk.Button(
    button_frame,
    text=f"{option_text[option]}",
    bg="#3a70c2",
    fg="black",
    font=("Helvetica", 14, "bold"),
    width=30,
    height=3,
    relief="groove",
    bd=2,
    cursor="hand2",
    command=swap_option,
)
option_btn.pack(padx=20, pady=10, fill="x")

# Hover effect
clear_btn.bind("<Enter>", lambda event: on_enter(event, "red"))
clear_btn.bind("<Leave>", lambda event: on_leave(event, "#ff8787"))

submit_btn.bind("<Enter>", lambda event: on_enter(event, "green"))
submit_btn.bind("<Leave>", lambda event: on_leave(event, "#69f18a"))

option_btn.bind("<Enter>", lambda event: on_enter(event, "blue"))
option_btn.bind("<Leave>", lambda event: on_leave(event, "#3a70c2"))

root.mainloop()