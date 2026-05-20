"""
╔══════════════════════════════════════════════════════╗
║   HexSoftwares Internship — Task 2                   ║
║   Face Detection using OpenCV in Python              ║
║   GUI Application (Tkinter + OpenCV)                 ║
╚══════════════════════════════════════════════════════╝

GitHub Repo Name: HexSoftwares_Face_Detection
"""

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import threading

# ─── Load Haar Cascade (OpenCV built-in) ───────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ─── Color Theme ───────────────────────────────────────
BG       = "#1e1e2e"
CARD     = "#2a2a3e"
ACCENT   = "#00d4aa"
BTN_BG   = "#00d4aa"
BTN_FG   = "#1e1e2e"
TEXT     = "#ffffff"
SUBTEXT  = "#aaaacc"
RED      = "#ff6b6b"

# ═══════════════════════════════════════════════════════
#  MAIN APP CLASS
# ═══════════════════════════════════════════════════════
class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HexSoftwares — Face Detection AI")
        self.root.geometry("1000x680")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self.cap = None                  # webcam capture
        self.webcam_running = False
        self.current_image = None        # original loaded image
        self.processed_image = None      # image after detection
        self.face_count = 0

        self._build_ui()

    # ─── BUILD UI ────────────────────────────────────────
    def _build_ui(self):
        # ── Top Header ──
        header = tk.Frame(self.root, bg=ACCENT, height=55)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(header, text="🎯  Face Detection using OpenCV  |  HexSoftwares Internship",
                 font=("Segoe UI", 14, "bold"), bg=ACCENT, fg=BG).pack(side="left", padx=20, pady=12)

        # ── Main Layout: Left Panel + Right Canvas ──
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # LEFT PANEL
        left = tk.Frame(main, bg=CARD, width=230, bd=0)
        left.pack(side="left", fill="y", padx=(0,10))
        left.pack_propagate(False)

        tk.Label(left, text="⚙  Controls", font=("Segoe UI", 12, "bold"),
                 bg=CARD, fg=ACCENT).pack(pady=(18,8), padx=10)

        self._separator(left)

        # ── Buttons ──
        self._btn(left, "📁  Load Image", self.load_image)
        self._btn(left, "🔍  Detect Faces", self.detect_faces_image)
        self._btn(left, "💾  Save Result", self.save_result)
        self._separator(left)
        self._btn(left, "📷  Start Webcam", self.start_webcam, color="#5599ff")
        self._btn(left, "⏹  Stop Webcam",  self.stop_webcam,  color=RED)
        self._separator(left)

        # ── Settings ──
        tk.Label(left, text="🎛  Settings", font=("Segoe UI", 10, "bold"),
                 bg=CARD, fg=ACCENT).pack(pady=(6,2), padx=10, anchor="w")

        # Scale Factor
        tk.Label(left, text="Scale Factor:", bg=CARD, fg=SUBTEXT,
                 font=("Segoe UI", 9)).pack(padx=14, anchor="w")
        self.scale_var = tk.DoubleVar(value=1.1)
        tk.Scale(left, from_=1.05, to=1.5, resolution=0.05,
                 orient="horizontal", variable=self.scale_var,
                 bg=CARD, fg=TEXT, highlightthickness=0,
                 troughcolor=BG, activebackground=ACCENT).pack(fill="x", padx=14)

        # Min Neighbours
        tk.Label(left, text="Min Neighbours:", bg=CARD, fg=SUBTEXT,
                 font=("Segoe UI", 9)).pack(padx=14, anchor="w")
        self.neighbours_var = tk.IntVar(value=5)
        tk.Scale(left, from_=1, to=10, resolution=1,
                 orient="horizontal", variable=self.neighbours_var,
                 bg=CARD, fg=TEXT, highlightthickness=0,
                 troughcolor=BG, activebackground=ACCENT).pack(fill="x", padx=14)

        # Detect Eyes checkbox
        self.detect_eyes_var = tk.BooleanVar(value=False)
        tk.Checkbutton(left, text="Detect Eyes too",
                       variable=self.detect_eyes_var,
                       bg=CARD, fg=TEXT, selectcolor=BG,
                       activebackground=CARD,
                       font=("Segoe UI", 9)).pack(padx=14, anchor="w", pady=4)

        self._separator(left)

        # ── Stats Card ──
        tk.Label(left, text="📊  Stats", font=("Segoe UI", 10, "bold"),
                 bg=CARD, fg=ACCENT).pack(pady=(4,4), padx=10, anchor="w")

        self.faces_label = tk.Label(left, text="Faces: 0",
                                    font=("Segoe UI", 22, "bold"),
                                    bg=CARD, fg=ACCENT)
        self.faces_label.pack()

        self.status_label = tk.Label(left, text="Load an image to begin",
                                     font=("Segoe UI", 8), bg=CARD, fg=SUBTEXT,
                                     wraplength=200, justify="center")
        self.status_label.pack(pady=6, padx=8)

        # RIGHT: Canvas area
        right = tk.Frame(main, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        # Canvas with scrollbars
        canvas_frame = tk.Frame(right, bg=CARD, bd=2, relief="flat")
        canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#111122",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)

        # Placeholder text
        self.placeholder_id = self.canvas.create_text(
            450, 300,
            text="📷  Load an image or start webcam\n\nFaces will be detected automatically",
            font=("Segoe UI", 14), fill=SUBTEXT, justify="center"
        )

        # Bottom status bar
        self.statusbar = tk.Label(self.root,
                                  text="Ready  |  HexSoftwares Face Detection  |  OpenCV " + cv2.__version__,
                                  font=("Segoe UI", 8), bg="#111122", fg=SUBTEXT, anchor="w")
        self.statusbar.pack(fill="x", side="bottom", padx=10, pady=2)

    # ─── HELPER WIDGETS ──────────────────────────────────
    def _btn(self, parent, text, cmd, color=BTN_BG):
        btn = tk.Button(parent, text=text, command=cmd,
                        font=("Segoe UI", 9, "bold"),
                        bg=color, fg=BTN_FG if color == BTN_BG else TEXT,
                        relief="flat", bd=0, cursor="hand2",
                        activebackground=ACCENT, activeforeground=BG,
                        pady=7)
        btn.pack(fill="x", padx=14, pady=4)

    def _separator(self, parent):
        tk.Frame(parent, bg="#444466", height=1).pack(fill="x", padx=10, pady=6)

    # ─── STATUS HELPERS ──────────────────────────────────
    def _set_status(self, msg, color=SUBTEXT):
        self.status_label.config(text=msg, fg=color)
        self.statusbar.config(text=msg)

    def _set_face_count(self, n):
        self.face_count = n
        self.faces_label.config(text=f"Faces: {n}",
                                fg=ACCENT if n > 0 else RED)

    # ─── SHOW IMAGE ON CANVAS ────────────────────────────
    def _show_on_canvas(self, cv_img):
        """Convert OpenCV BGR image → display on Tkinter Canvas."""
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Fit to canvas size
        cw = self.canvas.winfo_width()  or 760
        ch = self.canvas.winfo_height() or 580
        pil_img.thumbnail((cw, ch), Image.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        x = cw // 2
        y = ch // 2
        self.canvas.create_image(x, y, anchor="center", image=self.tk_img)

    # ─── DETECT FACES (core logic) ───────────────────────
    def _run_detection(self, frame):
        """
        Run Haar-cascade face (and optionally eye) detection.
        Returns annotated frame + face count.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)           # improve low-light

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_var.get(),
            minNeighbors=self.neighbours_var.get(),
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        result = frame.copy()
        count = 0

        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                count += 1
                # Bounding box
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 212, 170), 2)

                # Label background
                label = f"Face #{i+1}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(result, (x, y-lh-10), (x+lw+8, y), (0, 212, 170), -1)
                cv2.putText(result, label, (x+4, y-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 40), 1)

                # Eye detection inside face ROI
                if self.detect_eyes_var.get():
                    roi_gray  = gray[y:y+h, x:x+w]
                    roi_color = result[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4,
                                                        minSize=(20, 20))
                    for (ex, ey, ew, eh) in eyes:
                        cv2.circle(roi_color,
                                   (ex + ew//2, ey + eh//2),
                                   ew//2, (255, 170, 0), 2)

        # Counter overlay (top-right)
        overlay_text = f"Detected: {count} face{'s' if count != 1 else ''}"
        cv2.rectangle(result, (0, 0), (250, 36), (0,0,0), -1)
        cv2.putText(result, overlay_text, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 212, 170) if count > 0 else (255, 80, 80), 2)

        return result, count

    # ─── LOAD IMAGE ──────────────────────────────────────
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff")]
        )
        if not path:
            return

        self.current_image = cv2.imread(path)
        if self.current_image is None:
            messagebox.showerror("Error", "Could not load image!")
            return

        self._show_on_canvas(self.current_image)
        self._set_status(f"Loaded: {os.path.basename(path)}", ACCENT)
        self._set_face_count(0)
        self.processed_image = None

    # ─── DETECT ON IMAGE ─────────────────────────────────
    def detect_faces_image(self):
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first!")
            return

        self._set_status("Detecting faces...", ACCENT)
        result, count = self._run_detection(self.current_image)
        self.processed_image = result
        self._show_on_canvas(result)
        self._set_face_count(count)

        if count == 0:
            self._set_status("No faces detected. Try adjusting settings.", RED)
        else:
            self._set_status(f"✅  {count} face(s) detected!", ACCENT)

    # ─── SAVE RESULT ─────────────────────────────────────
    def save_result(self):
        img = self.processed_image if self.processed_image is not None else self.current_image
        if img is None:
            messagebox.showwarning("Nothing to save", "No result to save yet!")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            initialfile="face_detected_result"
        )
        if path:
            cv2.imwrite(path, img)
            self._set_status(f"💾 Saved: {os.path.basename(path)}", ACCENT)
            messagebox.showinfo("Saved!", f"Result saved to:\n{path}")

    # ─── WEBCAM ──────────────────────────────────────────
    def start_webcam(self):
        if self.webcam_running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam Error",
                                 "Could not open webcam!\nMake sure it is connected.")
            return
        self.webcam_running = True
        self._set_status("📷 Webcam running — detecting live...", ACCENT)
        threading.Thread(target=self._webcam_loop, daemon=True).start()

    def _webcam_loop(self):
        while self.webcam_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            result, count = self._run_detection(frame)
            # Update GUI from main thread
            self.root.after(0, self._update_webcam_frame, result, count)

        if self.cap:
            self.cap.release()

    def _update_webcam_frame(self, frame, count):
        self._show_on_canvas(frame)
        self._set_face_count(count)

    def stop_webcam(self):
        self.webcam_running = False
        self._set_status("Webcam stopped.", SUBTEXT)

    # ─── CLOSE CLEANLY ───────────────────────────────────
    def on_close(self):
        self.webcam_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ═══════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
