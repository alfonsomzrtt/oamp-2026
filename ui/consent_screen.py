"""
ui/consent_screen.py — Informed Consent modal window.

Dipanggil sekali saat startup jika consent belum diterima.
on_accept_callback() dipanggil setelah user setuju.
Jika user tolak → sys.exit(0).
"""

from __future__ import annotations
import sys
import customtkinter
from ui.theme import CLR, FONT, RADIUS
from core.game_logic import mark_consent_accepted

_CONSENT_TEXT = (
    "PERSETUJUAN PENGGUNA (INFORMED CONSENT) &\n"
    "KEBIJAKAN PRIVASI APLIKASI OAMP DESKTOP\n"
    "--------------------------------------------------------------------------------\n\n"
    "A. FORMULIR PERSETUJUAN PENGGUNA (INFORMED CONSENT)\n\n"
    "Dengan ini, saya menyatakan secara sadar dan sukarela bahwa:\n"
    "1. Saya menyetujui untuk berpartisipasi dalam sesi permainan"
    " balok desain (Block Design Test) Otak-Atik Merah Putih"
    " menggunakan aplikasi OAMP Desktop.\n"
    "2. Saya memahami bahwa aplikasi OAMP Desktop memanfaatkan pemrosesan"
    " Citra Digital/Kamera (Webcam) dan AI untuk mendeteksi"
    " susunan balok secara otomatis.\n"
    "3. Saya mengizinkan sistem untuk mencatat durasi waktu penyelesaian"
    " tes, level yang berhasil dicapai, serta skor penilaian"
    " visuospasial.\n"
    "4. Saya memahami bahwa partisipasi ini bersifat sukarela dan saya"
    " berhak menghentikan tes sewaktu-waktu jika merasa tidak nyaman.\n\n"
    "--------------------------------------------------------------------------------\n\n"
    "B. SYARAT & KETENTUAN PENGGUNAAN (USER AGREEMENT)\n\n"
    "1. Lisensi Penggunaan: Aplikasi OAMP Desktop disediakan khusus"
    " untuk keperluan asesmen kognitif, kegiatan edukasi, penelitian"
    " ilmiah, atau kompetisi resmi OAMP. Pengguna dilarang meretas,"
    " merusak, atau menyalahgunakan sistem backend API.\n\n"
    "2. Integritas Tes: Pengguna dilarang melakukan manipulasi terhadap"
    " umpan kamera maupun data konfigurasi lokal untuk merubah hasil"
    " penilaian kognitif secara tidak sah.\n\n"
    "3. Batasan Penilaian: Indikator 'Usia Kognitif' (Cognitive Age"
    " Index) dalam aplikasi disediakan semata-mata untuk analisis"
    " diagnostik awal dan statistik performa, serta TIDAK menggantikan"
    " diagnosis klinis/medis resmi dari tenaga profesional.\n\n"
    "--------------------------------------------------------------------------------\n\n"
    "C. PENGUMPULAN & PERLINDUNGAN DATA\n\n"
    "1. Data yang Dikumpulkan:\n"
    "   • Data Identitas Peserta: Nickname, Jenis Kelamin, dan UID.\n"
    "   • Data Performa Tes: Waktu penyelesaian per level, skor akhir.\n"
    "   • Configuration Log: Pengaturan perangkat lokal.\n\n"
    "2. Pemrosesan Video & Sensor Kamera:\n"
    "   • STREAM KAMERA HANYA DIPROSES SECARA REAL-TIME DI MEMORI LOKAL.\n"
    "   • SISTEM TIDAK MEREKAM, TIDAK MENYIMPAN, TIDAK MENGUNGGAH"
    " FOTO/VIDEO KE CLOUD ATAU PIHAK KETIGA.\n\n"
    "3. Penyimpanan Data Hasil Tes:\n"
    "   • Mode Online: Dikirim ke REST API / WebSocket Server OAMP.\n"
    "   • Mode Offline: Disimpan lokal di folder 'results/' (JSON).\n"
    "--------------------------------------------------------------------------------"
)


class ConsentScreen(customtkinter.CTkToplevel):
    """
    Modal informed consent.

    Cara pakai:
        root = customtkinter.CTk()
        root.withdraw()
        ConsentScreen(root, on_accept_callback=lambda: root.quit())
        root.mainloop()
    """

    def __init__(self, parent, on_accept_callback):
        super().__init__(parent)
        self.title("Persetujuan Pengguna & Informed Consent — OAMP Desktop")
        self.geometry("650x520")
        self.resizable(False, False)
        self._on_accept = on_accept_callback
        self.grab_set()
        self._center()
        self._build_ui()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _center(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth()  // 2) - (w // 2)
        y = (self.winfo_screenheight() // 2) - (h // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        customtkinter.CTkLabel(
            self,
            text="Formulir Persetujuan Pengguna & Kebijakan Privasi",
            font=customtkinter.CTkFont(size=18, weight="bold"),
            text_color=CLR.TEXT,
        ).pack(pady=(15, 5))

        # Scrollable consent text
        textbox = customtkinter.CTkTextbox(
            self, width=600, height=320,
            fg_color=CLR.CARD, text_color=CLR.TEXT,
            corner_radius=RADIUS.DEFAULT,
            font=customtkinter.CTkFont(family=FONT.P, size=13),
        )
        textbox.pack(pady=5, padx=20)
        textbox.insert("0.0", _CONSENT_TEXT)
        textbox.configure(state="disabled")

        # Agree checkbox
        self._agreed = customtkinter.StringVar(value="off")
        customtkinter.CTkCheckBox(
            self,
            text="Saya telah membaca, memahami, dan menyetujui seluruh ketentuan di atas.",
            variable=self._agreed,
            onvalue="on", offvalue="off",
            command=self._on_toggle,
            text_color=CLR.TEXT,
            fg_color=CLR.ACCENT,
            hover_color=CLR.ACCENT2,
        ).pack(pady=12)

        # Action buttons
        btn_row = customtkinter.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=5)

        customtkinter.CTkButton(
            btn_row, text="Tolak & Keluar",
            fg_color=CLR.DANGER, hover_color=CLR.DANGER_HOVER,
            command=self._on_cancel, width=140,
        ).pack(side="left", padx=10)

        self._accept_btn = customtkinter.CTkButton(
            btn_row, text="Setuju & Lanjutkan",
            state="disabled",
            fg_color=CLR.SUCCESS, hover_color=CLR.SUCCESS_HOVER,
            command=self._on_accept_click, width=160,
        )
        self._accept_btn.pack(side="right", padx=10)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_toggle(self):
        state = "normal" if self._agreed.get() == "on" else "disabled"
        self._accept_btn.configure(state=state)

    def _on_accept_click(self):
        mark_consent_accepted()
        self.grab_release()
        self.destroy()
        self._on_accept()

    def _on_cancel(self):
        sys.exit(0)