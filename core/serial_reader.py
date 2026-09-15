"""
core/serial_reader.py — ESP32 serial reader thread.

Membaca pesan dari ESP32 via USB/Bluetooth RFCOMM secara non-blocking.
Pesan dimasukkan ke message_queue, diambil oleh main thread via get_message().
"""

from __future__ import annotations
import os
import platform
import queue
import threading
import serial
import serial.tools.list_ports


class SerialReaderThread(threading.Thread):
    """
    Daemon thread yang membaca satu baris per loop dari serial port.
    
    Cara pakai:
        t = SerialReaderThread()
        t.start()
        ...
        msg = t.get_message()   # non-blocking, return None jika kosong
        t.stop()
    """

    def __init__(self, port: str | None = None, baudrate: int = 115200):
        super().__init__(daemon=True, name="SerialReaderThread")
        self.baudrate      = baudrate
        self._conn: serial.Serial | None = None
        self._running      = True
        self.message_queue: queue.Queue[str] = queue.Queue(maxsize=10)

        resolved = port or self._auto_detect()
        if resolved:
            try:
                self._conn = serial.Serial(resolved, baudrate, timeout=0.1)
                print(f">>> [Serial] Connected to {resolved} @ {baudrate} baud")
            except Exception as e:
                print(f">>> [Serial] Failed to open {resolved}: {e}")
        else:
            print(">>> [Serial] No device found — SerialReaderThread idle")

    # ── Auto-detection ────────────────────────────────────────────────────────

    @staticmethod
    def _auto_detect() -> str | None:
        # Linux Bluetooth RFCOMM
        if platform.system() == "Linux" and os.path.exists("/dev/rfcomm0"):
            print(">>> [Serial] Found Bluetooth RFCOMM at /dev/rfcomm0")
            return "/dev/rfcomm0"

        _IDENTIFIERS = ("USB", "CH340", "CP210", "UART", "Serial", "Bluetooth")
        for port in serial.tools.list_ports.comports():
            if any(k in port.description for k in _IDENTIFIERS):
                print(f">>> [Serial] Found device at {port.device}: {port.description}")
                return port.device

        return None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def stop(self):
        self._running = False
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        print(">>> [Serial] Thread stopped")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_message(self) -> str | None:
        """Non-blocking. Return pesan terbaru atau None."""
        try:
            return self.message_queue.get_nowait()
        except queue.Empty:
            return None

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        if not self._conn:
            return  # tidak ada koneksi — thread langsung exit

        print(">>> [Serial] Reader started")
        while self._running:
            try:
                if self._conn.in_waiting > 0:
                    line = self._conn.readline().decode("utf-8", errors="replace").strip()
                    if line:
                        if not self.message_queue.full():
                            self.message_queue.put(line)
                        if line == "disable_image":
                            print(f">>> [Serial] Received: {line}")
            except Exception as e:
                print(f">>> [Serial] Read error: {e}")
                import time; import time as _t; _t.sleep(0.1)
            else:
                import time; time.sleep(0.01)