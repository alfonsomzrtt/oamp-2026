# ── Bagian dari ui/game_screen.py — TimeIn.__init__ dan _poll ────────────────

class TimeIn(customtkinter.CTk):

    def __init__(self):
        super().__init__()
        # ... semua setup UI yang ada ...

        # ── Thread queues ─────────────────────────────────────────────────────
        self._frame_queue     = queue.Queue(maxsize=2)  # CameraThread → DetectionThread
        self._detection_queue = queue.Queue(maxsize=2)  # DetectionThread → main thread

        # ── Face assets untuk overlay ─────────────────────────────────────────
        # Format: list of (face_img_rgb, mask) index 0-based (face 1 = index 0)
        _face_assets = [
            (face_01, mask_face_01),
            (face_02, mask_face_02),
            (face_03, mask_face_03),
            (face_04, mask_face_04),
            (face_05, mask_face_05),
            (face_06, mask_face_06),
        ]

        # ── Worker threads ────────────────────────────────────────────────────
        self._camera_thread = CameraThread(
            self._frame_queue,
            mirror_x=CAMERA_MIRROR_X,
            mirror_y=CAMERA_MIRROR_Y,
            zoom=CAMERA_ZOOM,
            calibration=CAMERA_CALIBRATION,
            brightness=CAMERA_BRIGHTNESS,
            contrast=CAMERA_CONTRAST,
            saturation=CAMERA_SATURATION,
        )

        self._detection_thread = DetectionThread(
            model_yolo,
            USE_BANTAL_MODEL,
            self._frame_queue,
            self._detection_queue,
            yolo_infer_size=YOLO_INFER_SIZE,
            conf_threshold=0.7,
            face_assets=tuple(_face_assets),
        )

        # ── State ─────────────────────────────────────────────────────────────
        self._latest_result: DetectionResult | None = None
        self._game_running  = False
        self._poll_job      = None          # after() handle — untuk cancel

    # ── Game start / stop ─────────────────────────────────────────────────────

    def _start_game(self):
        """Dipanggil setelah countdown selesai."""
        self._game_running = True
        self._camera_thread.start()
        self._detection_thread.start()
        self._schedule_poll()

        # ... setup level, timer, dst ...

    def _stop_game(self):
        self._game_running = False
        if self._poll_job:
            self.after_cancel(self._poll_job)
            self._poll_job = None
        self._camera_thread.stop()
        self._detection_thread.stop()

    # ── Poll loop — SATU-SATUNYA entry point dari background ke GUI ───────────

    def _schedule_poll(self):
        """Reschedule poll. 33ms ≈ 30fps UI refresh."""
        if self._game_running:
            self._poll_job = self.after(33, self._poll)

    def _poll(self):
        """
        Main thread GUI poll.
        Ambil hasil terbaru dari detection_queue, update UI, cek jawaban.
        NON-BLOCKING — tidak ada I/O, tidak ada sleep.
        """
        # Drain queue — ambil yang paling baru
        result: DetectionResult | None = None
        while not self._detection_queue.empty():
            try:
                result = self._detection_queue.get_nowait()
            except queue.Empty:
                break

        if result is not None:
            self._latest_result = result

            # Update camera display
            self._update_camera_display(result.img_display)

            # Cek jawaban jika 4 block valid
            if result.sorted_design:
                self._check_answer(result.sorted_design)

        self._schedule_poll()

    # ── Camera display update ─────────────────────────────────────────────────

    def _update_camera_display(self, frame: np.ndarray):
        if HIDE_CAMERA or self.camera is None:
            return
        try:
            cw = self.video_frame_1.winfo_width()
            ch = self.video_frame_1.winfo_height()
            if cw < 2 or ch < 2:
                return
            h, w = frame.shape[:2]
            scale = min(cw / w, ch / h)
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_LINEAR
                )
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self._last_photo_ref = imgtk   # cegah GC
            self.camera.configure(image=imgtk)
        except Exception:
            pass

    # ── Answer checker — dipanggil dari main thread ───────────────────────────

    def _check_answer(self, sorted_design: list):
        """
        Pure state machine — tidak ada I/O.
        Semua sfx dipanggil via threading.Thread agar tidak blocking.
        """
        lvl = self.current_question
        flag_attr = f"task_flag_{lvl:02d}"
        timer_attr = f"timer_task_{lvl:02d}"

        if not getattr(self, flag_attr, False):
            return

        expected = LEVEL_ANSWERS.get(self.current_variant, [])
        if sorted_design != expected:
            return

        # ── Level complete ────────────────────────────────────────────────────
        elapsed = round(time.time() - self.start_task - timer_return, 2)
        setattr(self, timer_attr, elapsed)
        self.timer_task_all.append(elapsed)
        self.cognitive_age_list.append(self.estimate_cognitive_age(elapsed))
        self.variant_played_list.append(self.current_variant)
        setattr(self, flag_attr, False)

        print(f"TASK {lvl} COMPLETED in {elapsed}s")
        self._api_event("level_complete", level=lvl, time_sec=elapsed)

        # SFX di thread terpisah — sd.play() tidak blocking tapi sfx function
        # kadang lambat karena numpy concat; aman dijalankan di daemon thread
        self._play_sfx_async(elapsed)

        if lvl >= self.max_level:
            self.end_test()
            return

        # ── Advance level ─────────────────────────────────────────────────────
        self.current_question = lvl + 1
        variant = self.get_random_variant(self.current_question)
        self.current_variant = variant

        self.load_level_image(variant)
        self.current_level_button.grid_remove()
        self.show_current_level_button(self.current_question)
        self.start_task = time.time()
        self._api_event("level_start", level=self.current_question)
        self.reset_timer()
        self.start_timer()
        self._update_level_badge(lvl, state="completed")

        # next_level sfx setelah jeda kecil
        self.after(200, lambda: threading.Thread(
            target=lambda: play_sfx("next_level"), daemon=True
        ).start())

    def _play_sfx_async(self, elapsed: float):
        """Pilih dan play sfx di daemon thread."""
        if   elapsed < 10: effect = "amazing"
        elif elapsed < 15: effect = "great"
        elif elapsed < 20: effect = "solid"
        elif elapsed < 25: effect = "good"
        elif elapsed < 30: effect = "keep_going"
        else:              effect = "dont_give_up"
        threading.Thread(target=lambda: play_sfx(effect), daemon=True).start()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self):
        self._stop_game()
        if hasattr(self, "mp_hands_detector") and self.mp_hands_detector:
            self.mp_hands_detector.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()