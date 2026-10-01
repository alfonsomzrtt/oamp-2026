from PIL import Image

# 1. Buka file gambar hasil download/generasi
input_image_path = "Gemini_Generated_Image_2douia2douia2dou.jpg"  # Sesuaikan dengan nama file gambarmu

# 2. Buka gambar dengan Pillow
img = Image.open(input_image_path)

# 3. Simpan ulang sebagai file .ICO Windows yang sah (lengkap dari 256x256 sampai 16x16)
output_ico_path = "oamp_app.ico"
img.save(
    output_ico_path,
    format="ICO",
    sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)],
)

print(
    f"✅ File berhasil dikonversi ke .ICO resmi Windows: {output_ico_path}"
)