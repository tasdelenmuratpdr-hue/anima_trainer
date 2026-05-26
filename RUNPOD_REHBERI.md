# RunPod Çalıştırma Rehberi

## 1. Pod Aç

- [runpod.io](https://runpod.io) → **Deploy** → **GPU Cloud**
- Template: **RunPod PyTorch 2.4.0**
- GPU: **RTX A5000** (~$0.27/saat)
- **Expose HTTP Port**: `7860`
- → **Deploy**

---

## 2. Terminali Aç

Pod hazır olduktan sonra → **Connect → Start Web Terminal**

---

## 3. İlk Kurulum (sadece ilk seferde)

```bash
bash <(curl -s https://raw.githubusercontent.com/tasdelenmuratpdr-hue/anima_trainer/main/runpod_start.sh)
```

Bu komut her şeyi yapar:
- Repo'yu indirir
- sd-scripts ve paketleri kurar
- Text encoder + VAE indirir (~1.5 GB)
- Arayüzü başlatır

Terminalde `Running on http://0.0.0.0:7860` yazınca hazır.

---

## 4. Arayüze Bağlan

Pod sayfası → **Connect → HTTP Service [7860]**

---

## 5. Sonraki Oturumlarda (pod kapatılıp açıldıysa)

```bash
bash /workspace/anima-trainer/runpod_start.sh
```

Paket kurulumu atlanır, direkt başlar.

---

## 6. Eğitim Ayarları (önerilen)

| Ayar | Değer | Not |
|------|-------|-----|
| Repeats | 5 | |
| Max Epochs | 4 | 5. epoch stili aşırı pişiriyor |
| Resolution | 768 | **1024'ü geçme** — siyah çıktıya yol açar |
| Network Dim | 32 | |
| Network Alpha | 32 | |
| Optimizer | AdamW | AdamW8bit da çalışır |
| Mixed Precision | fp16 | **bf16 kullanma** — nan loss ve siyah çıktıya yol açar |

> Adım hesabı: `resim_sayısı × repeats × epoch` = 300 × 5 × 4 = 6000 adım

### Advanced Settings (kritik ayarlar)

- **Mixed Precision** → `fp16` (bf16 nan loss yapar → siyah çıktı)
- **Optimizer** → `AdamW`
- **Noise Offset** → `0` (sorun çıkarsa sıfırla)

> **Not:** accelerate_gpu.yaml dosyası da fp16 olarak ayarlandı. UI'daki Mixed Precision ile çakışmaması için her zaman ikisi aynı olmalı.

---

## 7. Özel Model Kullanmak (novaanimeanima gibi)

.safetensors dosyasını şu klasöre at:
```
/workspace/anima-trainer/models/anima/dit/
```

Arayüzde **Base Model** dropdown'ında otomatik görünür.

Terminalle yüklemek için (HuggingFace veya başka linkten):
```bash
wget -O /workspace/anima-trainer/models/anima/dit/novaanimeanima.safetensors "LINK_BURAYA"
```

---

## 8. Eğitim Bitti → Dosyayı İndir

Arayüzde **Downloads** sekmesi → **Refresh** → dosyayı indir.

---

## 9. Pod'u Durdur

Eğitim bittikten sonra pod sayfasında → **Stop** (Terminate değil, dosyalar silinir).

---

## Sıradaki Adımlar Sırası (unutma!)

1. **Configure Training** butonuna bas → ✓ yazısı çıksın
2. **Start Training** butonuna bas

> Configure Training atlanırsa `❌ No training config found` hatası çıkar.

---

## Sorun Giderme

| Hata | Çözüm |
|------|-------|
| `sd-scripts: No such file or directory` | Aşağıya bak |
| `Training script not found: .../anima_train_network.py` | Aşağıya bak |
| `accelerate not found` | Script otomatik bulur, sorun devam ederse `pip install accelerate` |
| Arayüz açılmıyor | Pod sayfasında port 7860 expose edilmiş mi kontrol et |
| `avr_loss=nan` + siyah çıktı | Mixed Precision'ı fp16 yap, bf16 kullanma |
| Resolution 1024+ → siyah çıktı | Resolution'ı 768'de tut |
| Terminal boş satırda bekliyor | Bekle — büyük repo indiriyor olabilir (2-3 dk) |

### sd-scripts eksik hatası

Script çalıştıktan sonra bu hata çıkarsa submodule tam inmemiş demektir:

```bash
cd /workspace/anima-trainer && git submodule update --init --recursive
```

Bittikten sonra (terminal prompt'a dönünce) scripti yeniden çalıştır:

```bash
bash /workspace/anima-trainer/runpod_start.sh
```

> Not: Submodule inderken terminal boş satırda durur, bu normal. 2-3 dakika bekle.
