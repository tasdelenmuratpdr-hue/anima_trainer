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

## 6. Eğitim Ayarları (300 resim için öneri)

| Ayar | Değer |
|------|-------|
| Repeats | 5 |
| Max Epochs | 3 |
| Resolution | 768 |
| Network Dim | 32 |
| Network Alpha | 32 |
| Optimizer | AdamW |
| Toplam süre | ~83 dakika |

> Adım hesabı: `resim_sayısı × repeats × epoch` = 300 × 5 × 3 = 4500 adım

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
| `avr_loss=nan` | Optimizer'ı AdamW yap, AdamW8bit deneme |
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
