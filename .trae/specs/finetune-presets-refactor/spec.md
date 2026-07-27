# Fine-Tune Presets Refactor Spec

## Why
Mevcut fine-tune presetleri kafa karıştırıcı ve ayarlar uygulama yeniden başlatıldığında kayboluyor. Ayrıca, kullanıcıların yanlışlıkla mouse tekerleği (scroll wheel) ile ince ayar değerlerini değiştirmesi kötü bir kullanıcı deneyimine (UX) yol açıyor. Presetlerin basitleştirilmesi, "Custom" seçeneğinin kaydedilmesi ve UX iyileştirmeleri gerekiyor.

## What Changes
- Mevcut tüm presetler silinecek.
- Yeni 5 preset eklenecek: `Ultra`, `Good`, `Mid`, `Low`, `Custom`.
- `Ultra` preseti için MLX donanım sınırlarını zorlayan "maksimumun maksimumu" değerler atanacak.
- `Good`, `Mid` ve `Low` için kademeli olarak düşen güvenli değerler tanımlanacak.
- Uygulama yeniden başlatıldığında, eğer `Custom` harici bir preset seçiliyse, her şey orijinal (hardcoded) değerlerine geri dönecek.
- Eğer `Custom` seçiliyse, kullanıcının son ayarladığı değerler `QSettings` üzerinden geri yüklenecek.
- Kullanıcı herhangi bir değeri manuel değiştirirse, preset otomatik olarak `Custom`'a geçecek ve bu yeni değerler kaydedilecek.
- Tüm Fine-Tune SpinBox / DoubleSpinBox ve Slider bileşenlerinde mouse wheel (tekerlek) ile değer değiştirme özelliği iptal edilecek (`wheelEvent` override edilerek).

## Impact
- Affected specs: Fine-tune ayarları yönetimi, UI etkileşimleri.
- Affected code: `main.py` (Arayüz kurulumu, `_apply_ft_preset` metodu, QSettings entegrasyonu).

## ADDED Requirements
### Requirement: Yeni Preset Seçenekleri ve Kayıt Mekanizması
Sistem, ince ayar konfigürasyonlarını yönetmek için 5 net seçenek sunmalıdır.
#### Scenario: Uygulama Başlatma
- **WHEN** uygulama başlar
- **THEN** son seçilen preset aktif olmalı, eğer `Custom` ise kaydedilen custom değerler spinboxlara dolmalı, değilse seçili presetin hardcoded değerleri uygulanmalı.

#### Scenario: Değer Değişimi
- **WHEN** kullanıcı manuel olarak bir spinbox değerini değiştirir
- **THEN** preset combobox otomatik olarak `Custom` seçeneğine geçmeli ve yeni custom değerler `QSettings`'e kaydedilmelidir.

## MODIFIED Requirements
### Requirement: Mouse Wheel Etkileşimi
- Fine-tune alanındaki Spinbox'ların mouse wheel olayları yoksayılmalı (ignore), böylece kullanıcı paneli kaydırırken yanlışlıkla değerleri değiştirmemelidir.
