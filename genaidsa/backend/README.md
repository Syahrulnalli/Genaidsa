# Local endpoint contract

Azure Functions Python v2 exposes `POST http://localhost:7071/api/process-image` with anonymous local authorization. JSON request:

```json
{"image_base64":"<base64 PNG/JPEG/WebP bytes>","filename":"photo.png","noise_level":20,"filter":"Gaussian Filter","bit_depth":8}
```

`filename` is informational. `noise_level` must be 0–50, `filter` one of `None`, `Median Filter`, `Gaussian Filter`, and `bit_depth` one of 4, 8, 16. Success returns `original_png`, `noisy_png` and `reconstructed_png` (Base64 PNG), `metrics`, and `metadata`. Errors return `{"error":"readable explanation"}` with HTTP 400 for invalid input or 500 for model/inference errors.

Example PowerShell request (replace the path):

```powershell
$bytes = [IO.File]::ReadAllBytes('C:\path\to\photo.png')
$body = @{ image_base64 = [Convert]::ToBase64String($bytes); filename = 'photo.png'; noise_level = 20; filter = 'Gaussian Filter'; bit_depth = 8 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://localhost:7071/api/process-image' -ContentType 'application/json' -Body $body
```
