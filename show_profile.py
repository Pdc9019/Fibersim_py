import json

log = json.load(open('logs/simlog_2025-11-06_15-10-27.json'))
profile = log['result']['profile']

print("Perfil de OSNR a lo largo del enlace:")
print("="*50)
for i, p in enumerate(profile):
    if i % 10 == 0 or p['OSNR_dB'] is not None:  # Mostrar cada 10 o cuando hay OSNR
        osnr_str = f"{p['OSNR_dB']:.2f}" if p['OSNR_dB'] is not None else "null"
        print(f"z = {p['z_km']:6.1f} km: OSNR = {osnr_str:>8s} dB")
