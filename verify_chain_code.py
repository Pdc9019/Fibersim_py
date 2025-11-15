"""
Test directo: Verificar si chain.py tiene la corrección
"""

import sys
sys.path.insert(0, 'src')

# Forzar reload
import importlib
if 'fibersim.core.chain' in sys.modules:
    del sys.modules['fibersim.core.chain']

# Importar chain
from fibersim.core import chain

# Leer el código fuente
import inspect
source = inspect.getsource(chain.run_chain)

print("="*70)
print("VERIFICACIÓN DEL CÓDIGO FUENTE DE chain.py")
print("="*70)

# Buscar la línea crítica
if "P_ase_total * G_lin + P_ase_added" in source:
    print("\n✅ CORRECCIÓN ENCONTRADA EN EL CÓDIGO:")
    print("   P_ase_total = P_ase_total * G_lin + P_ase_added")
    print("\n   El código está correcto en el archivo.")
elif "P_ase_total += P_ase_added" in source:
    print("\n❌ CÓDIGO VIEJO ENCONTRADO:")
    print("   P_ase_total += P_ase_added")
    print("\n   El archivo NO tiene la corrección!")
else:
    print("\n⚠️  No se encontró ninguna de las dos versiones")
    print("   Mostrando sección relevante:\n")
    lines = source.split('\n')
    for i, line in enumerate(lines):
        if 'P_ase' in line and 'edfa' in source[max(0,i-10):i]:
            print(f"   Línea {i}: {line}")
