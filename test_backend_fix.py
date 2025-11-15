#!/usr/bin/env python3
"""
Test script para verificar que la corrección del backend funciona correctamente.
"""

import sys
import os

# Agregar src al path para imports
sys.path.insert(0, 'src')

def test_cpu_backend():
    """Test básico del backend CPU."""
    print("Testing CPU backend...")
    
    # Forzar CPU
    os.environ["FIBERSIM_GPU"] = "0"
    
    # Import después de configurar entorno
    from fibersim.core import array_api as ap
    
    # Re-configurar backend
    backend_str = ap.set_backend(use_gpu=False)
    print(f"Backend configurado: {backend_str}")
    print(f"Backend name: {ap.backend_name}")
    print(f"xp module: {ap.xp.__name__}")
    
    # Test array operations
    test_array = ap.xp.array([1.0, 2.0, 3.0])
    result = ap.xp.sqrt(test_array)
    print(f"Test array: {test_array}")
    print(f"Sqrt result: {result}")
    
    # Test conversion function
    converted = ap.to_backend(test_array)
    print(f"Converted array type: {type(converted)}")
    
    return True

def test_edfa_compatibility():
    """Test que EDFA funcione sin errores de tipo."""
    print("\nTesting EDFA compatibility...")
    
    from fibersim.core.edfa import edfa_block
    from fibersim.core import array_api as ap
    
    # Crear señal de prueba
    test_signal = ap.xp.array([1.0 + 1j, 2.0 + 1j], dtype=ap.xp.complex128)
    
    # Parámetros de prueba
    info_in = {"Fs": 1e12, "Rb": 32e9}
    par = {"G_dB": 10.0, "nsp": 1.5}
    
    try:
        result_signal, result_info = edfa_block(test_signal, info_in, par)
        print(f"EDFA test successful!")
        print(f"Input power: {ap.xp.mean(ap.xp.abs(test_signal)**2):.6f}")
        print(f"Output power: {ap.xp.mean(ap.xp.abs(result_signal)**2):.6f}")
        print(f"Gain applied: {result_info['G_dB']} dB")
        return True
    except Exception as e:
        print(f"EDFA test failed: {e}")
        return False

def main():
    """Ejecutar todas las pruebas."""
    print("=== Test de corrección de backend ===")
    
    try:
        success1 = test_cpu_backend()
        success2 = test_edfa_compatibility()
        
        if success1 and success2:
            print("\n✅ Todas las pruebas pasaron exitosamente!")
            print("La corrección del backend está funcionando correctamente.")
        else:
            print("\n❌ Algunas pruebas fallaron.")
            
    except Exception as e:
        print(f"\n❌ Error durante las pruebas: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
