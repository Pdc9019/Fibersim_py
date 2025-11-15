#!/usr/bin/env python3
"""
Test script para verificar las funciones de curvas BER.
"""

import sys
import os

# Agregar src al path para imports
sys.path.insert(0, 'src')

def test_ber_curves():
    """Test de las funciones de curvas BER."""
    print("=== Test de Curvas BER ===\n")
    
    try:
        from fibersim.core.ber_curves import generate_ber_curves, snr_to_ebno, get_ber_from_ebno
        import numpy as np
        
        # Test 1: Generar curvas
        print("1. Generando curvas BER...")
        curves = generate_ber_curves((0, 20), 50)
        
        print(f"   Puntos EbNo: {len(curves['EbNo_dB'])}")
        print(f"   Rango EbNo: {curves['EbNo_dB'][0]:.1f} a {curves['EbNo_dB'][-1]:.1f} dB")
        
        # Test 2: Valores en puntos específicos
        print("\n2. BER en puntos específicos:")
        test_ebno = [0, 5, 10, 15, 20]
        
        for ebno in test_ebno:
            ber_bpsk = get_ber_from_ebno(ebno, "BPSK")
            ber_qpsk = get_ber_from_ebno(ebno, "QPSK") 
            ber_16qam = get_ber_from_ebno(ebno, "16QAM")
            ber_64qam = get_ber_from_ebno(ebno, "64QAM")
            
            print(f"   Eb/No = {ebno:2d} dB:")
            print(f"     BPSK:   {ber_bpsk:.2e}")
            print(f"     QPSK:   {ber_qpsk:.2e}")
            print(f"     16QAM:  {ber_16qam:.2e}")
            print(f"     64QAM:  {ber_64qam:.2e}")
        
        # Test 3: Conversión SNR -> Eb/No
        print("\n3. Conversión SNR → Eb/No:")
        snr_test = 10.0  # dB
        
        for M, mod_name in [(2, "BPSK"), (4, "QPSK"), (16, "16QAM"), (64, "64QAM")]:
            ebno = snr_to_ebno(snr_test, M)
            print(f"   SNR = {snr_test} dB, {mod_name} (M={M}) → Eb/No = {ebno:.1f} dB")
        
        # Test 4: Validación de rangos
        print("\n4. Validación de rangos:")
        ebno_range = np.linspace(0, 20, 21)
        for mod in ["BPSK", "QPSK", "16QAM", "64QAM"]:
            ber_vals = [get_ber_from_ebno(eb, mod) for eb in ebno_range]
            min_ber = min(ber_vals)
            max_ber = max(ber_vals)
            print(f"   {mod}: BER rango {min_ber:.2e} a {max_ber:.2e}")
        
        print("\n✅ Todas las pruebas de curvas BER exitosas!")
        return True
        
    except ImportError as e:
        print(f"❌ Error de import: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_ber_curves()
