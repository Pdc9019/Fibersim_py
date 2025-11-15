# Backend Inconsistency Fix

## Problem Description

User encountered the following error during CPU execution:

```
TypeError: Unsupported type <class 'numpy.ndarray'>
Traceback:
  File "src/fibersim/core/edfa.py", line 11, in edfa_block
    A = xp.sqrt(G) * Ain
```

The root cause was backend inconsistency where CuPy functions were being applied to NumPy arrays.

## Root Cause Analysis

1. **Inconsistent Array Types**: Arrays were not being consistently converted to the active backend
2. **Missing Backend Validation**: No checks to ensure arrays match the backend before operations
3. **Import Pattern Issues**: Direct imports of `xp` without runtime backend checking

## Solutions Applied

### 1. Enhanced `array_api.py`

**File**: `src/fibersim/core/array_api.py`

**Changes**:
- Added `set_backend(use_gpu: bool)` function for explicit backend configuration
- Added `to_backend(array)` function for safe array conversion
- Added `ensure_compatible_arrays()` for multiple array conversion
- Added `backend_name` global variable for backend tracking
- Enhanced error handling with fallback to CPU

**New Functions**:
```python
def set_backend(use_gpu: bool = False) -> str:
    """Configura el backend de arrays CPU/GPU."""
    
def to_backend(array, target_backend=None):
    """Convierte array al backend activo."""
    
def ensure_compatible_arrays(*arrays):
    """Asegura que todos los arrays sean del mismo backend."""
```

### 2. Fixed `edfa.py`

**File**: `src/fibersim/core/edfa.py`

**Changes**:
- Import pattern changed from `from .array_api import xp` to `from . import array_api as ap`
- Added explicit array conversion: `A = ap.to_backend(Ain)`
- Ensure G is in correct backend: `xp.sqrt(xp.array(G)) * A`

**Key Fix**:
```python
# OLD (problematic)
A = xp.sqrt(G) * Ain

# NEW (fixed)
A = ap.to_backend(Ain)  # Convert to backend
A = xp.sqrt(xp.array(G)) * A  # Ensure G is in backend
```

### 3. Enhanced `main.py`

**File**: `src/fibersim/main.py`

**Changes**:
- Added explicit backend configuration in `_prepare_backend()`
- Use new `set_backend()` function before module reloads
- Better backend detection and reporting

### 4. Updated Import Patterns

**Files**: `pulse.py`, `chain.py`

**Changes**:
- Changed from direct `xp` imports to safer `array_api as ap` pattern
- Added array conversion calls where needed
- Runtime backend access with `ap.xp` instead of import-time binding

## Testing

A test script has been created: `test_backend_fix.py`

**To run the test**:
```bash
cd "C:\Users\benja\Desktop\sim fibra\fibra sim\Fibersim_py"
python test_backend_fix.py
```

**What it tests**:
1. CPU backend configuration
2. Array operations work correctly
3. EDFA module compatibility
4. Type conversion functions

## Verification Steps

1. **Restart the GUI** completely to clear any cached imports
2. **Select CPU backend** explicitly in the interface
3. **Run a simple simulation** (e.g., BPSK 80km)
4. **Check console output** for backend confirmation
5. **Verify no TypeError** during execution

## Expected Behavior After Fix

### Before Fix
```
❌ TypeError: Unsupported type <class 'numpy.ndarray'>
```

### After Fix
```
✅ Backend: CPU (NumPy)
✅ Simulación terminada
✅ Results displayed correctly
```

## Backend Selection Logic

```python
# CPU Mode (use_gpu=False)
os.environ["FIBERSIM_GPU"] = "0"
backend = "CPU (NumPy)"
xp = numpy

# GPU Mode (use_gpu=True)
os.environ["FIBERSIM_GPU"] = "1"
if cupy_available:
    backend = "GPU (CuPy)"
    xp = cupy
else:
    backend = "CPU (NumPy)"  # Fallback
    xp = numpy
```

## Files Modified

1. ✅ `src/fibersim/core/array_api.py` - Enhanced backend management
2. ✅ `src/fibersim/core/edfa.py` - Fixed array conversion
3. ✅ `src/fibersim/main.py` - Better backend initialization
4. ✅ `src/fibersim/core/pulse.py` - Safer imports
5. ✅ `src/fibersim/core/chain.py` - Array conversion
6. ✅ `test_backend_fix.py` - Test script (new)
7. ✅ `BACKEND_FIX_README.md` - This documentation (new)

## Status

✅ **COMPLETED** - All fixes applied and ready for testing

**Next Steps**: Run the test script and then test with the GUI to verify the fix works correctly.
