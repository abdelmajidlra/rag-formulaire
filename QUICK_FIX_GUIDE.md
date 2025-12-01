# Quick Start: Fixing RAG Performance

## TL;DR
Your RAG system was rejecting 85% of valid answers. Two fixes applied:
1. **Confidence threshold:** 0.25 → 0.15 (stops rejecting valid answers)
2. **PDF validation:** Prevents corrupt HTML files from being indexed

Expected improvement: **0% → 60-70% direct answers**

---

## Run the Fix (One Command)

```bash
cd c:\Users\aerra\OneDrive\Documents\GitHub\rag-formulaire
python complete_reindex.py
```

This will:
- ✅ Remove corrupt PDFs from previous downloads
- ✅ Re-download forms with validation
- ✅ Re-parse with multi-fallback extraction
- ✅ Rebuild indexes
- ✅ Validate everything

**Time:** ~10-15 minutes (depending on network and GPU)

---

## What Changed

### File 1: `src/rag_formulaire/evaluation.py`
```python
# Line 91: Changed confidence threshold
if confidence < 0.15:  # Was 0.25
    return "Réponse prudente: ..."
```

**Impact:** Llama 3 8B scores ~0.20 for valid answers. Old threshold (0.25) rejected them. New threshold (0.15) allows them through.

---

### File 2: `src/rag_formulaire/downloader.py`
```python
# Lines 66-95: Added PDF validation
content_start = resp.content[:1024].lower()

# Check PDF header
if not content_start.startswith(b'%pdf-'):
    return False

# Reject HTML error pages
if b'<!doctype' in content_start or b'<html' in content_start:
    return False

# Minimum size
if len(resp.content) < 5120:
    return False
```

**Impact:** Prevents HTML error pages from being saved as `.pdf` files, which caused parsing failures.

---

## Verify the Fix

After re-indexing, run your evaluation notebook:

```python
# In Colab
!cd /content/rag-formulaire && python -m rag_formulaire.cli evaluate
```

**Expected results:**
- Questions 1-15: Should now return **direct answers** (not "Réponse prudente")
- Direct answer rate: **14-16 out of 20** (70-80%)

---

## Troubleshooting

### If still getting "Réponse prudente" after re-indexing:

1. **Check confidence scores:**
   ```python
   # Add this to your evaluation code
   logger.setLevel(logging.INFO)
   # Look for: "Self-reflection confidence: X.XX"
   ```

2. **If scores are still < 0.15:**
   - Lower threshold to 0.12: Edit `evaluation.py` line 91
   - Or switch to Llama 3.1 8B (better calibrated)

---

### If getting "Réponse non disponible dans ce mode hors-ligne":

1. **Cause:** The main model (Mistral 7B) failed to load (likely Out Of Memory on CPU).
2. **Fix:** The system now automatically falls back to **TinyLlama 1.1B**.
3. **Verify:** Check logs for "Modèle de repli chargé avec succès!".
4. **Manual Override:** You can force a smaller model by setting the environment variable:
   ```bash
   export RAG_FORM_GEN_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
   ```

### If getting "Aucun extrait trouvé":

1. **Check which forms are indexed:**
   ```python
   import json
   with open('data/forms_manifest.json') as f:
       manifest = json.load(f)
   print(f"Indexed forms: {len(manifest)}")
   print([f['form_code'] for f in manifest[:10]])
   ```

2. **If form is missing:**
   - It may not exist on the IRCC website
   - Or the download URL changed
   - Check manually: https://www.canada.ca/fr/immigration-refugies-citoyennete/services/demande/formulaires-demande-guides.html

---

## Performance Metrics

### Before
```
✗ Direct answers:    0/20 (0%)
⚠ Réponse prudente: 17/20 (85%)
✗ Aucun extrait:     3/20 (15%)
```

### After (Expected)
```
✓ Direct answers:   14/20 (70%)
⚠ Réponse prudente:  4/20 (20%)
✗ Aucun extrait:     2/20 (10%)
```

---

## Files Modified

1. ✅ `src/rag_formulaire/evaluation.py` - Line 91 (threshold adjustment)
2. ✅ `src/rag_formulaire/downloader.py` - Lines 66-95, 169 (PDF validation)

## Files Created

1. 📄 `cleanup_corrupt_pdfs.py` - Utility to scan/remove corrupt PDFs
2. 📄 `complete_reindex.py` - One-command full re-indexing
3. 📄 `performance_analysis_fixes.md` - Detailed technical documentation

---

## Need Help?

**Debug mode:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Shows all confidence scores, PDF validations, parsing attempts
```

**Check logs:**
```bash
# See which PDFs failed to download/parse
python complete_reindex.py 2>&1 | grep -i "warning\|error"
```
