"""
Level 4: LLVM IR Similarity Metrics

Measures how close predicted LLVM IR is to the ground truth using:
  - BLEU score (token-level)
  - Edit distance (character-level and token-level)
  - Exact match (after normalisation)

These metrics don't prove correctness but are useful for tracking model
quality over training and comparing different checkpoints.
"""

import pytest
import re
from typing import Optional
from .conftest import discover_test_cases

# ---------------------------------------------------------------------------
# Try to import metric libraries — degrade gracefully if missing
# ---------------------------------------------------------------------------

try:
    import editdistance

    HAS_EDITDISTANCE = True
except ImportError:
    HAS_EDITDISTANCE = False

try:
    import sacrebleu

    HAS_BLEU = True
except ImportError:
    HAS_BLEU = False


# ---------------------------------------------------------------------------
# Normalisation (matching Forklift's approach)
# ---------------------------------------------------------------------------

def normalize_ir(ir: str) -> str:
    """
    Normalise LLVM IR for fair comparison:
      - Strip module-level metadata (source_filename, target, attributes, !)
      - Normalise struct names (%struct.Foo → %struct.struct0)
      - Strip comments (lines starting with ;)
      - Normalise whitespace
    """
    if not ir:
        return ""

    lines = []
    for line in ir.splitlines():
        stripped = line.strip()
        # Skip metadata lines
        if stripped.startswith(";"):
            continue
        if stripped.startswith("source_filename"):
            continue
        if stripped.startswith("target datalayout"):
            continue
        if stripped.startswith("target triple"):
            continue
        if stripped.startswith("attributes"):
            continue
        if stripped.startswith("!"):
            continue
        # Remove inline metadata references like !tbaa !5
        line_clean = re.sub(r",?\s*![a-zA-Z0-9]+(\s+![0-9]+)?", "", stripped)
        # Remove attribute group references like #0
        line_clean = re.sub(r"#\d+", "", line_clean)
        line_clean = line_clean.strip()
        if line_clean:
            lines.append(line_clean)

    ir_text = "\n".join(lines)

    # Normalise struct names
    ir_text = _normalize_struct_names(ir_text)

    return ir_text


def _normalize_struct_names(ir: str) -> str:
    """Replace %struct.XYZ with %struct.struct0, %struct.struct1, etc."""
    struct_dict = {}
    counter = 0
    struct_pattern = re.compile(r"(%struct\.[a-zA-Z0-9_]+)")

    def replacer(match):
        nonlocal counter
        name = match.group(1)
        if name not in struct_dict:
            struct_dict[name] = f"%struct.struct{counter}"
            counter += 1
        return struct_dict[name]

    return struct_pattern.sub(replacer, ir)


def tokenize_ir(ir: str) -> list:
    """
    Simple whitespace tokenisation of LLVM IR.
    Splits on whitespace and punctuation boundaries.
    """
    # Split on whitespace, keeping tokens like %0, @f, i32, etc.
    return ir.split()


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def compute_bleu(predicted: str, reference: str) -> Optional[float]:
    """
    Compute BLEU score between predicted and reference IR.
    Both are normalised before comparison.
    Uses sacrebleu for corpus-level BLEU.
    
    Returns:
        BLEU score (0-100 scale), or None if sacrebleu not available.
    """
    if not HAS_BLEU:
        return None

    pred_norm = normalize_ir(predicted)
    ref_norm = normalize_ir(reference)

    if not pred_norm or not ref_norm:
        return 0.0

    try:
        score = sacrebleu.corpus_bleu([pred_norm], [[ref_norm]])
        return score.score
    except Exception:
        return None


def compute_edit_distance(predicted: str, reference: str, token_level: bool = True) -> Optional[int]:
    """
    Compute edit distance between predicted and reference IR.
    
    Args:
        token_level: If True, compute over tokens. If False, over characters.
    """
    if not HAS_EDITDISTANCE:
        return None

    pred_norm = normalize_ir(predicted)
    ref_norm = normalize_ir(reference)

    if token_level:
        pred_tokens = tokenize_ir(pred_norm)
        ref_tokens = tokenize_ir(ref_norm)
        return editdistance.eval(pred_tokens, ref_tokens)
    else:
        return editdistance.eval(pred_norm, ref_norm)


def compute_normalised_edit_distance(predicted: str, reference: str) -> Optional[float]:
    """
    Compute normalised edit distance (0.0 = identical, 1.0 = completely different).
    Normalised by the max length of the two sequences.
    """
    if not HAS_EDITDISTANCE:
        return None

    pred_norm = normalize_ir(predicted)
    ref_norm = normalize_ir(reference)

    pred_tokens = tokenize_ir(pred_norm)
    ref_tokens = tokenize_ir(ref_norm)

    if not pred_tokens and not ref_tokens:
        return 0.0

    dist = editdistance.eval(pred_tokens, ref_tokens)
    max_len = max(len(pred_tokens), len(ref_tokens))
    return dist / max_len if max_len > 0 else 0.0


def compute_exact_match(predicted: str, reference: str) -> bool:
    """Check if predicted and reference IR are identical after normalisation."""
    return normalize_ir(predicted) == normalize_ir(reference)


def compute_all_metrics(predicted: str, reference: str) -> dict:
    """
    Compute all available similarity metrics.
    
    Returns:
        {
            'bleu': float or None,
            'edit_distance_tokens': int or None,
            'normalised_edit_distance': float or None,
            'edit_distance_chars': int or None,
            'exact_match': bool,
        }
    """
    return {
        "bleu": compute_bleu(predicted, reference),
        "edit_distance_tokens": compute_edit_distance(predicted, reference, token_level=True),
        "normalised_edit_distance": compute_normalised_edit_distance(predicted, reference),
        "edit_distance_chars": compute_edit_distance(predicted, reference, token_level=False),
        "exact_match": compute_exact_match(predicted, reference),
    }


# ---------------------------------------------------------------------------
# Discover test cases with both predicted and ground-truth IR
# ---------------------------------------------------------------------------

SIMILARITY_CASES = [
    tc for tc in discover_test_cases()
    if tc.predicted_ll_path and tc.predicted_ll_path.exists()
    and tc.ground_truth_ll_path and tc.ground_truth_ll_path.exists()
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIRSimilarity:
    """Compute and report similarity metrics between predicted and ground-truth IR."""

    @pytest.mark.parametrize(
        "test_case",
        SIMILARITY_CASES,
        ids=[tc.name for tc in SIMILARITY_CASES],
    )
    def test_similarity_report(self, test_case):
        """Compute all metrics and report them (always passes — informational)."""
        predicted = test_case.predicted_ll
        reference = test_case.ground_truth_ll

        metrics = compute_all_metrics(predicted, reference)

        # Print metrics for visibility in pytest output (use -s flag)
        print(f"\n{'='*60}")
        print(f"Similarity report: {test_case.name}")
        print(f"{'='*60}")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print(f"{'='*60}")

    @pytest.mark.parametrize(
        "test_case",
        SIMILARITY_CASES,
        ids=[tc.name for tc in SIMILARITY_CASES],
    )
    def test_nonzero_bleu(self, test_case):
        """Predicted IR should have non-zero BLEU against ground truth."""
        if not HAS_BLEU:
            pytest.skip("bleu package not installed")

        bleu = compute_bleu(test_case.predicted_ll, test_case.ground_truth_ll)
        assert bleu is not None, "BLEU computation failed"
        assert bleu > 0.0, f"BLEU score is 0 for {test_case.name}"
        print(f"\n  BLEU({test_case.name}) = {bleu:.4f}")

    @pytest.mark.parametrize(
        "test_case",
        SIMILARITY_CASES,
        ids=[tc.name for tc in SIMILARITY_CASES],
    )
    def test_edit_distance_bounded(self, test_case):
        """
        Normalised edit distance should be below 1.0 
        (i.e., predicted IR shares SOME structure with ground truth).
        """
        if not HAS_EDITDISTANCE:
            pytest.skip("editdistance package not installed")

        ned = compute_normalised_edit_distance(test_case.predicted_ll, test_case.ground_truth_ll)
        assert ned is not None, "Edit distance computation failed"
        assert ned < 1.0, (
            f"Normalised edit distance is {ned:.4f} for {test_case.name} "
            f"(predicted shares no structure with ground truth)"
        )
        print(f"\n  NED({test_case.name}) = {ned:.4f}")


# ---------------------------------------------------------------------------
# Test normalisation itself
# ---------------------------------------------------------------------------

class TestNormalisation:
    """Unit tests for the IR normalisation functions."""

    def test_strip_metadata(self):
        ir = """
source_filename = "test.c"
target datalayout = "e-m:e-..."
target triple = "x86_64-pc-linux-gnu"

define void @f(i32* %0) {
  ret void
}

attributes #0 = { nounwind }
!0 = !{i32 1}
"""
        normalised = normalize_ir(ir)
        assert "source_filename" not in normalised
        assert "target datalayout" not in normalised
        assert "attributes" not in normalised
        assert "!0" not in normalised
        assert "define void @f" in normalised

    def test_normalize_structs(self):
        ir = "%struct.MyStruct = type { i32 }\n%struct.AnotherStruct = type { i64 }"
        normalised = _normalize_struct_names(ir)
        assert "%struct.struct0" in normalised
        assert "%struct.struct1" in normalised
        assert "%struct.MyStruct" not in normalised

    def test_exact_match_after_normalisation(self):
        ir1 = "define void @f(i32* %0) {\n  ret void\n}"
        ir2 = "define void @f(i32* %0) {\n  ret void\n}"
        assert compute_exact_match(ir1, ir2)

    def test_not_exact_match(self):
        ir1 = "define void @f(i32* %0) {\n  ret void\n}"
        ir2 = "define i32 @f(i32* %0) {\n  ret i32 0\n}"
        assert not compute_exact_match(ir1, ir2)
