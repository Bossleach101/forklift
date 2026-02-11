"""
AArch64 → LLVM IR Lifting Pipeline

End-to-end inference: takes a raw AArch64 assembly file (or string),
preprocesses it to match the Forklift training data format, tokenizes
it as an arm_ir-ir pair, runs the model, and returns the predicted
LLVM IR.

Two modes of operation:
  1. From .s file:  Pipeline.from_file('func.s', 'f')
  2. From C source: Pipeline.from_c_source('int f(int x) { return x+1; }', 'f')
     (compiles to AArch64 asm first, then lifts)

Usage:
    from neurel_deob.aarch64_pipeline import Pipeline
    
    pipe = Pipeline(model_path='jordiae/clang_opt3_ir_optz-ir_optz-2024-01-15-0959-e1bf-bc2b')
    result = pipe.from_file('func.s', function_name='f')
    print(result.predicted_ir)
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import BartForConditionalGeneration

from forklift.par_data import DP
from neurel_deob.asm_preprocessor import AsmPreprocessor, Arch


@dataclass
class LiftResult:
    """Result of lifting AArch64 assembly to LLVM IR."""
    predicted_ir: str                # Best prediction (detokenized)
    all_predictions: List[str]       # All n-best predictions
    source_asm: str                  # Preprocessed assembly that was fed to model
    function_name: str
    arch: Arch

    def save_ir(self, path: str):
        """Save the best prediction to a .ll file."""
        with open(path, 'w') as f:
            f.write(self.predicted_ir)

    def __repr__(self):
        lines = self.predicted_ir.strip().count('\n') + 1
        return (
            f"LiftResult(function='{self.function_name}', "
            f"arch={self.arch.value}, ir_lines={lines})"
        )


@dataclass 
class PipelineConfig:
    """Configuration for the lifting pipeline."""
    model_path: str
    pair: str = 'arm_ir-ir'
    beam: int = 5
    nbest: int = 1
    max_new_tokens: int = 2048
    early_stopping: bool = True
    length_penalty: float = 1.0
    min_length: int = 1
    device: str = 'cpu'  # 'cpu', 'cuda', 'cuda:0', etc.


class Pipeline:
    """
    AArch64 assembly → LLVM IR lifting pipeline.
    
    Wraps the Forklift model with preprocessing specific to AArch64 input.
    """

    def __init__(self, model_path: str, config: Optional[PipelineConfig] = None):
        if config is None:
            config = PipelineConfig(model_path=model_path)
        self.config = config

        # Load tokenizer
        tok_path = os.path.join(model_path, 'tokenizer.json')
        if os.path.exists(tok_path):
            self.tokenizer = Tokenizer.from_file(tok_path)
        else:
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem()
            self.tokenizer = Tokenizer.from_str(
                fs.open(os.path.join(model_path, 'tokenizer.json'), 'r').read()
            )

        # Load model
        self.model = BartForConditionalGeneration.from_pretrained(model_path).eval()
        if config.device != 'cpu':
            self.model = self.model.to(config.device)

        # Data processor for tokenization
        self.dp = DP(tokenizer=self.tokenizer)
        self.pad_id = self.tokenizer.get_vocab()['<pad>']

    def from_file(
        self,
        file_path: str,
        function_name: str = 'f',
        arch: Optional[Arch] = None,
    ) -> LiftResult:
        """
        Lift a function from a .s assembly file to LLVM IR.

        Args:
            file_path: Path to the assembly file.
            function_name: Name of the function to extract.
            arch: Architecture override (auto-detected if None).

        Returns:
            LiftResult with predicted IR and metadata.
        """
        preprocessor = AsmPreprocessor(
            file_path=file_path,
            target_function=function_name,
            arch=arch,
        )
        return self._lift(preprocessor.func_asm, function_name, preprocessor.arch)

    def from_asm(
        self,
        asm_code: str,
        function_name: str = 'f',
        arch: Optional[Arch] = None,
    ) -> LiftResult:
        """
        Lift a function from a raw assembly string to LLVM IR.

        Args:
            asm_code: Raw assembly code (full file or extracted function).
            function_name: Name of the function to extract.
            arch: Architecture override (auto-detected if None).

        Returns:
            LiftResult with predicted IR and metadata.
        """
        preprocessor = AsmPreprocessor(
            asm_code=asm_code,
            target_function=function_name,
            arch=arch,
        )
        return self._lift(preprocessor.func_asm, function_name, preprocessor.arch)

    def from_c_source(
        self,
        c_code: str,
        function_name: str = 'f',
    ) -> LiftResult:
        """
        Compile C source to AArch64 assembly, then lift to LLVM IR.
        Requires clang with AArch64 cross-compilation support.

        Args:
            c_code: C source code containing the target function.
            function_name: Name of the function to extract.

        Returns:
            LiftResult with predicted IR and metadata.
        """
        from forklift.asm import Compiler
        compiler = Compiler.factory('clang', arch='arm', o='0')
        result = compiler.get_func_asm(c_code, function_name)
        if result.is_err():
            raise RuntimeError(f"Compilation failed: {result.val}")
        return self._lift(result.val.func_asm, function_name, Arch.AARCH64)

    def _lift(self, preprocessed_asm: str, function_name: str, arch: Arch) -> LiftResult:
        """
        Core lifting: tokenize preprocessed assembly → model → detokenize IR.
        """
        pair = self.config.pair

        # Build a fake row that get_par_data can consume.
        # We need the exact keys that get_par_data() will look up.
        # When row=None, get_par_data returns the keys it expects.
        source_key, target_key, _, _ = self.dp.get_par_data(
            row=None, pair=pair, asm_key='angha', fPIC=False,
        )

        row = {
            'asm': {
                'target': [source_key, target_key],
                'code': [preprocessed_asm, ''],  # Empty target (will be masked)
            }
        }

        # Tokenize using the standard pipeline
        source, target, tok_source, tok_target = self.dp.get_par_data(
            row, pair, asm_key='angha', fPIC=False,
            tokenize_ids=True, do_normalize_ir_structs=False,
        )

        # Check length
        if len(tok_source) > self.model.config.max_position_embeddings:
            raise ValueError(
                f"Input too long ({len(tok_source)} tokens, "
                f"max {self.model.config.max_position_embeddings}). "
                f"Try a simpler function."
            )

        # Run inference
        batch = torch.tensor([tok_source])
        if self.config.device != 'cpu':
            batch = batch.to(self.config.device)

        with torch.no_grad():
            output = self.model.generate(
                batch,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.beam,
                num_return_sequences=self.config.nbest,
                early_stopping=self.config.early_stopping,
                length_penalty=self.config.length_penalty,
                min_length=self.config.min_length,
            )

        # Detokenize
        output = output.cpu()
        predictions = []
        for out in output:
            detokenized = self.dp.detokenize(out.tolist())
            predictions.append(detokenized)

        return LiftResult(
            predicted_ir=predictions[0] if predictions else '',
            all_predictions=predictions,
            source_asm=preprocessed_asm,
            function_name=function_name,
            arch=arch,
        )

    def lift_batch(
        self,
        items: List[dict],
    ) -> List[LiftResult]:
        """
        Batch lifting for multiple functions.

        Args:
            items: List of dicts with keys:
                - 'file_path' or 'asm_code': Assembly input
                - 'function_name': Target function (default 'f')
                - 'arch': Optional architecture override

        Returns:
            List of LiftResult objects.
        """
        pair = self.config.pair
        preprocessed = []
        tokenized = []
        too_long = []

        for item in items:
            fname = item.get('function_name', 'f')
            arch = item.get('arch', None)

            if 'file_path' in item:
                p = AsmPreprocessor(file_path=item['file_path'], target_function=fname, arch=arch)
            elif 'asm_code' in item:
                p = AsmPreprocessor(asm_code=item['asm_code'], target_function=fname, arch=arch)
            else:
                raise ValueError("Each item must have 'file_path' or 'asm_code'")

            preprocessed.append((p.func_asm, fname, p.arch))

            # Build fake row and tokenize using the keys get_par_data expects
            source_key, target_key, _, _ = self.dp.get_par_data(
                row=None, pair=pair, asm_key='angha', fPIC=False,
            )
            row = {
                'asm': {
                    'target': [source_key, target_key],
                    'code': [p.func_asm, ''],
                }
            }
            _, _, tok_source, _ = self.dp.get_par_data(
                row, pair, asm_key='angha', fPIC=False,
                tokenize_ids=True, do_normalize_ir_structs=False,
            )

            if len(tok_source) > self.model.config.max_position_embeddings:
                too_long.append(True)
            else:
                tokenized.append(torch.tensor(tok_source))
                too_long.append(False)

        # Batch inference
        if tokenized:
            batch = pad_sequence(tokenized, True, self.pad_id)
            if self.config.device != 'cpu':
                batch = batch.to(self.config.device)

            with torch.no_grad():
                output = self.model.generate(
                    batch,
                    max_new_tokens=self.config.max_new_tokens,
                    num_beams=self.config.beam,
                    num_return_sequences=self.config.nbest,
                    early_stopping=self.config.early_stopping,
                    length_penalty=self.config.length_penalty,
                    min_length=self.config.min_length,
                )

            output = output.view(len(tokenized), self.config.nbest, -1).cpu()

        # Build results
        results = []
        tok_idx = 0
        for i, (asm, fname, arch) in enumerate(preprocessed):
            if too_long[i]:
                results.append(LiftResult(
                    predicted_ir='',
                    all_predictions=[''],
                    source_asm=asm,
                    function_name=fname,
                    arch=arch,
                ))
            else:
                preds = []
                for out in output[tok_idx]:
                    preds.append(self.dp.detokenize(out.tolist()))
                results.append(LiftResult(
                    predicted_ir=preds[0] if preds else '',
                    all_predictions=preds,
                    source_asm=asm,
                    function_name=fname,
                    arch=arch,
                ))
                tok_idx += 1

        return results


# =====================================================================
# CLI entry point
# =====================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Lift AArch64 assembly to LLVM IR using the Forklift neural lifter.'
    )
    parser.add_argument('input', help='Path to .s assembly file')
    parser.add_argument('-f', '--function', default='f',
                        help='Function name to extract (default: f)')
    parser.add_argument('-m', '--model',
                        default='jordiae/clang_opt3_ir_optz-ir_optz-2024-01-15-0959-e1bf-bc2b',
                        help='HuggingFace model path or local directory')
    parser.add_argument('-o', '--output', help='Output .ll file path')
    parser.add_argument('--beam', type=int, default=5, help='Beam size')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--nbest', type=int, default=1,
                        help='Number of hypotheses to generate')

    args = parser.parse_args()

    config = PipelineConfig(
        model_path=args.model,
        beam=args.beam,
        device=args.device,
        nbest=args.nbest,
    )

    print(f"Loading model from {args.model}...")
    pipe = Pipeline(args.model, config=config)

    print(f"Lifting function '{args.function}' from {args.input}...")
    result = pipe.from_file(args.input, function_name=args.function)

    print(f"\n{'='*60}")
    print(f"Predicted LLVM IR ({result.arch.value}):")
    print(f"{'='*60}")
    for i, pred in enumerate(result.all_predictions):
        if args.nbest > 1:
            print(f"\n--- Hypothesis {i+1} ---")
        print(pred)

    if args.output:
        result.save_ir(args.output)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
