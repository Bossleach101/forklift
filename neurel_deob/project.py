from forklift.evaluator import Evaluator, Config
import os
from forklift.asm import AsmAdder, FuncDataclass
from forklift.utils import normalize_structs, InferenceDataset

DIRECTION = 'clang_opt3_ir_optz-ir_optz'

MODELS = {'clang_opt3_ir_optz-ir_optz': 'jordiae/clang_opt3_ir_optz-ir_optz-2024-01-15-0959-e1bf-bc2b'
          }



