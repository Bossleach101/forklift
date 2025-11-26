from forklift.evaluator import Evaluator, Config
import os
from forklift.asm import AsmAdder, FuncDataclass
from forklift.utils import normalize_structs, InferenceDataset

DIRECTION = 'clang_opt3_ir_optz-ir_optz'

MODELS = {'clang_opt3_ir_optz-ir_optz': 'jordiae/clang_opt3_ir_optz-ir_optz-2024-01-15-0959-e1bf-bc2b'
          }


def load_sample_from_file(filepath, fname='f'):
    """Load C function from a file and create a sample dict."""
    with open(filepath, 'r') as f:
        func_def = f.read()
        
    
    return dict(
        func_def=func_def,
        deps='',
        fname=fname,
    )


def run(sample, batch_size=1):
    config = Config(hf_model_path=MODELS[DIRECTION],
                                pairs=[DIRECTION],
                                )

    evaluator = Evaluator(config)
    predictions = []
    fnames = []
    batch = []
    rows = []
    pair = DIRECTION
    samples = [sample]
    for row in InferenceDataset(samples, compilers_keys=['clang_ir_Oz', 'clang_x86_O3']):
        batch.append((row, pair))
        fnames.append(sample['fname'])
        rows.append(row)
        
        if len(batch) == batch_size:
            predictions.extend(evaluator.predict_batch(batch))
            batch = []
    if len(batch) > 0:
        predictions.extend(evaluator.predict_batch(batch))
    return predictions[0]


if __name__ == '__main__':
    import sys
    
    # Default to inline sample if no file provided
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        fname = sys.argv[2] if len(sys.argv) > 2 else 'f'
        sample = load_sample_from_file(input_file, fname)
    else:
        # Original inline sample
        sample = dict(func_def='''
void f(int *list, int val, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        list[i] += val;
    }
}
''',
                      deps='',
                      fname='f',
                      )
    
    predicted = run(sample)
    for idx, lifted in enumerate(predicted):
        print(lifted)
        print('_____')