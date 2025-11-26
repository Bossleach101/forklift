from forklift.evaluator import Evaluator, Config
from forklift.asm import AsmAdder, FuncDataclass
from forklift.utils import InferenceDataset
import tempfile
import os

# Read the full assembly from your file
with open('f_ground_truth.s', 'r') as f:
    assembly_code = f.read()

# Create a minimal C function as a template
# The AsmAdder will help us create the proper structure
SAMPLE = dict(
    func_def='''
void f(void) {
    int i;
    for (i = 0; i < n; ++i) {
        list[i] -= val;
    }
}
''',
    fname='f',
)

DIRECTION = 'clang_opt3_ir_optz-ir_optz'

MODELS = {
    'clang_opt3_ir_optz-ir_optz': 'jordiae/clang_opt3_ir_optz-ir_optz-2024-01-15-0959-e1bf-bc2b'
}


def run_assembly_direct(assembly_code, batch_size=1):
    """Run inference by directly providing assembly and compiling a placeholder IR"""
    config = Config(
        hf_model_path=MODELS[DIRECTION],
        pairs=[DIRECTION],
    )

    evaluator = Evaluator(config)
    
    # Create a sample with the assembly injected
    sample = SAMPLE.copy()
    
    # Use AsmAdder to compile and get the proper structure
    adder = AsmAdder()
    samples_with_asm = []
    
    for s in [sample]:
        fd = FuncDataclass(func_def=s['func_def'],
                            fname=s['fname'],
                            path='',  # Empty string for path
                            func_head='',  # Empty string for func_head
                            signature=''  # Empty string for signature
        )
        adder.add_asm(fd)
        
        # Convert back to dict format
        result_dict = {
            'func_def': fd.func_def,
            'deps': '',  # Add deps back as empty string for InferenceDataset
            'fname': fd.fname,
            'asm': fd.asm
        }

        # Override the x86 assembly with our custom assembly
        if 'asm' in result_dict and result_dict['asm']:
            # Find the x86_O3 entry and replace it
            for i, target in enumerate(result_dict['asm']['target']):
                if 'x86_O3' in target:
                    result_dict['asm']['code'][i] = assembly_code
        samples_with_asm.append(result_dict)
    
    predictions = []
    batch = []
    pair = DIRECTION
    
    for row in InferenceDataset(samples_with_asm, compilers_keys=['clang_x86_O3', 'clang_ir_Oz']):
        batch.append((row, pair))
        if len(batch) == batch_size:
            predictions.extend(evaluator.predict_batch(batch))
            batch = []
    
    if len(batch) > 0:
        predictions.extend(evaluator.predict_batch(batch))

    return predictions[0] if predictions else None


if __name__ == '__main__':
    predicted = run_assembly_direct(assembly_code)
    if predicted:
        print("Predicted LLVM IR:")
        print("=" * 60)
        for idx, lifted in enumerate(predicted):
            print(lifted)
            print('_____')
    else:
        print("No predictions generated")