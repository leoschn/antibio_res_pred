from dataset.dataset_antibiores import reconstruct_synth

if '__main__' == __name__ :
    reconstruct_synth('/lustre/fsn1/projects/rech/bun/ucg81ws/output_ms_gen/eval_test_utility'
                      ,'/lustre/fsn1/projects/rech/bun/ucg81ws/output_ms_gen/eval_test_utility_reconstructed'
                      ,'.pkl'
                      ,9)