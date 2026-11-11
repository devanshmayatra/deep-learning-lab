"""Inception experiment results for plotting. Percentages use the 0–100 scale.

Use current_experiments for comparisons; archive contains superseded trials.
Missing test scores are None, not zero.
"""

results = {'schema_version': '2.0',
 'scope': 'All Inception/CIFAR-100 experiments represented in this session; current results plus '
          'historical trials',
 'evidence': 'Saved notebook outputs and user-provided logs; no independent training execution',
 'dataset': {'name': 'CIFAR-100',
             'train_examples': 45000,
             'validation_examples': 5000,
             'test_examples': 10000,
             'split_seed': 42,
             'corrected_loader_counts_verified_in_uploaded_output': True,
             'test_evaluated': True},
 'shared_planned_settings': {'batch_size': 32,
                             'learning_rate_epochs_1_to_10': 0.001,
                             'learning_rate_epochs_11_to_20': 0.0001,
                             'selection_metric': 'validation accuracy',
                             'caution': 'Historical source edits and incomplete RNG provenance '
                                        'prevent claiming all runs are tightly controlled '
                                        'comparisons'},
 'current_experiments': [{'experiment_id': 'adamw_crop_flip',
                          'status': 'Retained: user confirms run preceded leaking loader change; '
                                    'saved source has historical naming/argument mismatches',
                          'optimizer': 'AdamW',
                          'weight_decay': 0.0001,
                          'batchnorm': False,
                          'trainable_parameters': 818500,
                          'augmentation': 'crop + flip; user clarified historical run',
                          'selected_result': {'epoch': 19,
                                              'training_accuracy_pct': 74.15,
                                              'validation_accuracy_pct': 57.08,
                                              'validation_loss': 1.8555},
                          'printed_checkpoint': 'best_inception_cifar100_ep20.pt',
                          'epoch_history': [{'epoch': 1,
                                             'train_loss': 3.9006,
                                             'validation_loss': 3.5059,
                                             'validation_accuracy_pct': 15.62,
                                             'epoch_seconds': 96.2},
                                            {'epoch': 2,
                                             'train_loss': 3.2595,
                                             'validation_loss': 2.9903,
                                             'validation_accuracy_pct': 24.8,
                                             'epoch_seconds': 95.6},
                                            {'epoch': 3,
                                             'train_loss': 2.7533,
                                             'validation_loss': 2.7236,
                                             'validation_accuracy_pct': 31.92,
                                             'epoch_seconds': 96.4},
                                            {'epoch': 4,
                                             'train_loss': 2.4242,
                                             'validation_loss': 2.4323,
                                             'validation_accuracy_pct': 38.12,
                                             'epoch_seconds': 95.4},
                                            {'epoch': 5,
                                             'train_loss': 2.1792,
                                             'validation_loss': 2.2442,
                                             'validation_accuracy_pct': 41.66,
                                             'epoch_seconds': 96.5},
                                            {'epoch': 6,
                                             'train_loss': 2.0019,
                                             'validation_loss': 2.095,
                                             'validation_accuracy_pct': 44.94,
                                             'epoch_seconds': 97.5},
                                            {'epoch': 7,
                                             'train_loss': 1.8633,
                                             'validation_loss': 2.0016,
                                             'validation_accuracy_pct': 46.94,
                                             'epoch_seconds': 96.8},
                                            {'epoch': 8,
                                             'train_loss': 1.7377,
                                             'validation_loss': 2.0137,
                                             'validation_accuracy_pct': 47.84,
                                             'epoch_seconds': 101.4},
                                            {'epoch': 9,
                                             'train_loss': 1.6384,
                                             'validation_loss': 1.8784,
                                             'validation_accuracy_pct': 50.22,
                                             'epoch_seconds': 97.1},
                                            {'epoch': 10,
                                             'train_loss': 1.5491,
                                             'validation_loss': 1.8901,
                                             'validation_accuracy_pct': 50.42,
                                             'epoch_seconds': 97.5},
                                            {'epoch': 11,
                                             'train_loss': 1.1769,
                                             'validation_loss': 1.7387,
                                             'validation_accuracy_pct': 55.38,
                                             'epoch_seconds': 96.1},
                                            {'epoch': 12,
                                             'train_loss': 1.1029,
                                             'validation_loss': 1.7441,
                                             'validation_accuracy_pct': 56.36,
                                             'epoch_seconds': 96.0},
                                            {'epoch': 13,
                                             'train_loss': 1.0578,
                                             'validation_loss': 1.7835,
                                             'validation_accuracy_pct': 56.08,
                                             'epoch_seconds': 94.9},
                                            {'epoch': 14,
                                             'train_loss': 1.0269,
                                             'validation_loss': 1.8073,
                                             'validation_accuracy_pct': 56.74,
                                             'epoch_seconds': 95.1},
                                            {'epoch': 15,
                                             'train_loss': 1.0046,
                                             'validation_loss': 1.7884,
                                             'validation_accuracy_pct': 56.62,
                                             'epoch_seconds': 94.6},
                                            {'epoch': 16,
                                             'train_loss': 0.9746,
                                             'validation_loss': 1.8253,
                                             'validation_accuracy_pct': 56.5,
                                             'epoch_seconds': 94.9},
                                            {'epoch': 17,
                                             'train_loss': 0.9576,
                                             'validation_loss': 1.8386,
                                             'validation_accuracy_pct': 56.84,
                                             'epoch_seconds': 94.3},
                                            {'epoch': 18,
                                             'train_loss': 0.9312,
                                             'validation_loss': 1.869,
                                             'validation_accuracy_pct': 56.7,
                                             'epoch_seconds': 95.1},
                                            {'epoch': 19,
                                             'train_loss': 0.9135,
                                             'validation_loss': 1.8555,
                                             'validation_accuracy_pct': 57.08,
                                             'epoch_seconds': 94.1},
                                            {'epoch': 20,
                                             'train_loss': 0.8915,
                                             'validation_loss': 1.8723,
                                             'validation_accuracy_pct': 56.68,
                                             'epoch_seconds': 94.7}],
                          'sum_printed_epoch_seconds': 1920.2,
                          'source_cells': [9, 11],
                          'test_accuracy_pct': None},
                         {'experiment_id': 'adamw_flip_baseline',
                          'status': 'Retained notebook output; historical code/output mismatches '
                                    'limit exact provenance',
                          'optimizer': 'AdamW',
                          'weight_decay': 0.0001,
                          'batchnorm': False,
                          'trainable_parameters': 818500,
                          'augmentation': 'flip; inferred from notebook default loader',
                          'selected_result': {'epoch': 13,
                                              'training_accuracy_pct': 83.23,
                                              'validation_accuracy_pct': 57.08,
                                              'validation_loss': 1.855},
                          'printed_checkpoint': 'best_inception_cifar100_base_seed42_ep20.pt',
                          'epoch_history': [{'epoch': 1,
                                             'train_loss': 3.9623,
                                             'validation_loss': 3.5226,
                                             'validation_accuracy_pct': 15.04,
                                             'epoch_seconds': 94.5},
                                            {'epoch': 2,
                                             'train_loss': 3.1807,
                                             'validation_loss': 2.8592,
                                             'validation_accuracy_pct': 27.76,
                                             'epoch_seconds': 94.1},
                                            {'epoch': 3,
                                             'train_loss': 2.5825,
                                             'validation_loss': 2.4429,
                                             'validation_accuracy_pct': 36.02,
                                             'epoch_seconds': 93.4},
                                            {'epoch': 4,
                                             'train_loss': 2.2235,
                                             'validation_loss': 2.2024,
                                             'validation_accuracy_pct': 41.82,
                                             'epoch_seconds': 94.6},
                                            {'epoch': 5,
                                             'train_loss': 1.966,
                                             'validation_loss': 2.0452,
                                             'validation_accuracy_pct': 44.96,
                                             'epoch_seconds': 98.6},
                                            {'epoch': 6,
                                             'train_loss': 1.7722,
                                             'validation_loss': 2.0038,
                                             'validation_accuracy_pct': 46.42,
                                             'epoch_seconds': 97.0},
                                            {'epoch': 7,
                                             'train_loss': 1.6067,
                                             'validation_loss': 1.916,
                                             'validation_accuracy_pct': 49.0,
                                             'epoch_seconds': 92.5},
                                            {'epoch': 8,
                                             'train_loss': 1.4744,
                                             'validation_loss': 1.8854,
                                             'validation_accuracy_pct': 50.14,
                                             'epoch_seconds': 93.5},
                                            {'epoch': 9,
                                             'train_loss': 1.3485,
                                             'validation_loss': 1.8492,
                                             'validation_accuracy_pct': 50.64,
                                             'epoch_seconds': 93.3},
                                            {'epoch': 10,
                                             'train_loss': 1.2377,
                                             'validation_loss': 1.8276,
                                             'validation_accuracy_pct': 52.34,
                                             'epoch_seconds': 93.3},
                                            {'epoch': 11,
                                             'train_loss': 0.797,
                                             'validation_loss': 1.7569,
                                             'validation_accuracy_pct': 56.68,
                                             'epoch_seconds': 93.8},
                                            {'epoch': 12,
                                             'train_loss': 0.689,
                                             'validation_loss': 1.8108,
                                             'validation_accuracy_pct': 56.72,
                                             'epoch_seconds': 93.0},
                                            {'epoch': 13,
                                             'train_loss': 0.632,
                                             'validation_loss': 1.855,
                                             'validation_accuracy_pct': 57.08,
                                             'epoch_seconds': 93.6},
                                            {'epoch': 14,
                                             'train_loss': 0.5815,
                                             'validation_loss': 1.9319,
                                             'validation_accuracy_pct': 56.36,
                                             'epoch_seconds': 93.2},
                                            {'epoch': 15,
                                             'train_loss': 0.5419,
                                             'validation_loss': 1.9587,
                                             'validation_accuracy_pct': 57.08,
                                             'epoch_seconds': 93.5},
                                            {'epoch': 16,
                                             'train_loss': 0.4997,
                                             'validation_loss': 2.0396,
                                             'validation_accuracy_pct': 56.3,
                                             'epoch_seconds': 91.9},
                                            {'epoch': 17,
                                             'train_loss': 0.4655,
                                             'validation_loss': 2.0644,
                                             'validation_accuracy_pct': 56.76,
                                             'epoch_seconds': 89.6},
                                            {'epoch': 18,
                                             'train_loss': 0.4273,
                                             'validation_loss': 2.1732,
                                             'validation_accuracy_pct': 56.16,
                                             'epoch_seconds': 89.3},
                                            {'epoch': 19,
                                             'train_loss': 0.3934,
                                             'validation_loss': 2.2585,
                                             'validation_accuracy_pct': 56.46,
                                             'epoch_seconds': 89.8},
                                            {'epoch': 20,
                                             'train_loss': 0.3588,
                                             'validation_loss': 2.3294,
                                             'validation_accuracy_pct': 55.64,
                                             'epoch_seconds': 89.5}],
                          'sum_printed_epoch_seconds': 1862.0,
                          'source_cells': [14, 16],
                          'test_accuracy_pct': None},
                         {'experiment_id': 'adam_crop_corrected',
                          'status': 'Corrected 45000-image continuation; epoch10 checkpoint '
                                    'restored with optimizer',
                          'optimizer': 'Adam',
                          'weight_decay': 0,
                          'batchnorm': False,
                          'trainable_parameters': 818500,
                          'augmentation': 'crop + flip in corrected continuation; stage-one '
                                          'historical augmentation not established solely by saved '
                                          'source',
                          'selected_result': {'epoch': 20,
                                              'training_accuracy_pct': 75.97,
                                              'validation_accuracy_pct': 56.66,
                                              'validation_loss': 1.9344},
                          'printed_checkpoint': 'best_inception_cifar100_crop_adam_ep20.pt',
                          'epoch_history': [{'epoch': 1,
                                             'train_loss': 3.9532,
                                             'validation_loss': 3.6288,
                                             'validation_accuracy_pct': 12.98,
                                             'epoch_seconds': 93.2},
                                            {'epoch': 2,
                                             'train_loss': 3.3044,
                                             'validation_loss': 3.1249,
                                             'validation_accuracy_pct': 23.66,
                                             'epoch_seconds': 92.0},
                                            {'epoch': 3,
                                             'train_loss': 2.8167,
                                             'validation_loss': 2.7124,
                                             'validation_accuracy_pct': 32.04,
                                             'epoch_seconds': 92.8},
                                            {'epoch': 4,
                                             'train_loss': 2.4548,
                                             'validation_loss': 2.3621,
                                             'validation_accuracy_pct': 38.5,
                                             'epoch_seconds': 92.9},
                                            {'epoch': 5,
                                             'train_loss': 2.2013,
                                             'validation_loss': 2.2763,
                                             'validation_accuracy_pct': 41.92,
                                             'epoch_seconds': 93.7},
                                            {'epoch': 6,
                                             'train_loss': 2.0128,
                                             'validation_loss': 2.1502,
                                             'validation_accuracy_pct': 44.18,
                                             'epoch_seconds': 93.2},
                                            {'epoch': 7,
                                             'train_loss': 1.86,
                                             'validation_loss': 2.0323,
                                             'validation_accuracy_pct': 46.82,
                                             'epoch_seconds': 93.0},
                                            {'epoch': 8,
                                             'train_loss': 1.7378,
                                             'validation_loss': 2.0498,
                                             'validation_accuracy_pct': 47.0,
                                             'epoch_seconds': 93.4},
                                            {'epoch': 9,
                                             'train_loss': 1.63,
                                             'validation_loss': 1.9688,
                                             'validation_accuracy_pct': 48.76,
                                             'epoch_seconds': 93.8},
                                            {'epoch': 10,
                                             'train_loss': 1.5359,
                                             'validation_loss': 1.9822,
                                             'validation_accuracy_pct': 50.84,
                                             'epoch_seconds': 93.2},
                                            {'epoch': 11,
                                             'train_loss': 1.1562,
                                             'validation_loss': 1.8119,
                                             'validation_accuracy_pct': 55.22,
                                             'epoch_seconds': 90.0},
                                            {'epoch': 12,
                                             'train_loss': 1.0778,
                                             'validation_loss': 1.7886,
                                             'validation_accuracy_pct': 55.96,
                                             'epoch_seconds': 89.0},
                                            {'epoch': 13,
                                             'train_loss': 1.0361,
                                             'validation_loss': 1.829,
                                             'validation_accuracy_pct': 55.88,
                                             'epoch_seconds': 89.6},
                                            {'epoch': 14,
                                             'train_loss': 1.0045,
                                             'validation_loss': 1.8169,
                                             'validation_accuracy_pct': 56.18,
                                             'epoch_seconds': 90.7},
                                            {'epoch': 15,
                                             'train_loss': 0.9728,
                                             'validation_loss': 1.8723,
                                             'validation_accuracy_pct': 56.06,
                                             'epoch_seconds': 89.8},
                                            {'epoch': 16,
                                             'train_loss': 0.9517,
                                             'validation_loss': 1.8811,
                                             'validation_accuracy_pct': 56.4,
                                             'epoch_seconds': 89.9},
                                            {'epoch': 17,
                                             'train_loss': 0.9252,
                                             'validation_loss': 1.9158,
                                             'validation_accuracy_pct': 56.36,
                                             'epoch_seconds': 88.2},
                                            {'epoch': 18,
                                             'train_loss': 0.9047,
                                             'validation_loss': 1.8875,
                                             'validation_accuracy_pct': 56.52,
                                             'epoch_seconds': 89.0},
                                            {'epoch': 19,
                                             'train_loss': 0.8847,
                                             'validation_loss': 1.9162,
                                             'validation_accuracy_pct': 56.48,
                                             'epoch_seconds': 88.8},
                                            {'epoch': 20,
                                             'train_loss': 0.86,
                                             'validation_loss': 1.9344,
                                             'validation_accuracy_pct': 56.66,
                                             'epoch_seconds': 89.1}],
                          'sum_printed_epoch_seconds': 1825.3,
                          'source_cells': [19, 21],
                          'test_accuracy_pct': None,
                          'evaluation': {'accuracy_pct': 56.66,
                                         'incorrect_predictions': 2167,
                                         'mean_confidence_incorrect_pct': 58.87,
                                         'mean_confidence_pct': 73.8,
                                         'nll': 1.9344,
                                         'brier_sum_over_classes': 0.6044,
                                         'ece_15_bins_pct': 17.14,
                                         'raw_outputs': {'22': 'Validation examples: 5,000\n'
                                                               'Accuracy: 56.66%\n'
                                                               'Incorrect predictions: 2,167\n'
                                                               'Mean confidence on incorrect '
                                                               'predictions: 58.87%\n'
                                                               '\n'
                                                               'Top 10 most confident mistakes:\n'
                                                               'Position 2761 | True: 32 | '
                                                               'Predicted: 91 | Confidence: '
                                                               '100.00%\n'
                                                               'Position  860 | True: 99 | '
                                                               'Predicted: 75 | Confidence: '
                                                               '100.00%\n'
                                                               'Position 2311 | True: 82 | '
                                                               'Predicted: 60 | Confidence: '
                                                               '100.00%\n'
                                                               'Position  563 | True: 76 | '
                                                               'Predicted: 23 | Confidence: '
                                                               '99.99%\n'
                                                               'Position 3288 | True: 25 | '
                                                               'Predicted: 20 | Confidence: '
                                                               '99.97%\n'
                                                               'Position 2413 | True: 47 | '
                                                               'Predicted: 96 | Confidence: '
                                                               '99.96%\n'
                                                               'Position 2796 | True: 16 | '
                                                               'Predicted:  9 | Confidence: '
                                                               '99.94%\n'
                                                               'Position 2866 | True: 71 | '
                                                               'Predicted: 60 | Confidence: '
                                                               '99.94%\n'
                                                               'Position 3752 | True:  3 | '
                                                               'Predicted: 43 | Confidence: '
                                                               '99.94%\n'
                                                               'Position 1100 | True: 35 | '
                                                               'Predicted: 36 | Confidence: '
                                                               '99.90%\n',
                                                         '23': '10 lowest-recall classes:\n'
                                                               'otter                | Recall: '
                                                               '15.22% | Precision: 16.28% | '
                                                               'Support: 46\n'
                                                               'seal                 | Recall: '
                                                               '15.56% | Precision: 21.21% | '
                                                               'Support: 45\n'
                                                               'beaver               | Recall: '
                                                               '17.39% | Precision: 17.78% | '
                                                               'Support: 46\n'
                                                               'lizard               | Recall: '
                                                               '21.28% | Precision: 30.30% | '
                                                               'Support: 47\n'
                                                               'mouse                | Recall: '
                                                               '26.00% | Precision: 37.14% | '
                                                               'Support: 50\n'
                                                               'boy                  | Recall: '
                                                               '31.48% | Precision: 45.95% | '
                                                               'Support: 54\n'
                                                               'tulip                | Recall: '
                                                               '32.00% | Precision: 51.61% | '
                                                               'Support: 50\n'
                                                               'man                  | Recall: '
                                                               '32.08% | Precision: 53.12% | '
                                                               'Support: 53\n'
                                                               'kangaroo             | Recall: '
                                                               '34.69% | Precision: 48.57% | '
                                                               'Support: 49\n'
                                                               'rabbit               | Recall: '
                                                               '35.19% | Precision: 43.18% | '
                                                               'Support: 54\n',
                                                         '24': 'Most common confusion for each '
                                                               'low-recall class:\n'
                                                               'otter        → beaver       | 4 / '
                                                               '46 examples\n'
                                                               'seal         → otter        | 8 / '
                                                               '45 examples\n'
                                                               'beaver       → shrew        | 6 / '
                                                               '46 examples\n'
                                                               'lizard       → shrew        | 3 / '
                                                               '47 examples\n'
                                                               'mouse        → shrew        | 6 / '
                                                               '50 examples\n'
                                                               'boy          → girl         | 8 / '
                                                               '54 examples\n'
                                                               'tulip        → rose         | 10 / '
                                                               '50 examples\n'
                                                               'man          → woman        | 14 / '
                                                               '53 examples\n'
                                                               'kangaroo     → rabbit       | 3 / '
                                                               '49 examples\n'
                                                               'rabbit       → fox          | 3 / '
                                                               '54 examples\n',
                                                         '26': 'Confidence bins:\n'
                                                               '[0.07, 0.13) | n=   4 | '
                                                               'Accuracy=0.00% | '
                                                               'Confidence=12.09%\n'
                                                               '[0.13, 0.20) | n=  36 | '
                                                               'Accuracy=8.33% | '
                                                               'Confidence=17.43%\n'
                                                               '[0.20, 0.27) | n= 135 | '
                                                               'Accuracy=13.33% | '
                                                               'Confidence=23.54%\n'
                                                               '[0.27, 0.33) | n= 210 | '
                                                               'Accuracy=14.76% | '
                                                               'Confidence=30.03%\n'
                                                               '[0.33, 0.40) | n= 237 | '
                                                               'Accuracy=23.21% | '
                                                               'Confidence=36.97%\n'
                                                               '[0.40, 0.47) | n= 296 | '
                                                               'Accuracy=23.99% | '
                                                               'Confidence=43.44%\n'
                                                               '[0.47, 0.53) | n= 350 | '
                                                               'Accuracy=33.14% | '
                                                               'Confidence=50.08%\n'
                                                               '[0.53, 0.60) | n= 326 | '
                                                               'Accuracy=35.28% | '
                                                               'Confidence=56.49%\n'
                                                               '[0.60, 0.67) | n= 295 | '
                                                               'Accuracy=40.00% | '
                                                               'Confidence=63.28%\n'
                                                               '[0.67, 0.73) | n= 314 | '
                                                               'Accuracy=43.95% | '
                                                               'Confidence=70.12%\n'
                                                               '[0.73, 0.80) | n= 305 | '
                                                               'Accuracy=51.48% | '
                                                               'Confidence=76.61%\n'
                                                               '[0.80, 0.87) | n= 331 | '
                                                               'Accuracy=57.40% | '
                                                               'Confidence=83.38%\n'
                                                               '[0.87, 0.93) | n= 393 | '
                                                               'Accuracy=66.92% | '
                                                               'Confidence=90.33%\n'
                                                               '[0.93, 1.00] | n=1768 | '
                                                               'Accuracy=88.12% | '
                                                               'Confidence=98.50%\n'
                                                               '\n'
                                                               'Validation examples: 5,000\n'
                                                               'Accuracy:           56.66%\n'
                                                               'Mean confidence:    73.80%\n'
                                                               'NLL:                1.9344\n'
                                                               'Brier score:        0.6044\n'
                                                               'ECE (15 bins):      17.14%\n',
                                                         '27': 'Device: cuda:0\n'
                                                               'GPU: NVIDIA GeForce RTX 5070 Ti '
                                                               'Laptop GPU\n'
                                                               'Dtype: torch.float32\n'
                                                               '\n'
                                                               'Batch size: 1\n'
                                                               'Mean batch latency:   3.295 ms\n'
                                                               'Median batch latency: 3.180 ms\n'
                                                               'P95 batch latency:    4.372 ms\n'
                                                               'Throughput:           303.5 '
                                                               'images/s\n'
                                                               'Peak allocated GPU memory: 179.65 '
                                                               'MiB\n'
                                                               'Peak increase above baseline: 7.56 '
                                                               'MiB\n'
                                                               '\n'
                                                               'Batch size: 32\n'
                                                               'Mean batch latency:   25.796 ms\n'
                                                               'Median batch latency: 25.580 ms\n'
                                                               'P95 batch latency:    27.194 ms\n'
                                                               'Throughput:           1240.5 '
                                                               'images/s\n'
                                                               'Peak allocated GPU memory: 606.92 '
                                                               'MiB\n'
                                                               'Peak increase above baseline: '
                                                               '229.06 MiB\n'}},
                          'inference_benchmark': {'device': 'RTX 5070 Ti Laptop GPU',
                                                  'dtype': 'float32',
                                                  'warmup': 20,
                                                  'repeats': 100,
                                                  'scope': 'forward only; excludes loading and '
                                                           'transfer',
                                                  'measurements': [{'batch_size': 1,
                                                                    'mean_batch_ms': 3.295,
                                                                    'median_batch_ms': 3.18,
                                                                    'p95_batch_ms': 4.372,
                                                                    'images_per_second': 303.5,
                                                                    'peak_allocated_mib': 179.65,
                                                                    'peak_increase_mib': 7.56},
                                                                   {'batch_size': 32,
                                                                    'mean_batch_ms': 25.796,
                                                                    'median_batch_ms': 25.58,
                                                                    'p95_batch_ms': 27.194,
                                                                    'images_per_second': 1240.5,
                                                                    'peak_allocated_mib': 606.92,
                                                                    'peak_increase_mib': 229.06}]}},
                         {'experiment_id': 'adam_batchnorm_crop_corrected',
                          'status': 'Corrected 45000-image continuation; epoch10 checkpoint '
                                    'restored with optimizer',
                          'optimizer': 'Adam',
                          'weight_decay': 0,
                          'batchnorm': True,
                          'trainable_parameters': 822660,
                          'augmentation': 'crop + flip in corrected continuation; stage-one '
                                          'historical augmentation not established solely by saved '
                                          'source',
                          'selected_result': {'epoch': 19,
                                              'training_accuracy_pct': 86.49,
                                              'validation_accuracy_pct': 68.16,
                                              'validation_loss': 1.205},
                          'printed_checkpoint': 'best_inception_bn_cifar100_crop_adam_20ep.pt',
                          'epoch_history': [{'epoch': 1,
                                             'train_loss': 3.2956,
                                             'validation_loss': 3.1128,
                                             'validation_accuracy_pct': 24.7,
                                             'epoch_seconds': 126.2},
                                            {'epoch': 2,
                                             'train_loss': 2.4175,
                                             'validation_loss': 2.3616,
                                             'validation_accuracy_pct': 38.8,
                                             'epoch_seconds': 125.5},
                                            {'epoch': 3,
                                             'train_loss': 2.0137,
                                             'validation_loss': 1.9491,
                                             'validation_accuracy_pct': 46.44,
                                             'epoch_seconds': 125.5},
                                            {'epoch': 4,
                                             'train_loss': 1.7586,
                                             'validation_loss': 1.8081,
                                             'validation_accuracy_pct': 50.6,
                                             'epoch_seconds': 125.7},
                                            {'epoch': 5,
                                             'train_loss': 1.5895,
                                             'validation_loss': 1.6897,
                                             'validation_accuracy_pct': 54.04,
                                             'epoch_seconds': 124.8},
                                            {'epoch': 6,
                                             'train_loss': 1.4507,
                                             'validation_loss': 1.6459,
                                             'validation_accuracy_pct': 55.72,
                                             'epoch_seconds': 124.1},
                                            {'epoch': 7,
                                             'train_loss': 1.3315,
                                             'validation_loss': 1.6862,
                                             'validation_accuracy_pct': 55.3,
                                             'epoch_seconds': 124.2},
                                            {'epoch': 8,
                                             'train_loss': 1.2393,
                                             'validation_loss': 1.5024,
                                             'validation_accuracy_pct': 59.14,
                                             'epoch_seconds': 124.2},
                                            {'epoch': 9,
                                             'train_loss': 1.1575,
                                             'validation_loss': 1.4996,
                                             'validation_accuracy_pct': 60.38,
                                             'epoch_seconds': 125.6},
                                            {'epoch': 10,
                                             'train_loss': 1.0797,
                                             'validation_loss': 1.4892,
                                             'validation_accuracy_pct': 61.18,
                                             'epoch_seconds': 126.6},
                                            {'epoch': 11,
                                             'train_loss': 0.7986,
                                             'validation_loss': 1.2033,
                                             'validation_accuracy_pct': 66.48,
                                             'epoch_seconds': 125.4},
                                            {'epoch': 12,
                                             'train_loss': 0.7358,
                                             'validation_loss': 1.184,
                                             'validation_accuracy_pct': 66.92,
                                             'epoch_seconds': 127.5},
                                            {'epoch': 13,
                                             'train_loss': 0.7039,
                                             'validation_loss': 1.1814,
                                             'validation_accuracy_pct': 67.42,
                                             'epoch_seconds': 127.2},
                                            {'epoch': 14,
                                             'train_loss': 0.6817,
                                             'validation_loss': 1.1885,
                                             'validation_accuracy_pct': 67.42,
                                             'epoch_seconds': 127.1},
                                            {'epoch': 15,
                                             'train_loss': 0.6586,
                                             'validation_loss': 1.191,
                                             'validation_accuracy_pct': 67.56,
                                             'epoch_seconds': 127.0},
                                            {'epoch': 16,
                                             'train_loss': 0.6397,
                                             'validation_loss': 1.1856,
                                             'validation_accuracy_pct': 67.92,
                                             'epoch_seconds': 127.3},
                                            {'epoch': 17,
                                             'train_loss': 0.6204,
                                             'validation_loss': 1.1817,
                                             'validation_accuracy_pct': 67.86,
                                             'epoch_seconds': 127.4},
                                            {'epoch': 18,
                                             'train_loss': 0.6053,
                                             'validation_loss': 1.19,
                                             'validation_accuracy_pct': 68.02,
                                             'epoch_seconds': 126.9},
                                            {'epoch': 19,
                                             'train_loss': 0.588,
                                             'validation_loss': 1.205,
                                             'validation_accuracy_pct': 68.16,
                                             'epoch_seconds': 129.1},
                                            {'epoch': 20,
                                             'train_loss': 0.5771,
                                             'validation_loss': 1.2059,
                                             'validation_accuracy_pct': 67.72,
                                             'epoch_seconds': 127.5}],
                          'sum_printed_epoch_seconds': 2524.8,
                          'source_cells': [31, 33],
                          'test_accuracy_pct': None},
                         {'experiment_id': 'adamw_batchnorm_crop',
                          'status': 'Completed: implementation, training, validation error '
                                    'analysis, calibration, inference benchmarking and final test '
                                    'evaluation; user-reported results',
                          'optimizer': 'AdamW',
                          'batchnorm': True,
                          'trainable_parameters': 822660,
                          'details': {'schema_version': '1.0',
                                      'experiment': 'inception_batchnorm_adamw_cifar100_crop_flip',
                                      'evidence': 'User-reported training logs; not independently '
                                                  'executed',
                                      'model': {'architecture': 'CIFAR-adapted Inception-style '
                                                                'classifier with BatchNorm',
                                                'trainable_parameters': 822660,
                                                'parameter_count_source': 'Previously reported '
                                                                          'matching BatchNorm '
                                                                          'architecture'},
                                      'dataset': {'name': 'CIFAR-100',
                                                  'training_examples': 45000,
                                                  'validation_examples': 5000,
                                                  'test_examples': 10000,
                                                  'split_seed': 42,
                                                  'augmentation': ['RandomCrop(32, padding=4)',
                                                                   'RandomHorizontalFlip(p=0.5)',
                                                                   'ToTensor'],
                                                  'validation_transform': ['ToTensor'],
                                                  'training_loader': 'Corrected training subset; '
                                                                     'excludes validation indices, '
                                                                     'per session setup',
                                                  'test_evaluated': True},
                                      'training': {'optimizer': 'AdamW',
                                                   'weight_decay': 0.0001,
                                                   'batch_size': 32,
                                                   'settings_source': 'Agreed experiment '
                                                                      'configuration',
                                                   'learning_rate_schedule': [{'first_epoch': 1,
                                                                               'last_epoch': 10,
                                                                               'learning_rate': 0.001},
                                                                              {'first_epoch': 11,
                                                                               'last_epoch': 20,
                                                                               'learning_rate': 0.0001}],
                                                   'epochs_completed': 20,
                                                   'checkpoint_selection': 'Highest validation '
                                                                           'accuracy',
                                                   'resume': {'reason': 'WSL interruption after '
                                                                        'epoch 10',
                                                              'restored_epoch': 10,
                                                              'epoch10_was_stage_best': True,
                                                              'model_and_optimizer_restored': 'User '
                                                                                              'confirmed '
                                                                                              'in '
                                                                                              'conversation',
                                                              'rng_state_restoration': 'Not '
                                                                                       'verified'},
                                                   'stage1_checkpoint': 'best_inception_bn_cifar100_crop_adamw_ep10.pt',
                                                   'stage2_checkpoint': 'best_inception_bn_cifar100_crop_adamw_ep20.pt'},
                                      'selected_result': {'epoch': 18,
                                                          'training_accuracy_pct': 85.94,
                                                          'validation_accuracy_pct': 68.98,
                                                          'validation_correct': 3449,
                                                          'validation_incorrect': 1551,
                                                          'validation_loss': 1.184,
                                                          'test_accuracy_pct': 68.08},
                                      'lowest_validation_loss': {'epoch': 14,
                                                                 'validation_loss': 1.1626,
                                                                 'validation_accuracy_pct': 68.22},
                                      'comparison': {'reference': 'BatchNorm + Adam corrected '
                                                                  'continuation',
                                                     'reference_validation_accuracy_pct': 68.16,
                                                     'improvement_percentage_points': 0.82,
                                                     'additional_correct_validation_examples': 41,
                                                     'limitation': 'Single runs; complete matched '
                                                                   'RNG conditions not verified'},
                                      'timing': {'sum_reported_epoch_seconds': 3449.4,
                                                 'scope': 'Training plus validation per epoch, as '
                                                          'defined by notebook helper; excludes '
                                                          'interruption downtime'},
                                      'epoch_history': [{'epoch': 1,
                                                         'train_loss': 3.3119,
                                                         'validation_loss': 2.6429,
                                                         'validation_accuracy_pct': 32.1,
                                                         'epoch_seconds': 165.9},
                                                        {'epoch': 2,
                                                         'train_loss': 2.4183,
                                                         'validation_loss': 2.3021,
                                                         'validation_accuracy_pct': 39.42,
                                                         'epoch_seconds': 169.5},
                                                        {'epoch': 3,
                                                         'train_loss': 2.0054,
                                                         'validation_loss': 1.9253,
                                                         'validation_accuracy_pct': 47.82,
                                                         'epoch_seconds': 175.2},
                                                        {'epoch': 4,
                                                         'train_loss': 1.7526,
                                                         'validation_loss': 1.7238,
                                                         'validation_accuracy_pct': 52.04,
                                                         'epoch_seconds': 174.0},
                                                        {'epoch': 5,
                                                         'train_loss': 1.577,
                                                         'validation_loss': 1.6669,
                                                         'validation_accuracy_pct': 54.56,
                                                         'epoch_seconds': 176.4},
                                                        {'epoch': 6,
                                                         'train_loss': 1.4373,
                                                         'validation_loss': 1.5844,
                                                         'validation_accuracy_pct': 57.4,
                                                         'epoch_seconds': 174.4},
                                                        {'epoch': 7,
                                                         'train_loss': 1.3205,
                                                         'validation_loss': 1.5083,
                                                         'validation_accuracy_pct': 58.76,
                                                         'epoch_seconds': 175.1},
                                                        {'epoch': 8,
                                                         'train_loss': 1.2325,
                                                         'validation_loss': 1.5881,
                                                         'validation_accuracy_pct': 58.16,
                                                         'epoch_seconds': 173.8},
                                                        {'epoch': 9,
                                                         'train_loss': 1.1392,
                                                         'validation_loss': 1.4759,
                                                         'validation_accuracy_pct': 59.68,
                                                         'epoch_seconds': 168.8},
                                                        {'epoch': 10,
                                                         'train_loss': 1.0653,
                                                         'validation_loss': 1.4312,
                                                         'validation_accuracy_pct': 61.78,
                                                         'epoch_seconds': 159.7},
                                                        {'epoch': 11,
                                                         'train_loss': 0.7981,
                                                         'validation_loss': 1.1765,
                                                         'validation_accuracy_pct': 67.62,
                                                         'epoch_seconds': 171.8},
                                                        {'epoch': 12,
                                                         'train_loss': 0.729,
                                                         'validation_loss': 1.1692,
                                                         'validation_accuracy_pct': 68.02,
                                                         'epoch_seconds': 173.1},
                                                        {'epoch': 13,
                                                         'train_loss': 0.6962,
                                                         'validation_loss': 1.1671,
                                                         'validation_accuracy_pct': 68.38,
                                                         'epoch_seconds': 176.0},
                                                        {'epoch': 14,
                                                         'train_loss': 0.6777,
                                                         'validation_loss': 1.1626,
                                                         'validation_accuracy_pct': 68.22,
                                                         'epoch_seconds': 174.8},
                                                        {'epoch': 15,
                                                         'train_loss': 0.6568,
                                                         'validation_loss': 1.1677,
                                                         'validation_accuracy_pct': 68.12,
                                                         'epoch_seconds': 174.9},
                                                        {'epoch': 16,
                                                         'train_loss': 0.6317,
                                                         'validation_loss': 1.1698,
                                                         'validation_accuracy_pct': 68.66,
                                                         'epoch_seconds': 175.4},
                                                        {'epoch': 17,
                                                         'train_loss': 0.6195,
                                                         'validation_loss': 1.1818,
                                                         'validation_accuracy_pct': 68.3,
                                                         'epoch_seconds': 176.6},
                                                        {'epoch': 18,
                                                         'train_loss': 0.6047,
                                                         'validation_loss': 1.184,
                                                         'validation_accuracy_pct': 68.98,
                                                         'epoch_seconds': 171.3},
                                                        {'epoch': 19,
                                                         'train_loss': 0.5876,
                                                         'validation_loss': 1.1716,
                                                         'validation_accuracy_pct': 68.84,
                                                         'epoch_seconds': 172.7},
                                                        {'epoch': 20,
                                                         'train_loss': 0.5731,
                                                         'validation_loss': 1.1838,
                                                         'validation_accuracy_pct': 68.62,
                                                         'epoch_seconds': 170.0}],
                                      'pending': [],
                                      'test_evaluation': {'examples': 10000,
                                                          'accuracy_pct': 68.08,
                                                          'cross_entropy_loss': 1.1754,
                                                          'correct_predictions': 6808,
                                                          'incorrect_predictions': 3192,
                                                          'selected_epoch': 18,
                                                          'evidence': 'User-reported final '
                                                                      'evaluation; eval mode '
                                                                      'discussed and confirmed '
                                                                      'workflow'}},
                          'selected_result': {'epoch': 18,
                                              'training_accuracy_pct': 85.94,
                                              'validation_accuracy_pct': 68.98,
                                              'validation_correct': 3449,
                                              'validation_incorrect': 1551,
                                              'validation_loss': 1.184,
                                              'test_accuracy_pct': 68.08},
                          'test_evaluation': {'examples': 10000,
                                              'accuracy_pct': 68.08,
                                              'cross_entropy_loss': 1.1754,
                                              'correct_predictions': 6808,
                                              'incorrect_predictions': 3192,
                                              'selected_epoch': 18,
                                              'evidence': 'User-reported final evaluation; eval '
                                                          'mode discussed and confirmed workflow'},
                          'evaluation': {'split': 'validation',
                                         'examples': 5000,
                                         'accuracy_pct': 68.98,
                                         'incorrect_predictions': 1551,
                                         'mean_confidence_incorrect_pct': 58.12,
                                         'mean_confidence_pct': 77.48,
                                         'nll': 1.184,
                                         'brier_sum_over_classes': 0.4363,
                                         'ece_15_equal_width_bins_pct': 8.59,
                                         'lowest_recall_classes': [{'class': 'seal',
                                                                    'recall_pct': 28.89,
                                                                    'precision_pct': 24.53,
                                                                    'support': 45},
                                                                   {'class': 'otter',
                                                                    'recall_pct': 30.43,
                                                                    'precision_pct': 36.84,
                                                                    'support': 46},
                                                                   {'class': 'bowl',
                                                                    'recall_pct': 33.33,
                                                                    'precision_pct': 42.11,
                                                                    'support': 48},
                                                                   {'class': 'beaver',
                                                                    'recall_pct': 34.78,
                                                                    'precision_pct': 51.61,
                                                                    'support': 46},
                                                                   {'class': 'woman',
                                                                    'recall_pct': 35.85,
                                                                    'precision_pct': 37.25,
                                                                    'support': 53},
                                                                   {'class': 'bear',
                                                                    'recall_pct': 39.13,
                                                                    'precision_pct': 39.13,
                                                                    'support': 46},
                                                                   {'class': 'tulip',
                                                                    'recall_pct': 40,
                                                                    'precision_pct': 52.63,
                                                                    'support': 50},
                                                                   {'class': 'crab',
                                                                    'recall_pct': 41.67,
                                                                    'precision_pct': 52.63,
                                                                    'support': 48},
                                                                   {'class': 'lizard',
                                                                    'recall_pct': 42.55,
                                                                    'precision_pct': 41.67,
                                                                    'support': 47},
                                                                   {'class': 'rabbit',
                                                                    'recall_pct': 42.59,
                                                                    'precision_pct': 47.92,
                                                                    'support': 54}]},
                          'inference_benchmark': {'device': 'NVIDIA GeForce RTX 5070 Ti Laptop GPU',
                                                  'dtype': 'float32',
                                                  'warmup': 20,
                                                  'repeats': 100,
                                                  'scope': 'Forward only; loading and transfers '
                                                           'excluded; peak allocation includes '
                                                           'notebook tensors',
                                                  'measurements': [{'batch_size': 1,
                                                                    'mean_batch_ms': 8.841,
                                                                    'median_batch_ms': 8.53,
                                                                    'p95_batch_ms': 12.182,
                                                                    'images_per_second': 113.1,
                                                                    'peak_allocated_mib': 92.77,
                                                                    'peak_increase_mib': 7.06},
                                                                   {'batch_size': 32,
                                                                    'mean_batch_ms': 30.955,
                                                                    'median_batch_ms': 30.854,
                                                                    'p95_batch_ms': 37.452,
                                                                    'images_per_second': 1033.8,
                                                                    'peak_allocated_mib': 602.07,
                                                                    'peak_increase_mib': 232.81}]}}],
 'best_observed_retained_result': {'experiment_id': 'adamw_batchnorm_crop',
                                   'epoch': 18,
                                   'validation_accuracy_pct': 68.98,
                                   'training_accuracy_pct': 85.94,
                                   'validation_loss': 1.184,
                                   'test_accuracy_pct': 68.08},
 'archive': {'leakage_affected_superseded_runs': [{'run_id': 'crop_adam_reported',
                                                   'source_cell_indices': [19, 21],
                                                   'epoch_history': [{'epoch': 1,
                                                                      'train_loss': 3.9532,
                                                                      'validation_loss': 3.6288,
                                                                      'validation_accuracy_pct': 12.98,
                                                                      'epoch_seconds': 93.2,
                                                                      'source_cell_index': 19},
                                                                     {'epoch': 2,
                                                                      'train_loss': 3.3044,
                                                                      'validation_loss': 3.1249,
                                                                      'validation_accuracy_pct': 23.66,
                                                                      'epoch_seconds': 92.0,
                                                                      'source_cell_index': 19},
                                                                     {'epoch': 3,
                                                                      'train_loss': 2.8167,
                                                                      'validation_loss': 2.7124,
                                                                      'validation_accuracy_pct': 32.04,
                                                                      'epoch_seconds': 92.8,
                                                                      'source_cell_index': 19},
                                                                     {'epoch': 4,
                                                                      'train_loss': 2.4548,
                                                                      'validation_loss': 2.3621,
                                                                      'validation_accuracy_pct': 38.5,
                                                                      'epoch_seconds': 92.9,
                                                                      'source_cell_index': 19},
                                                                     {'epoch': 5,
                                                                      'train_loss': 2.2013,
                                                                      'validation_loss': 2.2763,
                                                                      'validation_accuracy_pct': 41.92,
                                                                      'epoch_seconds': 93.7,
                                                                      'source_cell_index': 19},
                                                                     {'epoch': 6,
                                                                      'train_loss': 2.0128,
                                                                      'validation_loss': 2.1502,
                                                                      'validation_accuracy_pct': 44.18,
                                                                      'epoch_seconds': 93.2,
                                                                      'source_cell_index': 19},
                                                                     {'epoch': 7,
                                                                      'train_loss': 1.86,
                                                                      'validation_loss': 2.0323,
                                                                      'validation_accuracy_pct': 46.82,
                                                                      'epoch_seconds': 93.0,
                                                                      'source_cell_index': 19},
                                                                     {'epoch': 8,
                                                                      'train_loss': 1.7378,
                                                                      'validation_loss': 2.0498,
                                                                      'validation_accuracy_pct': 47.0,
                                                                      'epoch_seconds': 93.4,
                                                                      'source_cell_index': 19},
                                                                     {'epoch': 9,
                                                                      'train_loss': 1.63,
                                                                      'validation_loss': 1.9688,
                                                                      'validation_accuracy_pct': 48.76,
                                                                      'epoch_seconds': 93.8,
                                                                      'source_cell_index': 19},
                                                                     {'epoch': 10,
                                                                      'train_loss': 1.5359,
                                                                      'validation_loss': 1.9822,
                                                                      'validation_accuracy_pct': 50.84,
                                                                      'epoch_seconds': 93.2,
                                                                      'source_cell_index': 19},
                                                                     {'epoch': 11,
                                                                      'train_loss': 1.2132,
                                                                      'validation_loss': 1.7014,
                                                                      'validation_accuracy_pct': 56.94,
                                                                      'epoch_seconds': 107.0,
                                                                      'source_cell_index': 21},
                                                                     {'epoch': 12,
                                                                      'train_loss': 1.1365,
                                                                      'validation_loss': 1.5988,
                                                                      'validation_accuracy_pct': 58.62,
                                                                      'epoch_seconds': 102.3,
                                                                      'source_cell_index': 21},
                                                                     {'epoch': 13,
                                                                      'train_loss': 1.0953,
                                                                      'validation_loss': 1.5593,
                                                                      'validation_accuracy_pct': 59.9,
                                                                      'epoch_seconds': 97.0,
                                                                      'source_cell_index': 21},
                                                                     {'epoch': 14,
                                                                      'train_loss': 1.0634,
                                                                      'validation_loss': 1.4947,
                                                                      'validation_accuracy_pct': 60.5,
                                                                      'epoch_seconds': 95.8,
                                                                      'source_cell_index': 21},
                                                                     {'epoch': 15,
                                                                      'train_loss': 1.0317,
                                                                      'validation_loss': 1.4676,
                                                                      'validation_accuracy_pct': 61.44,
                                                                      'epoch_seconds': 95.8,
                                                                      'source_cell_index': 21},
                                                                     {'epoch': 16,
                                                                      'train_loss': 1.0079,
                                                                      'validation_loss': 1.4326,
                                                                      'validation_accuracy_pct': 62.56,
                                                                      'epoch_seconds': 95.0,
                                                                      'source_cell_index': 21},
                                                                     {'epoch': 17,
                                                                      'train_loss': 0.9813,
                                                                      'validation_loss': 1.3909,
                                                                      'validation_accuracy_pct': 63.1,
                                                                      'epoch_seconds': 95.8,
                                                                      'source_cell_index': 21},
                                                                     {'epoch': 18,
                                                                      'train_loss': 0.9609,
                                                                      'validation_loss': 1.3378,
                                                                      'validation_accuracy_pct': 64.12,
                                                                      'epoch_seconds': 95.7,
                                                                      'source_cell_index': 21},
                                                                     {'epoch': 19,
                                                                      'train_loss': 0.9382,
                                                                      'validation_loss': 1.3207,
                                                                      'validation_accuracy_pct': 64.52,
                                                                      'epoch_seconds': 95.4,
                                                                      'source_cell_index': 21},
                                                                     {'epoch': 20,
                                                                      'train_loss': 0.9197,
                                                                      'validation_loss': 1.2977,
                                                                      'validation_accuracy_pct': 65.46,
                                                                      'epoch_seconds': 96.3,
                                                                      'source_cell_index': 21}],
                                                   'selected_epoch': 20,
                                                   'training_accuracy_pct': 75.27,
                                                   'validation_accuracy_pct': 65.46,
                                                   'validation_loss': 1.2977,
                                                   'sum_printed_epoch_seconds': 1907.3,
                                                   'printed_checkpoint': 'best_inception_cifar100_crop_adam_ep20.pt',
                                                   'checkpoint_in_current_source': 'best_inception_cifar100_crop_adam_ep20.pt',
                                                   'validation_integrity': 'contaminated by full '
                                                                           '50000-image cropped '
                                                                           'loader at epochs '
                                                                           '11–20'},
                                                  {'run_id': 'crop_adam_batchnorm_reported',
                                                   'source_cell_indices': [31, 33],
                                                   'epoch_history': [{'epoch': 1,
                                                                      'train_loss': 3.2956,
                                                                      'validation_loss': 3.1128,
                                                                      'validation_accuracy_pct': 24.7,
                                                                      'epoch_seconds': 126.2,
                                                                      'source_cell_index': 31},
                                                                     {'epoch': 2,
                                                                      'train_loss': 2.4175,
                                                                      'validation_loss': 2.3616,
                                                                      'validation_accuracy_pct': 38.8,
                                                                      'epoch_seconds': 125.5,
                                                                      'source_cell_index': 31},
                                                                     {'epoch': 3,
                                                                      'train_loss': 2.0137,
                                                                      'validation_loss': 1.9491,
                                                                      'validation_accuracy_pct': 46.44,
                                                                      'epoch_seconds': 125.5,
                                                                      'source_cell_index': 31},
                                                                     {'epoch': 4,
                                                                      'train_loss': 1.7586,
                                                                      'validation_loss': 1.8081,
                                                                      'validation_accuracy_pct': 50.6,
                                                                      'epoch_seconds': 125.7,
                                                                      'source_cell_index': 31},
                                                                     {'epoch': 5,
                                                                      'train_loss': 1.5895,
                                                                      'validation_loss': 1.6897,
                                                                      'validation_accuracy_pct': 54.04,
                                                                      'epoch_seconds': 124.8,
                                                                      'source_cell_index': 31},
                                                                     {'epoch': 6,
                                                                      'train_loss': 1.4507,
                                                                      'validation_loss': 1.6459,
                                                                      'validation_accuracy_pct': 55.72,
                                                                      'epoch_seconds': 124.1,
                                                                      'source_cell_index': 31},
                                                                     {'epoch': 7,
                                                                      'train_loss': 1.3315,
                                                                      'validation_loss': 1.6862,
                                                                      'validation_accuracy_pct': 55.3,
                                                                      'epoch_seconds': 124.2,
                                                                      'source_cell_index': 31},
                                                                     {'epoch': 8,
                                                                      'train_loss': 1.2393,
                                                                      'validation_loss': 1.5024,
                                                                      'validation_accuracy_pct': 59.14,
                                                                      'epoch_seconds': 124.2,
                                                                      'source_cell_index': 31},
                                                                     {'epoch': 9,
                                                                      'train_loss': 1.1575,
                                                                      'validation_loss': 1.4996,
                                                                      'validation_accuracy_pct': 60.38,
                                                                      'epoch_seconds': 125.6,
                                                                      'source_cell_index': 31},
                                                                     {'epoch': 10,
                                                                      'train_loss': 1.0797,
                                                                      'validation_loss': 1.4892,
                                                                      'validation_accuracy_pct': 61.18,
                                                                      'epoch_seconds': 126.6,
                                                                      'source_cell_index': 31},
                                                                     {'epoch': 11,
                                                                      'train_loss': 0.8532,
                                                                      'validation_loss': 1.1172,
                                                                      'validation_accuracy_pct': 68.38,
                                                                      'epoch_seconds': 139.2,
                                                                      'source_cell_index': 33},
                                                                     {'epoch': 12,
                                                                      'train_loss': 0.7834,
                                                                      'validation_loss': 1.0612,
                                                                      'validation_accuracy_pct': 69.8,
                                                                      'epoch_seconds': 137.4,
                                                                      'source_cell_index': 33},
                                                                     {'epoch': 13,
                                                                      'train_loss': 0.7496,
                                                                      'validation_loss': 1.0038,
                                                                      'validation_accuracy_pct': 71.4,
                                                                      'epoch_seconds': 135.0,
                                                                      'source_cell_index': 33},
                                                                     {'epoch': 14,
                                                                      'train_loss': 0.73,
                                                                      'validation_loss': 0.9665,
                                                                      'validation_accuracy_pct': 72.16,
                                                                      'epoch_seconds': 134.3,
                                                                      'source_cell_index': 33},
                                                                     {'epoch': 15,
                                                                      'train_loss': 0.7038,
                                                                      'validation_loss': 0.9381,
                                                                      'validation_accuracy_pct': 72.52,
                                                                      'epoch_seconds': 134.1,
                                                                      'source_cell_index': 33},
                                                                     {'epoch': 16,
                                                                      'train_loss': 0.6848,
                                                                      'validation_loss': 0.8939,
                                                                      'validation_accuracy_pct': 74.04,
                                                                      'epoch_seconds': 134.5,
                                                                      'source_cell_index': 33},
                                                                     {'epoch': 17,
                                                                      'train_loss': 0.6634,
                                                                      'validation_loss': 0.8577,
                                                                      'validation_accuracy_pct': 74.92,
                                                                      'epoch_seconds': 134.1,
                                                                      'source_cell_index': 33},
                                                                     {'epoch': 18,
                                                                      'train_loss': 0.6499,
                                                                      'validation_loss': 0.8257,
                                                                      'validation_accuracy_pct': 75.9,
                                                                      'epoch_seconds': 134.1,
                                                                      'source_cell_index': 33},
                                                                     {'epoch': 19,
                                                                      'train_loss': 0.6343,
                                                                      'validation_loss': 0.7985,
                                                                      'validation_accuracy_pct': 76.26,
                                                                      'epoch_seconds': 133.7,
                                                                      'source_cell_index': 33},
                                                                     {'epoch': 20,
                                                                      'train_loss': 0.6229,
                                                                      'validation_loss': 0.7719,
                                                                      'validation_accuracy_pct': 77.12,
                                                                      'epoch_seconds': 134.6,
                                                                      'source_cell_index': 33}],
                                                   'selected_epoch': 20,
                                                   'training_accuracy_pct': 86.76,
                                                   'validation_accuracy_pct': 77.12,
                                                   'validation_loss': 0.7719,
                                                   'sum_printed_epoch_seconds': 2603.4,
                                                   'printed_checkpoint': 'best_inception_bn_cifar100_crop_adam_20ep.pt',
                                                   'checkpoint_in_current_source': 'best_inception_bn_cifar100_crop_adam_20ep.pt',
                                                   'validation_integrity': 'contaminated by full '
                                                                           '50000-image cropped '
                                                                           'loader at epochs '
                                                                           '11–20'}],
             'leakage_affected_superseded_adam_analysis': {'validation_integrity': 'contaminated',
                                                           'accuracy_pct': 65.46,
                                                           'incorrect': 1727,
                                                           'mean_confidence_wrong_pct': 53.63,
                                                           'mean_confidence_pct': 73.24,
                                                           'nll': 1.2977,
                                                           'brier_sum_over_classes': 0.4673,
                                                           'ece_15_bins_pct': 7.78,
                                                           'raw_outputs': {'22': 'Validation '
                                                                                 'examples: 5,000\n'
                                                                                 'Accuracy: '
                                                                                 '65.46%\n'
                                                                                 'Incorrect '
                                                                                 'predictions: '
                                                                                 '1,727\n'
                                                                                 'Mean confidence '
                                                                                 'on incorrect '
                                                                                 'predictions: '
                                                                                 '53.63%\n'
                                                                                 '\n'
                                                                                 'Top 10 most '
                                                                                 'confident '
                                                                                 'mistakes:\n'
                                                                                 'Position 2761 | '
                                                                                 'True: 32 | '
                                                                                 'Predicted: 91 | '
                                                                                 'Confidence: '
                                                                                 '100.00%\n'
                                                                                 'Position  563 | '
                                                                                 'True: 76 | '
                                                                                 'Predicted: 23 | '
                                                                                 'Confidence: '
                                                                                 '99.71%\n'
                                                                                 'Position 3752 | '
                                                                                 'True:  3 | '
                                                                                 'Predicted: 43 | '
                                                                                 'Confidence: '
                                                                                 '99.68%\n'
                                                                                 'Position 4213 | '
                                                                                 'True: 44 | '
                                                                                 'Predicted: 61 | '
                                                                                 'Confidence: '
                                                                                 '99.47%\n'
                                                                                 'Position 4844 | '
                                                                                 'True: 63 | '
                                                                                 'Predicted: 42 | '
                                                                                 'Confidence: '
                                                                                 '99.42%\n'
                                                                                 'Position 1027 | '
                                                                                 'True:  4 | '
                                                                                 'Predicted: 74 | '
                                                                                 'Confidence: '
                                                                                 '99.39%\n'
                                                                                 'Position 3288 | '
                                                                                 'True: 25 | '
                                                                                 'Predicted: 20 | '
                                                                                 'Confidence: '
                                                                                 '99.26%\n'
                                                                                 'Position 1501 | '
                                                                                 'True: 83 | '
                                                                                 'Predicted: 53 | '
                                                                                 'Confidence: '
                                                                                 '99.25%\n'
                                                                                 'Position 2413 | '
                                                                                 'True: 47 | '
                                                                                 'Predicted: 96 | '
                                                                                 'Confidence: '
                                                                                 '99.15%\n'
                                                                                 'Position  234 | '
                                                                                 'True:  4 | '
                                                                                 'Predicted: 88 | '
                                                                                 'Confidence: '
                                                                                 '99.11%\n',
                                                                           '23': '10 lowest-recall '
                                                                                 'classes:\n'
                                                                                 'otter                '
                                                                                 '| Recall: 19.57% '
                                                                                 '| Precision: '
                                                                                 '22.50% | '
                                                                                 'Support: 46\n'
                                                                                 'beaver               '
                                                                                 '| Recall: 26.09% '
                                                                                 '| Precision: '
                                                                                 '36.36% | '
                                                                                 'Support: 46\n'
                                                                                 'lizard               '
                                                                                 '| Recall: 27.66% '
                                                                                 '| Precision: '
                                                                                 '38.24% | '
                                                                                 'Support: 47\n'
                                                                                 'seal                 '
                                                                                 '| Recall: 33.33% '
                                                                                 '| Precision: '
                                                                                 '34.09% | '
                                                                                 'Support: 45\n'
                                                                                 'mouse                '
                                                                                 '| Recall: 40.00% '
                                                                                 '| Precision: '
                                                                                 '42.55% | '
                                                                                 'Support: 50\n'
                                                                                 'couch                '
                                                                                 '| Recall: 41.18% '
                                                                                 '| Precision: '
                                                                                 '61.76% | '
                                                                                 'Support: 51\n'
                                                                                 'woman                '
                                                                                 '| Recall: 43.40% '
                                                                                 '| Precision: '
                                                                                 '50.00% | '
                                                                                 'Support: 53\n'
                                                                                 'turtle               '
                                                                                 '| Recall: 43.64% '
                                                                                 '| Precision: '
                                                                                 '51.06% | '
                                                                                 'Support: 55\n'
                                                                                 'man                  '
                                                                                 '| Recall: 45.28% '
                                                                                 '| Precision: '
                                                                                 '60.00% | '
                                                                                 'Support: 53\n'
                                                                                 'ray                  '
                                                                                 '| Recall: 45.83% '
                                                                                 '| Precision: '
                                                                                 '61.11% | '
                                                                                 'Support: 48\n',
                                                                           '24': 'Most common '
                                                                                 'confusion for '
                                                                                 'each low-recall '
                                                                                 'class:\n'
                                                                                 'otter        → '
                                                                                 'crocodile    | 3 '
                                                                                 '/ 46 examples\n'
                                                                                 'beaver       → '
                                                                                 'shrew        | 6 '
                                                                                 '/ 46 examples\n'
                                                                                 'lizard       → '
                                                                                 'crocodile    | 5 '
                                                                                 '/ 47 examples\n'
                                                                                 'seal         → '
                                                                                 'otter        | 8 '
                                                                                 '/ 45 examples\n'
                                                                                 'mouse        → '
                                                                                 'shrew        | 5 '
                                                                                 '/ 50 examples\n'
                                                                                 'couch        → '
                                                                                 'bed          | 5 '
                                                                                 '/ 51 examples\n'
                                                                                 'woman        → '
                                                                                 'man          | 6 '
                                                                                 '/ 53 examples\n'
                                                                                 'turtle       → '
                                                                                 'shark        | 5 '
                                                                                 '/ 55 examples\n'
                                                                                 'man          → '
                                                                                 'woman        | 9 '
                                                                                 '/ 53 examples\n'
                                                                                 'ray          → '
                                                                                 'shark        | 4 '
                                                                                 '/ 48 examples\n',
                                                                           '26': 'Confidence '
                                                                                 'bins:\n'
                                                                                 '[0.07, 0.13) | '
                                                                                 'n=   1 | '
                                                                                 'Accuracy=0.00% | '
                                                                                 'Confidence=13.01%\n'
                                                                                 '[0.13, 0.20) | '
                                                                                 'n=  47 | '
                                                                                 'Accuracy=14.89% '
                                                                                 '| '
                                                                                 'Confidence=17.93%\n'
                                                                                 '[0.20, 0.27) | '
                                                                                 'n= 135 | '
                                                                                 'Accuracy=17.04% '
                                                                                 '| '
                                                                                 'Confidence=23.57%\n'
                                                                                 '[0.27, 0.33) | '
                                                                                 'n= 210 | '
                                                                                 'Accuracy=17.14% '
                                                                                 '| '
                                                                                 'Confidence=30.07%\n'
                                                                                 '[0.33, 0.40) | '
                                                                                 'n= 296 | '
                                                                                 'Accuracy=26.69% '
                                                                                 '| '
                                                                                 'Confidence=36.61%\n'
                                                                                 '[0.40, 0.47) | '
                                                                                 'n= 303 | '
                                                                                 'Accuracy=36.96% '
                                                                                 '| '
                                                                                 'Confidence=43.32%\n'
                                                                                 '[0.47, 0.53) | '
                                                                                 'n= 321 | '
                                                                                 'Accuracy=38.01% '
                                                                                 '| '
                                                                                 'Confidence=50.08%\n'
                                                                                 '[0.53, 0.60) | '
                                                                                 'n= 304 | '
                                                                                 'Accuracy=49.34% '
                                                                                 '| '
                                                                                 'Confidence=56.62%\n'
                                                                                 '[0.60, 0.67) | '
                                                                                 'n= 314 | '
                                                                                 'Accuracy=49.68% '
                                                                                 '| '
                                                                                 'Confidence=63.28%\n'
                                                                                 '[0.67, 0.73) | '
                                                                                 'n= 309 | '
                                                                                 'Accuracy=55.99% '
                                                                                 '| '
                                                                                 'Confidence=70.05%\n'
                                                                                 '[0.73, 0.80) | '
                                                                                 'n= 269 | '
                                                                                 'Accuracy=65.80% '
                                                                                 '| '
                                                                                 'Confidence=76.68%\n'
                                                                                 '[0.80, 0.87) | '
                                                                                 'n= 350 | '
                                                                                 'Accuracy=72.00% '
                                                                                 '| '
                                                                                 'Confidence=83.37%\n'
                                                                                 '[0.87, 0.93) | '
                                                                                 'n= 433 | '
                                                                                 'Accuracy=79.91% '
                                                                                 '| '
                                                                                 'Confidence=90.23%\n'
                                                                                 '[0.93, 1.00] | '
                                                                                 'n=1708 | '
                                                                                 'Accuracy=96.02% '
                                                                                 '| '
                                                                                 'Confidence=98.49%\n'
                                                                                 '\n'
                                                                                 'Validation '
                                                                                 'examples: 5,000\n'
                                                                                 'Accuracy:           '
                                                                                 '65.46%\n'
                                                                                 'Mean '
                                                                                 'confidence:    '
                                                                                 '73.24%\n'
                                                                                 'NLL:                '
                                                                                 '1.2977\n'
                                                                                 'Brier '
                                                                                 'score:        '
                                                                                 '0.4673\n'
                                                                                 'ECE (15 '
                                                                                 'bins):      '
                                                                                 '7.78%\n',
                                                                           '27': 'Device: cuda:0\n'
                                                                                 'GPU: NVIDIA '
                                                                                 'GeForce RTX 5070 '
                                                                                 'Ti Laptop GPU\n'
                                                                                 'Dtype: '
                                                                                 'torch.float32\n'
                                                                                 '\n'
                                                                                 'Batch size: 1\n'
                                                                                 'Mean batch '
                                                                                 'latency:   3.150 '
                                                                                 'ms\n'
                                                                                 'Median batch '
                                                                                 'latency: 3.057 '
                                                                                 'ms\n'
                                                                                 'P95 batch '
                                                                                 'latency:    '
                                                                                 '3.894 ms\n'
                                                                                 'Throughput:           '
                                                                                 '317.5 images/s\n'
                                                                                 'Peak allocated '
                                                                                 'GPU memory: '
                                                                                 '90.50 MiB\n'
                                                                                 'Peak increase '
                                                                                 'above baseline: '
                                                                                 '7.06 MiB\n'
                                                                                 '\n'
                                                                                 'Batch size: 32\n'
                                                                                 'Mean batch '
                                                                                 'latency:   '
                                                                                 '24.961 ms\n'
                                                                                 'Median batch '
                                                                                 'latency: 24.937 '
                                                                                 'ms\n'
                                                                                 'P95 batch '
                                                                                 'latency:    '
                                                                                 '26.155 ms\n'
                                                                                 'Throughput:           '
                                                                                 '1282.0 images/s\n'
                                                                                 'Peak allocated '
                                                                                 'GPU memory: '
                                                                                 '519.65 MiB\n'
                                                                                 'Peak increase '
                                                                                 'above baseline: '
                                                                                 '232.69 MiB\n'}},
             'explanation': '65.46% Adam and 77.12% BatchNorm outputs used the full 50000-image '
                            'cropped loader and are excluded from clean comparisons.',
             'earlier_conversation_trials': [{'label': 'flip_adamw_initial_10ep',
                                              'best_epoch': 10,
                                              'training_accuracy_pct': 67.68,
                                              'validation_accuracy_pct': 50.58,
                                              'validation_loss': 1.9228},
                                             {'label': 'flip_adamw_fresh_10ep',
                                              'best_epoch': 10,
                                              'training_accuracy_pct': 70.2,
                                              'validation_accuracy_pct': 52.14,
                                              'validation_loss': 1.8473},
                                             {'label': 'crop_trial_10ep',
                                              'best_epoch': 9,
                                              'training_accuracy_pct': 60.15,
                                              'validation_accuracy_pct': 51.12,
                                              'validation_loss': 1.9204},
                                             {'label': 'rerun_10ep_augmentation_uncertain',
                                              'best_epoch': 10,
                                              'training_accuracy_pct': 61.24,
                                              'validation_accuracy_pct': 51.74,
                                              'validation_loss': 1.9089},
                                             {'label': 'crop_adamw_previous_20ep',
                                              'best_epoch': 19,
                                              'training_accuracy_pct': 76.09,
                                              'validation_accuracy_pct': 56.96,
                                              'validation_loss': 1.8092,
                                              'optimizer_resume_provenance': 'Earlier '
                                                                             'optimizer-reset '
                                                                             'discussion; not '
                                                                             'established as '
                                                                             'uninterrupted '
                                                                             'continuation'},
                                             {'label': 'crop_adam_previous_20ep',
                                              'best_epoch': 20,
                                              'training_accuracy_pct': 76.28,
                                              'validation_accuracy_pct': 57.1,
                                              'validation_loss': 1.917,
                                              'optimizer_resume_provenance': 'Earlier '
                                                                             'optimizer-reset '
                                                                             'discussion; '
                                                                             'superseded by '
                                                                             'corrected '
                                                                             'continuation'}]},
 'pending': [],
 'final_test_result': {'examples': 10000,
                       'accuracy_pct': 68.08,
                       'cross_entropy_loss': 1.1754,
                       'correct_predictions': 6808,
                       'incorrect_predictions': 3192,
                       'selected_epoch': 18,
                       'evidence': 'User-reported final evaluation; eval mode discussed and '
                                   'confirmed workflow'},
 'experiment_status': 'Inception experiment complete; full GoogLeNet reproduction not claimed'}

# Convenient references for figures.
experiments = results["current_experiments"]
final_test_result = results["final_test_result"]
