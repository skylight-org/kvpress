# Experiment 5 — upstream error of KV cache compression

Generated from 148 runs, 7,224,320 (head, query) rows.

Models: ['Llama-3.1-8B-Instruct', 'Qwen3-4B-Instruct-2507']. Datasets: ['ruler32k', 'scbench']. GPUs: ['NVIDIA A100-SXM4-80GB', 'NVIDIA A40', 'NVIDIA L40S'].


## Realised memory (the x axis of every comparison)

| model | dataset | ratio | arm | kept fraction |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | ruler32k | 4x | bernoulli | 0.2500 |
| Llama-3.1-8B-Instruct | ruler32k | 8x | bernoulli | 0.1250 |
| Llama-3.1-8B-Instruct | ruler32k | 16x | bernoulli | 0.0625 |
| Llama-3.1-8B-Instruct | ruler32k | 4x | kvzip | 0.2500 |
| Llama-3.1-8B-Instruct | ruler32k | 8x | kvzip | 0.1250 |
| Llama-3.1-8B-Instruct | ruler32k | 16x | kvzip | 0.0625 |
| Llama-3.1-8B-Instruct | ruler32k | 4x | perhead_bern | 0.2500 |
| Llama-3.1-8B-Instruct | ruler32k | 8x | perhead_bern | 0.1250 |
| Llama-3.1-8B-Instruct | ruler32k | 16x | perhead_bern | 0.0625 |
| Llama-3.1-8B-Instruct | ruler32k | 8x | vkvwr | 0.1994 |
| Llama-3.1-8B-Instruct | ruler32k | 16x | vkvwr | 0.1523 |
| Llama-3.1-8B-Instruct | ruler32k | 32x | vkvwr | 0.1247 |
| Llama-3.1-8B-Instruct | ruler32k | 64x | vkvwr | 0.1049 |
| Llama-3.1-8B-Instruct | ruler32k | 200x | vkvwr | 0.0706 |
| Qwen3-4B-Instruct-2507 | ruler32k | 4x | bernoulli | 0.2500 |
| Qwen3-4B-Instruct-2507 | ruler32k | 8x | bernoulli | 0.1250 |
| Qwen3-4B-Instruct-2507 | ruler32k | 16x | bernoulli | 0.0625 |
| Qwen3-4B-Instruct-2507 | ruler32k | 4x | kvzip | 0.2500 |
| Qwen3-4B-Instruct-2507 | ruler32k | 8x | kvzip | 0.1250 |
| Qwen3-4B-Instruct-2507 | ruler32k | 16x | kvzip | 0.0625 |
| Qwen3-4B-Instruct-2507 | ruler32k | 4x | perhead_bern | 0.2500 |
| Qwen3-4B-Instruct-2507 | ruler32k | 8x | perhead_bern | 0.1250 |
| Qwen3-4B-Instruct-2507 | ruler32k | 16x | perhead_bern | 0.0625 |
| Qwen3-4B-Instruct-2507 | ruler32k | 8x | vkvwr | 0.2355 |
| Qwen3-4B-Instruct-2507 | ruler32k | 16x | vkvwr | 0.1970 |
| Qwen3-4B-Instruct-2507 | ruler32k | 32x | vkvwr | 0.1730 |
| Qwen3-4B-Instruct-2507 | ruler32k | 64x | vkvwr | 0.1484 |
| Qwen3-4B-Instruct-2507 | ruler32k | 200x | vkvwr | 0.1001 |
| Llama-3.1-8B-Instruct | scbench | 8x | bernoulli | 0.1250 |
| Llama-3.1-8B-Instruct | scbench | 8x | kvzip | 0.1250 |
| Llama-3.1-8B-Instruct | scbench | 8x | perhead_bern | 0.1250 |
| Llama-3.1-8B-Instruct | scbench | 8x | vkvwr | 0.2206 |
| Llama-3.1-8B-Instruct | scbench | 16x | vkvwr | 0.1646 |
| Llama-3.1-8B-Instruct | scbench | 32x | vkvwr | 0.1232 |
| Qwen3-4B-Instruct-2507 | scbench | 8x | bernoulli | 0.1250 |
| Qwen3-4B-Instruct-2507 | scbench | 8x | kvzip | 0.1250 |
| Qwen3-4B-Instruct-2507 | scbench | 8x | perhead_bern | 0.1250 |
| Qwen3-4B-Instruct-2507 | scbench | 8x | vkvwr | 0.2067 |
| Qwen3-4B-Instruct-2507 | scbench | 16x | vkvwr | 0.1561 |
| Qwen3-4B-Instruct-2507 | scbench | 32x | vkvwr | 0.1224 |

## denominator |B_hat-B|/B: median [p25, p75] over heads

| model | dataset | task | ratio | arm | all layers | layers 0-12 | layers 13+ |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 4x | bernoulli | 0.0105 [0.0038, 0.0241] | 0.0130 [0.0048, 0.0293] | 0.0090 [0.0034, 0.0208] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 8x | bernoulli | 0.0420 [0.0176, 0.0897] | 0.0390 [0.0145, 0.0897] | 0.0438 [0.0198, 0.0897] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 16x | bernoulli | 0.0661 [0.0275, 0.1597] | 0.0652 [0.0255, 0.1533] | 0.0667 [0.0287, 0.1647] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 4x | kvzip | 0.0849 [0.0343, 0.1964] | 0.1967 [0.0910, 0.3715] | 0.0528 [0.0260, 0.1021] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 8x | kvzip | 0.2154 [0.1099, 0.3923] | 0.3494 [0.1839, 0.5913] | 0.1607 [0.0877, 0.2761] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 16x | kvzip | 0.3618 [0.1876, 0.6220] | 0.5307 [0.2816, 0.7692] | 0.2860 [0.1554, 0.4867] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 4x | perhead_bern | 0.0045 [0.0010, 0.0275] | 0.0023 [0.0007, 0.0109] | 0.0079 [0.0014, 0.0449] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 8x | perhead_bern | 0.0071 [0.0017, 0.0411] | 0.0044 [0.0014, 0.0170] | 0.0115 [0.0021, 0.0675] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 16x | perhead_bern | 0.0103 [0.0026, 0.0473] | 0.0069 [0.0023, 0.0234] | 0.0156 [0.0029, 0.0722] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 8x | vkvwr | 0.0064 [0.0018, 0.0266] | 0.0047 [0.0015, 0.0153] | 0.0085 [0.0020, 0.0370] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 16x | vkvwr | 0.0089 [0.0025, 0.0336] | 0.0071 [0.0025, 0.0211] | 0.0111 [0.0025, 0.0445] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 32x | vkvwr | 0.0100 [0.0028, 0.0390] | 0.0083 [0.0028, 0.0259] | 0.0122 [0.0028, 0.0506] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 64x | vkvwr | 0.0114 [0.0031, 0.0453] | 0.0088 [0.0029, 0.0277] | 0.0148 [0.0033, 0.0612] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 200x | vkvwr | 0.0137 [0.0036, 0.0543] | 0.0102 [0.0034, 0.0318] | 0.0185 [0.0038, 0.0766] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 4x | bernoulli | 0.0063 [0.0022, 0.0158] | 0.0078 [0.0027, 0.0187] | 0.0055 [0.0019, 0.0139] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 8x | bernoulli | 0.0180 [0.0062, 0.0457] | 0.0235 [0.0083, 0.0589] | 0.0151 [0.0053, 0.0376] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 16x | bernoulli | 0.0354 [0.0125, 0.0909] | 0.0454 [0.0160, 0.1158] | 0.0300 [0.0107, 0.0760] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 4x | kvzip | 0.0969 [0.0405, 0.2194] | 0.1778 [0.0740, 0.3546] | 0.0691 [0.0288, 0.1375] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 8x | kvzip | 0.2202 [0.1000, 0.4701] | 0.3833 [0.1658, 0.6863] | 0.1639 [0.0747, 0.3224] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 16x | kvzip | 0.3588 [0.1653, 0.6546] | 0.5035 [0.2229, 0.8199] | 0.2956 [0.1416, 0.5392] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 4x | perhead_bern | 0.0037 [0.0008, 0.0178] | 0.0021 [0.0006, 0.0093] | 0.0057 [0.0012, 0.0261] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 8x | perhead_bern | 0.0061 [0.0015, 0.0272] | 0.0042 [0.0012, 0.0158] | 0.0085 [0.0018, 0.0378] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 16x | perhead_bern | 0.0086 [0.0023, 0.0346] | 0.0064 [0.0020, 0.0219] | 0.0112 [0.0027, 0.0464] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 8x | vkvwr | 0.0056 [0.0015, 0.0207] | 0.0043 [0.0013, 0.0144] | 0.0069 [0.0018, 0.0263] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 16x | vkvwr | 0.0073 [0.0021, 0.0249] | 0.0066 [0.0020, 0.0199] | 0.0079 [0.0021, 0.0296] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 32x | vkvwr | 0.0089 [0.0024, 0.0304] | 0.0083 [0.0024, 0.0259] | 0.0093 [0.0025, 0.0346] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 64x | vkvwr | 0.0096 [0.0026, 0.0323] | 0.0089 [0.0025, 0.0271] | 0.0101 [0.0026, 0.0371] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 200x | vkvwr | 0.0106 [0.0028, 0.0378] | 0.0094 [0.0027, 0.0298] | 0.0118 [0.0029, 0.0448] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 4x | bernoulli | 0.0021 [0.0008, 0.0052] | 0.0021 [0.0007, 0.0047] | 0.0022 [0.0008, 0.0056] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 8x | bernoulli | 0.0038 [0.0014, 0.0092] | 0.0041 [0.0015, 0.0100] | 0.0037 [0.0014, 0.0088] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 16x | bernoulli | 0.0064 [0.0023, 0.0157] | 0.0073 [0.0026, 0.0175] | 0.0058 [0.0021, 0.0145] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 4x | kvzip | 0.0232 [0.0096, 0.0522] | 0.0403 [0.0168, 0.0793] | 0.0168 [0.0077, 0.0346] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 8x | kvzip | 0.0466 [0.0203, 0.1013] | 0.0794 [0.0352, 0.1603] | 0.0338 [0.0159, 0.0684] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 16x | kvzip | 0.0712 [0.0313, 0.1534] | 0.1084 [0.0492, 0.2310] | 0.0541 [0.0251, 0.1127] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 4x | perhead_bern | 0.0017 [0.0005, 0.0050] | 0.0012 [0.0004, 0.0035] | 0.0022 [0.0007, 0.0061] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 8x | perhead_bern | 0.0027 [0.0009, 0.0076] | 0.0022 [0.0007, 0.0060] | 0.0031 [0.0010, 0.0087] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 16x | perhead_bern | 0.0042 [0.0014, 0.0124] | 0.0039 [0.0013, 0.0107] | 0.0045 [0.0014, 0.0136] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 8x | vkvwr | 0.0025 [0.0009, 0.0068] | 0.0022 [0.0007, 0.0058] | 0.0028 [0.0010, 0.0074] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 16x | vkvwr | 0.0034 [0.0012, 0.0093] | 0.0034 [0.0011, 0.0092] | 0.0035 [0.0012, 0.0094] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 32x | vkvwr | 0.0041 [0.0013, 0.0115] | 0.0042 [0.0014, 0.0120] | 0.0040 [0.0013, 0.0111] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 64x | vkvwr | 0.0048 [0.0015, 0.0140] | 0.0050 [0.0016, 0.0145] | 0.0047 [0.0015, 0.0138] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 200x | vkvwr | 0.0062 [0.0018, 0.0200] | 0.0063 [0.0019, 0.0205] | 0.0061 [0.0018, 0.0198] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 4x | bernoulli | 0.0064 [0.0024, 0.0158] | 0.0063 [0.0023, 0.0158] | 0.0065 [0.0024, 0.0158] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 8x | bernoulli | 0.0326 [0.0128, 0.0691] | 0.0260 [0.0090, 0.0652] | 0.0366 [0.0160, 0.0713] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 16x | bernoulli | 0.0656 [0.0263, 0.1460] | 0.0518 [0.0183, 0.1337] | 0.0731 [0.0335, 0.1547] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 4x | kvzip | 0.0912 [0.0375, 0.2156] | 0.1212 [0.0477, 0.2784] | 0.0762 [0.0324, 0.1771] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 8x | kvzip | 0.2500 [0.1313, 0.4924] | 0.2610 [0.1158, 0.5538] | 0.2458 [0.1396, 0.4548] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 16x | kvzip | 0.3886 [0.1841, 0.6530] | 0.3737 [0.1592, 0.6877] | 0.3958 [0.1988, 0.6346] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 4x | perhead_bern | 0.0037 [0.0007, 0.0190] | 0.0014 [0.0003, 0.0075] | 0.0069 [0.0016, 0.0291] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 8x | perhead_bern | 0.0063 [0.0012, 0.0328] | 0.0023 [0.0005, 0.0127] | 0.0121 [0.0026, 0.0498] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 16x | perhead_bern | 0.0094 [0.0019, 0.0467] | 0.0036 [0.0009, 0.0177] | 0.0190 [0.0039, 0.0699] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 8x | vkvwr | 0.0057 [0.0014, 0.0243] | 0.0028 [0.0008, 0.0112] | 0.0095 [0.0024, 0.0339] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 16x | vkvwr | 0.0081 [0.0021, 0.0320] | 0.0043 [0.0012, 0.0161] | 0.0130 [0.0032, 0.0446] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 32x | vkvwr | 0.0099 [0.0025, 0.0369] | 0.0052 [0.0013, 0.0198] | 0.0156 [0.0039, 0.0492] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 64x | vkvwr | 0.0104 [0.0027, 0.0399] | 0.0051 [0.0014, 0.0202] | 0.0168 [0.0045, 0.0546] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 200x | vkvwr | 0.0123 [0.0028, 0.0491] | 0.0057 [0.0015, 0.0228] | 0.0212 [0.0049, 0.0680] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 4x | bernoulli | 0.0163 [0.0047, 0.0435] | 0.0221 [0.0067, 0.0572] | 0.0136 [0.0039, 0.0367] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 8x | bernoulli | 0.0838 [0.0292, 0.1848] | 0.0794 [0.0227, 0.1932] | 0.0860 [0.0331, 0.1811] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 16x | bernoulli | 0.1579 [0.0559, 0.3300] | 0.1252 [0.0370, 0.3058] | 0.1753 [0.0696, 0.3407] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 4x | kvzip | 0.1032 [0.0304, 0.2848] | 0.2872 [0.1060, 0.6249] | 0.0613 [0.0198, 0.1545] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 8x | kvzip | 0.3374 [0.1281, 0.6241] | 0.6102 [0.2982, 0.9313] | 0.2417 [0.0944, 0.4559] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 16x | kvzip | 0.6440 [0.3962, 0.9092] | 0.9283 [0.5189, 0.9837] | 0.5543 [0.3697, 0.7832] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 4x | perhead_bern | 0.0053 [0.0008, 0.0358] | 0.0039 [0.0009, 0.0144] | 0.0073 [0.0007, 0.0710] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 8x | perhead_bern | 0.0116 [0.0021, 0.0756] | 0.0082 [0.0022, 0.0271] | 0.0167 [0.0020, 0.1469] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 16x | perhead_bern | 0.0201 [0.0037, 0.1186] | 0.0146 [0.0040, 0.0461] | 0.0287 [0.0034, 0.2191] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 8x | vkvwr | 0.0087 [0.0017, 0.0382] | 0.0084 [0.0023, 0.0262] | 0.0090 [0.0014, 0.0504] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 16x | vkvwr | 0.0112 [0.0022, 0.0467] | 0.0108 [0.0030, 0.0336] | 0.0115 [0.0018, 0.0585] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 32x | vkvwr | 0.0143 [0.0028, 0.0603] | 0.0131 [0.0038, 0.0395] | 0.0155 [0.0023, 0.0790] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 64x | vkvwr | 0.0159 [0.0032, 0.0684] | 0.0135 [0.0040, 0.0413] | 0.0184 [0.0027, 0.0971] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 200x | vkvwr | 0.0214 [0.0044, 0.0938] | 0.0172 [0.0053, 0.0501] | 0.0266 [0.0039, 0.1406] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 4x | bernoulli | 0.0066 [0.0015, 0.0214] | 0.0130 [0.0034, 0.0350] | 0.0044 [0.0011, 0.0148] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 8x | bernoulli | 0.0356 [0.0121, 0.0878] | 0.0530 [0.0151, 0.1214] | 0.0297 [0.0110, 0.0696] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 16x | bernoulli | 0.1286 [0.0457, 0.2628] | 0.1220 [0.0326, 0.2850] | 0.1313 [0.0531, 0.2519] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 4x | kvzip | 0.1093 [0.0210, 0.3407] | 0.3020 [0.0933, 0.5499] | 0.0569 [0.0120, 0.1974] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 8x | kvzip | 0.3120 [0.0834, 0.6863] | 0.6510 [0.2776, 0.9156] | 0.1845 [0.0526, 0.4598] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 16x | kvzip | 0.5885 [0.3163, 0.8608] | 0.7987 [0.4488, 0.9745] | 0.4891 [0.2777, 0.7549] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 4x | perhead_bern | 0.0043 [0.0006, 0.0227] | 0.0041 [0.0008, 0.0136] | 0.0045 [0.0005, 0.0339] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 8x | perhead_bern | 0.0090 [0.0015, 0.0451] | 0.0088 [0.0021, 0.0278] | 0.0093 [0.0012, 0.0701] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 16x | perhead_bern | 0.0158 [0.0027, 0.0780] | 0.0150 [0.0035, 0.0454] | 0.0166 [0.0024, 0.1290] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 8x | vkvwr | 0.0088 [0.0016, 0.0340] | 0.0104 [0.0025, 0.0309] | 0.0077 [0.0011, 0.0364] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 16x | vkvwr | 0.0127 [0.0023, 0.0467] | 0.0154 [0.0037, 0.0450] | 0.0110 [0.0017, 0.0482] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 32x | vkvwr | 0.0145 [0.0027, 0.0529] | 0.0169 [0.0041, 0.0495] | 0.0130 [0.0021, 0.0556] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 64x | vkvwr | 0.0166 [0.0031, 0.0588] | 0.0181 [0.0044, 0.0525] | 0.0155 [0.0025, 0.0644] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 200x | vkvwr | 0.0199 [0.0037, 0.0733] | 0.0205 [0.0049, 0.0578] | 0.0195 [0.0031, 0.0869] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 4x | bernoulli | 0.0024 [0.0006, 0.0076] | 0.0038 [0.0008, 0.0118] | 0.0019 [0.0005, 0.0058] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 8x | bernoulli | 0.0058 [0.0015, 0.0179] | 0.0094 [0.0023, 0.0271] | 0.0046 [0.0013, 0.0137] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 16x | bernoulli | 0.0115 [0.0031, 0.0330] | 0.0179 [0.0047, 0.0494] | 0.0091 [0.0026, 0.0256] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 4x | kvzip | 0.0335 [0.0076, 0.1133] | 0.0803 [0.0179, 0.1997] | 0.0215 [0.0058, 0.0698] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 8x | kvzip | 0.0674 [0.0169, 0.2106] | 0.1637 [0.0375, 0.4007] | 0.0445 [0.0127, 0.1313] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 16x | kvzip | 0.1016 [0.0271, 0.2888] | 0.2253 [0.0588, 0.5687] | 0.0698 [0.0201, 0.1889] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 4x | perhead_bern | 0.0015 [0.0003, 0.0060] | 0.0025 [0.0004, 0.0103] | 0.0012 [0.0003, 0.0045] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 8x | perhead_bern | 0.0037 [0.0008, 0.0129] | 0.0056 [0.0011, 0.0208] | 0.0030 [0.0006, 0.0099] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 16x | perhead_bern | 0.0076 [0.0017, 0.0254] | 0.0109 [0.0025, 0.0363] | 0.0063 [0.0013, 0.0206] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 8x | vkvwr | 0.0028 [0.0006, 0.0106] | 0.0053 [0.0010, 0.0202] | 0.0021 [0.0004, 0.0074] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 16x | vkvwr | 0.0041 [0.0008, 0.0152] | 0.0075 [0.0015, 0.0284] | 0.0031 [0.0006, 0.0108] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 32x | vkvwr | 0.0053 [0.0011, 0.0196] | 0.0089 [0.0020, 0.0308] | 0.0040 [0.0008, 0.0145] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 64x | vkvwr | 0.0071 [0.0015, 0.0244] | 0.0102 [0.0024, 0.0338] | 0.0057 [0.0012, 0.0201] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 200x | vkvwr | 0.0121 [0.0028, 0.0382] | 0.0163 [0.0042, 0.0463] | 0.0101 [0.0022, 0.0335] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 4x | bernoulli | 0.0089 [0.0023, 0.0259] | 0.0107 [0.0026, 0.0311] | 0.0081 [0.0021, 0.0231] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 8x | bernoulli | 0.0438 [0.0136, 0.1076] | 0.0496 [0.0127, 0.1223] | 0.0413 [0.0139, 0.0993] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 16x | bernoulli | 0.1167 [0.0408, 0.2550] | 0.1115 [0.0283, 0.3003] | 0.1186 [0.0482, 0.2367] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 4x | kvzip | 0.0940 [0.0260, 0.2731] | 0.2351 [0.0638, 0.5123] | 0.0630 [0.0192, 0.1621] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 8x | kvzip | 0.2798 [0.0922, 0.5785] | 0.4472 [0.1499, 0.8163] | 0.2263 [0.0757, 0.4728] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 16x | kvzip | 0.5169 [0.2442, 0.7924] | 0.6509 [0.2760, 0.9155] | 0.4593 [0.2330, 0.7233] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 4x | perhead_bern | 0.0034 [0.0004, 0.0225] | 0.0026 [0.0004, 0.0102] | 0.0044 [0.0004, 0.0400] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 8x | perhead_bern | 0.0079 [0.0011, 0.0454] | 0.0057 [0.0012, 0.0212] | 0.0104 [0.0011, 0.0786] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 16x | perhead_bern | 0.0146 [0.0022, 0.0767] | 0.0102 [0.0022, 0.0350] | 0.0201 [0.0022, 0.1249] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 8x | vkvwr | 0.0065 [0.0011, 0.0282] | 0.0059 [0.0013, 0.0222] | 0.0070 [0.0010, 0.0335] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 16x | vkvwr | 0.0096 [0.0017, 0.0385] | 0.0090 [0.0019, 0.0312] | 0.0101 [0.0016, 0.0436] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 32x | vkvwr | 0.0118 [0.0021, 0.0441] | 0.0105 [0.0023, 0.0367] | 0.0125 [0.0020, 0.0493] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 64x | vkvwr | 0.0138 [0.0026, 0.0519] | 0.0116 [0.0026, 0.0388] | 0.0155 [0.0025, 0.0622] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 200x | vkvwr | 0.0177 [0.0033, 0.0684] | 0.0139 [0.0032, 0.0446] | 0.0210 [0.0034, 0.0891] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 8x | bernoulli | 0.0378 [0.0142, 0.0802] | 0.0287 [0.0099, 0.0678] | 0.0450 [0.0182, 0.0876] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 8x | kvzip | 0.3343 [0.1708, 0.7250] | 0.8075 [0.4641, 0.9703] | 0.2239 [0.1319, 0.3509] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 8x | perhead_bern | 0.0059 [0.0012, 0.0296] | 0.0024 [0.0006, 0.0110] | 0.0110 [0.0022, 0.0598] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 8x | vkvwr | 0.0054 [0.0012, 0.0216] | 0.0023 [0.0007, 0.0087] | 0.0095 [0.0022, 0.0341] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 16x | vkvwr | 0.0064 [0.0016, 0.0260] | 0.0033 [0.0010, 0.0125] | 0.0103 [0.0026, 0.0403] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 32x | vkvwr | 0.0075 [0.0020, 0.0310] | 0.0045 [0.0014, 0.0160] | 0.0110 [0.0028, 0.0466] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 8x | bernoulli | 0.0176 [0.0044, 0.0571] | 0.0300 [0.0052, 0.0958] | 0.0141 [0.0042, 0.0426] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 8x | kvzip | 0.1383 [0.0314, 0.3960] | 0.3269 [0.0895, 0.7989] | 0.0838 [0.0212, 0.2572] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 8x | perhead_bern | 0.0017 [0.0002, 0.0138] | 0.0016 [0.0002, 0.0080] | 0.0017 [0.0002, 0.0249] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 8x | vkvwr | 0.0016 [0.0002, 0.0097] | 0.0019 [0.0003, 0.0089] | 0.0015 [0.0002, 0.0105] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 16x | vkvwr | 0.0028 [0.0004, 0.0153] | 0.0034 [0.0006, 0.0144] | 0.0025 [0.0004, 0.0160] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 32x | vkvwr | 0.0041 [0.0006, 0.0226] | 0.0052 [0.0009, 0.0223] | 0.0036 [0.0005, 0.0229] |

## attention ||A/B_hat - A/B||/||A/B||: median [p25, p75] over heads

| model | dataset | task | ratio | arm | all layers | layers 0-12 | layers 13+ |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 4x | bernoulli | 0.0599 [0.0233, 0.1481] | 0.0644 [0.0242, 0.1847] | 0.0570 [0.0228, 0.1325] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 8x | bernoulli | 0.1841 [0.0796, 0.3976] | 0.1689 [0.0720, 0.4063] | 0.1943 [0.0853, 0.3935] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 16x | bernoulli | 0.3033 [0.1262, 0.6434] | 0.2563 [0.1128, 0.5901] | 0.3401 [0.1393, 0.6671] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 4x | kvzip | 0.1522 [0.0525, 0.3901] | 0.3036 [0.0888, 0.7465] | 0.1094 [0.0432, 0.2357] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 8x | kvzip | 0.3616 [0.1617, 0.7069] | 0.5593 [0.2276, 0.9989] | 0.2870 [0.1401, 0.5257] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 16x | kvzip | 0.6311 [0.3524, 0.9408] | 0.8351 [0.4673, 1.1600] | 0.5405 [0.3003, 0.7938] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 4x | perhead_bern | 0.0310 [0.0077, 0.1293] | 0.0172 [0.0052, 0.0559] | 0.0505 [0.0112, 0.2083] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 8x | perhead_bern | 0.0487 [0.0138, 0.1821] | 0.0322 [0.0111, 0.0900] | 0.0708 [0.0175, 0.2965] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 16x | perhead_bern | 0.0614 [0.0203, 0.2041] | 0.0472 [0.0176, 0.1216] | 0.0780 [0.0232, 0.3073] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 8x | vkvwr | 0.0440 [0.0141, 0.1220] | 0.0333 [0.0121, 0.0844] | 0.0539 [0.0163, 0.1534] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 16x | vkvwr | 0.0569 [0.0190, 0.1542] | 0.0458 [0.0173, 0.1169] | 0.0669 [0.0206, 0.1883] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 32x | vkvwr | 0.0667 [0.0223, 0.1851] | 0.0544 [0.0203, 0.1438] | 0.0780 [0.0242, 0.2188] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 64x | vkvwr | 0.0736 [0.0242, 0.2107] | 0.0569 [0.0212, 0.1521] | 0.0904 [0.0271, 0.2631] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 200x | vkvwr | 0.0876 [0.0287, 0.2676] | 0.0666 [0.0252, 0.1811] | 0.1108 [0.0327, 0.3482] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 4x | bernoulli | 0.0354 [0.0149, 0.0821] | 0.0367 [0.0169, 0.0804] | 0.0345 [0.0134, 0.0833] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 8x | bernoulli | 0.1036 [0.0455, 0.2239] | 0.1065 [0.0503, 0.2284] | 0.1013 [0.0419, 0.2210] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 16x | bernoulli | 0.1914 [0.0836, 0.4039] | 0.1910 [0.0877, 0.4025] | 0.1916 [0.0809, 0.4051] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 4x | kvzip | 0.1838 [0.0747, 0.3851] | 0.2608 [0.1193, 0.5040] | 0.1418 [0.0560, 0.3082] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 8x | kvzip | 0.4204 [0.1968, 0.7243] | 0.5865 [0.3024, 0.8630] | 0.3321 [0.1523, 0.5972] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 16x | kvzip | 0.6460 [0.3508, 0.9139] | 0.7831 [0.4505, 0.9881] | 0.5595 [0.2969, 0.8341] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 4x | perhead_bern | 0.0237 [0.0065, 0.0803] | 0.0124 [0.0041, 0.0358] | 0.0410 [0.0102, 0.1262] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 8x | perhead_bern | 0.0388 [0.0120, 0.1227] | 0.0238 [0.0089, 0.0612] | 0.0627 [0.0163, 0.1842] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 16x | perhead_bern | 0.0532 [0.0182, 0.1592] | 0.0346 [0.0140, 0.0878] | 0.0780 [0.0238, 0.2335] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 8x | vkvwr | 0.0378 [0.0129, 0.0939] | 0.0253 [0.0101, 0.0584] | 0.0522 [0.0163, 0.1267] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 16x | vkvwr | 0.0483 [0.0184, 0.1139] | 0.0374 [0.0161, 0.0787] | 0.0604 [0.0208, 0.1465] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 32x | vkvwr | 0.0575 [0.0220, 0.1333] | 0.0451 [0.0198, 0.0937] | 0.0711 [0.0244, 0.1691] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 64x | vkvwr | 0.0616 [0.0233, 0.1461] | 0.0479 [0.0205, 0.0987] | 0.0777 [0.0263, 0.1892] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 200x | vkvwr | 0.0704 [0.0264, 0.1700] | 0.0533 [0.0231, 0.1119] | 0.0907 [0.0298, 0.2244] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 4x | bernoulli | 0.0100 [0.0038, 0.0266] | 0.0088 [0.0033, 0.0219] | 0.0110 [0.0041, 0.0303] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 8x | bernoulli | 0.0204 [0.0082, 0.0479] | 0.0189 [0.0073, 0.0447] | 0.0214 [0.0089, 0.0503] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 16x | bernoulli | 0.0367 [0.0154, 0.0833] | 0.0333 [0.0134, 0.0760] | 0.0390 [0.0168, 0.0893] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 4x | kvzip | 0.0503 [0.0171, 0.1352] | 0.0697 [0.0235, 0.1823] | 0.0421 [0.0143, 0.1047] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 8x | kvzip | 0.0992 [0.0401, 0.2354] | 0.1434 [0.0572, 0.3393] | 0.0809 [0.0331, 0.1783] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 16x | kvzip | 0.1530 [0.0667, 0.3284] | 0.2045 [0.0895, 0.4354] | 0.1296 [0.0568, 0.2639] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 4x | perhead_bern | 0.0080 [0.0028, 0.0212] | 0.0055 [0.0020, 0.0133] | 0.0106 [0.0038, 0.0292] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 8x | perhead_bern | 0.0140 [0.0057, 0.0324] | 0.0104 [0.0043, 0.0237] | 0.0169 [0.0070, 0.0398] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 16x | perhead_bern | 0.0236 [0.0100, 0.0517] | 0.0180 [0.0078, 0.0392] | 0.0286 [0.0121, 0.0608] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 8x | vkvwr | 0.0129 [0.0053, 0.0298] | 0.0107 [0.0044, 0.0245] | 0.0147 [0.0062, 0.0341] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 16x | vkvwr | 0.0183 [0.0079, 0.0397] | 0.0160 [0.0070, 0.0354] | 0.0200 [0.0087, 0.0427] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 32x | vkvwr | 0.0238 [0.0101, 0.0496] | 0.0212 [0.0092, 0.0455] | 0.0256 [0.0109, 0.0523] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 64x | vkvwr | 0.0288 [0.0122, 0.0593] | 0.0252 [0.0109, 0.0521] | 0.0315 [0.0134, 0.0635] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 200x | vkvwr | 0.0407 [0.0163, 0.0842] | 0.0336 [0.0138, 0.0707] | 0.0467 [0.0187, 0.0938] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 4x | bernoulli | 0.0346 [0.0138, 0.0812] | 0.0315 [0.0131, 0.0713] | 0.0374 [0.0144, 0.0888] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 8x | bernoulli | 0.1326 [0.0632, 0.2649] | 0.1163 [0.0531, 0.2394] | 0.1440 [0.0708, 0.2838] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 16x | bernoulli | 0.2432 [0.1150, 0.4760] | 0.2028 [0.0932, 0.4168] | 0.2721 [0.1345, 0.5129] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 4x | kvzip | 0.1804 [0.0619, 0.3997] | 0.2234 [0.0801, 0.4539] | 0.1516 [0.0512, 0.3610] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 8x | kvzip | 0.4415 [0.2032, 0.7646] | 0.4928 [0.2193, 0.8307] | 0.4112 [0.1938, 0.7131] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 16x | kvzip | 0.5821 [0.3202, 0.8924] | 0.6453 [0.3390, 0.9580] | 0.5480 [0.3090, 0.8384] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 4x | perhead_bern | 0.0212 [0.0051, 0.0822] | 0.0073 [0.0022, 0.0295] | 0.0415 [0.0114, 0.1203] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 8x | perhead_bern | 0.0347 [0.0089, 0.1286] | 0.0126 [0.0042, 0.0465] | 0.0655 [0.0189, 0.1966] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 16x | perhead_bern | 0.0512 [0.0144, 0.1800] | 0.0204 [0.0070, 0.0701] | 0.0898 [0.0291, 0.2753] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 8x | vkvwr | 0.0354 [0.0107, 0.0978] | 0.0163 [0.0057, 0.0484] | 0.0572 [0.0197, 0.1311] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 16x | vkvwr | 0.0481 [0.0159, 0.1241] | 0.0241 [0.0089, 0.0676] | 0.0730 [0.0277, 0.1635] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 32x | vkvwr | 0.0573 [0.0187, 0.1482] | 0.0285 [0.0108, 0.0834] | 0.0869 [0.0321, 0.1944] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 64x | vkvwr | 0.0616 [0.0192, 0.1653] | 0.0297 [0.0111, 0.0900] | 0.0948 [0.0335, 0.2153] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 200x | vkvwr | 0.0676 [0.0209, 0.1922] | 0.0317 [0.0119, 0.0982] | 0.1071 [0.0367, 0.2615] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 4x | bernoulli | 0.0508 [0.0197, 0.1236] | 0.0606 [0.0231, 0.1406] | 0.0463 [0.0182, 0.1133] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 8x | bernoulli | 0.1537 [0.0645, 0.3389] | 0.1489 [0.0615, 0.3163] | 0.1564 [0.0665, 0.3533] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 16x | bernoulli | 0.2453 [0.1048, 0.5144] | 0.2173 [0.0891, 0.4611] | 0.2620 [0.1148, 0.5450] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 4x | kvzip | 0.1106 [0.0406, 0.2751] | 0.2295 [0.0971, 0.5535] | 0.0746 [0.0301, 0.1690] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 8x | kvzip | 0.3096 [0.1367, 0.6351] | 0.5172 [0.2337, 1.0477] | 0.2397 [0.1103, 0.4655] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 16x | kvzip | 0.6489 [0.3547, 0.9645] | 0.8361 [0.4213, 1.3147] | 0.5829 [0.3305, 0.8518] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 4x | perhead_bern | 0.0165 [0.0038, 0.0947] | 0.0118 [0.0039, 0.0361] | 0.0235 [0.0037, 0.1674] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 8x | perhead_bern | 0.0355 [0.0092, 0.1717] | 0.0256 [0.0096, 0.0715] | 0.0494 [0.0090, 0.2990] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 16x | perhead_bern | 0.0542 [0.0151, 0.2150] | 0.0420 [0.0164, 0.1105] | 0.0709 [0.0142, 0.3324] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 8x | vkvwr | 0.0278 [0.0080, 0.0958] | 0.0265 [0.0099, 0.0667] | 0.0289 [0.0068, 0.1288] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 16x | vkvwr | 0.0344 [0.0101, 0.1145] | 0.0341 [0.0128, 0.0862] | 0.0346 [0.0085, 0.1459] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 32x | vkvwr | 0.0429 [0.0124, 0.1448] | 0.0416 [0.0150, 0.1056] | 0.0441 [0.0109, 0.1874] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 64x | vkvwr | 0.0468 [0.0136, 0.1582] | 0.0424 [0.0159, 0.1078] | 0.0506 [0.0121, 0.2160] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 200x | vkvwr | 0.0568 [0.0171, 0.1994] | 0.0480 [0.0187, 0.1213] | 0.0665 [0.0160, 0.2862] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 4x | bernoulli | 0.0219 [0.0081, 0.0506] | 0.0305 [0.0105, 0.0633] | 0.0185 [0.0072, 0.0431] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 8x | bernoulli | 0.0750 [0.0321, 0.1593] | 0.0928 [0.0358, 0.1900] | 0.0676 [0.0307, 0.1408] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 16x | bernoulli | 0.1686 [0.0779, 0.3255] | 0.1717 [0.0697, 0.3440] | 0.1672 [0.0820, 0.3156] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 4x | kvzip | 0.1041 [0.0346, 0.2434] | 0.1814 [0.0657, 0.3552] | 0.0764 [0.0261, 0.1806] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 8x | kvzip | 0.2905 [0.1170, 0.5981] | 0.4755 [0.2048, 0.8168] | 0.2250 [0.0915, 0.4589] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 16x | kvzip | 0.5481 [0.2948, 0.8588] | 0.6723 [0.3129, 0.9941] | 0.5034 [0.2876, 0.7775] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 4x | perhead_bern | 0.0110 [0.0024, 0.0532] | 0.0086 [0.0027, 0.0240] | 0.0144 [0.0023, 0.0907] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 8x | perhead_bern | 0.0216 [0.0055, 0.0882] | 0.0180 [0.0062, 0.0443] | 0.0267 [0.0050, 0.1370] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 16x | perhead_bern | 0.0346 [0.0092, 0.1276] | 0.0288 [0.0101, 0.0691] | 0.0419 [0.0086, 0.1983] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 8x | vkvwr | 0.0239 [0.0066, 0.0689] | 0.0236 [0.0082, 0.0535] | 0.0241 [0.0058, 0.0829] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 16x | vkvwr | 0.0332 [0.0095, 0.0888] | 0.0350 [0.0119, 0.0775] | 0.0319 [0.0083, 0.0994] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 32x | vkvwr | 0.0356 [0.0104, 0.0968] | 0.0375 [0.0126, 0.0857] | 0.0341 [0.0093, 0.1063] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 64x | vkvwr | 0.0370 [0.0111, 0.1033] | 0.0382 [0.0131, 0.0892] | 0.0362 [0.0100, 0.1164] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 200x | vkvwr | 0.0426 [0.0124, 0.1235] | 0.0413 [0.0143, 0.0947] | 0.0437 [0.0114, 0.1484] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 4x | bernoulli | 0.0080 [0.0029, 0.0186] | 0.0089 [0.0028, 0.0205] | 0.0076 [0.0029, 0.0176] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 8x | bernoulli | 0.0195 [0.0076, 0.0434] | 0.0214 [0.0079, 0.0461] | 0.0185 [0.0075, 0.0416] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 16x | bernoulli | 0.0367 [0.0149, 0.0792] | 0.0398 [0.0157, 0.0811] | 0.0351 [0.0145, 0.0777] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 4x | kvzip | 0.0383 [0.0115, 0.1010] | 0.0618 [0.0180, 0.1501] | 0.0298 [0.0100, 0.0752] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 8x | kvzip | 0.0724 [0.0256, 0.1746] | 0.1198 [0.0387, 0.2663] | 0.0568 [0.0222, 0.1309] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 16x | kvzip | 0.1060 [0.0408, 0.2458] | 0.1715 [0.0610, 0.3848] | 0.0842 [0.0353, 0.1843] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 4x | perhead_bern | 0.0048 [0.0015, 0.0136] | 0.0052 [0.0014, 0.0163] | 0.0046 [0.0015, 0.0124] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 8x | perhead_bern | 0.0112 [0.0039, 0.0291] | 0.0117 [0.0038, 0.0325] | 0.0110 [0.0039, 0.0275] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 16x | perhead_bern | 0.0228 [0.0082, 0.0562] | 0.0227 [0.0080, 0.0581] | 0.0229 [0.0083, 0.0554] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 8x | vkvwr | 0.0092 [0.0032, 0.0226] | 0.0116 [0.0033, 0.0336] | 0.0083 [0.0031, 0.0186] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 16x | vkvwr | 0.0133 [0.0046, 0.0321] | 0.0166 [0.0050, 0.0462] | 0.0122 [0.0044, 0.0270] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 32x | vkvwr | 0.0171 [0.0061, 0.0403] | 0.0197 [0.0063, 0.0536] | 0.0160 [0.0059, 0.0354] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 64x | vkvwr | 0.0224 [0.0082, 0.0503] | 0.0229 [0.0078, 0.0576] | 0.0222 [0.0083, 0.0473] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 200x | vkvwr | 0.0368 [0.0136, 0.0770] | 0.0345 [0.0129, 0.0758] | 0.0380 [0.0141, 0.0776] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 4x | bernoulli | 0.0279 [0.0103, 0.0646] | 0.0205 [0.0076, 0.0474] | 0.0332 [0.0125, 0.0758] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 8x | bernoulli | 0.0985 [0.0391, 0.2135] | 0.0699 [0.0277, 0.1558] | 0.1174 [0.0495, 0.2459] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 16x | bernoulli | 0.1900 [0.0807, 0.3821] | 0.1328 [0.0535, 0.2770] | 0.2288 [0.1035, 0.4404] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 4x | kvzip | 0.0970 [0.0353, 0.2318] | 0.1505 [0.0502, 0.3494] | 0.0801 [0.0295, 0.1770] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 8x | kvzip | 0.2672 [0.1083, 0.5195] | 0.3167 [0.1087, 0.6563] | 0.2492 [0.1082, 0.4581] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 16x | kvzip | 0.4487 [0.2149, 0.7790] | 0.4576 [0.1976, 0.8695] | 0.4444 [0.2235, 0.7355] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 4x | perhead_bern | 0.0092 [0.0017, 0.0759] | 0.0042 [0.0012, 0.0150] | 0.0198 [0.0023, 0.1571] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 8x | perhead_bern | 0.0209 [0.0042, 0.1368] | 0.0098 [0.0031, 0.0312] | 0.0460 [0.0057, 0.2488] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 16x | perhead_bern | 0.0363 [0.0080, 0.1967] | 0.0177 [0.0058, 0.0534] | 0.0761 [0.0109, 0.3362] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 8x | vkvwr | 0.0189 [0.0044, 0.0761] | 0.0108 [0.0036, 0.0314] | 0.0308 [0.0054, 0.1119] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 16x | vkvwr | 0.0274 [0.0067, 0.0964] | 0.0167 [0.0054, 0.0475] | 0.0409 [0.0079, 0.1324] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 32x | vkvwr | 0.0326 [0.0081, 0.1084] | 0.0202 [0.0064, 0.0571] | 0.0468 [0.0097, 0.1464] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 64x | vkvwr | 0.0368 [0.0091, 0.1243] | 0.0219 [0.0070, 0.0609] | 0.0550 [0.0113, 0.1723] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 200x | vkvwr | 0.0456 [0.0112, 0.1631] | 0.0256 [0.0083, 0.0695] | 0.0715 [0.0145, 0.2406] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 8x | bernoulli | 0.0977 [0.0472, 0.1942] | 0.0892 [0.0428, 0.1957] | 0.1031 [0.0507, 0.1934] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 8x | kvzip | 0.4732 [0.2085, 0.9919] | 1.0159 [0.6685, 1.1403] | 0.2907 [0.1500, 0.5262] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 8x | perhead_bern | 0.0202 [0.0048, 0.0891] | 0.0105 [0.0030, 0.0345] | 0.0355 [0.0073, 0.1605] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 8x | vkvwr | 0.0183 [0.0051, 0.0622] | 0.0107 [0.0032, 0.0319] | 0.0293 [0.0074, 0.0923] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 16x | vkvwr | 0.0247 [0.0074, 0.0776] | 0.0165 [0.0053, 0.0449] | 0.0352 [0.0096, 0.1128] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 32x | vkvwr | 0.0276 [0.0088, 0.0854] | 0.0213 [0.0070, 0.0572] | 0.0334 [0.0104, 0.1122] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 8x | bernoulli | 0.0422 [0.0167, 0.0984] | 0.0485 [0.0168, 0.1156] | 0.0395 [0.0166, 0.0891] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 8x | kvzip | 0.1913 [0.0700, 0.4631] | 0.3056 [0.1044, 0.7073] | 0.1533 [0.0595, 0.3507] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 8x | perhead_bern | 0.0049 [0.0010, 0.0318] | 0.0038 [0.0008, 0.0146] | 0.0061 [0.0011, 0.0691] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 8x | vkvwr | 0.0053 [0.0012, 0.0227] | 0.0047 [0.0012, 0.0160] | 0.0056 [0.0011, 0.0301] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 16x | vkvwr | 0.0090 [0.0021, 0.0328] | 0.0089 [0.0022, 0.0262] | 0.0091 [0.0020, 0.0400] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 32x | vkvwr | 0.0126 [0.0031, 0.0455] | 0.0134 [0.0034, 0.0383] | 0.0120 [0.0029, 0.0538] |

## numerator ||A_hat-A||/||A||: median [p25, p75] over heads

| model | dataset | task | ratio | arm | all layers | layers 0-12 | layers 13+ |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 4x | bernoulli | 0.0608 [0.0231, 0.1502] | 0.0647 [0.0240, 0.1859] | 0.0580 [0.0226, 0.1346] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 8x | bernoulli | 0.1902 [0.0781, 0.4167] | 0.1738 [0.0715, 0.4216] | 0.2015 [0.0835, 0.4143] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 16x | bernoulli | 0.3165 [0.1258, 0.6916] | 0.2634 [0.1120, 0.6268] | 0.3588 [0.1401, 0.7199] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 4x | kvzip | 0.1387 [0.0437, 0.3619] | 0.2560 [0.0637, 0.6706] | 0.1044 [0.0372, 0.2324] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 8x | kvzip | 0.3445 [0.1466, 0.6640] | 0.4746 [0.1887, 0.8826] | 0.2887 [0.1287, 0.5355] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 16x | kvzip | 0.6170 [0.3183, 0.8829] | 0.7260 [0.3954, 0.9764] | 0.5517 [0.2737, 0.8002] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 4x | perhead_bern | 0.0310 [0.0076, 0.1346] | 0.0172 [0.0052, 0.0569] | 0.0509 [0.0108, 0.2240] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 8x | perhead_bern | 0.0488 [0.0136, 0.1901] | 0.0321 [0.0109, 0.0913] | 0.0711 [0.0171, 0.3147] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 16x | perhead_bern | 0.0615 [0.0198, 0.2091] | 0.0472 [0.0173, 0.1240] | 0.0780 [0.0225, 0.3200] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 8x | vkvwr | 0.0445 [0.0140, 0.1297] | 0.0337 [0.0119, 0.0877] | 0.0555 [0.0160, 0.1654] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 16x | vkvwr | 0.0582 [0.0187, 0.1670] | 0.0466 [0.0171, 0.1226] | 0.0690 [0.0201, 0.2067] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 32x | vkvwr | 0.0684 [0.0220, 0.1990] | 0.0552 [0.0202, 0.1505] | 0.0809 [0.0240, 0.2409] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 64x | vkvwr | 0.0751 [0.0239, 0.2271] | 0.0582 [0.0211, 0.1600] | 0.0940 [0.0267, 0.2889] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 200x | vkvwr | 0.0893 [0.0284, 0.2854] | 0.0677 [0.0251, 0.1889] | 0.1140 [0.0322, 0.3801] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 4x | bernoulli | 0.0364 [0.0149, 0.0843] | 0.0382 [0.0173, 0.0827] | 0.0352 [0.0132, 0.0854] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 8x | bernoulli | 0.1080 [0.0461, 0.2330] | 0.1118 [0.0515, 0.2391] | 0.1052 [0.0425, 0.2295] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 16x | bernoulli | 0.2009 [0.0846, 0.4268] | 0.1990 [0.0877, 0.4282] | 0.2024 [0.0824, 0.4262] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 4x | kvzip | 0.1926 [0.0716, 0.4057] | 0.2752 [0.1137, 0.5307] | 0.1482 [0.0527, 0.3262] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 8x | kvzip | 0.4355 [0.1905, 0.7470] | 0.6015 [0.2916, 0.8725] | 0.3467 [0.1456, 0.6236] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 16x | kvzip | 0.6672 [0.3493, 0.9115] | 0.7652 [0.4324, 0.9638] | 0.6050 [0.3028, 0.8539] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 4x | perhead_bern | 0.0244 [0.0064, 0.0859] | 0.0126 [0.0041, 0.0375] | 0.0424 [0.0102, 0.1353] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 8x | perhead_bern | 0.0400 [0.0120, 0.1309] | 0.0242 [0.0089, 0.0648] | 0.0643 [0.0164, 0.1959] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 16x | perhead_bern | 0.0546 [0.0182, 0.1675] | 0.0352 [0.0139, 0.0923] | 0.0800 [0.0240, 0.2472] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 8x | vkvwr | 0.0390 [0.0129, 0.1012] | 0.0261 [0.0101, 0.0622] | 0.0540 [0.0163, 0.1368] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 16x | vkvwr | 0.0504 [0.0184, 0.1231] | 0.0384 [0.0161, 0.0844] | 0.0630 [0.0209, 0.1583] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 32x | vkvwr | 0.0599 [0.0221, 0.1441] | 0.0465 [0.0197, 0.1005] | 0.0742 [0.0243, 0.1825] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 64x | vkvwr | 0.0640 [0.0233, 0.1569] | 0.0496 [0.0205, 0.1065] | 0.0810 [0.0265, 0.2029] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 200x | vkvwr | 0.0731 [0.0266, 0.1807] | 0.0549 [0.0230, 0.1201] | 0.0943 [0.0304, 0.2399] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 4x | bernoulli | 0.0100 [0.0037, 0.0267] | 0.0088 [0.0032, 0.0222] | 0.0109 [0.0040, 0.0305] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 8x | bernoulli | 0.0209 [0.0083, 0.0491] | 0.0193 [0.0073, 0.0461] | 0.0219 [0.0090, 0.0513] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 16x | bernoulli | 0.0378 [0.0156, 0.0859] | 0.0341 [0.0135, 0.0781] | 0.0405 [0.0170, 0.0923] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 4x | kvzip | 0.0510 [0.0159, 0.1403] | 0.0700 [0.0209, 0.1883] | 0.0428 [0.0134, 0.1091] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 8x | kvzip | 0.1016 [0.0386, 0.2446] | 0.1439 [0.0525, 0.3457] | 0.0841 [0.0329, 0.1875] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 16x | kvzip | 0.1577 [0.0661, 0.3412] | 0.2041 [0.0845, 0.4496] | 0.1369 [0.0578, 0.2793] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 4x | perhead_bern | 0.0081 [0.0028, 0.0215] | 0.0055 [0.0020, 0.0136] | 0.0106 [0.0037, 0.0297] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 8x | perhead_bern | 0.0142 [0.0056, 0.0336] | 0.0106 [0.0042, 0.0244] | 0.0173 [0.0070, 0.0411] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 16x | perhead_bern | 0.0242 [0.0099, 0.0541] | 0.0184 [0.0077, 0.0410] | 0.0294 [0.0121, 0.0641] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 8x | vkvwr | 0.0131 [0.0053, 0.0306] | 0.0108 [0.0043, 0.0251] | 0.0149 [0.0061, 0.0349] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 16x | vkvwr | 0.0187 [0.0079, 0.0411] | 0.0163 [0.0069, 0.0365] | 0.0207 [0.0087, 0.0444] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 32x | vkvwr | 0.0244 [0.0100, 0.0519] | 0.0217 [0.0091, 0.0475] | 0.0266 [0.0109, 0.0550] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 64x | vkvwr | 0.0298 [0.0123, 0.0624] | 0.0259 [0.0109, 0.0546] | 0.0330 [0.0135, 0.0673] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 200x | vkvwr | 0.0424 [0.0165, 0.0908] | 0.0348 [0.0138, 0.0753] | 0.0491 [0.0190, 0.1016] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 4x | bernoulli | 0.0352 [0.0138, 0.0835] | 0.0317 [0.0131, 0.0730] | 0.0381 [0.0144, 0.0914] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 8x | bernoulli | 0.1362 [0.0631, 0.2738] | 0.1187 [0.0531, 0.2487] | 0.1477 [0.0708, 0.2926] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 16x | bernoulli | 0.2495 [0.1137, 0.4988] | 0.2077 [0.0925, 0.4386] | 0.2797 [0.1329, 0.5362] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 4x | kvzip | 0.1792 [0.0575, 0.4147] | 0.2191 [0.0727, 0.4722] | 0.1565 [0.0479, 0.3783] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 8x | kvzip | 0.4440 [0.2026, 0.7679] | 0.4679 [0.2002, 0.8204] | 0.4291 [0.2036, 0.7340] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 16x | kvzip | 0.5915 [0.3119, 0.8661] | 0.6143 [0.3061, 0.9093] | 0.5777 [0.3165, 0.8349] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 4x | perhead_bern | 0.0215 [0.0051, 0.0842] | 0.0073 [0.0022, 0.0303] | 0.0418 [0.0114, 0.1226] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 8x | perhead_bern | 0.0352 [0.0089, 0.1317] | 0.0125 [0.0041, 0.0486] | 0.0658 [0.0189, 0.2017] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 16x | perhead_bern | 0.0517 [0.0143, 0.1842] | 0.0204 [0.0069, 0.0732] | 0.0904 [0.0289, 0.2836] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 8x | vkvwr | 0.0360 [0.0106, 0.1020] | 0.0163 [0.0057, 0.0507] | 0.0585 [0.0197, 0.1362] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 16x | vkvwr | 0.0491 [0.0158, 0.1300] | 0.0243 [0.0088, 0.0704] | 0.0752 [0.0278, 0.1716] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 32x | vkvwr | 0.0584 [0.0186, 0.1550] | 0.0289 [0.0107, 0.0880] | 0.0892 [0.0321, 0.2024] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 64x | vkvwr | 0.0628 [0.0190, 0.1716] | 0.0301 [0.0110, 0.0941] | 0.0975 [0.0335, 0.2241] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 200x | vkvwr | 0.0685 [0.0207, 0.1986] | 0.0319 [0.0117, 0.1014] | 0.1088 [0.0361, 0.2723] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 4x | bernoulli | 0.0518 [0.0193, 0.1295] | 0.0601 [0.0222, 0.1453] | 0.0479 [0.0181, 0.1194] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 8x | bernoulli | 0.1603 [0.0608, 0.3693] | 0.1534 [0.0584, 0.3357] | 0.1651 [0.0624, 0.3920] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 16x | bernoulli | 0.2544 [0.0950, 0.5597] | 0.2216 [0.0824, 0.4877] | 0.2755 [0.1029, 0.6063] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 4x | kvzip | 0.0944 [0.0331, 0.2480] | 0.1754 [0.0583, 0.4786] | 0.0696 [0.0257, 0.1666] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 8x | kvzip | 0.2832 [0.1059, 0.5987] | 0.4421 [0.1573, 0.7704] | 0.2291 [0.0882, 0.4858] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 16x | kvzip | 0.5848 [0.2736, 0.8666] | 0.7360 [0.3155, 0.9175] | 0.5215 [0.2559, 0.8147] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 4x | perhead_bern | 0.0165 [0.0037, 0.0991] | 0.0118 [0.0037, 0.0370] | 0.0233 [0.0036, 0.1836] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 8x | perhead_bern | 0.0353 [0.0089, 0.1811] | 0.0255 [0.0091, 0.0741] | 0.0487 [0.0087, 0.3265] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 16x | perhead_bern | 0.0533 [0.0146, 0.2163] | 0.0420 [0.0157, 0.1147] | 0.0673 [0.0137, 0.3362] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 8x | vkvwr | 0.0283 [0.0078, 0.1052] | 0.0268 [0.0095, 0.0706] | 0.0296 [0.0067, 0.1484] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 16x | vkvwr | 0.0349 [0.0098, 0.1252] | 0.0342 [0.0123, 0.0915] | 0.0355 [0.0084, 0.1659] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 32x | vkvwr | 0.0440 [0.0120, 0.1602] | 0.0423 [0.0145, 0.1122] | 0.0457 [0.0107, 0.2166] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 64x | vkvwr | 0.0483 [0.0132, 0.1745] | 0.0436 [0.0154, 0.1135] | 0.0526 [0.0118, 0.2501] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 200x | vkvwr | 0.0578 [0.0165, 0.2140] | 0.0487 [0.0181, 0.1268] | 0.0668 [0.0153, 0.3149] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 4x | bernoulli | 0.0226 [0.0080, 0.0535] | 0.0319 [0.0104, 0.0675] | 0.0190 [0.0072, 0.0449] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 8x | bernoulli | 0.0784 [0.0315, 0.1716] | 0.0990 [0.0358, 0.2078] | 0.0698 [0.0298, 0.1503] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 16x | bernoulli | 0.1768 [0.0726, 0.3603] | 0.1840 [0.0659, 0.3780] | 0.1733 [0.0760, 0.3501] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 4x | kvzip | 0.1008 [0.0311, 0.2571] | 0.1722 [0.0562, 0.4038] | 0.0764 [0.0239, 0.1915] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 8x | kvzip | 0.2947 [0.1053, 0.6313] | 0.4671 [0.1714, 0.8232] | 0.2347 [0.0839, 0.5041] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 16x | kvzip | 0.5382 [0.2570, 0.8357] | 0.6314 [0.2667, 0.9199] | 0.4987 [0.2524, 0.7798] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 4x | perhead_bern | 0.0112 [0.0024, 0.0568] | 0.0088 [0.0026, 0.0256] | 0.0144 [0.0022, 0.0965] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 8x | perhead_bern | 0.0217 [0.0053, 0.0903] | 0.0183 [0.0059, 0.0470] | 0.0258 [0.0049, 0.1395] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 16x | perhead_bern | 0.0341 [0.0088, 0.1249] | 0.0289 [0.0097, 0.0740] | 0.0398 [0.0082, 0.1894] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 8x | vkvwr | 0.0245 [0.0064, 0.0752] | 0.0246 [0.0079, 0.0581] | 0.0244 [0.0056, 0.0914] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 16x | vkvwr | 0.0341 [0.0092, 0.0978] | 0.0364 [0.0116, 0.0851] | 0.0324 [0.0080, 0.1113] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 32x | vkvwr | 0.0364 [0.0101, 0.1062] | 0.0389 [0.0122, 0.0937] | 0.0346 [0.0090, 0.1167] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 64x | vkvwr | 0.0381 [0.0107, 0.1112] | 0.0396 [0.0127, 0.0965] | 0.0370 [0.0097, 0.1259] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 200x | vkvwr | 0.0426 [0.0118, 0.1291] | 0.0423 [0.0136, 0.1021] | 0.0428 [0.0108, 0.1547] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 4x | bernoulli | 0.0082 [0.0028, 0.0194] | 0.0090 [0.0027, 0.0214] | 0.0078 [0.0029, 0.0183] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 8x | bernoulli | 0.0202 [0.0076, 0.0459] | 0.0218 [0.0076, 0.0487] | 0.0194 [0.0076, 0.0441] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 16x | bernoulli | 0.0383 [0.0149, 0.0843] | 0.0409 [0.0152, 0.0861] | 0.0370 [0.0148, 0.0830] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 4x | kvzip | 0.0373 [0.0105, 0.1032] | 0.0585 [0.0145, 0.1515] | 0.0303 [0.0094, 0.0790] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 8x | kvzip | 0.0707 [0.0239, 0.1802] | 0.1065 [0.0317, 0.2734] | 0.0588 [0.0216, 0.1406] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 16x | kvzip | 0.1055 [0.0384, 0.2537] | 0.1519 [0.0489, 0.3874] | 0.0891 [0.0347, 0.2009] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 4x | perhead_bern | 0.0049 [0.0014, 0.0144] | 0.0052 [0.0013, 0.0173] | 0.0047 [0.0015, 0.0131] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 8x | perhead_bern | 0.0114 [0.0038, 0.0311] | 0.0116 [0.0037, 0.0348] | 0.0113 [0.0039, 0.0292] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 16x | perhead_bern | 0.0233 [0.0080, 0.0602] | 0.0229 [0.0077, 0.0616] | 0.0236 [0.0081, 0.0595] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 8x | vkvwr | 0.0093 [0.0031, 0.0241] | 0.0115 [0.0032, 0.0356] | 0.0086 [0.0030, 0.0198] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 16x | vkvwr | 0.0136 [0.0045, 0.0341] | 0.0166 [0.0048, 0.0485] | 0.0126 [0.0044, 0.0291] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 32x | vkvwr | 0.0176 [0.0059, 0.0433] | 0.0198 [0.0060, 0.0565] | 0.0166 [0.0059, 0.0382] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 64x | vkvwr | 0.0232 [0.0080, 0.0545] | 0.0232 [0.0075, 0.0610] | 0.0231 [0.0083, 0.0517] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 200x | vkvwr | 0.0384 [0.0134, 0.0847] | 0.0354 [0.0125, 0.0812] | 0.0401 [0.0140, 0.0868] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 4x | bernoulli | 0.0290 [0.0101, 0.0683] | 0.0206 [0.0072, 0.0499] | 0.0347 [0.0127, 0.0807] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 8x | bernoulli | 0.1035 [0.0378, 0.2317] | 0.0699 [0.0258, 0.1657] | 0.1267 [0.0495, 0.2698] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 16x | bernoulli | 0.2010 [0.0755, 0.4197] | 0.1311 [0.0481, 0.3005] | 0.2482 [0.1021, 0.4842] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 4x | kvzip | 0.0946 [0.0311, 0.2376] | 0.1217 [0.0363, 0.3322] | 0.0853 [0.0285, 0.1966] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 8x | kvzip | 0.2758 [0.0924, 0.5576] | 0.2458 [0.0754, 0.6214] | 0.2902 [0.1029, 0.5351] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 16x | kvzip | 0.4432 [0.1705, 0.7926] | 0.3847 [0.1382, 0.7822] | 0.4795 [0.1961, 0.7962] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 4x | perhead_bern | 0.0094 [0.0016, 0.0828] | 0.0040 [0.0011, 0.0157] | 0.0206 [0.0023, 0.1700] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 8x | perhead_bern | 0.0216 [0.0040, 0.1486] | 0.0098 [0.0030, 0.0332] | 0.0478 [0.0057, 0.2694] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 16x | perhead_bern | 0.0374 [0.0077, 0.2145] | 0.0174 [0.0055, 0.0561] | 0.0792 [0.0107, 0.3568] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 8x | vkvwr | 0.0194 [0.0042, 0.0824] | 0.0107 [0.0033, 0.0338] | 0.0321 [0.0053, 0.1222] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 16x | vkvwr | 0.0282 [0.0064, 0.1046] | 0.0164 [0.0051, 0.0515] | 0.0429 [0.0078, 0.1459] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 32x | vkvwr | 0.0340 [0.0078, 0.1189] | 0.0199 [0.0060, 0.0619] | 0.0501 [0.0096, 0.1627] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 64x | vkvwr | 0.0382 [0.0088, 0.1367] | 0.0216 [0.0066, 0.0649] | 0.0581 [0.0112, 0.1901] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 200x | vkvwr | 0.0474 [0.0108, 0.1764] | 0.0254 [0.0080, 0.0747] | 0.0754 [0.0145, 0.2619] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 8x | bernoulli | 0.0970 [0.0440, 0.1997] | 0.0895 [0.0405, 0.1997] | 0.1021 [0.0469, 0.1996] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 8x | kvzip | 0.4633 [0.1935, 0.9471] | 0.9764 [0.5853, 0.9999] | 0.2819 [0.1345, 0.5240] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 8x | perhead_bern | 0.0199 [0.0047, 0.0863] | 0.0104 [0.0029, 0.0351] | 0.0346 [0.0071, 0.1543] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 8x | vkvwr | 0.0179 [0.0049, 0.0614] | 0.0105 [0.0031, 0.0322] | 0.0280 [0.0070, 0.0902] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 16x | vkvwr | 0.0243 [0.0071, 0.0769] | 0.0164 [0.0053, 0.0460] | 0.0339 [0.0092, 0.1099] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 32x | vkvwr | 0.0267 [0.0083, 0.0829] | 0.0209 [0.0068, 0.0582] | 0.0320 [0.0098, 0.1066] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 8x | bernoulli | 0.0415 [0.0160, 0.1016] | 0.0474 [0.0159, 0.1228] | 0.0391 [0.0160, 0.0909] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 8x | kvzip | 0.1691 [0.0612, 0.4209] | 0.2617 [0.0787, 0.6187] | 0.1417 [0.0537, 0.3234] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 8x | perhead_bern | 0.0048 [0.0009, 0.0327] | 0.0036 [0.0008, 0.0151] | 0.0060 [0.0010, 0.0732] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 8x | vkvwr | 0.0051 [0.0011, 0.0236] | 0.0046 [0.0011, 0.0166] | 0.0055 [0.0011, 0.0313] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 16x | vkvwr | 0.0087 [0.0020, 0.0346] | 0.0085 [0.0020, 0.0276] | 0.0090 [0.0019, 0.0417] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 32x | vkvwr | 0.0121 [0.0029, 0.0481] | 0.0128 [0.0031, 0.0398] | 0.0117 [0.0028, 0.0562] |

## numerator vs Abar: median [p25, p75] over heads

| model | dataset | task | ratio | arm | all layers | layers 0-12 | layers 13+ |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 4x | bernoulli | 0.0234 [0.0110, 0.0469] | 0.0242 [0.0109, 0.0520] | 0.0230 [0.0110, 0.0432] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 8x | bernoulli | 0.0699 [0.0348, 0.1305] | 0.0630 [0.0313, 0.1218] | 0.0742 [0.0377, 0.1372] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 16x | bernoulli | 0.1097 [0.0533, 0.2215] | 0.0934 [0.0459, 0.1886] | 0.1220 [0.0598, 0.2443] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 4x | kvzip | 0.0494 [0.0200, 0.0937] | 0.0726 [0.0281, 0.1267] | 0.0395 [0.0175, 0.0718] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 8x | kvzip | 0.1034 [0.0603, 0.1529] | 0.1163 [0.0664, 0.1677] | 0.0960 [0.0568, 0.1427] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 16x | kvzip | 0.1485 [0.1017, 0.2124] | 0.1525 [0.1087, 0.2142] | 0.1450 [0.0973, 0.2106] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 4x | perhead_bern | 0.0100 [0.0032, 0.0426] | 0.0055 [0.0020, 0.0142] | 0.0170 [0.0048, 0.0728] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 8x | perhead_bern | 0.0156 [0.0058, 0.0566] | 0.0107 [0.0043, 0.0229] | 0.0238 [0.0074, 0.0971] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 16x | perhead_bern | 0.0210 [0.0086, 0.0588] | 0.0164 [0.0070, 0.0327] | 0.0275 [0.0101, 0.0922] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 8x | vkvwr | 0.0149 [0.0061, 0.0385] | 0.0116 [0.0048, 0.0232] | 0.0187 [0.0072, 0.0565] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 16x | vkvwr | 0.0197 [0.0082, 0.0479] | 0.0162 [0.0071, 0.0333] | 0.0233 [0.0093, 0.0688] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 32x | vkvwr | 0.0233 [0.0096, 0.0576] | 0.0188 [0.0083, 0.0411] | 0.0273 [0.0109, 0.0790] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 64x | vkvwr | 0.0254 [0.0104, 0.0655] | 0.0196 [0.0087, 0.0425] | 0.0314 [0.0122, 0.0927] |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | 200x | vkvwr | 0.0293 [0.0122, 0.0792] | 0.0223 [0.0101, 0.0480] | 0.0372 [0.0143, 0.1164] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 4x | bernoulli | 0.0146 [0.0071, 0.0285] | 0.0148 [0.0077, 0.0277] | 0.0144 [0.0065, 0.0291] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 8x | bernoulli | 0.0426 [0.0217, 0.0796] | 0.0438 [0.0225, 0.0806] | 0.0417 [0.0211, 0.0788] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 16x | bernoulli | 0.0778 [0.0390, 0.1454] | 0.0773 [0.0376, 0.1469] | 0.0783 [0.0401, 0.1442] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 4x | kvzip | 0.0729 [0.0317, 0.1337] | 0.0946 [0.0480, 0.1606] | 0.0587 [0.0242, 0.1140] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 8x | kvzip | 0.1434 [0.0801, 0.2316] | 0.1761 [0.1053, 0.2811] | 0.1247 [0.0668, 0.1996] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 16x | kvzip | 0.2021 [0.1274, 0.2962] | 0.2195 [0.1386, 0.3317] | 0.1919 [0.1189, 0.2757] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 4x | perhead_bern | 0.0092 [0.0028, 0.0305] | 0.0047 [0.0016, 0.0125] | 0.0158 [0.0049, 0.0450] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 8x | perhead_bern | 0.0150 [0.0052, 0.0470] | 0.0088 [0.0034, 0.0215] | 0.0238 [0.0080, 0.0655] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 16x | perhead_bern | 0.0205 [0.0079, 0.0590] | 0.0131 [0.0056, 0.0302] | 0.0298 [0.0113, 0.0822] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 8x | vkvwr | 0.0149 [0.0056, 0.0348] | 0.0098 [0.0040, 0.0218] | 0.0200 [0.0080, 0.0453] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 16x | vkvwr | 0.0193 [0.0082, 0.0418] | 0.0143 [0.0065, 0.0302] | 0.0236 [0.0103, 0.0524] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 32x | vkvwr | 0.0228 [0.0099, 0.0486] | 0.0171 [0.0080, 0.0359] | 0.0282 [0.0121, 0.0596] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 64x | vkvwr | 0.0247 [0.0105, 0.0529] | 0.0180 [0.0084, 0.0374] | 0.0309 [0.0130, 0.0649] |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | 200x | vkvwr | 0.0275 [0.0119, 0.0605] | 0.0198 [0.0094, 0.0418] | 0.0351 [0.0148, 0.0767] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 4x | bernoulli | 0.0043 [0.0018, 0.0100] | 0.0035 [0.0015, 0.0079] | 0.0050 [0.0021, 0.0117] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 8x | bernoulli | 0.0090 [0.0041, 0.0184] | 0.0075 [0.0034, 0.0163] | 0.0099 [0.0048, 0.0200] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 16x | bernoulli | 0.0164 [0.0078, 0.0323] | 0.0135 [0.0063, 0.0277] | 0.0184 [0.0091, 0.0356] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 4x | kvzip | 0.0215 [0.0080, 0.0511] | 0.0270 [0.0096, 0.0644] | 0.0189 [0.0071, 0.0424] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 8x | kvzip | 0.0422 [0.0192, 0.0865] | 0.0542 [0.0231, 0.1158] | 0.0366 [0.0173, 0.0714] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 16x | kvzip | 0.0640 [0.0319, 0.1182] | 0.0753 [0.0364, 0.1460] | 0.0587 [0.0293, 0.1042] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 4x | perhead_bern | 0.0035 [0.0013, 0.0085] | 0.0023 [0.0009, 0.0051] | 0.0049 [0.0019, 0.0115] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 8x | perhead_bern | 0.0061 [0.0027, 0.0132] | 0.0043 [0.0019, 0.0093] | 0.0077 [0.0036, 0.0161] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 16x | perhead_bern | 0.0103 [0.0047, 0.0211] | 0.0073 [0.0034, 0.0154] | 0.0128 [0.0061, 0.0252] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 8x | vkvwr | 0.0057 [0.0026, 0.0118] | 0.0044 [0.0019, 0.0093] | 0.0068 [0.0032, 0.0134] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 16x | vkvwr | 0.0081 [0.0038, 0.0159] | 0.0066 [0.0031, 0.0134] | 0.0092 [0.0045, 0.0172] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 32x | vkvwr | 0.0104 [0.0048, 0.0199] | 0.0086 [0.0041, 0.0172] | 0.0118 [0.0055, 0.0213] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 64x | vkvwr | 0.0126 [0.0058, 0.0239] | 0.0102 [0.0048, 0.0201] | 0.0145 [0.0069, 0.0261] |
| Llama-3.1-8B-Instruct | ruler32k | vt | 200x | vkvwr | 0.0177 [0.0077, 0.0353] | 0.0135 [0.0060, 0.0278] | 0.0210 [0.0095, 0.0401] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 4x | bernoulli | 0.0135 [0.0058, 0.0280] | 0.0130 [0.0060, 0.0257] | 0.0139 [0.0056, 0.0304] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 8x | bernoulli | 0.0509 [0.0263, 0.0926] | 0.0477 [0.0241, 0.0881] | 0.0530 [0.0280, 0.0961] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 16x | bernoulli | 0.0906 [0.0493, 0.1631] | 0.0814 [0.0407, 0.1554] | 0.0960 [0.0555, 0.1681] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 4x | kvzip | 0.0658 [0.0230, 0.1298] | 0.0775 [0.0326, 0.1395] | 0.0577 [0.0169, 0.1226] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 8x | kvzip | 0.1390 [0.0756, 0.2257] | 0.1427 [0.0767, 0.2399] | 0.1367 [0.0749, 0.2183] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 16x | kvzip | 0.1761 [0.1087, 0.2627] | 0.1829 [0.1067, 0.2820] | 0.1722 [0.1101, 0.2527] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 4x | perhead_bern | 0.0078 [0.0022, 0.0253] | 0.0028 [0.0009, 0.0105] | 0.0133 [0.0047, 0.0357] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 8x | perhead_bern | 0.0129 [0.0039, 0.0408] | 0.0049 [0.0017, 0.0169] | 0.0218 [0.0082, 0.0568] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 16x | perhead_bern | 0.0192 [0.0063, 0.0567] | 0.0079 [0.0029, 0.0254] | 0.0309 [0.0125, 0.0788] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 8x | vkvwr | 0.0132 [0.0047, 0.0321] | 0.0064 [0.0024, 0.0175] | 0.0193 [0.0085, 0.0398] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 16x | vkvwr | 0.0181 [0.0069, 0.0399] | 0.0094 [0.0038, 0.0248] | 0.0248 [0.0120, 0.0498] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 32x | vkvwr | 0.0219 [0.0082, 0.0475] | 0.0112 [0.0045, 0.0303] | 0.0300 [0.0141, 0.0569] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 64x | vkvwr | 0.0235 [0.0084, 0.0524] | 0.0115 [0.0046, 0.0329] | 0.0327 [0.0147, 0.0630] |
| Llama-3.1-8B-Instruct | ruler32k | cwe | 200x | vkvwr | 0.0256 [0.0091, 0.0610] | 0.0122 [0.0050, 0.0352] | 0.0367 [0.0161, 0.0769] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 4x | bernoulli | 0.0253 [0.0110, 0.0527] | 0.0271 [0.0120, 0.0527] | 0.0242 [0.0104, 0.0527] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 8x | bernoulli | 0.0738 [0.0331, 0.1478] | 0.0651 [0.0305, 0.1218] | 0.0799 [0.0349, 0.1694] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 16x | bernoulli | 0.1106 [0.0492, 0.2247] | 0.0911 [0.0416, 0.1756] | 0.1239 [0.0549, 0.2613] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 4x | kvzip | 0.0438 [0.0179, 0.0905] | 0.0670 [0.0292, 0.1309] | 0.0346 [0.0145, 0.0714] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 8x | kvzip | 0.1083 [0.0529, 0.1866] | 0.1259 [0.0650, 0.2012] | 0.0988 [0.0471, 0.1744] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 16x | kvzip | 0.1728 [0.1050, 0.2611] | 0.1711 [0.1051, 0.2592] | 0.1737 [0.1049, 0.2620] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 4x | perhead_bern | 0.0075 [0.0020, 0.0408] | 0.0054 [0.0019, 0.0130] | 0.0110 [0.0020, 0.0764] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 8x | perhead_bern | 0.0155 [0.0048, 0.0697] | 0.0113 [0.0046, 0.0251] | 0.0227 [0.0049, 0.1271] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 16x | perhead_bern | 0.0238 [0.0080, 0.0824] | 0.0184 [0.0082, 0.0392] | 0.0314 [0.0078, 0.1317] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 8x | vkvwr | 0.0127 [0.0043, 0.0396] | 0.0114 [0.0050, 0.0249] | 0.0141 [0.0038, 0.0631] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 16x | vkvwr | 0.0160 [0.0055, 0.0473] | 0.0150 [0.0066, 0.0323] | 0.0170 [0.0048, 0.0683] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 32x | vkvwr | 0.0198 [0.0067, 0.0599] | 0.0181 [0.0079, 0.0376] | 0.0216 [0.0060, 0.0921] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 64x | vkvwr | 0.0214 [0.0073, 0.0651] | 0.0188 [0.0082, 0.0388] | 0.0245 [0.0067, 0.1054] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | 200x | vkvwr | 0.0254 [0.0090, 0.0800] | 0.0211 [0.0095, 0.0430] | 0.0310 [0.0086, 0.1291] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 4x | bernoulli | 0.0114 [0.0046, 0.0229] | 0.0148 [0.0058, 0.0265] | 0.0099 [0.0042, 0.0203] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 8x | bernoulli | 0.0389 [0.0179, 0.0727] | 0.0456 [0.0194, 0.0810] | 0.0357 [0.0173, 0.0673] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 16x | bernoulli | 0.0842 [0.0398, 0.1530] | 0.0831 [0.0345, 0.1486] | 0.0848 [0.0423, 0.1557] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 4x | kvzip | 0.0485 [0.0175, 0.1046] | 0.0723 [0.0293, 0.1448] | 0.0387 [0.0139, 0.0849] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 8x | kvzip | 0.1254 [0.0552, 0.2336] | 0.1761 [0.0814, 0.2913] | 0.1040 [0.0462, 0.1983] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 16x | kvzip | 0.2006 [0.1155, 0.3056] | 0.2177 [0.1140, 0.3356] | 0.1923 [0.1163, 0.2903] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 4x | perhead_bern | 0.0052 [0.0013, 0.0252] | 0.0040 [0.0013, 0.0103] | 0.0072 [0.0013, 0.0426] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 8x | perhead_bern | 0.0102 [0.0029, 0.0388] | 0.0084 [0.0030, 0.0190] | 0.0127 [0.0027, 0.0607] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 16x | perhead_bern | 0.0160 [0.0049, 0.0533] | 0.0134 [0.0052, 0.0296] | 0.0193 [0.0046, 0.0792] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 8x | vkvwr | 0.0115 [0.0035, 0.0310] | 0.0110 [0.0040, 0.0231] | 0.0118 [0.0032, 0.0391] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 16x | vkvwr | 0.0160 [0.0051, 0.0406] | 0.0163 [0.0060, 0.0334] | 0.0158 [0.0046, 0.0474] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 32x | vkvwr | 0.0171 [0.0056, 0.0441] | 0.0173 [0.0065, 0.0374] | 0.0170 [0.0052, 0.0498] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 64x | vkvwr | 0.0179 [0.0059, 0.0457] | 0.0178 [0.0067, 0.0377] | 0.0180 [0.0055, 0.0527] |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | 200x | vkvwr | 0.0201 [0.0065, 0.0530] | 0.0192 [0.0073, 0.0403] | 0.0209 [0.0061, 0.0653] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 4x | bernoulli | 0.0040 [0.0016, 0.0086] | 0.0041 [0.0014, 0.0090] | 0.0040 [0.0017, 0.0084] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 8x | bernoulli | 0.0099 [0.0042, 0.0200] | 0.0099 [0.0040, 0.0201] | 0.0099 [0.0044, 0.0199] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 16x | bernoulli | 0.0187 [0.0084, 0.0364] | 0.0184 [0.0080, 0.0352] | 0.0189 [0.0086, 0.0373] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 4x | kvzip | 0.0185 [0.0059, 0.0444] | 0.0265 [0.0079, 0.0630] | 0.0154 [0.0054, 0.0354] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 8x | kvzip | 0.0347 [0.0135, 0.0754] | 0.0471 [0.0169, 0.1078] | 0.0297 [0.0124, 0.0617] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 16x | kvzip | 0.0508 [0.0216, 0.1043] | 0.0666 [0.0257, 0.1449] | 0.0446 [0.0200, 0.0872] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 4x | perhead_bern | 0.0024 [0.0008, 0.0065] | 0.0025 [0.0006, 0.0076] | 0.0024 [0.0009, 0.0059] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 8x | perhead_bern | 0.0055 [0.0020, 0.0138] | 0.0055 [0.0018, 0.0151] | 0.0056 [0.0022, 0.0132] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 16x | perhead_bern | 0.0112 [0.0043, 0.0264] | 0.0105 [0.0039, 0.0259] | 0.0116 [0.0045, 0.0267] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 8x | vkvwr | 0.0046 [0.0017, 0.0108] | 0.0053 [0.0016, 0.0153] | 0.0043 [0.0017, 0.0092] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 16x | vkvwr | 0.0066 [0.0025, 0.0154] | 0.0076 [0.0025, 0.0204] | 0.0062 [0.0025, 0.0134] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 32x | vkvwr | 0.0085 [0.0033, 0.0192] | 0.0092 [0.0032, 0.0233] | 0.0083 [0.0033, 0.0176] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 64x | vkvwr | 0.0110 [0.0044, 0.0241] | 0.0105 [0.0039, 0.0252] | 0.0113 [0.0046, 0.0236] |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | 200x | vkvwr | 0.0178 [0.0072, 0.0374] | 0.0159 [0.0064, 0.0334] | 0.0192 [0.0077, 0.0395] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 4x | bernoulli | 0.0137 [0.0057, 0.0283] | 0.0100 [0.0040, 0.0213] | 0.0161 [0.0071, 0.0328] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 8x | bernoulli | 0.0489 [0.0209, 0.0944] | 0.0340 [0.0146, 0.0697] | 0.0587 [0.0272, 0.1086] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 16x | bernoulli | 0.0923 [0.0405, 0.1724] | 0.0623 [0.0263, 0.1262] | 0.1113 [0.0541, 0.1975] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 4x | kvzip | 0.0427 [0.0166, 0.0948] | 0.0552 [0.0194, 0.1197] | 0.0378 [0.0153, 0.0815] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 8x | kvzip | 0.1085 [0.0475, 0.2038] | 0.0996 [0.0395, 0.2083] | 0.1126 [0.0535, 0.2019] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 16x | kvzip | 0.1701 [0.0829, 0.2881] | 0.1399 [0.0662, 0.2546] | 0.1891 [0.0964, 0.3005] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 4x | perhead_bern | 0.0043 [0.0009, 0.0344] | 0.0020 [0.0006, 0.0065] | 0.0090 [0.0012, 0.0626] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 8x | perhead_bern | 0.0099 [0.0022, 0.0607] | 0.0048 [0.0016, 0.0138] | 0.0204 [0.0031, 0.0993] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 16x | perhead_bern | 0.0172 [0.0042, 0.0840] | 0.0084 [0.0030, 0.0237] | 0.0344 [0.0059, 0.1288] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 8x | vkvwr | 0.0092 [0.0024, 0.0321] | 0.0053 [0.0019, 0.0145] | 0.0144 [0.0030, 0.0458] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 16x | vkvwr | 0.0131 [0.0036, 0.0408] | 0.0080 [0.0028, 0.0217] | 0.0193 [0.0044, 0.0547] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 32x | vkvwr | 0.0158 [0.0043, 0.0467] | 0.0096 [0.0033, 0.0262] | 0.0223 [0.0054, 0.0614] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 64x | vkvwr | 0.0177 [0.0049, 0.0531] | 0.0105 [0.0037, 0.0278] | 0.0258 [0.0063, 0.0714] |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | 200x | vkvwr | 0.0217 [0.0060, 0.0677] | 0.0122 [0.0044, 0.0314] | 0.0334 [0.0081, 0.0977] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 8x | bernoulli | 0.0407 [0.0209, 0.0728] | 0.0373 [0.0188, 0.0711] | 0.0428 [0.0225, 0.0738] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 8x | kvzip | 0.1623 [0.0812, 0.3292] | 0.3360 [0.1856, 0.4660] | 0.1115 [0.0589, 0.1824] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 8x | perhead_bern | 0.0081 [0.0021, 0.0337] | 0.0041 [0.0013, 0.0121] | 0.0147 [0.0034, 0.0574] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 8x | vkvwr | 0.0075 [0.0023, 0.0226] | 0.0042 [0.0014, 0.0118] | 0.0117 [0.0035, 0.0339] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 16x | vkvwr | 0.0099 [0.0033, 0.0285] | 0.0066 [0.0023, 0.0171] | 0.0142 [0.0045, 0.0405] |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | 32x | vkvwr | 0.0112 [0.0040, 0.0311] | 0.0085 [0.0030, 0.0219] | 0.0139 [0.0048, 0.0391] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 8x | bernoulli | 0.0212 [0.0090, 0.0476] | 0.0249 [0.0095, 0.0566] | 0.0195 [0.0088, 0.0429] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 8x | kvzip | 0.0774 [0.0327, 0.1579] | 0.1131 [0.0439, 0.2197] | 0.0644 [0.0285, 0.1275] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 8x | perhead_bern | 0.0025 [0.0005, 0.0141] | 0.0019 [0.0005, 0.0069] | 0.0030 [0.0006, 0.0315] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 8x | vkvwr | 0.0027 [0.0006, 0.0104] | 0.0024 [0.0006, 0.0076] | 0.0028 [0.0006, 0.0136] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 16x | vkvwr | 0.0046 [0.0012, 0.0153] | 0.0045 [0.0012, 0.0128] | 0.0046 [0.0011, 0.0181] |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | 32x | vkvwr | 0.0064 [0.0017, 0.0212] | 0.0068 [0.0019, 0.0182] | 0.0061 [0.0016, 0.0244] |

## Signed denominator error (is the estimator biased?)

Top-k drops mass and never reweights, so it must be negative. Bernoulli and VKvWR are unbiased by construction, so they should straddle zero.

| model | dataset | ratio | arm | mean B_hat/B | median signed err |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | ruler32k | 4x | bernoulli | 0.9935 | -0.0022 |
| Llama-3.1-8B-Instruct | ruler32k | 8x | bernoulli | 0.9931 | -0.0041 |
| Llama-3.1-8B-Instruct | ruler32k | 16x | bernoulli | 0.9937 | -0.0090 |
| Llama-3.1-8B-Instruct | ruler32k | 4x | kvzip | 0.8705 | -0.0649 |
| Llama-3.1-8B-Instruct | ruler32k | 8x | kvzip | 0.7478 | -0.1662 |
| Llama-3.1-8B-Instruct | ruler32k | 16x | kvzip | 0.6519 | -0.2612 |
| Llama-3.1-8B-Instruct | ruler32k | 4x | perhead_bern | 0.9933 | -0.0011 |
| Llama-3.1-8B-Instruct | ruler32k | 8x | perhead_bern | 0.9932 | -0.0015 |
| Llama-3.1-8B-Instruct | ruler32k | 16x | perhead_bern | 0.9915 | -0.0021 |
| Llama-3.1-8B-Instruct | ruler32k | 8x | vkvwr | 0.9916 | -0.0016 |
| Llama-3.1-8B-Instruct | ruler32k | 16x | vkvwr | 0.9909 | -0.0021 |
| Llama-3.1-8B-Instruct | ruler32k | 32x | vkvwr | 0.9906 | -0.0023 |
| Llama-3.1-8B-Instruct | ruler32k | 64x | vkvwr | 0.9914 | -0.0024 |
| Llama-3.1-8B-Instruct | ruler32k | 200x | vkvwr | 0.9928 | -0.0025 |
| Qwen3-4B-Instruct-2507 | ruler32k | 4x | bernoulli | 1.0001 | -0.0002 |
| Qwen3-4B-Instruct-2507 | ruler32k | 8x | bernoulli | 1.0001 | -0.0011 |
| Qwen3-4B-Instruct-2507 | ruler32k | 16x | bernoulli | 0.9993 | -0.0047 |
| Qwen3-4B-Instruct-2507 | ruler32k | 4x | kvzip | 0.8264 | -0.0778 |
| Qwen3-4B-Instruct-2507 | ruler32k | 8x | kvzip | 0.6733 | -0.2241 |
| Qwen3-4B-Instruct-2507 | ruler32k | 16x | kvzip | 0.5203 | -0.4561 |
| Qwen3-4B-Instruct-2507 | ruler32k | 4x | perhead_bern | 0.9988 | -0.0000 |
| Qwen3-4B-Instruct-2507 | ruler32k | 8x | perhead_bern | 1.0003 | -0.0001 |
| Qwen3-4B-Instruct-2507 | ruler32k | 16x | perhead_bern | 1.0018 | -0.0004 |
| Qwen3-4B-Instruct-2507 | ruler32k | 8x | vkvwr | 0.9985 | -0.0001 |
| Qwen3-4B-Instruct-2507 | ruler32k | 16x | vkvwr | 0.9973 | -0.0002 |
| Qwen3-4B-Instruct-2507 | ruler32k | 32x | vkvwr | 0.9955 | -0.0004 |
| Qwen3-4B-Instruct-2507 | ruler32k | 64x | vkvwr | 0.9944 | -0.0005 |
| Qwen3-4B-Instruct-2507 | ruler32k | 200x | vkvwr | 0.9934 | -0.0007 |
| Llama-3.1-8B-Instruct | scbench | 8x | bernoulli | 0.9888 | -0.0103 |
| Llama-3.1-8B-Instruct | scbench | 8x | kvzip | 0.5603 | -0.3343 |
| Llama-3.1-8B-Instruct | scbench | 8x | perhead_bern | 0.9896 | -0.0027 |
| Llama-3.1-8B-Instruct | scbench | 8x | vkvwr | 0.9891 | -0.0028 |
| Llama-3.1-8B-Instruct | scbench | 16x | vkvwr | 0.9842 | -0.0035 |
| Llama-3.1-8B-Instruct | scbench | 32x | vkvwr | 0.9784 | -0.0038 |
| Qwen3-4B-Instruct-2507 | scbench | 8x | bernoulli | 0.9979 | -0.0007 |
| Qwen3-4B-Instruct-2507 | scbench | 8x | kvzip | 0.7378 | -0.1383 |
| Qwen3-4B-Instruct-2507 | scbench | 8x | perhead_bern | 1.0000 | -0.0000 |
| Qwen3-4B-Instruct-2507 | scbench | 8x | vkvwr | 0.9949 | -0.0000 |
| Qwen3-4B-Instruct-2507 | scbench | 16x | vkvwr | 0.9939 | -0.0000 |
| Qwen3-4B-Instruct-2507 | scbench | 32x | vkvwr | 0.9910 | -0.0001 |

## Error versus REALISED memory (the comparison that is not confounded)

kvzip, bernoulli and perhead_bern realise their target exactly, so their nominal ratios are matched memory. VKvWR does not: its ratio only sizes topk_frac and the pilot, while the CLT top-up sets the rest. Read the arms off this table at equal kept fraction.

| model | dataset | task | arm | nominal | realised kept | denominator err | attention err |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | ruler32k | cwe | perhead_bern | 16x | 0.0625 | 0.0094 | 0.0512 |
| Llama-3.1-8B-Instruct | ruler32k | cwe | kvzip | 16x | 0.0625 | 0.3886 | 0.5821 |
| Llama-3.1-8B-Instruct | ruler32k | cwe | bernoulli | 16x | 0.0625 | 0.0656 | 0.2432 |
| Llama-3.1-8B-Instruct | ruler32k | cwe | vkvwr | 200x | 0.0666 | 0.0123 | 0.0676 |
| Llama-3.1-8B-Instruct | ruler32k | cwe | vkvwr | 64x | 0.0963 | 0.0104 | 0.0616 |
| Llama-3.1-8B-Instruct | ruler32k | cwe | vkvwr | 32x | 0.1162 | 0.0099 | 0.0573 |
| Llama-3.1-8B-Instruct | ruler32k | cwe | perhead_bern | 8x | 0.1250 | 0.0063 | 0.0347 |
| Llama-3.1-8B-Instruct | ruler32k | cwe | kvzip | 8x | 0.1250 | 0.2500 | 0.4415 |
| Llama-3.1-8B-Instruct | ruler32k | cwe | bernoulli | 8x | 0.1250 | 0.0326 | 0.1326 |
| Llama-3.1-8B-Instruct | ruler32k | cwe | vkvwr | 16x | 0.1419 | 0.0081 | 0.0481 |
| Llama-3.1-8B-Instruct | ruler32k | cwe | vkvwr | 8x | 0.1921 | 0.0057 | 0.0354 |
| Llama-3.1-8B-Instruct | ruler32k | cwe | perhead_bern | 4x | 0.2500 | 0.0037 | 0.0212 |
| Llama-3.1-8B-Instruct | ruler32k | cwe | kvzip | 4x | 0.2500 | 0.0912 | 0.1804 |
| Llama-3.1-8B-Instruct | ruler32k | cwe | bernoulli | 4x | 0.2500 | 0.0064 | 0.0346 |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | perhead_bern | 16x | 0.0625 | 0.0086 | 0.0532 |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | kvzip | 16x | 0.0625 | 0.3588 | 0.6460 |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | bernoulli | 16x | 0.0625 | 0.0354 | 0.1914 |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | vkvwr | 200x | 0.0688 | 0.0106 | 0.0704 |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | vkvwr | 64x | 0.1005 | 0.0096 | 0.0616 |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | vkvwr | 32x | 0.1167 | 0.0089 | 0.0575 |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | perhead_bern | 8x | 0.1250 | 0.0061 | 0.0388 |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | kvzip | 8x | 0.1250 | 0.2202 | 0.4204 |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | bernoulli | 8x | 0.1250 | 0.0180 | 0.1036 |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | vkvwr | 16x | 0.1424 | 0.0073 | 0.0483 |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | vkvwr | 8x | 0.1856 | 0.0056 | 0.0378 |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | perhead_bern | 4x | 0.2500 | 0.0037 | 0.0237 |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | kvzip | 4x | 0.2500 | 0.0969 | 0.1838 |
| Llama-3.1-8B-Instruct | ruler32k | niah_multikey_2 | bernoulli | 4x | 0.2500 | 0.0063 | 0.0354 |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | kvzip | 16x | 0.0625 | 0.3618 | 0.6311 |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | perhead_bern | 16x | 0.0625 | 0.0103 | 0.0614 |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | bernoulli | 16x | 0.0625 | 0.0661 | 0.3033 |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | vkvwr | 200x | 0.0858 | 0.0137 | 0.0876 |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | bernoulli | 8x | 0.1250 | 0.0420 | 0.1841 |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | kvzip | 8x | 0.1250 | 0.2154 | 0.3616 |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | perhead_bern | 8x | 0.1250 | 0.0071 | 0.0487 |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | vkvwr | 64x | 0.1323 | 0.0114 | 0.0736 |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | vkvwr | 32x | 0.1601 | 0.0100 | 0.0667 |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | vkvwr | 16x | 0.1960 | 0.0089 | 0.0569 |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | vkvwr | 8x | 0.2456 | 0.0064 | 0.0440 |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | perhead_bern | 4x | 0.2500 | 0.0045 | 0.0310 |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | kvzip | 4x | 0.2500 | 0.0849 | 0.1522 |
| Llama-3.1-8B-Instruct | ruler32k | niah_single_3 | bernoulli | 4x | 0.2500 | 0.0105 | 0.0599 |
| Llama-3.1-8B-Instruct | ruler32k | vt | vkvwr | 200x | 0.0612 | 0.0062 | 0.0407 |
| Llama-3.1-8B-Instruct | ruler32k | vt | kvzip | 16x | 0.0625 | 0.0712 | 0.1530 |
| Llama-3.1-8B-Instruct | ruler32k | vt | bernoulli | 16x | 0.0625 | 0.0064 | 0.0367 |
| Llama-3.1-8B-Instruct | ruler32k | vt | perhead_bern | 16x | 0.0625 | 0.0042 | 0.0236 |
| Llama-3.1-8B-Instruct | ruler32k | vt | vkvwr | 64x | 0.0905 | 0.0048 | 0.0288 |
| Llama-3.1-8B-Instruct | ruler32k | vt | vkvwr | 32x | 0.1057 | 0.0041 | 0.0238 |
| Llama-3.1-8B-Instruct | ruler32k | vt | kvzip | 8x | 0.1250 | 0.0466 | 0.0992 |
| Llama-3.1-8B-Instruct | ruler32k | vt | bernoulli | 8x | 0.1250 | 0.0038 | 0.0204 |
| Llama-3.1-8B-Instruct | ruler32k | vt | perhead_bern | 8x | 0.1250 | 0.0027 | 0.0140 |
| Llama-3.1-8B-Instruct | ruler32k | vt | vkvwr | 16x | 0.1288 | 0.0034 | 0.0183 |
| Llama-3.1-8B-Instruct | ruler32k | vt | vkvwr | 8x | 0.1743 | 0.0025 | 0.0129 |
| Llama-3.1-8B-Instruct | ruler32k | vt | bernoulli | 4x | 0.2500 | 0.0021 | 0.0100 |
| Llama-3.1-8B-Instruct | ruler32k | vt | kvzip | 4x | 0.2500 | 0.0232 | 0.0503 |
| Llama-3.1-8B-Instruct | ruler32k | vt | perhead_bern | 4x | 0.2500 | 0.0017 | 0.0080 |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | vkvwr | 32x | 0.1232 | 0.0075 | 0.0276 |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | bernoulli | 8x | 0.1250 | 0.0378 | 0.0977 |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | kvzip | 8x | 0.1250 | 0.3343 | 0.4732 |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | perhead_bern | 8x | 0.1250 | 0.0059 | 0.0202 |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | vkvwr | 16x | 0.1646 | 0.0064 | 0.0247 |
| Llama-3.1-8B-Instruct | scbench | scbench_prefix_suffix | vkvwr | 8x | 0.2206 | 0.0054 | 0.0183 |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | perhead_bern | 16x | 0.0625 | 0.0146 | 0.0363 |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | kvzip | 16x | 0.0625 | 0.5169 | 0.4487 |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | bernoulli | 16x | 0.0625 | 0.1167 | 0.1900 |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | vkvwr | 200x | 0.0994 | 0.0177 | 0.0456 |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | perhead_bern | 8x | 0.1250 | 0.0079 | 0.0209 |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | bernoulli | 8x | 0.1250 | 0.0438 | 0.0985 |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | kvzip | 8x | 0.1250 | 0.2798 | 0.2672 |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | vkvwr | 64x | 0.1496 | 0.0138 | 0.0368 |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | vkvwr | 32x | 0.1748 | 0.0118 | 0.0326 |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | vkvwr | 16x | 0.1971 | 0.0096 | 0.0274 |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | vkvwr | 8x | 0.2370 | 0.0065 | 0.0189 |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | perhead_bern | 4x | 0.2500 | 0.0034 | 0.0092 |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | kvzip | 4x | 0.2500 | 0.0940 | 0.0970 |
| Qwen3-4B-Instruct-2507 | ruler32k | cwe | bernoulli | 4x | 0.2500 | 0.0089 | 0.0279 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | perhead_bern | 16x | 0.0625 | 0.0158 | 0.0346 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | bernoulli | 16x | 0.0625 | 0.1286 | 0.1686 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | kvzip | 16x | 0.0625 | 0.5885 | 0.5481 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | vkvwr | 200x | 0.0950 | 0.0199 | 0.0426 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | perhead_bern | 8x | 0.1250 | 0.0090 | 0.0216 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | bernoulli | 8x | 0.1250 | 0.0356 | 0.0750 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | kvzip | 8x | 0.1250 | 0.3120 | 0.2905 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | vkvwr | 64x | 0.1335 | 0.0166 | 0.0370 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | vkvwr | 32x | 0.1528 | 0.0145 | 0.0356 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | vkvwr | 16x | 0.1736 | 0.0127 | 0.0332 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | vkvwr | 8x | 0.2109 | 0.0088 | 0.0239 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | perhead_bern | 4x | 0.2500 | 0.0043 | 0.0110 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | kvzip | 4x | 0.2500 | 0.1093 | 0.1041 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_multikey_2 | bernoulli | 4x | 0.2500 | 0.0066 | 0.0219 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | bernoulli | 16x | 0.0625 | 0.1579 | 0.2453 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | kvzip | 16x | 0.0625 | 0.6440 | 0.6489 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | perhead_bern | 16x | 0.0625 | 0.0201 | 0.0542 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | vkvwr | 200x | 0.1186 | 0.0214 | 0.0568 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | perhead_bern | 8x | 0.1250 | 0.0116 | 0.0355 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | kvzip | 8x | 0.1250 | 0.3374 | 0.3096 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | bernoulli | 8x | 0.1250 | 0.0838 | 0.1537 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | vkvwr | 64x | 0.1839 | 0.0159 | 0.0468 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | vkvwr | 32x | 0.2193 | 0.0143 | 0.0429 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | perhead_bern | 4x | 0.2500 | 0.0053 | 0.0165 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | kvzip | 4x | 0.2500 | 0.1032 | 0.1106 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | bernoulli | 4x | 0.2500 | 0.0163 | 0.0508 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | vkvwr | 16x | 0.2531 | 0.0112 | 0.0344 |
| Qwen3-4B-Instruct-2507 | ruler32k | niah_single_3 | vkvwr | 8x | 0.2910 | 0.0087 | 0.0278 |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | perhead_bern | 16x | 0.0625 | 0.0076 | 0.0228 |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | kvzip | 16x | 0.0625 | 0.1016 | 0.1060 |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | bernoulli | 16x | 0.0625 | 0.0115 | 0.0367 |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | vkvwr | 200x | 0.0875 | 0.0121 | 0.0368 |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | perhead_bern | 8x | 0.1250 | 0.0037 | 0.0112 |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | bernoulli | 8x | 0.1250 | 0.0058 | 0.0195 |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | kvzip | 8x | 0.1250 | 0.0674 | 0.0724 |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | vkvwr | 64x | 0.1266 | 0.0071 | 0.0224 |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | vkvwr | 32x | 0.1449 | 0.0053 | 0.0171 |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | vkvwr | 16x | 0.1641 | 0.0041 | 0.0133 |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | vkvwr | 8x | 0.2029 | 0.0028 | 0.0092 |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | perhead_bern | 4x | 0.2500 | 0.0015 | 0.0048 |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | kvzip | 4x | 0.2500 | 0.0335 | 0.0383 |
| Qwen3-4B-Instruct-2507 | ruler32k | vt | bernoulli | 4x | 0.2500 | 0.0024 | 0.0080 |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | vkvwr | 32x | 0.1224 | 0.0041 | 0.0126 |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | kvzip | 8x | 0.1250 | 0.1383 | 0.1913 |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | bernoulli | 8x | 0.1250 | 0.0176 | 0.0422 |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | perhead_bern | 8x | 0.1250 | 0.0017 | 0.0049 |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | vkvwr | 16x | 0.1561 | 0.0028 | 0.0090 |
| Qwen3-4B-Instruct-2507 | scbench | scbench_prefix_suffix | vkvwr | 8x | 0.2067 | 0.0016 | 0.0053 |

## C7: is the win the estimator or the allocation?

perhead_bern is the shipped Bernoulli estimator with a per-head budget instead of one global c. Against bernoulli it isolates the ALLOCATION; against vkvwr at matched realised memory it isolates the ESTIMATOR.

| model | dataset | ratio | comparison | mean diff | SE | t | contexts |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | ruler32k | 4x | allocation (perhead - global) | +0.0118 | 0.0021 | +5.69 | 80 |
| Llama-3.1-8B-Instruct | ruler32k | 8x | allocation (perhead - global) | -0.0611 | 0.0050 | -12.16 | 90 |
| Llama-3.1-8B-Instruct | ruler32k | 8x | estimator (perhead - vkvwr) | +0.0378 | 0.0033 | +11.45 | 90 |
| Llama-3.1-8B-Instruct | ruler32k | 16x | allocation (perhead - global) | -0.1538 | 0.0111 | -13.87 | 80 |
| Llama-3.1-8B-Instruct | ruler32k | 16x | estimator (perhead - vkvwr) | +0.0515 | 0.0045 | +11.40 | 80 |

## C1 / C2: paired against top-k, clustered at the context

Negative means the sampled arm has LOWER error, i.e. it wins.

| metric | arm | ratio | mean diff vs top-k | SE | t | p | Holm | contexts | head-rows | win rate |
|---|---|---|---|---|---|---|---|---|---|---|
| den_err | bernoulli | 4x | -0.1334 | 0.0055 | -24.28 | 2.17e-38 | 2.04e-37 | 80 | 348,160 | 96.69% |
| den_err | bernoulli | 8x | -0.2358 | 0.0088 | -26.92 | 1.536e-44 | 3.686e-43 | 90 | 565,760 | 94.14% |
| den_err | bernoulli | 16x | -0.2888 | 0.0107 | -27.03 | 1.12e-41 | 1.904e-40 | 80 | 348,160 | 94.24% |
| den_err | perhead_bern | 4x | -0.1202 | 0.0048 | -24.87 | 4.072e-39 | 5.701e-38 | 80 | 348,160 | 85.80% |
| den_err | perhead_bern | 8x | -0.2473 | 0.0094 | -26.39 | 7.45e-44 | 1.564e-42 | 90 | 565,760 | 90.77% |
| den_err | perhead_bern | 16x | -0.3488 | 0.0148 | -23.64 | 1.396e-37 | 9.769e-37 | 80 | 348,160 | 94.76% |
| den_err | vkvwr | 8x | -0.2703 | 0.0104 | -25.87 | 3.586e-43 | 7.172e-42 | 90 | 565,760 | 94.79% |
| den_err | vkvwr | 16x | -0.3816 | 0.0170 | -22.49 | 4.43e-36 | 2.329e-35 | 80 | 348,160 | 97.89% |
| attn_err | bernoulli | 4x | -0.1400 | 0.0057 | -24.46 | 1.313e-38 | 1.444e-37 | 80 | 348,160 | 90.90% |
| attn_err | bernoulli | 8x | -0.2182 | 0.0089 | -24.60 | 1.834e-41 | 2.935e-40 | 90 | 565,760 | 88.69% |
| attn_err | bernoulli | 16x | -0.2349 | 0.0091 | -25.93 | 2.161e-40 | 3.242e-39 | 80 | 348,160 | 84.48% |
| attn_err | perhead_bern | 4x | -0.1282 | 0.0058 | -22.20 | 1.049e-35 | 4.196e-35 | 80 | 348,160 | 79.59% |
| attn_err | perhead_bern | 8x | -0.2793 | 0.0110 | -25.29 | 2.108e-42 | 4.006e-41 | 90 | 565,760 | 86.51% |
| attn_err | perhead_bern | 16x | -0.3887 | 0.0160 | -24.31 | 2.04e-38 | 2.04e-37 | 80 | 348,160 | 89.98% |
| attn_err | vkvwr | 8x | -0.3170 | 0.0120 | -26.43 | 6.616e-44 | 1.456e-42 | 90 | 565,760 | 92.23% |
| attn_err | vkvwr | 16x | -0.4401 | 0.0177 | -24.80 | 4.963e-39 | 6.452e-38 | 80 | 348,160 | 95.81% |
| num_err | bernoulli | 4x | -0.1309 | 0.0058 | -22.53 | 3.881e-36 | 2.329e-35 | 80 | 348,160 | 89.96% |
| num_err | bernoulli | 8x | -0.1837 | 0.0094 | -19.54 | 5.885e-34 | 1.766e-33 | 90 | 565,760 | 86.98% |
| num_err | bernoulli | 16x | -0.1593 | 0.0088 | -18.11 | 7.628e-30 | 1.526e-29 | 80 | 348,160 | 81.24% |
| num_err | perhead_bern | 4x | -0.1100 | 0.0061 | -17.95 | 1.313e-29 | 1.526e-29 | 80 | 348,160 | 79.64% |
| num_err | perhead_bern | 8x | -0.2389 | 0.0107 | -22.33 | 3.094e-38 | 2.475e-37 | 90 | 565,760 | 86.81% |
| num_err | perhead_bern | 16x | -0.3245 | 0.0132 | -24.66 | 7.369e-39 | 8.842e-38 | 80 | 348,160 | 90.32% |
| num_err | vkvwr | 8x | -0.2862 | 0.0108 | -26.52 | 5.014e-44 | 1.153e-42 | 90 | 565,760 | 92.26% |
| num_err | vkvwr | 16x | -0.3867 | 0.0142 | -27.32 | 5.228e-42 | 9.411e-41 | 80 | 348,160 | 95.65% |
