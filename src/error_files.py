from collections import defaultdict

mpii_error_slices = defaultdict(set)
mpii_error_slices['Bust'] = {'nl_6019a', 'nl_5595a', 'csr4239a', 'CSR0094A', 'CSR1289A','CSR1226A', 'CSR1334A', 'CSR1622A', 'CSR1742A', 'CSR2825A', 'CSR2833A', 'CSR2845A'}

mpii_error_slices['Aux_UnderBust_Bust_0'] = {'csr4793a','CSR0017A','CSR1178A','CSR1444A','CSR1473A','CSR1604A','CSR1721A','CSR1827A','CSR1906A','CSR2149A','CSR2300A',
                                             'CSR2603A','CSR2825A','CSR2834A','CSR2862A','CSR2997A','CSR3010A','CSR3027A','nl_1088a','nl_1229a''nl_5218a','nl_5949a'}

ucsc_error_slices  = defaultdict(set)
ucsc_error_slices['UnderBust'] = {'SPRING1894', 'SPRING2391', 'SPRING2622'}
